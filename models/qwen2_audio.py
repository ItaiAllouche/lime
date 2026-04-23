import os

# set HF cache BEFORE importing transformers
os.environ["HF_HOME"] = "/tmp/hf_cache_fresh"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

import torch
import torch.nn as nn
from transformers import AutoProcessor
import librosa
import gc

import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  
from modeling.modeling_qwen2_audio_lime import Qwen2AudioForConditionalGenerationLIME
from descriptor import LimeDesc
from lrp.patches import patch_qwen2_audio

QWEN2_AUDIO_ID = 'Qwen/Qwen2-Audio-7B-Instruct'
MAX_AUDIO_DURATION = 30
class Qwen2AudioLIME(nn.Module):  
    def __init__(
        self,
        verbose: bool = True
    ):
        super().__init__()
        self.model = Qwen2AudioForConditionalGenerationLIME.from_pretrained(
            QWEN2_AUDIO_ID,
            attn_implementation="eager",
        )
        
        # lrp patching
        patch_qwen2_audio(verbose)
        
        self.processor = AutoProcessor.from_pretrained(QWEN2_AUDIO_ID)
        self.audio_encoder = self.model.audio_tower
        
        # make teacher exact sanpshot of loaded LM - for KL 
        self.model.reference_model.load_state_dict(self.model.language_model.state_dict())
        self.model.reference_model.eval()
        for p in self.model.reference_model.parameters():
            p.requires_grad = False        
        
        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def get_inputs_for_forward(
        self,
        instruction: str,
        wav_path: str,
        device_num: int = 0,
    ):
        device = f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu'
        sr = self.processor.feature_extractor.sampling_rate
        audio_array, _ = librosa.load(wav_path, sr=sr)

        # trim audio to 30 sec - audio encoder limit
        if len(audio_array) > MAX_AUDIO_DURATION*sr:
            audio_array = audio_array[:MAX_AUDIO_DURATION*sr]

        # create conversation with proper audio token format
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": instruction},
            ]},
        ]

        prmopt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prmopt, audio=[audio_array], sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        return inputs
             
    def _outputs_for_relevance(
        self,
        inputs: dict,
        desc: LimeDesc,
        position_ids,
    ):
        model = self.model
        
        stash = {}
        def _cap_inputs_embeds(mod, args, kwargs):
            x = kwargs.get("inputs_embeds", None)
            if x is not None and "merged" not in stash:
                x.requires_grad_(True)
                x.retain_grad()
                stash["merged"] = x
                kwargs["inputs_embeds"] = x
            return args, kwargs

        h = model.language_model.register_forward_pre_hook(_cap_inputs_embeds, with_kwargs=True)

        model.zero_grad(set_to_none=True)
        outputs = model(
            **inputs,
            use_cache=False,
            output_attentions=False,
            position_ids=position_ids,
            desc=desc
        )

        h.remove()

        return outputs, stash     
   
    def _top_k_top_p_filtering(
        self,
        logits,
        top_k=0,
        top_p=1.0,
        filter_value=-float("inf")
    ):
        # logits: [batch, vocab]
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            min_vals = v[:, -1, None]
            logits = torch.where(logits < min_vals, torch.full_like(logits, filter_value), logits)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = nn.functional.softmax(sorted_logits, dim=-1)
            cumprobs = probs.cumsum(dim=-1)

            # mask tokens with cumulative prob > top_p
            mask = cumprobs > top_p
            # always keep at least one token
            mask[..., 0] = False

            sorted_logits = torch.where(mask, torch.full_like(sorted_logits, filter_value), sorted_logits)
            # unsort back
            logits = torch.full_like(logits, filter_value)
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        return logits
        
    def _decode(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
        top_k: float,
        top_p: float
    ):
        with torch.no_grad():
            next_token_logits = logits[:, -1, :] 

            # apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(next_token_logits.size(0)):
                    prev_tokens = input_ids[b]
                    # apply penalty only on previously generated tokens
                    prev_tokens_unique = prev_tokens.unique()
                    token_logits = next_token_logits[b, prev_tokens_unique]
                    negative = token_logits < 0
                    token_logits[negative] *= repetition_penalty
                    token_logits[~negative] /= repetition_penalty
                    next_token_logits[b, prev_tokens_unique] = token_logits

            # apply temperature
            next_token_logits = next_token_logits / temperature

            # apply top_k and top_p
            if do_sample:
                filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k, top_p)
                probs = nn.functional.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # greedy decoding
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            return next_token

    def generate(
        self,
        inputs: dict,
        opt_steps: int = 7,
        opt_lr: float = 0.0005,
        lambda_kl: float = 0.007,
        deltas_layers: list = list(range(0,32)),
        max_new_tokens: int = 50, 
        verbose: bool = False,
        output_relevance: bool = False
    ):    
        relevances = [] if output_relevance else None
        
        if verbose:
            print(f'Start generate response with:\nopt_steps={opt_steps} | opt_lr={opt_lr} | lambda_kl={lambda_kl}')

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        input_ids = inputs["input_ids"].to(device) 
        
        # features (audio inputs) stay fixed
        fixed_inputs = {
            "input_features": inputs.get("input_features", None),
            "feature_attention_mask": inputs.get("feature_attention_mask", None),
        }        
        
        _, T = inputs["input_ids"].shape
        head_dim = self.model.language_model.config.hidden_size // \
                   self.model.language_model.config.num_attention_heads
        
        # audio span indices
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
        modality_bos_idx = next(i for i, t in enumerate(tokens) if t == "<|audio_bos|>")
        modality_eos_idx = next(i for i, t in enumerate(tokens) if t == "<|audio_eos|>")
        eos_id = self.processor.tokenizer.eos_token_id
        if verbose:
            print(f'modality_bos_idx: {modality_bos_idx} | audio_eos_idx: {modality_eos_idx}')
        
        # initlizte trainable kv deltas per layer - not registerd in the model
        kv_deltas = {}
        delta_params = []
        for layer_idx in deltas_layers:
            delta_k = torch.zeros(
                1,
                self.model.language_model.config.num_attention_heads,
                input_ids.shape[1],
                head_dim,
                device=device,
                requires_grad=True,
            )
            delta_v = torch.zeros(
                1,
                self.model.language_model.config.num_attention_heads,
                input_ids.shape[1],
                head_dim,
                device=device,
                requires_grad=True,
            )            
            kv_deltas[layer_idx] = (delta_k, delta_v)
            delta_params.extend([delta_k, delta_v])

        self.model.eval()

        # adam optimizer for deltas kv tuning
        optimizer = torch.optim.Adam(delta_params, lr=opt_lr)

        # parameters for generation
        do_sample = self.model.generation_config.do_sample
        temperature = self.model.generation_config.temperature if do_sample else 1.0
        repetition_penalty = self.model.generation_config.repetition_penalty if do_sample else 1.0
        top_k = self.model.generation_config.top_k if do_sample else 0
        top_p = self.model.generation_config.top_p if do_sample else 1.0

        # initilize descriptor
        desc = LimeDesc(
            deltas_layers=deltas_layers,
            lambda_kl=lambda_kl,
            approach='opt',
            modality_bos_idx=modality_bos_idx,
            modality_eos_idx=modality_eos_idx,
            prompt_len=T,
        )
        desc.set_kv_deltas(kv_deltas)
        desc.set_reference_logits(None)
        
        # generation process
        for step in range(max_new_tokens):
            if verbose:
                print(f'\n---------------------')
                print(f'Generation step: {step}')

            # optimize k and v by Adam opt
            for opt_step in range(opt_steps):

                # inputs for current forward
                attention_mask = torch.ones_like(input_ids, device=device)
                position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    **fixed_inputs,
                }

                # forward + optimization step
                optimizer.zero_grad()

                if verbose:
                    print(f'Adam step: {opt_step}')
                
                # compute relevance using LRP
                outputs, stash = self._outputs_for_relevance(
                    inputs=model_inputs,
                    desc=desc,
                    position_ids=position_ids
                )
                kl_loss = desc.kl_loss

                assert kl_loss is not None, "KL loss is None"

                nonpad_idx = torch.where(model_inputs['attention_mask'][0].to(torch.bool))[0].tolist()
                target_pos = nonpad_idx[-1] # last non-padded token

                with torch.no_grad():
                    next_token = self._decode(
                        outputs.logits,
                        input_ids,
                        do_sample,
                        temperature,
                        repetition_penalty,
                        top_k,
                        top_p
                    )

                # next predicted token id at that position (for batch 0)
                next_id = int(next_token[0, 0].item())
                target_logit = outputs.logits[0, target_pos, next_id]
                merged = stash["merged"]

                grad_merged = torch.autograd.grad(
                    outputs=target_logit,
                    inputs=merged,
                    # retain_graph=True, # same as create_graph
                    create_graph=True,# false
                    allow_unused=False,
                )[0]

                # token-level relevance as grad * input
                relevance = (grad_merged[0] * merged[0]).sum(-1)

                # in the last opt step we save the normalized relevance and attnetions
                if output_relevance and opt_step + 1 == opt_steps:
                    # normalized_relevance = relevance / (relevance.abs().max().detach() + 1e-12)
                    relevances.append(relevance.detach().to(torch.float32).cpu().numpy())
                    
                tau = 0.1
                log_probs = nn.functional.log_softmax(relevance / tau, dim=-1)
                relevance_loss = - (log_probs[desc.modality_bos_idx:desc.modality_eos_idx+1].mean())
                loss = lambda_kl * kl_loss + relevance_loss

                if verbose:
                    print(f'KL: {kl_loss.item():.4f} | Audio Relevance: {float(-relevance_loss):.4f} | Overall loss: {loss.item():.4f}')

                loss.backward()
                optimizer.step()

            # decode next token from last logits
            with torch.no_grad():
                next_token = self._decode(
                    outputs.logits,
                    input_ids,
                    do_sample,
                    temperature,
                    repetition_penalty,
                    top_k,
                    top_p
                )

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # early stop if EOS everywhere
            if eos_id is not None and (next_token == eos_id).all():
                if verbose:
                    print(f'---------------------')
                break      

            # decode current answer
            answer_ids = input_ids[:, T:]
            decoded_answer = self.processor.batch_decode(
                answer_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]            

            if verbose:
                print(f"Partial answer: {decoded_answer}")
                print(f'---------------------')

            # cleanup
            desc.set_reference_logits(None)
            desc.kv_deltas_cleanup()
            kv_deltas.clear()
            delta_params.clear()

            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()

            del optimizer
            gc.collect()
            torch.cuda.empty_cache()

            # initilaize kv deltas
            kv_deltas = {}
            delta_params = []
            for layer_idx in deltas_layers:
                delta_k = torch.zeros(
                    1,
                    self.model.language_model.config.num_attention_heads,
                    input_ids.shape[1],
                    head_dim,
                    device=device,
                    requires_grad=True,
                )
                delta_v = torch.zeros(
                    1,
                    self.model.language_model.config.num_attention_heads,
                    input_ids.shape[1],
                    head_dim,
                    device=device,
                    requires_grad=True,
                )
                kv_deltas[layer_idx] = (delta_k, delta_v)
                delta_params.extend([delta_k, delta_v])

            optimizer = torch.optim.Adam(delta_params, lr=opt_lr)
            desc.set_kv_deltas(kv_deltas)
            
        # parse output
        gen_ids = input_ids[:, T:]
        response = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        generated_token_ids = gen_ids[0].detach().cpu().tolist()
        generated_tokens = self.processor.tokenizer.convert_ids_to_tokens(generated_token_ids)

        if verbose:
            print(f"Model's output: {response}" )
         
        output = {
            'response': response,
            'relevance': relevances,
            'modality_bos_idx': modality_bos_idx,
            'modality_eos_idx': modality_eos_idx,
            'tokens': generated_tokens         
        }
        
        return output


