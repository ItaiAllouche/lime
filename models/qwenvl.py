from transformers import AutoTokenizer
import torch
import torch.nn as nn
import gc
import copy

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  
from descriptor import LimeDesc
from modeling.modeling_qwen_lime import QWenLMHeadModel, make_context, get_stop_words_ids, decode_tokens
from lrp.patch_qwenvl import patch_qwenvl
from utils import compare_model_to_reference_qwenvl

torch.manual_seed(42)

QWEN_VL_ID = "Qwen/Qwen-VL-Chat"
class QwenVLLIME(nn.Module):
    def __init__(
        self,
        verbose: bool = True
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_VL_ID, trust_remote_code=True)
        self.model = QWenLMHeadModel.from_pretrained(
            QWEN_VL_ID,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        # # create reference model AFTER loading weights
        self.model.reference_transformer = copy.deepcopy(self.model.transformer)
        self.model.reference_transformer.eval()

        if self.model.config.bf16:
                self.model.reference_transformer.bfloat16()
        if self.model.config.fp16:
                self.model.reference_transformer.half()
    
        for p in self.model.reference_transformer.parameters():
            p.requires_grad = False

        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False
            
        # lrp patching
        patch_qwenvl(verbose)

        # validate equality of between model and reference model
        compare_model_to_reference_qwenvl(self.model)

    def get_inputs_for_forward(
        self,
        instruction: str,
        image_path: str,
        device_num: int = 0
    ):
        query = self.tokenizer.from_list_format([
            {'image': image_path},
            {'text': instruction},
        ])

        return query
    
    def _outputs_for_relevance(
        self,
        inputs: dict,
        desc: LimeDesc,
        position_ids
    ):
        model = self.model
        stash = {}

        def _cap_merged_hidden(mod, args, kwargs):
            # args[0] is hidden_states entering block 0
            x = args[0]
            if torch.is_tensor(x) and "merged" not in stash:
                x.requires_grad_(True)
                x.retain_grad()
                stash["merged"] = x
                args = (x,) + args[1:]
            return args, kwargs

        # capture at first transformer block (not at transformer kwargs inputs_embeds)
        h = model.transformer.h[0].register_forward_pre_hook(_cap_merged_hidden, with_kwargs=True)

        model.zero_grad(set_to_none=True)
        outputs = model(
            **inputs,
            use_cache=False,
            output_attentions=False,
            position_ids=position_ids,
            desc=desc
        )

        h.remove()

        if "merged" not in stash:
            raise RuntimeError("Failed to capture merged hidden states in block 0.")

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
                    for prev_token_id in set(input_ids[b].tolist()):
                        next_token_logits[b, prev_token_id] /= repetition_penalty

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
        inputs: str,
        opt_steps: int = 7,
        opt_lr: float = 0.0004,
        lambda_kl: float = 0.1,
        deltas_layers: list =list(range(0,32)), # Qwen-VL-Chat has 32 decoder layers
        max_new_tokens: int = 50, 
        plot: bool = False,
        output_relevance: bool = False        
    ):
        relevances = [] if output_relevance else None
        
        device = next(self.model.parameters()).device

        generation_config = self.model.generation_config
    
        raw_text, context_tokens = make_context(
            self.tokenizer,
            query=inputs,
            history=None,
            system="You are a helpful assistant.",
            max_window_size=generation_config.max_window_size,
            chat_format=generation_config.chat_format,
        )

        # get stop words for proper termination
        stop_words_ids = get_stop_words_ids(
            generation_config.chat_format, 
            self.tokenizer
        )

        # use context_tokens directly (already properly tokenized by make_context)
        input_ids = torch.tensor([context_tokens]).to(device)
        T = len(context_tokens)  # prompt length from context_tokens
        raw_text_len = len(raw_text)
        context_length = len(context_tokens)      
        
        head_dim = self.model.config.hidden_size // \
                   self.model.config.num_attention_heads
        
        # get image token boundaries from visual config
        visual_config = self.model.config.visual
        image_start_id = visual_config['image_start_id']
        image_size = visual_config['image_size']
        patch_size = visual_config['patch_size']
        num_image_tokens = (image_size // patch_size) ** 2 + 1  # +1 for class token
        
        # fsind modality token positions
        input_ids_list = input_ids[0].tolist()
        modality_bos_idx = input_ids_list.index(image_start_id)
        modality_eos_idx = modality_bos_idx + num_image_tokens
        modality_eos_idx = min(modality_eos_idx, len(input_ids_list) - 1)
        
        if plot:
            print(f'modality_bos_idx: {modality_bos_idx} | modality_eos_idx: {modality_eos_idx}')
        
        # initialize trainable KV deltas per layer
        kv_deltas = {}
        delta_params = []
        for layer_idx in deltas_layers:
            delta_k = torch.zeros(
                1,
                self.model.transformer.config.num_attention_heads,
                input_ids.shape[1],
                head_dim,
                device=device,
                requires_grad=True
            )
            delta_v = torch.zeros(
                1,
                self.model.transformer.config.num_attention_heads,
                input_ids.shape[1],
                head_dim,
                device=device,
                requires_grad=True
            )            
            kv_deltas[layer_idx] = (delta_k, delta_v)
            delta_params.extend([delta_k, delta_v])

        self.model.eval()

        # Adam optimizer for deltas KV tuning
        optimizer = torch.optim.Adam(delta_params, lr=opt_lr)

        # Generation parameters
        do_sample = getattr(generation_config, 'do_sample', False)
        temperature = getattr(generation_config, 'temperature', 1.0) if do_sample else 1.0
        repetition_penalty = getattr(generation_config, 'repetition_penalty', 1.0) if do_sample else 1.0
        top_k = getattr(generation_config, 'top_k', 0) if do_sample else 0
        top_p = getattr(generation_config, 'top_p', 1.0) if do_sample else 1.0

        # Initialize KVOpt description
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
        
        eos_id = self.tokenizer.eos_token_id
        
        # Generation process
        for step in range(max_new_tokens):
            if plot:
                print(f'\n---------------------')
                print(f'Generation step: {step}')

            # optimize K and V by Adam opt
            for opt_step in range(opt_steps):
                
                # inputs for current forward
                attention_mask = torch.ones_like(input_ids, device=device)
                position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                # Forward + optimization step
                optimizer.zero_grad()

                if plot:
                    print(f'Adam step: {opt_step}')
                
                # Compute relevance using LRP
                outputs, stash = self._outputs_for_relevance(
                    inputs=model_inputs,
                    desc=desc, 
                    position_ids=position_ids
                )
                kl_loss = desc.kl_loss

                assert kl_loss is not None, "KL loss is None"

                nonpad_idx = torch.where(model_inputs['attention_mask'][0].to(torch.bool))[0].tolist()
                target_pos = nonpad_idx[-1]  # last non-padded token

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
                    create_graph=True,
                    allow_unused=False,
                )[0]        

                # Token-level relevance as grad * input
                relevance = (grad_merged[0] * merged[0]).sum(-1)

                if output_relevance and opt_step + 1 == opt_steps:
                    relevances.append(relevance.detach().to(torch.float32).cpu().numpy())  

                tau = 0.1
                log_probs = nn.functional.log_softmax(relevance / tau, dim=-1)
                relevance_loss = -(log_probs[desc.modality_bos_idx:desc.modality_eos_idx+1].mean())
                loss = lambda_kl * kl_loss + relevance_loss

                if plot:
                    print(f'KL: {kl_loss.item():.4f} | Image Relevance: {float(-relevance_loss):.4f} | Overall loss: {loss.item():.4f}')
                
                loss.backward()
                optimizer.step()

            # Decode next token from last logits
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

            # Check if next token is in stop words
            if stop_words_ids:
                is_stop = False
                for stop_word_id in stop_words_ids:
                    if next_token[0, 0].item() == stop_word_id:
                        is_stop = True
                        break
                if is_stop:
                    print(f"Stopping: Hit stop word token")
                    break
            
            # Check if EOS
            if eos_id is not None and next_token[0, 0].item() == eos_id:
                if plot:
                    print(f"Stopping: Hit EOS token")
                break            
            
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode current answer
            answer_ids = input_ids[:, T:]
            decoded_answer = self.tokenizer.batch_decode(
                answer_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]            

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

            # Reinitialize KV deltas
            kv_deltas = {}
            delta_params = []
            for layer_idx in deltas_layers:
                delta_k = torch.zeros(
                    1,
                    self.model.transformer.config.num_attention_heads,
                    input_ids.shape[1],
                    head_dim,
                    device=device,
                    requires_grad=True
                )
                delta_v = torch.zeros(
                    1,
                    self.model.transformer.config.num_attention_heads,
                    input_ids.shape[1],
                    head_dim,
                    device=device,
                    requires_grad=True
                )            
                kv_deltas[layer_idx] = (delta_k, delta_v)
                delta_params.extend([delta_k, delta_v])

            optimizer = torch.optim.Adam(delta_params, lr=opt_lr)
            desc.set_kv_deltas(kv_deltas)

        gen_ids = input_ids[0].tolist()
        response = decode_tokens(
            gen_ids,
            self.tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            chat_format=generation_config.chat_format,
            verbose=False,
            errors='replace'
        )

        generated_token_ids = gen_ids[0].detach().cpu().tolist()
        generated_tokens = self.processor.tokenizer.convert_ids_to_tokens(generated_token_ids)        

        if plot:
            print(f"Model's output: {response}")
        
        output = {
            'response': response,
            'relevance': relevances,
            'modality_bos_idx': modality_bos_idx,
            'modality_eos_idx': modality_eos_idx,
            'tokens': generated_tokens
        }
        
        return output