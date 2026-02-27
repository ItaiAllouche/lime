import torch
import torch.nn as nn
from typing import Literal, Optional
import soundfile as sf
import librosa

from .src.model import SALMONN
from ...descriptor import KVOptDesc

import sys
sys.path.append('/app/dev/')
# from lrp.patch_salmonn import patch_salmonn7b
from plot import plot_relevance

class Salmonn7B(nn.Module):
    def __init__(
            self,
            device_num: int = 0,
    ):
        super().__init__()
        self.device = f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu'
        self.model = SALMONN(
            ckpt='/app/dev/models/speechlms/salmonn7b/src/salmonn_7b_v0.pth',
            whisper_path='openai/whisper-large-v2',
            beats_path='/app/dev/models/speechlms/salmonn7b/src/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
            vicuna_path='lmsys/vicuna-7b-v1.5',
            lora=False,
            low_resource=False,
            device_num=device_num,
            lrp=False
    )
    
        self.model.to(self.device)
        self.model.eval()
    
    def get_inputs_for_forward(
            self,
            instruction: str,
            wav_path: str
    ):

        inputs = {'instruction': instruction, 'wav_path': wav_path}
        return inputs

    def generate(
            self,
            inputs: dict,
            plot: bool = False,
            max_new_tokens: int = 150
    ):
        # with torch.inference_mode():
        response = self.model.generate(
            wav_path=inputs['wav_path'],
            prompt=inputs['instruction'],
            device=self.device,
            max_length=max_new_tokens
        )[0]
        if plot:
            print(f"Model's output: {response}")
        return response
    
class Salmonn7BKVOpt(nn.Module):
    def __init__(
            self,
            device_num: int = 0,
            verbose: bool = False,
    ):
        super().__init__()
        self.device = f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu'
        self.model = SALMONN(
            ckpt='/app/dev/models/speechlms/salmonn7b/src/salmonn_7b_v0.pth',
            whisper_path='openai/whisper-large-v2',
            beats_path='/app/dev/models/speechlms/salmonn7b/src/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
            vicuna_path='lmsys/vicuna-7b-v1.5',
            lora=False,
            low_resource=False,
            device_num=device_num,
            lrp=True
        )

        self.model.llama_model.reference_model.load_state_dict(self.model.llama_model.model.state_dict())
        self.model.llama_model.reference_model.eval()        

        # lrp patching
        # patch_salmonn7b(verbose)
        
        self.model.to(self.device)
        self.model.eval()

        # freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def _outputs_for_relevance(
            self,
            inputs: dict,
            desc: KVOptDesc
        ):

        model = self.model
        
        stash = {}
        def _cap_inputs_embeds(mod, args, kwargs):
            x = kwargs.get("inputs_embeds", None)
            if x is not None and "merged" not in stash:
                x = x.clone()
                x.requires_grad_(True)
                x.retain_grad()
                stash["merged"] = x
                kwargs["inputs_embeds"] = x
            return args, kwargs

        h = model.llama_model.register_forward_pre_hook(_cap_inputs_embeds, with_kwargs=True)

        model.zero_grad(set_to_none=True)
        outputs = model(**inputs, use_cache=False, output_attentions=False, desc=desc)

        h.remove()

        return outputs, stash

    def get_inputs_for_forward(
            self,
            wav_path: str,
            instruction: str,
            prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
            device_num: int = 0
    ):
        device = f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu'
        
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        raw_wav = torch.from_numpy(wav).to(device).unsqueeze(0)

        # whisper
        spectrogram = self.model.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device)
        speech_embeds = self.model.speech_encoder(spectrogram, return_dict=True).last_hidden_state
    
        # beats
        audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
        audio_embeds, _ = self.model.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

        # auditory embeds
        speech_embeds = self.model.ln_speech(speech_embeds)
        audio_embeds = self.model.ln_audio(audio_embeds)
        audio_embeds = nn.functional.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # split frames
        B, T, C = speech_embeds.shape
        kernel = round(T * self.model.second_per_frame / 30.0)
        stride = round(T * self.model.second_stride / 30.0)
        kernel = (1, kernel)
        stride = (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = nn.functional.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        # Qformer
        query_tokens = self.model.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.model.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )

        speech_embeds = self.model.speech_llama_proj(query_output.last_hidden_state.to(self.model.speech_llama_proj.weight.dtype))
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.model.llama_model.model.model.embed_tokens if self.model.lora else self.model.llama_model.model.embed_tokens
        prompt_left, prompts_right = prompt_pattern.format(instruction).split('<SpeechHere>')
        prompt_left_ids = self.model.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_right_ids = self.model.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)

        bos_embeds = self.model.llama_model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.model.llama_tokenizer.bos_token_id
        ) if not self.model.lora else self.model.llama_model.model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=device,
            ) * self.model.llama_tokenizer.bos_token_id
        )

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        inputs = {
            'embeds': embeds,
            'atts': atts,
            'modality_bos_idx': bos_embeds.shape[1]+prompt_left_embeds.shape[1], 
            'modality_eos_idx': bos_embeds.shape[1]+prompt_left_embeds.shape[1]+speech_embeds.shape[1]
        }
        
        return inputs

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
            generated_ids: torch.Tensor,
            do_sample: bool,
            temperature: float,
            repetition_penalty: float,
            top_p: float
    ):
        with torch.no_grad():
            next_token_logits = logits[:, -1, :] 

            # apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(next_token_logits.size(0)):
                    prev_tokens = generated_ids[b]
                    # apply penalty only on previously generated tokens
                    prev_tokens_unique = prev_tokens.unique()
                    token_logits = next_token_logits[b, prev_tokens_unique]
                    negative = token_logits < 0
                    token_logits[negative] *= repetition_penalty
                    token_logits[~negative] /= repetition_penalty
                    next_token_logits[b, prev_tokens_unique] = token_logits

            # apply temperature
            next_token_logits = next_token_logits / temperature

            # apply top_k and top_p (note: top_k is not used in SALMONN, set to 0)
            if do_sample:
                filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k=0, top_p=top_p)
                probs = nn.functional.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # greedy decoding
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            return next_token
    
    def generate(
            self,
            embeds: torch.Tensor,
            atts: torch.Tensor,
            modality_bos_idx: int,
            modality_eos_idx: int,
            approach: Literal['opt', 'vanila'],
            deltas_layers: list = list(range(0,32)),
            opt_steps: int = 3,
            opt_lr: int = 1e-3,
            lambda_kl: float = 1,
            lambda_relevance_multimodal: float = 0.7,
            lambda_relevance_text: float = 0.7,
            example: str = '',
            max_new_tokens: int = 50,             
            plot: bool = False
    ):

        relevances = []

        device = embeds.device

        T = embeds.shape[1]
        head_dim = self.model.llama_model.config.hidden_size // \
                   self.model.llama_model.config.num_attention_heads
        
        print(f'modality_bos_idx: {modality_bos_idx} | modality_eos_idx: {modality_eos_idx}')
        
        # initlizte trainable kv deltas per layer - not registerd in the model 
        kv_deltas = {}
        delta_params = []
        for layer_idx in deltas_layers:
            delta_k = torch.zeros(
                1, 
                self.model.llama_model.config.num_attention_heads,
                embeds.shape[1],
                # modality_eos_idx-modality_bos_idx+1,
                head_dim,
                device=device,
                dtype=torch.float32,
                requires_grad=True
            )
            delta_v = torch.zeros(
                1,
                self.model.llama_model.config.num_attention_heads,
                embeds.shape[1],
                # modality_eos_idx-modality_bos_idx+1,
                head_dim,
                device=device,
                dtype=torch.float32,
                requires_grad=True
            )
            nn.init.normal_(delta_k, mean=0.0, std=1e-4)# new
            nn.init.normal_(delta_v, mean=0.0, std=1e-4)    
            kv_deltas[layer_idx] = (delta_k, delta_v)
            delta_params.extend([delta_k, delta_v])

        self.model.eval()

        # adam optimizer for deltas kv tuning
        optimizer = torch.optim.Adam(delta_params, lr=opt_lr)

        # parameters for generation
        do_sample = True
        temperature = 1.0
        repetition_penalty = 1.0
        top_p = 0.9
        eos_id = self.model.llama_tokenizer.eos_token_id

        # initilize KVOpt description
        desc = KVOptDesc(
            deltas_layers=deltas_layers,
            lambda_kl=lambda_kl,
            lambda_relevance_multimodal=lambda_relevance_multimodal,
            lambda_relevance_text=lambda_relevance_text,
            approach=approach,
            modality_bos_idx=modality_bos_idx,
            modality_eos_idx=modality_eos_idx,
            prompt_len=T,
        )
        desc.set_kv_deltas(kv_deltas)
        desc.set_reference_logits(None)

        # initialize tracking of generated token IDs (empty at start)
        generated_ids = torch.empty((embeds.shape[0], 0), dtype=torch.long, device=device)

        # get embedding layer for converting tokens to embeddings
        embed_tokens = self.model.llama_model.model.model.embed_tokens if self.model.lora else self.model.llama_model.model.embed_tokens

        # generation process
        for step in range(max_new_tokens):
            print(f'\n---------------------')
            print(f'Generation step: {step}')

            # optimize k and v by Adam opt
            for opt_step in range(opt_steps):
                model_inputs = {
                    'embeds': embeds.detach(),
                    'atts': atts
                }

                # forward + optimization step
                optimizer.zero_grad()

                if approach == 'opt':
                    print(f'Adam step: {opt_step}')

                     # new
                    with torch.amp.autocast('cuda', enabled=False):                    
                        outputs, stash = self._outputs_for_relevance(model_inputs, desc)
                        # new

                        # compute relevance using LRP
                        # outputs, stash = self._outputs_for_relevance(model_inputs, desc)
                        kl_loss = desc.kl_loss

                        assert kl_loss is not None, "KL loss is None when approach='opt'."                    

                        nonpad_idx = torch.where(model_inputs['atts'][0].to(torch.bool))[0].tolist()
                        target_pos = nonpad_idx[-1]  # last non-padded token       

                        with torch.no_grad():
                            next_token = self._decode(
                                outputs.logits,
                                generated_ids,
                                do_sample,
                                temperature,
                                repetition_penalty,
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
                        
                        # token-level relevance as grad * input, normalized to [-1, 1]
                        relevance = (grad_merged[0] * merged[0]).sum(-1)
                        # relevance = relevance / (relevance.abs().max().detach() + 1e-12)
                        audio_relevance = relevance[desc.modality_bos_idx:desc.modality_eos_idx+1]
                        text_relevance = torch.cat((relevance[:desc.modality_bos_idx], relevance[desc.modality_eos_idx+1:]))

                        audio_relevance = audio_relevance.abs().sum()
                        text_relevance = text_relevance.abs().sum()
                        audio_relevance = audio_relevance / (relevance.abs().sum() + 1e-12)
                        text_relevance = text_relevance / (relevance.abs().sum() + 1e-12)

                        
                        loss = lambda_kl * kl_loss - lambda_relevance_multimodal * audio_relevance + lambda_relevance_text * text_relevance
                        print(f'KL: {kl_loss.item():.4f} | Audio Relevance: {float(audio_relevance):.4f} | Text Relevance: {float(text_relevance):.4f} | Audio/Text:  {(float(audio_relevance)/float(text_relevance)):.4f} | Overall loss: {loss.item():.4f}')
                        print(f'λ_kl X kl: {float(lambda_kl * kl_loss):.4f} | λ_relevance_audio X audio_relevance: {float(lambda_relevance_multimodal * audio_relevance):.4f} | λ_relevance_text X text_relevance: {float(lambda_relevance_text * text_relevance):.4f}')
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(delta_params, max_norm=0.1)
                    optimizer.step()
                    
                # approach = vanila
                else:
                    outputs = self.model(**model_inputs, use_cache=False, desc=desc)                    

            # decode next token from last logits
            with torch.no_grad():
                next_token = self._decode(outputs.logits, generated_ids, do_sample, temperature, repetition_penalty, top_p)

            # here we do another forward pass to calculate the lrp relevance for plotting
            if plot:
                outputs_plot, stash_plot = self._outputs_for_relevance(model_inputs, desc)
                nonpad_idx = torch.where(model_inputs['atts'][0].to(torch.bool))[0].tolist()
                target_pos = nonpad_idx[-1]  # last non-padded token
                next_id = int(next_token[0, 0].item())
                target_logit = outputs_plot.logits[0, target_pos, next_id]
            
                merged = stash_plot["merged"]

                grad_merged = torch.autograd.grad(
                    outputs=target_logit,
                    inputs=merged,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            
                # token-level relevance as grad * input, normalized to [-1, 1]
                relevance = (grad_merged[0] * merged[0]).sum(-1)
                relevance = relevance / (relevance.abs().max().detach() + 1e-12)
                relevances.append(relevance.detach().to(torch.float32).cpu().numpy())
                plot_relevance(relevances, desc.modality_bos_idx, desc.modality_eos_idx+1, f'salmonn7B/{approach}_{example}')

            # track the generated token ID
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # convert token to embedding and concatenate to embeds
            next_token_embed = embed_tokens(next_token)
            embeds = torch.cat([embeds, next_token_embed], dim=1)
            atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

            # early stop if EOS everywhere
            if eos_id is not None and (next_token == eos_id).all():
                print(f'EOS reached at step {step}')
                break

            # decode current answer
            decoded_answer = self.model.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]            

            if plot:
                print(f"Partial answer: {decoded_answer}")
                print(f'---------------------')

            # resets for next token generation
            desc.set_reference_logits(None)
            kv_deltas = {}
            delta_params = []
            for layer_idx in deltas_layers:
                delta_k = torch.zeros(
                    1,
                    self.model.llama_model.config.num_attention_heads,
                    embeds.shape[1],
                    head_dim,
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True
                )
                delta_v = torch.zeros(
                    1,
                    self.model.llama_model.config.num_attention_heads,
                    embeds.shape[1],
                    head_dim,
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True
                )            
                kv_deltas[layer_idx] = (delta_k, delta_v)
                delta_params.extend([delta_k, delta_v])
            
            optimizer = torch.optim.Adam(delta_params, lr=opt_lr)
            desc.set_kv_deltas(kv_deltas)

        # parse output
        response = self.model.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if plot:
            print(f"\nModel's output: {response}")
        
        return response
