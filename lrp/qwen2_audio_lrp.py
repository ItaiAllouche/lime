import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Qwen2AudioConfig
from lxt.utils import pdf_heatmap, clean_tokens
from dev.lrp.utils import clean_gpu_cache

import sys
sys.path.append('/app/dev/lrp/')
from patches.qwen2_audio_patch import patch_qwen2_audio

MODEL_ID = 'Qwen/Qwen2-Audio-7B-Instruct'

def qwen2_audio_relevance(prompt: str, wav_path: str, target_token: int = -1):

    clean_gpu_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if (device == "cuda") else torch.float32

    def _update_cfg(cfg: Qwen2AudioConfig):
        if hasattr(cfg, "audio_config") and hasattr(cfg.audio_config, "_attn_implementation"):
            print(cfg.audio_config._attn_implementation)
            cfg.audio_config._attn_implementation = "eager"    
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "_attn_implementation"):
            cfg.text_config._attn_implementation = "eager" 

        if hasattr(cfg, "text_config"):
            if hasattr(cfg.text_config, "sliding_window"):
                cfg.text_config.sliding_window = None
            if hasattr(cfg.text_config, "use_cache"):
                cfg.text_config.use_cache = False

        return cfg

    # modify attention implementation from sdpa to eager
    cfg: Qwen2AudioConfig = _update_cfg(Qwen2AudioConfig.from_pretrained(MODEL_ID))
    model: Qwen2AudioForConditionalGeneration = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, config=cfg, device_map=device, torch_dtype=dtype).eval()

    # apply patching function to support lrp rules
    model = patch_qwen2_audio(model)

    # disable model grads - no need for lrp calculation
    for p in model.parameters():
        p.requires_grad_(False)

    # wraps a Qwen2Audio feature extractor and a Qwen2Audio
    # offers all the functionalities of WhisperFeatureExtractor and Qwen2TokenizerFast
    processor: AutoProcessor = AutoProcessor.from_pretrained(MODEL_ID)

    sr: int = processor.feature_extractor.sampling_rate
    audio_array, _ = librosa.load(wav_path, sr=sr)

    # create conversation with proper audio token format
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_array},
            {"type": "text", "text": prompt},
        ]},
    ]
   
   # tokenzie input
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio_array], sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()} # move to device

    # -----------------------
    fe_dev   = model.audio_tower.conv1.weight.device
    fe_dtype = model.audio_tower.conv1.weight.dtype
    inp_feats = inputs["input_features"].to(fe_dev, fe_dtype).detach()
    inp_feats.requires_grad_(True)
    inp_feats.retain_grad()
    inputs["input_features"] = inp_feats
    # -----------------------

    # capture merged (audio+text) embeddings sent to the LM
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

    # zero gradients before forward pass
    model.zero_grad(set_to_none=True)

    # forward pass
    outputs = model(
        input_ids=inputs.get("input_ids", None),
        input_features=inputs.get("input_features", None),
        attention_mask=inputs.get("attention_mask", None),
        feature_attention_mask=inputs.get("feature_attention_mask", None),
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    # remove hook
    h.remove()

    logits = outputs.logits
    attention_mask = outputs.attention_mask if getattr(outputs, "attention_mask", None) is not None else inputs.get("attention_mask")
    input_ids = inputs["input_ids"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    # --- resolve the actual time step to backprop from, based on target_token ---
    if attention_mask is not None:
        nonpad_idx = torch.where(attention_mask[0].to(torch.bool))[0].tolist()
    else:
        nonpad_idx = list(range(logits.shape[1]))

    if len(nonpad_idx) == 0:
        raise ValueError("No valid (non-padded) token positions found.")

    # Support negative indexing relative to non-padded span
    if target_token < 0:
        idx_in_nonpad = max(0, len(nonpad_idx) + target_token)  # e.g., -1 -> last nonpad
    else:
        idx_in_nonpad = min(target_token, len(nonpad_idx) - 1)  # clamp

    target_pos = nonpad_idx[idx_in_nonpad]

    next_id = int(torch.argmax(logits[0, target_pos], dim=-1).item())
    next_token = processor.tokenizer.convert_ids_to_tokens([next_id])[0]
    target_logit = logits[0, target_pos, next_id]

    # last_idx = attention_mask[0].nonzero()[target_token].item() if attention_mask is not None else (logits.shape[1] - 1)
    # next_id = int(torch.argmax(logits[0, last_idx], dim=-1).item())
    # next_token = processor.tokenizer.convert_ids_to_tokens([next_id])[0]
    # next_token_logit = logits[0, last_idx, next_id]

    print(f'Input tokens: {tokens}')
    print(f'Input tokens length: {len(tokens)}')
    print(f'logits shape: {logits.shape}')
    print(f'Next token: "{next_token}" | id: {next_id} | logit: {target_logit}')
    
    # backward pass (the relevance is initialized with the value of next_token_logit)
    target_logit.backward()

    merged = stash["merged"]                     

    # obtain and normalize relevance to be between -1 and 1
    relevance = (merged.grad[0] * merged[0]).sum(-1)
    relevance = relevance / (relevance.abs().max() + 1e-12)
    relevance = relevance.detach().to(torch.float32).cpu().numpy()

    # -----------------
    mel_rel = (inp_feats.grad[0] * inp_feats[0])
    mel_rel = mel_rel / (mel_rel.abs().max() + 1e-12)
    mel_rel = mel_rel.detach().float().cpu().numpy()
    # -----------------

  
    # remove special characters that are not compatible wiht LaTeX
    tokens = clean_tokens(tokens)
    print(f'Clear tokens: {tokens}')
    print(f'Mel relevance shape: {mel_rel.shape}')

    # same heatmap as pdf
    pdf_heatmap(tokens, relevance, path='/app/dev/figs/qwen2_audio_heatmap.pdf', backend='xelatex')

    return mel_rel