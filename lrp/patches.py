from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm
from lxt.efficient.patches import patch_method, patch_attention, rms_norm_forward, gated_mlp_forward

from models.modeling import modeling_qwen2_audio_lime, modeling_llama_lime, modeling_qwen2_5_vl_lime

def patch_qwen2_audio(verbose: bool = False): 
    if not getattr(modeling_qwen2_audio_lime, "_lxt_attn_patched", False):
        success = patch_attention(modeling_qwen2_audio_lime)
        setattr(modeling_qwen2_audio_lime, "_lxt_attn_patched", True)

        if not success:
            print(f"Failed to patch {modeling_qwen2_audio_lime.__name__}")

        if verbose and success:
            print(f"Patched {modeling_qwen2_audio_lime.__name__}")

    success = patch_method(gated_mlp_forward, Qwen2MLP)

    if not success:
        print(f"Failed to patch Qwen2MLP")

    if verbose and success:
        print("Patched Qwen2MLP")

    success = patch_method(rms_norm_forward, Qwen2RMSNorm)

    if not success:
        print(f"Failed to patch Qwen2RMSNorm")    

    if verbose and success:
        print("Patched Qwen2RMSNorm")

def patch_llava(verbose: bool = False): 
    if not getattr(modeling_llama_lime, "_lxt_attn_patched", False):
        success = patch_attention(modeling_llama_lime)
        setattr(modeling_llama_lime, "_lxt_attn_patched", True)
        
        if verbose and success:
            print(f"Patched {modeling_llama_lime.__name__}")        

    success = patch_method(gated_mlp_forward, modeling_llama_lime.LlamaMLP)

    if verbose and success:
        print("Patched LlamaMLP")

    success = patch_method(rms_norm_forward, modeling_llama_lime.LlamaRMSNorm)

    if verbose and success:
        print("Patched LlamaRMSNorm")

def patch_qwenvl2_5(verbose: bool = False):
    if not getattr(modeling_qwen2_5_vl_lime, "_lxt_attn_patched", False):
        success = patch_attention(modeling_qwen2_5_vl_lime)
        setattr(modeling_qwen2_5_vl_lime, "_lxt_attn_patched", True)

        if verbose and success:
            print(f"Patched {modeling_qwen2_5_vl_lime.__name__}")

    success = patch_method(gated_mlp_forward, modeling_qwen2_5_vl_lime.Qwen2MLP)

    if verbose and success:
        print("Patched Qwen2MLP")

    success = patch_method(rms_norm_forward, modeling_qwen2_5_vl_lime.Qwen2RMSNorm)

    if verbose and success:
        print("Patched Qwen2RMSNorm")


  