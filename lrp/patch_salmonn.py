from lxt.efficient.patches import patch_method, patch_attention, rms_norm_forward, gated_mlp_forward

import sys
sys.path.append('/app/dev/models')
from models.speechlms.salmonn7b import modeling_llama_kv_opt_salmonn

def patch_salmonn7b(verbose: bool = False): 
    if not getattr(modeling_llama_kv_opt_salmonn, "_lxt_attn_patched", False):
        success = patch_attention(modeling_llama_kv_opt_salmonn)
        setattr(modeling_llama_kv_opt_salmonn, "_lxt_attn_patched", True)

        if verbose and success:
            print(f"Patched {modeling_llama_kv_opt_salmonn.__name__}")

    success = patch_method(gated_mlp_forward, modeling_llama_kv_opt_salmonn.LlamaMLP)
    
    if verbose and success:
        print("Patched LlamaMLP")

    success = patch_method(rms_norm_forward, modeling_llama_kv_opt_salmonn.LlamaRMSNorm)

    if verbose and success:
        print("Patched LlamaRMSNorm")
