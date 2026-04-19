from lxt.efficient.patches import patch_method, patch_attention, rms_norm_forward, gated_mlp_forward

import sys
sys.path.append('/app/dev/models')
from models.visionlms.qwenvl import modeling_qwen_kv_opt

def patch_qwenvl(verbose: bool = False):
    if not getattr(modeling_qwen_kv_opt, "_lxt_attn_patched", False):
        success = patch_attention(modeling_qwen_kv_opt)
        setattr(modeling_qwen_kv_opt, "_lxt_attn_patched", True)

        if verbose and success:
            print(f"Patched {modeling_qwen_kv_opt.__name__}")

    success = patch_method(gated_mlp_forward, modeling_qwen_kv_opt.QWenMLP)

    if verbose and success:
        print("Patched QWenMLP")

    success = patch_method(rms_norm_forward, modeling_qwen_kv_opt.RMSNorm)

    if verbose and success:
        print("Patched RMSNorm")        
 