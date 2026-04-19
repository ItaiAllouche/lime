from lxt.efficient.patches import patch_method, patch_attention, rms_norm_forward, gated_mlp_forward

from models.modeling import modeling_qwen_lime

def patch_qwenvl(verbose: bool = False):
    if not getattr(modeling_qwen_lime, "_lxt_attn_patched", False):
        success = patch_attention(modeling_qwen_lime)
        setattr(modeling_qwen_lime, "_lxt_attn_patched", True)

        if verbose and success:
            print(f"Patched {modeling_qwen_lime.__name__}")

    success = patch_method(gated_mlp_forward, modeling_qwen_lime.QWenMLP)

    if verbose and success:
        print("Patched QWenMLP")

    success = patch_method(rms_norm_forward, modeling_qwen_lime.RMSNorm)

    if verbose and success:
        print("Patched RMSNorm")        
 