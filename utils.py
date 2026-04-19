import torch
import gc

def print_cuda_mem(tag=""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    max_reserv = torch.cuda.max_memory_reserved() / 1024**2
    print(f"[{tag}] allocated={alloc:.1f} MiB | reserved={reserv:.1f} MiB | "
          f"peak_alloc={max_alloc:.1f} MiB | peak_reserved={max_reserv:.1f} MiB")

def clean_gpu_cache(
        device_num: int = 0,
        print_stats: bool = True,
        print_summary: bool = False
):
    if not torch.cuda.is_available():
        if print_stats:
            print("CUDA not available, nothing to clean.")
        return

    torch.cuda.set_device(device_num)

    # Wait for all pending kernels to finish
    torch.cuda.synchronize()

    # Run Python garbage collector to remove unreferenced objects
    gc.collect()

    # Release cached blocks back to CUDA
    torch.cuda.empty_cache()

    # Collect any inter-process (IPC) memory handles
    torch.cuda.ipc_collect()

    if print_stats:
        alloc_gb = torch.cuda.memory_allocated(device_num) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device_num) / (1024 ** 3)
        print(f"[GPU {device_num}] allocated: {alloc_gb:.3f} GB")
        print(f"[GPU {device_num}] reserved : {reserved_gb:.3f} GB")

    if print_summary:
        # This can be long, but very useful for debugging fragmentation
        print(torch.cuda.memory_summary(device=device_num, abbreviated=True))

def cleanup(
        optimizer: torch.optim.Adam,
        kv_deltas: dict,
        delta_params: list,
):
    del optimizer
    del kv_deltas
    del delta_params
    torch.cuda.empty_cache()

def compare_model_to_reference(model, verbose: bool = True):

    print("\n=== Weight Comparison ===")
    for name, param in model.model.named_parameters():
        ref_param = dict(model.reference_model.named_parameters())[name]
        are_equal = torch.allclose(param, ref_param, rtol=1e-5, atol=1e-8)

        if verbose:
            print(f"{name}: {'✓ Equal' if are_equal else '✗ Different'}")
        if not are_equal and verbose:
            print(f"  Max diff: {(param - ref_param).abs().max().item()}")
        # Only check first 5 for quick verification
        if list(model.model.named_parameters()).index((name, param)) >= 4:
            if verbose:
                print("... (showing first 5, all others follow same pattern)")
            break

    # Method 2: Comprehensive check - verify all parameters match
    if verbose:
        print("\n=== Comprehensive Check ===")
    all_match = True
    mismatch_count = 0
    total_params = 0

    for (name, param), (ref_name, ref_param) in zip(
        model.model.named_parameters(),
        model.reference_model.named_parameters()
    ):
        total_params += 1
        if name != ref_name:
            if verbose:
                print(f"✗ Name mismatch: {name} vs {ref_name}")
            all_match = False
            mismatch_count += 1
            continue
        
        if not torch.equal(param, ref_param):
            if not torch.allclose(param, ref_param, rtol=1e-5, atol=1e-8):
                if verbose:
                    print(f"✗ Weights differ for {name}")
                    print(f"  Max absolute difference: {(param - ref_param).abs().max().item()}")
                all_match = False
                mismatch_count += 1

    if verbose:
        print(f"\nTotal parameters checked: {total_params}")
        print(f"Mismatches: {mismatch_count}")
        print(f"Result: {'✓ All weights match!' if all_match else '✗ Some weights differ'}")

        # Method 3: Verify reference model is frozen
        print("\n=== Reference Model Frozen Check ===")
    ref_requires_grad = [p.requires_grad for p in model.reference_model.parameters()]

    if verbose:
        print(f"All reference params frozen: {not any(ref_requires_grad)}")
        print(f"Reference model in eval mode: {not model.reference_model.training}")

def compare_model_to_reference_qwenvl(model, verbose: bool = True):
    """
    Compare QwenVLKVOpt model.transformer with model.reference_transformer.
    Similar to compare_model_to_reference but for models using transformer attribute.
    """
    if verbose:
        print("\n=== Weight Comparison (QwenVL) ===")
    for name, param in model.transformer.named_parameters():
        ref_param = dict(model.reference_transformer.named_parameters())[name]
        are_equal = torch.allclose(param, ref_param, rtol=1e-5, atol=1e-8)

        if verbose:
            print(f"{name}: {'✓ Equal' if are_equal else '✗ Different'}")
        if not are_equal and verbose:
            print(f"  Max diff: {(param - ref_param).abs().max().item()}")
        # Only check first 5 for quick verification
        if list(model.transformer.named_parameters()).index((name, param)) >= 4:
            if verbose:
                print("... (showing first 5, all others follow same pattern)")
            break

    # Method 2: Comprehensive check - verify all parameters match
    if verbose:
        print("\n=== Comprehensive Check (QwenVL) ===")
    all_match = True
    mismatch_count = 0
    total_params = 0

    for (name, param), (ref_name, ref_param) in zip(
        model.transformer.named_parameters(),
        model.reference_transformer.named_parameters()
    ):
        total_params += 1
        if name != ref_name:
            if verbose:
                print(f"✗ Name mismatch: {name} vs {ref_name}")
            all_match = False
            mismatch_count += 1
            continue
        
        if not torch.equal(param, ref_param):
            if not torch.allclose(param, ref_param, rtol=1e-5, atol=1e-8):
                if verbose:
                    print(f"✗ Weights differ for {name}")
                    print(f"  Max absolute difference: {(param - ref_param).abs().max().item()}")
                all_match = False
                mismatch_count += 1

    if verbose:
        print(f"\nTotal parameters checked: {total_params}")
        print(f"Mismatches: {mismatch_count}")
        print(f"Result: {'✓ All weights match!' if all_match else '✗ Some weights differ'}")

        # Method 3: Verify reference model is frozen
        print("\n=== Reference Transformer Frozen Check (QwenVL) ===")
    ref_requires_grad = [p.requires_grad for p in model.reference_transformer.parameters()]

    if verbose:
        print(f"All reference params frozen: {not any(ref_requires_grad)}")
        print(f"Reference transformer in eval mode: {not model.reference_transformer.training}")