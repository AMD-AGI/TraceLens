###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic CSV-driven harness for ``--from-report-dir`` mode.

For ops whose entry in ``OP_CALL_SPEC`` does *not* set
``use_existing_harness=True``, the parent ``validate_perf_model.py`` doesn't
know specific dimensional defaults; instead it forwards the raw ``Input
Dims`` and ``Input type`` columns from the trace's
``unified_perf_summary.csv`` and lets ``test_generic_simple_op`` build random
tensors of the right shapes and dtypes, then dispatch to the registered
callable here.

This replaces the runtime string template that
``generate_harness_from_csv`` previously emitted.
"""

import json

import torch

C10_TO_TORCH = {
    "c10::Float8_e4m3fnuz": torch.float8_e4m3fnuz,
    "c10::Float8_e4m3fn": torch.float8_e4m3fn,
    "c10::Float8_e5m2fnuz": torch.float8_e5m2fnuz,
    "c10::Float8_e5m2": torch.float8_e5m2,
    "c10::BFloat16": torch.bfloat16,
    "c10::Half": torch.float16,
    "c10::Float": torch.float32,
    "c10::Double": torch.float64,
    "float": torch.float32,
    "unsigned char": torch.uint8,
    "unsigned short": torch.int16,
    "int": torch.int32,
    "long": torch.int64,
}
_FP8_DTYPES = {
    torch.float8_e4m3fnuz,
    torch.float8_e4m3fn,
    torch.float8_e5m2fnuz,
    torch.float8_e5m2,
}


def _call_gemm_a8w8_blockscale(t):
    import aiter
    return aiter.gemm_a8w8_blockscale(t[0], t[1], t[2], t[3])


def _call_gemm_a16w16_asm(t):
    from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm
    return gemm_a16w16_asm(t[0], t[1], t[2])


def _call_aiter_silu_and_mul(t):
    import aiter
    return aiter.silu_and_mul(t[0], t[1])


def _call_aiter_gelu_and_mul(t):
    import aiter
    return aiter.gelu_and_mul(t[0], t[1])


def _call_aiter_gelu_tanh_and_mul(t):
    import aiter
    return aiter.gelu_tanh_and_mul(t[0], t[1])


def _call_aiter_rms_norm(t):
    import aiter
    return aiter.rms_norm(t[0], t[1], 1e-06)


def _call_aiter_fused_add_rms_norm(t):
    import aiter
    return aiter.fused_add_rms_norm_cu(t[0].clone(), t[1].clone(), t[4], 1e-06)


def _call_rmsnorm_dynamicquant(t):
    from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_dynamicquant
    return rmsnorm2d_fwd_with_dynamicquant(t[0], t[1], 1e-06)


def _call_dynamic_per_token_scaled_quant(t):
    import aiter
    return aiter.dynamic_per_token_scaled_quant(t[0], t[1], t[2])


def _call_flash_attn_func(t):
    import aiter
    return aiter.flash_attn_func(t[0], t[1], t[2], causal=True)


def _call_vllm_triton_group_quant_fp8(t):
    from vllm.model_executor.layers.quantization.utils import fp8_utils  # noqa: F401
    return torch.ops.vllm.rocm_aiter_triton_per_token_group_quant_fp8(t[0], t[1], t[2])


def _call_vllm_rmsnorm_fp8_group_quant(t):
    import vllm._aiter_ops  # noqa: F401
    return torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant(t[0], t[1], t[2], t[3], 1e-06, 128)


def _call_vllm_rmsnorm_add_fp8_group_quant(t):
    import vllm._aiter_ops  # noqa: F401
    return torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant(
        t[0], t[1], t[2], t[3], t[4], 1e-06, 128
    )


OP_CALL_SPEC = {
    "gemm_a8w8_blockscale": {"call": _call_gemm_a8w8_blockscale, "output_indices": [4], "skip_indices": []},
    "gemm_a16w16_atomic_": {"call": _call_gemm_a16w16_asm, "output_indices": [2], "skip_indices": []},
    "silu_and_mul": {"call": _call_aiter_silu_and_mul, "output_indices": [0], "skip_indices": []},
    "gelu_and_mul": {"call": _call_aiter_gelu_and_mul, "output_indices": [0], "skip_indices": []},
    "gelu_tanh_and_mul": {"call": _call_aiter_gelu_tanh_and_mul, "output_indices": [0], "skip_indices": []},
    "rms_norm": {"call": _call_aiter_rms_norm, "output_indices": [], "skip_indices": [2, 3]},
    "add_rmsnorm": {"call": _call_aiter_fused_add_rms_norm, "output_indices": [2, 3], "skip_indices": [5, 6]},
    "rmsnorm_dynamicquant": {"call": _call_rmsnorm_dynamicquant, "output_indices": [], "skip_indices": [2, 3, 4]},
    "dynamic_per_token_scaled_quant": {
        "call": _call_dynamic_per_token_scaled_quant,
        "output_indices": [0, 2],
        "skip_indices": [3, 4, 5, 6],
    },
    "_flash_attn_forward": {"call": _call_flash_attn_func, "output_indices": [], "skip_indices": []},
    "vllm_unquantized_gemm": {"call": _call_gemm_a16w16_asm, "output_indices": [2], "skip_indices": []},
    "vllm_triton_gemm_a8w8_blockscale": {"call": _call_gemm_a8w8_blockscale, "output_indices": [4], "skip_indices": []},
    "vllm_triton_group_quant_fp8": {
        "call": _call_vllm_triton_group_quant_fp8,
        "output_indices": [0, 2],
        "skip_indices": [3, 4, 5, 6],
    },
    "vllm_rmsnorm_fp8_group_quant": {
        "call": _call_vllm_rmsnorm_fp8_group_quant,
        "output_indices": [0, 2],
        "skip_indices": [4, 5],
    },
    "vllm_rmsnorm_add_fp8_group_quant": {
        "call": _call_vllm_rmsnorm_add_fp8_group_quant,
        "output_indices": [0, 2],
        "skip_indices": [5, 6],
    },
}
USE_EXISTING_HARNESS = {
    "ck_moe_stage1",
    "ck_moe_stage2",
    "mha_varlen_fwd",
    "unified_attention",
    "fmha_v3_varlen_fwd",
    "wrapper_fmha_v3_fwd",
    "vllm_unified_attention",
    "vllm_gdn_attention_core",
    "fmoe_fp8_blockscale_g1u1",
    "moe_cktile2stages_gemm1_ck",
    "moe_cktile2stages_gemm2_ck",
}


def _make_tensor(shape, dtype):
    """Allocate a random tensor of the given shape and torch dtype.

    FP8 dtypes can't be created directly via torch.randn, so we sample in
    float32 and cast. Integer dtypes use zeros (we don't care about content,
    only that the kernel sees a valid tensor with no NaNs).
    """
    device = "cuda"
    if dtype in _FP8_DTYPES:
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.zeros(shape, dtype=dtype, device=device)


def _make_empty(shape, dtype):
    """Allocate an uninitialized tensor (used for output positions)."""
    return torch.empty(shape, dtype=dtype, device="cuda")


def test_generic_simple_op(input_dims_json, input_types_json, registry_key=None, num_warmup=3, **_):
    """Generic harness driven by CSV ``Input Dims`` / ``Input type`` columns.

    Looks up ``registry_key`` in :data:`OP_CALL_SPEC` to find the call dispatch
    function and the output / skip index sets, builds tensors of the exact
    shapes and dtypes from the CSV, and runs warmup + measured iterations.
    """
    if registry_key is None:
        raise ValueError("test_generic_simple_op requires --registry-key")
    spec = OP_CALL_SPEC.get(registry_key)
    if spec is None:
        raise ValueError(
            f"No OP_CALL_SPEC entry for registry_key='{registry_key}'. "
            f"Known keys: {sorted(OP_CALL_SPEC)}"
        )
    input_dims = json.loads(input_dims_json)
    input_types = json.loads(input_types_json)
    output_indices = set(spec.get("output_indices", []))
    skip_indices = set(spec.get("skip_indices", []))
    print(f"test: __generic__ registry_key={registry_key} n_inputs={len(input_dims)}", flush=True)
    t = {}
    for i, (dims, dtype_str) in enumerate(zip(input_dims, input_types)):
        if i in skip_indices:
            continue
        if not dims:
            continue
        if not dtype_str or dtype_str in ("Scalar", ""):
            continue
        torch_dtype = C10_TO_TORCH.get(dtype_str)
        if torch_dtype is None:
            continue
        shape = list(dims)
        if i in output_indices:
            t[i] = _make_empty(shape, torch_dtype)
        else:
            t[i] = _make_tensor(shape, torch_dtype)
    fn = spec["call"]
    for _ in range(num_warmup):
        fn(t)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    fn(t)
    torch.cuda.synchronize()
    print("test: done", flush=True)
