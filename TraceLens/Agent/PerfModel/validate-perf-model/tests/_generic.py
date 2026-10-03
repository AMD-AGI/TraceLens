###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generic CSV-driven harness for ``--from-report-dir`` mode.

For report rows whose op was not registered with ``csv_harness=True`` in
``perf_model_harnesses.py``, ``validate_perf_model.py`` forwards the raw
``Input Dims`` and ``Input type`` columns from the trace's
``unified_perf_summary.csv``. ``test_generic_simple_op`` builds random
tensors of those shapes and dtypes and dispatches to the op's
``OP_CALL_SPEC`` entry here.
"""

import inspect
import json

import torch

C10_TO_TORCH = {
    "c10::Float8_e4m3fnuz": torch.float8_e4m3fnuz,
    "c10::Float8_e4m3fn": torch.float8_e4m3fn,
    "c10::Float8_e5m2fnuz": torch.float8_e5m2fnuz,
    "c10::Float8_e5m2": torch.float8_e5m2,
    "c10::Float8_e8m0fnu": torch.float8_e8m0fnu,
    "c10::Float4_e2m1fn_x2": torch.float4_e2m1fn_x2,
    "c10::BFloat16": torch.bfloat16,
    "c10::Half": torch.float16,
    "c10::Float": torch.float32,
    "c10::Double": torch.float64,
    "float": torch.float32,
    "unsigned char": torch.uint8,
    "unsigned short": torch.int16,
    "short": torch.int16,
    "int": torch.int32,
    "unsigned int": torch.int32,
    "long": torch.int64,
    # Spellings emitted by the PyTorch profiler for 64-bit integer operands.
    "long int": torch.int64,
    "long unsigned int": torch.int64,
    "bool": torch.bool,
}
# Narrow float formats that torch.randn cannot produce directly; sample in
# float32 and cast instead.
_FP8_DTYPES = {
    torch.float8_e4m3fnuz,
    torch.float8_e4m3fn,
    torch.float8_e5m2fnuz,
    torch.float8_e5m2,
    torch.float8_e8m0fnu,
    torch.float4_e2m1fn_x2,
}


def _call_gemm_a8w8_blockscale(t):
    import aiter

    return aiter.gemm_a8w8_blockscale(t[0], t[1], t[2], t[3])


def _call_gemm_a16w16_asm(t):
    from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm

    # Some traces record the output operand as an empty shape (it is allocated
    # inside the op), so materialise it from the A/B shapes when it is absent.
    out = t.get(2)
    if out is None:
        out = torch.empty(
            (t[0].shape[0], t[1].shape[0]), dtype=t[0].dtype, device="cuda"
        )
    return gemm_a16w16_asm(t[0], t[1], out)


def _call_vllm_rocm_unquantized_gemm(t):
    import vllm  # noqa: F401
    import vllm.model_executor.layers.utils  # noqa: F401  (registers the op)

    # Dispatch through the real vLLM op so the skinny-GEMM / torch.mm decision
    # matches the trace; the aiter ASM kernel alone rejects some N values.
    return torch.ops.vllm.rocm_unquantized_gemm(t[0], t[1], t.get(2))


def _call_aiter_silu_and_mul(t):
    import aiter

    return aiter.silu_and_mul(t[0], t[1])


def _call_sgl_kernel_silu_and_mul(t):
    import sgl_kernel

    # The trace records (out, input) while the Python wrapper takes
    # (input, out); the input is twice as wide, so pick by shape.
    a, b = t[0], t[1]
    inp, out = (b, a) if b.shape[-1] > a.shape[-1] else (a, b)
    return sgl_kernel.silu_and_mul(inp, out)


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

    return torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant(
        t[0], t[1], t[2], t[3], 1e-06, 128
    )


def _call_vllm_rmsnorm_add_fp8_group_quant(t):
    import vllm._aiter_ops  # noqa: F401

    return torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant(
        t[0], t[1], t[2], t[3], t[4], 1e-06, 128
    )


def _call_dsv3_fused_qk_rope_cat_and_cache_mla(t):
    from aiter.ops.triton.fusions.fused_kv_cache import (
        fused_qk_rope_cat_and_cache_mla,
    )

    q_nope = t[0]
    q_pe = t[1]
    k_nope = t[2]
    k_pe = t[3]
    kv_cache = t[4]
    cos = t[7]
    sin = t[8]
    T = q_nope.shape[0]
    b_cache = kv_cache.shape[0]
    slot = torch.randperm(b_cache, device="cuda", dtype=torch.int64)[:T].contiguous()
    pos = torch.randint(0, cos.shape[0], (T,), device="cuda", dtype=torch.int64)
    k_scale = torch.ones((1,), dtype=torch.float32, device="cuda")[0]
    return fused_qk_rope_cat_and_cache_mla(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        slot,
        pos,
        cos,
        sin,
        k_scale,
        is_neox=True,
        num_decode_toks_for_zeros=0,
        apply_scale=False,
    )


def _call_dsv3_dynamic_per_group_scaled_quant_fp4(t):
    import aiter

    x = t[1]
    M, N = x.shape
    out = torch.empty((M, N // 2), dtype=torch.uint8, device="cuda")
    scales = torch.empty((M, N // 32), dtype=torch.uint8, device="cuda")
    return aiter.dynamic_per_group_scaled_quant_fp4(out, x, scales, 32)


def _call_dsv3_quant_dynamic_mxfp4_quant(t):
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant

    return dynamic_mxfp4_quant(t[0])


def _call_aiter_rmsnorm(t):
    from aiter.ops.rmsnorm import rmsnorm

    return rmsnorm(t[0], t[1], t[2], 1e-06)


def _call_gemm_afp4wfp4(t):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_

    # Signature is (x, w, x_scales, w_scales, dtype, y): operand 4 in the trace
    # is the output ScalarType, not a tensor, and operand 5 is the output.
    out = t[5]
    return gemm_afp4wfp4_(t[0], t[1], t[2], t[3], out.dtype, out)


def _call_rope_cached_positions_2c_fwd_impl(t):
    from aiter.ops.rope import rope_cached_positions_2c_fwd_impl

    # Positions index the cos/sin caches, so they must stay in range; the
    # generic allocator zero-fills integer tensors, which is already valid.
    return rope_cached_positions_2c_fwd_impl(
        t[0],
        t[1],
        t[2],
        t[3],
        t[4],
        t[5],
        t[6],
        0,
        True,
        False,
    )


def _call_fused_rms_mxfp4_quant(t):
    from aiter.ops.triton.quant import fused_rms_mxfp4_quant

    # Two shapes occur: a QK-pair variant (x1, w1, x2, w2) and a single-input
    # variant that may carry a residual tensor in the last operand slot.
    x2 = t.get(2)
    x2_weight = t.get(3)
    res1 = t.get(4)
    if x2 is not None and x2_weight is not None:
        return fused_rms_mxfp4_quant(t[0], t[1], 1e-06, x2, x2_weight, 1e-06)
    return fused_rms_mxfp4_quant(t[0], t[1], 1e-06, res1=res1)


def _call_fused_flatten_mxfp4_quant(t):
    from aiter.ops.triton.quant import fused_flatten_mxfp4_quant

    return fused_flatten_mxfp4_quant(t[0])


def _call_batched_gemm_a16wfp4(t, c):
    from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import batched_gemm_a16wfp4_

    # (x, w, w_scales, dtype, y, config, transpose_bm, prequant). transpose_bm
    # decides whether y is (M, B, N) or (B, M, N), and it varies per call site.
    out = t[4]
    transpose_bm = c.get(6)
    if transpose_bm is None:
        transpose_bm = out.shape[0] != t[0].shape[0]
    return batched_gemm_a16wfp4_(
        t[0], t[1], t[2], out.dtype, out, None, bool(transpose_bm), bool(c.get(7, True))
    )


def _call_fused_qk_rmsnorm(t):
    from aiter.ops.fused_qk_norm_rope_cache_quant import fused_qk_rmsnorm

    return fused_qk_rmsnorm(t[0], t[1], 1e-06, t[3], t[4], 1e-06)


def _call_biased_grouped_topk(t, c):
    from aiter.ops.topk import biased_grouped_topk

    return biased_grouped_topk(
        t[0],
        t[1],
        t[2],
        t[3],
        c.get(4, 1),
        c.get(5, 1),
        c.get(6, True),
        c.get(7, 1.0),
    )


def _call_moe_sorting_fwd(t, c):
    from aiter.ops.moe_sorting import moe_sorting_fwd

    # (topk_ids, topk_weights, sorted_token_ids, sorted_weights,
    #  sorted_expert_ids, num_valid_ids, moe_buf, num_experts, unit_size, ...)
    num_experts = c.get(7, 256)
    unit_size = c.get(8, 32)
    # topk_ids must address a real expert; the generic allocator zero-fills
    # integer tensors, which routes every token to expert 0 and skews the sort,
    # so spread them across the expert range instead.
    topk_ids = torch.randint(
        0, num_experts, t[0].shape, dtype=t[0].dtype, device="cuda"
    )
    return moe_sorting_fwd(
        topk_ids,
        t[1],
        t[2],
        t[3],
        t[4],
        t[5],
        t[6],
        num_experts,
        unit_size,
        None,
        None,
        c.get(11, 0),
    )


def _call_mxfp4_moe_sort_hip(t, c):
    from aiter.ops.quant import mxfp4_moe_sort_hip

    # (out_scale, scale, sorted_ids, num_valid_ids, token_num, cols)
    token_num = c.get(4, t[1].shape[0])
    cols = c.get(5, t[1].shape[1] * 32)
    sorted_ids = torch.randint(
        0, t[1].shape[0], t[2].shape, dtype=t[2].dtype, device="cuda"
    )
    num_valid = torch.full(t[3].shape, t[2].shape[0], dtype=t[3].dtype, device="cuda")
    return mxfp4_moe_sort_hip(t[0], t[1], sorted_ids, num_valid, token_num, cols)


def _call_vllm_concat_and_cache_mla(t):
    import vllm._C  # noqa: F401  (registers torch.ops._C_cache_ops.*)

    kv_c, k_pe, kv_cache = t[0], t[1], t[2]
    # Distinct slots keep the write pattern representative; the traced
    # slot_mapping is a permutation over cache blocks.
    n_tokens = kv_c.shape[0]
    n_slots = kv_cache.shape[0]
    slot_mapping = torch.randperm(n_slots, device="cuda", dtype=torch.int64)[:n_tokens]
    scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    return torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping.contiguous(), "auto", scale
    )


def _call_vllm_concat_mla_q(t):
    import vllm._C  # noqa: F401

    return torch.ops._C_cache_ops.concat_mla_q(t[0], t[1], t[2])


def _call_vllm_per_token_group_fp8_quant(t):
    import vllm._C  # noqa: F401

    # Traced scalars: group_size=128, eps=1e-10, fp8 range +/-448, ue8m0 scales.
    group_size = t[0].shape[-1]
    return torch.ops._C.per_token_group_fp8_quant(
        t[0], t[1], t[2], group_size, 1e-10, -448.0, 448.0, True, False, False
    )


def _call_flash_attn_varlen_forward(t):
    import flash_attn  # noqa: F401  (registers torch.ops.flash_attn.*)

    q, k, v = t[0], t[1], t[2]
    # cu_seqlens has n_seqs+1 entries; the trace records only its length, so
    # split the packed tokens evenly across that many sequences.
    n_seqs = t[3].shape[0] - 1
    cu = torch.linspace(0, q.shape[0], n_seqs + 1, device="cuda")
    cu = cu.round().to(torch.int32).contiguous()
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    return torch.ops.flash_attn._flash_attn_varlen_forward(
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen,
        max_seqlen,
        0.0,
        q.shape[-1] ** -0.5,
        True,
        -1,
        -1,
        0.0,
        None,
        False,
    )


OP_CALL_SPEC = {
    "dsv3_fused_qk_rope_cat_and_cache_mla": {
        "call": _call_dsv3_fused_qk_rope_cat_and_cache_mla,
        "output_indices": [],
        "skip_indices": [],
    },
    "dsv3_dynamic_per_group_scaled_quant_fp4": {
        "call": _call_dsv3_dynamic_per_group_scaled_quant_fp4,
        "output_indices": [],
        "skip_indices": [0, 2],
    },
    "dsv3_quant_dynamic_mxfp4_quant": {
        "call": _call_dsv3_quant_dynamic_mxfp4_quant,
        "output_indices": [],
        "skip_indices": [],
    },
    "gemm_a8w8_blockscale": {
        "call": _call_gemm_a8w8_blockscale,
        "output_indices": [4],
        "skip_indices": [],
    },
    "gemm_a16w16_atomic_": {
        "call": _call_gemm_a16w16_asm,
        "output_indices": [2],
        "skip_indices": [],
    },
    "gemm_afp4wfp4": {
        "call": _call_gemm_afp4wfp4,
        "output_indices": [5],
        "skip_indices": [],
    },
    "rmsnorm": {"call": _call_aiter_rmsnorm, "output_indices": [0], "skip_indices": []},
    "rope_cached_positions_2c_fwd_impl": {
        "call": _call_rope_cached_positions_2c_fwd_impl,
        "output_indices": [0, 1],
        "skip_indices": [],
    },
    "fused_rms_mxfp4_quant": {
        "call": _call_fused_rms_mxfp4_quant,
        "output_indices": [],
        "skip_indices": [],
    },
    "fused_flatten_mxfp4_quant": {
        "call": _call_fused_flatten_mxfp4_quant,
        "output_indices": [],
        "skip_indices": [],
    },
    "batched_gemm_a16wfp4_": {
        "call": _call_batched_gemm_a16wfp4,
        "output_indices": [4],
        "skip_indices": [],
    },
    "fused_qk_rmsnorm": {
        "call": _call_fused_qk_rmsnorm,
        "output_indices": [],
        "skip_indices": [],
    },
    "biased_grouped_topk_hip": {
        "call": _call_biased_grouped_topk,
        "output_indices": [2, 3],
        "skip_indices": [],
    },
    "moe_sorting_fwd": {
        "call": _call_moe_sorting_fwd,
        "output_indices": [2, 3, 4, 5, 6],
        "skip_indices": [],
    },
    "mxfp4_moe_sort_hip": {
        "call": _call_mxfp4_moe_sort_hip,
        "output_indices": [0],
        "skip_indices": [],
    },
    "vllm_concat_and_cache_mla": {
        "call": _call_vllm_concat_and_cache_mla,
        "output_indices": [2],
        "skip_indices": [3],
    },
    "vllm_concat_mla_q": {
        "call": _call_vllm_concat_mla_q,
        "output_indices": [2],
        "skip_indices": [],
    },
    "vllm_per_token_group_fp8_quant": {
        "call": _call_vllm_per_token_group_fp8_quant,
        "output_indices": [1, 2],
        "skip_indices": [],
    },
    "flash_attn_varlen_forward": {
        "call": _call_flash_attn_varlen_forward,
        "output_indices": [],
        "skip_indices": [],
    },
    "silu_and_mul": {
        "call": _call_aiter_silu_and_mul,
        "output_indices": [0],
        "skip_indices": [],
    },
    "sgl_kernel_silu_and_mul": {
        "call": _call_sgl_kernel_silu_and_mul,
        "output_indices": [0],
        "skip_indices": [],
    },
    "gelu_and_mul": {
        "call": _call_aiter_gelu_and_mul,
        "output_indices": [0],
        "skip_indices": [],
    },
    "gelu_tanh_and_mul": {
        "call": _call_aiter_gelu_tanh_and_mul,
        "output_indices": [0],
        "skip_indices": [],
    },
    "rms_norm": {
        "call": _call_aiter_rms_norm,
        "output_indices": [],
        "skip_indices": [2, 3],
    },
    "add_rmsnorm": {
        "call": _call_aiter_fused_add_rms_norm,
        "output_indices": [2, 3],
        "skip_indices": [5, 6],
    },
    "rmsnorm_dynamicquant": {
        "call": _call_rmsnorm_dynamicquant,
        "output_indices": [],
        "skip_indices": [2, 3, 4],
    },
    "dynamic_per_token_scaled_quant": {
        "call": _call_dynamic_per_token_scaled_quant,
        "output_indices": [0, 2],
        "skip_indices": [3, 4, 5, 6],
    },
    "_flash_attn_forward": {
        "call": _call_flash_attn_func,
        "output_indices": [],
        "skip_indices": [],
    },
    "vllm_unquantized_gemm": {
        "call": _call_vllm_rocm_unquantized_gemm,
        "output_indices": [],
        "skip_indices": [],
    },
    "vllm_triton_gemm_a8w8_blockscale": {
        "call": _call_gemm_a8w8_blockscale,
        "output_indices": [4],
        "skip_indices": [],
    },
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


def _parse_concrete(raw_json):
    """Parse the traced non-tensor arguments into a positional lookup.

    Values arrive as strings ('384', 'True', ''); empty slots correspond to
    tensor operands and are dropped so callers can use ``.get(i)`` to mean
    "was a scalar recorded at this position".
    """
    out = {}
    if not raw_json:
        return out
    for i, raw in enumerate(json.loads(raw_json)):
        text = str(raw).strip()
        if not text:
            continue
        if text in ("True", "False"):
            out[i] = text == "True"
            continue
        try:
            out[i] = int(text)
        except ValueError:
            try:
                out[i] = float(text)
            except ValueError:
                out[i] = text
    return out


def test_generic_simple_op(
    input_dims_json,
    input_types_json,
    registry_key=None,
    num_warmup=3,
    concrete_inputs_json=None,
    **_,
):
    """Generic harness driven by CSV ``Input Dims`` / ``Input type`` columns.

    Looks up ``registry_key`` in :data:`OP_CALL_SPEC` to find the call dispatch
    function and the output / skip index sets, builds tensors of the exact
    shapes and dtypes from the CSV, and runs warmup + measured iterations.

    Dispatch functions take the tensor map alone, or ``(tensors, concrete)`` if
    they need the traced scalar arguments.
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
    print(
        f"test: __generic__ registry_key={registry_key} n_inputs={len(input_dims)}",
        flush=True,
    )
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
    concrete = _parse_concrete(concrete_inputs_json)
    takes_concrete = len(inspect.signature(fn).parameters) >= 2

    def _call():
        return fn(t, concrete) if takes_concrete else fn(t)

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print("test: done", flush=True)
