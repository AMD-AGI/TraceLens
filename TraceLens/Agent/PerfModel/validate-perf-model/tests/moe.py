###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mixture-of-Experts test harnesses for ``validate_perf_model``.

Each ``test_<op>`` function builds the input tensors for a single MoE kernel
in ``aiter``, runs a few warm-up iterations and then a measured iteration, and
synchronizes the GPU. The functions are imported by ``tests/_runner.py``,
which is what ``rocprofv3`` actually wraps.

Every test function is parameterized over input/weight/output dtype and the
relevant ``aiter`` ``QuantType`` so that perf-model validation can target any
of the production kernel variants. See ``tests/moe.md`` for the per-op
mapping of dtypes onto the upstream ``aiter`` implementation.

Shared helpers
--------------
* :func:`_init_moe_routing` -- random hidden + topk routing + sorted buffers.
* :func:`_quantize_weight_blockscale` -- per-token block-scale FP8/INT8 quant.
* :func:`_quantize_act_blockscale`    -- per-token block-scale FP8/INT8 quant
  for the activation tensor.
* :func:`_shuffle_moe_weights`        -- :func:`aiter.ops.shuffle.shuffle_weight`
  wrapper for the FMoE asm-kernel layout.
* :func:`_resolve_dtype`              -- string -> ``torch.dtype`` resolver
  that defers torch import.
"""

# NOTE: torch is imported lazily inside each function so that this module can
# be imported by the parent process for argv-building without paying the cost
# (and side effects) of importing torch.


# ---------------------------------------------------------------------------
# Dtype string resolution
# ---------------------------------------------------------------------------
#
# The actual resolvers live in ``tests/_dtypes.py`` so the gemm/attention/
# rmsnorm/other harnesses can reuse them. Re-exported here under the
# legacy ``_<name>`` aliases so existing references inside this file continue
# to work without touching every callsite.

from ._dtypes import (
    resolve_activation as _resolve_activation,
    resolve_dtype as _resolve_dtype,
    resolve_quant_type as _resolve_quant_type,
)


# ---------------------------------------------------------------------------
# Shared MoE input builders
# ---------------------------------------------------------------------------

def _init_moe_routing(M, K, E, topk, dtype, device="cuda"):
    """Build a random MoE routing buffer set.

    Returns a dict with ``hidden, w_score, topk_weights, topk_ids,
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out``.
    The ``out`` buffer is the standard FMoE output buffer
    (``[M, K]`` in ``dtype``) returned by ``moe_sorting``.
    """
    import torch  # noqa: F401  (used implicitly via aiter)
    from aiter.fused_moe import fused_topk, moe_sorting

    hidden = torch.randn(M, K, dtype=dtype, device=device)
    w_score = torch.randn(M, E, dtype=dtype, device=device)
    topk_weights, topk_ids = fused_topk(hidden, w_score, topk, True)
    (sorted_token_ids, sorted_weights, sorted_expert_ids,
     num_valid_ids, out) = moe_sorting(
        topk_ids, topk_weights, E, K, dtype,
    )
    return {
        "hidden": hidden,
        "w_score": w_score,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "out": out,
    }


def _quantize_weight_blockscale(w, BLOCK_N, BLOCK_K, quant_dtype):
    """Per-token block-scale quantize an MoE expert weight tensor.

    ``w`` has shape ``[E, dim_n, dim_k]``. We tile to
    ``[E, dim_n / BLOCK_N, dim_k / BLOCK_K, BLOCK_N, BLOCK_K]``,
    pertoken-quantize each tile, then reshape back to ``[E, dim_n, dim_k]``.
    Returns ``(w_q, w_scale)`` with ``w_scale`` shaped ``[E, num_blocks_total]``.
    """
    from aiter import pertoken_quant
    from einops import rearrange

    E, dim_n, dim_k = w.shape
    tmp = rearrange(
        w.view(E, dim_n // BLOCK_N, BLOCK_N, dim_k // BLOCK_K, BLOCK_K),
        "e nn bn nk bk -> e nn nk (bn bk)",
    ).contiguous()
    w_q, w_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
    w_q = rearrange(
        w_q.view(E, dim_n // BLOCK_N, dim_k // BLOCK_K, BLOCK_N, BLOCK_K),
        "e nn nk bn bk -> e (nn bn) (nk bk)",
    ).contiguous()
    w_scale = w_scale.view(E, -1)
    return w_q, w_scale


def _quantize_act_blockscale(input_hbm, K, BLOCK_K, quant_dtype):
    """Per-token block-scale quantize an MoE activation tensor ``[M, K]``."""
    from aiter import pertoken_quant

    a_q, a_scale = pertoken_quant(
        input_hbm.view(-1, K // BLOCK_K, BLOCK_K),
        quant_dtype=quant_dtype,
    )
    a_q = a_q.view(-1, K)
    a_scale = a_scale.squeeze(-1)
    return a_q, a_scale


def _shuffle_moe_weights(*weights, tile=(16, 16)):
    """Apply ``aiter.ops.shuffle.shuffle_weight`` to each weight tensor."""
    from aiter.ops.shuffle import shuffle_weight

    return [shuffle_weight(w, tile) for w in weights]


# ---------------------------------------------------------------------------
# 1. fmoe_fp8_blockscale_g1u1
# ---------------------------------------------------------------------------

def test_fmoe_fp8_blockscale_g1u1(
    M, N, K, E, topk,
    in_dtype="bf16",
    w_dtype="fp8",
    out_dtype="bf16",
    scale_dtype="fp32",
    activation="silu",
    block_n=128,
    block_k=128,
    num_warmup=3,
    **_,
):
    """``aiter.fmoe_fp8_blockscale_g1u1`` (fused FP8 block-scaled MoE, SwiGLU).

    Parameters
    ----------
    in_dtype : {"bf16", "fp8"}
        Input activation dtype. ``"bf16"`` exercises the in-kernel quant path;
        ``"fp8"`` does caller-side ``pertoken_quant``.
    w_dtype : {"fp8"}
        Expert weight dtype. The kernel only supports FP8 weights.
    out_dtype : {"bf16"}
        Output dtype. The kernel only supports BF16 output.
    scale_dtype : {"fp32"}
        Block-scale dtype.
    activation : {"silu", "gelu"}
    """
    import torch
    import aiter
    from aiter import dtypes

    torch.set_default_device("cuda")
    in_t  = _resolve_dtype(in_dtype)
    w_t   = _resolve_dtype(w_dtype)
    out_t = _resolve_dtype(out_dtype)
    if w_t != dtypes.fp8:
        raise ValueError(f"fmoe_fp8_blockscale_g1u1 requires fp8 weights, got {w_dtype}")
    if out_t != dtypes.bf16:
        raise ValueError(f"fmoe_fp8_blockscale_g1u1 requires bf16 output, got {out_dtype}")
    if in_t not in (dtypes.bf16, dtypes.fp8):
        raise ValueError(f"fmoe_fp8_blockscale_g1u1: unsupported in_dtype {in_dtype}")

    act = _resolve_activation(activation)
    print(
        f"test: fmoe_fp8_blockscale_g1u1 M={M} K={K} N={N} E={E} topk={topk} "
        f"in={in_dtype} w={w_dtype} out={out_dtype} act={activation}",
        flush=True,
    )

    routing = _init_moe_routing(M, K, E, topk, dtype=out_t)
    input_bf16 = routing["hidden"]

    w1_bf16 = torch.randn((E, N * 2, K), dtype=out_t) / 10
    w2_bf16 = torch.randn((E, K, N), dtype=out_t) / 10

    print("test: quantizing weights...", flush=True)
    w1_q, w1_scale = _quantize_weight_blockscale(w1_bf16, block_n, block_k, w_t)
    w2_q, w2_scale = _quantize_weight_blockscale(w2_bf16, block_n, block_k, w_t)
    a_q, a_scale = _quantize_act_blockscale(input_bf16, K, block_k, w_t)
    w1_shuf, w2_shuf = _shuffle_moe_weights(w1_q, w2_q)

    if in_t == dtypes.fp8:
        # Caller-quantized FP8 input path (production on gfx950).
        a_in = a_q
        a_scale_in = a_scale.t().contiguous()
    else:
        # In-kernel-quant path: kernel quantizes input internally; we still
        # provide a scale buffer of the right dtype (kernel ignores it for
        # bf16 input on the current configs).
        a_in = input_bf16
        a_scale_in = a_scale.t().contiguous()

    out = routing["out"]

    def _run():
        aiter.fmoe_fp8_blockscale_g1u1(
            out, a_in, w1_shuf, w2_shuf,
            routing["sorted_token_ids"], routing["sorted_weights"],
            routing["sorted_expert_ids"], routing["num_valid_ids"], topk,
            a_scale_in, w1_scale, w2_scale,
            "", block_n, block_k, None,
            activation=act.value,
        )

    for _ in range(num_warmup):
        _run()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _run()
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 2. moe_cktile2stages_gemm1_ck
# ---------------------------------------------------------------------------

def _build_cktile_fp4_weights(E, dim_n, dim_k, dtype_in, scale_dtype, device="cuda"):
    """Build ``(w_packed_fp4, w_scale)`` for the CK-Tile MoE kernels.

    ``aiter.moe_cktile2stages_gemm{1,2}_ck`` accept FP4-packed weights only
    (``torch.float4_e2m1fn_x2``), with per-128-block scales of dtype matching
    ``x_scale`` -- typically ``fp32`` for FP8 activations or
    ``float8_e8m0fnu`` (MXFP4) for BF16/FP16 activations.

    We start from a BF16 reference, quantize via :func:`_quantize_weight_blockscale`
    (which uses ``pertoken_quant``), then convert the quantized weight to
    ``torch.float4_e2m1fn_x2`` if available. If the running torch build does
    not expose ``float4_e2m1fn_x2``, raise -- the kernel cannot be launched.
    """
    import torch

    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError(
            "torch.float4_e2m1fn_x2 unavailable; cannot build FP4 weights for "
            "moe_cktile2stages_gemm*_ck. Use a torch build with FP4 support "
            "or run on gfx950+."
        )
    # Use aiter's reference MXFP4 quantizer to build *valid* FP4 weights
    # plus an E8M0 block-scale tensor. Random uint8 bit patterns happen to
    # decode to valid FP4 values but the corresponding scales must obey
    # the kernel's expected MX layout; using the upstream quantizer keeps
    # the layout in sync with whatever ``moe_cktile2stages_gemm{1,2}_ck``
    # currently expects (gfx950 GPU MAFs otherwise).
    from aiter import fp4_utils
    dynamic_mxfp4_quant = fp4_utils.dynamic_mxfp4_quant
    w_bf16 = (torch.randn((E, dim_n, dim_k), dtype=torch.bfloat16, device=device) / 10).contiguous()
    # ``dynamic_mxfp4_quant`` operates per-row on the trailing dim; we
    # quantize each (dim_n, dim_k) expert slab independently and stack.
    packed_slabs = []
    scale_slabs = []
    for e in range(E):
        x_fp4, x_scale = dynamic_mxfp4_quant(w_bf16[e])
        packed_slabs.append(x_fp4)
        scale_slabs.append(x_scale)
    w_packed = torch.stack(packed_slabs, dim=0).contiguous()
    w_scale_e8m0 = torch.stack(scale_slabs, dim=0).contiguous()
    # The cktile kernels accept either an ``e8m0`` (uint8-backed) scale
    # tensor or a per-block ``fp32`` tensor. We keep the e8m0 layout from
    # the quantizer (matches MX block tiling) and cast only when the
    # caller asked for a non-e8m0 dtype.
    target_scl = _resolve_dtype(scale_dtype)
    if target_scl == torch.float32 or target_scl == torch.float16 or target_scl == torch.bfloat16:
        # Decode e8m0 -> fp32 then cast.
        w_scale = fp4_utils.e8m0_to_f32(w_scale_e8m0).to(target_scl)
    else:
        w_scale = w_scale_e8m0
    return w_packed, w_scale


def test_moe_cktile2stages_gemm1_ck(
    M, N, K, E, topk,
    in_dtype="bf16",
    w_dtype="fp4x2",
    out_dtype="bf16",
    scale_dtype="fp32",
    activation="silu",
    block_m=32,
    split_k=1,
    num_warmup=3,
    **_,
):
    """``aiter.moe_cktile2stages_gemm1_ck`` (CK-Tile MoE up-projection).

    The kernel only supports FP4-packed weights and BF16 output. The
    activation can be BF16/FP16 (a16w4) or FP8 (a8w4).
    """
    import torch
    import aiter

    torch.set_default_device("cuda")
    in_t   = _resolve_dtype(in_dtype)
    w_t    = _resolve_dtype(w_dtype)
    out_t  = _resolve_dtype(out_dtype)
    if w_dtype != "fp4x2":
        raise ValueError(
            f"moe_cktile2stages_gemm1_ck only supports fp4x2 weights, got {w_dtype}"
        )
    if out_t != torch.bfloat16:
        raise ValueError(
            f"moe_cktile2stages_gemm1_ck only supports bf16 output, got {out_dtype}"
        )
    act = _resolve_activation(activation)
    print(
        f"test: moe_cktile2stages_gemm1_ck M={M} K={K} N={N} E={E} topk={topk} "
        f"in={in_dtype} w={w_dtype} out={out_dtype} act={activation} split_k={split_k}",
        flush=True,
    )

    routing = _init_moe_routing(M, K, E, topk, dtype=out_t)
    hidden = routing["hidden"]
    if in_t != torch.bfloat16:
        # Caller-side activation cast (a8w4 path uses FP8 hidden).
        from aiter import pertoken_quant
        from aiter import dtypes
        hidden_q, x_scale = pertoken_quant(
            hidden.view(-1, K // 128, 128), quant_dtype=dtypes.fp8,
        )
        hidden = hidden_q.view(-1, K)
        x_scale = x_scale.squeeze(-1).to(_resolve_dtype(scale_dtype))
    else:
        x_scale = None

    w1_packed, w1_scale = _build_cktile_fp4_weights(
        E, N * 2, K, in_t, scale_dtype,
    )
    Y = torch.empty(M, topk, N, dtype=out_t, device="cuda")

    def _run():
        aiter.moe_cktile2stages_gemm1_ck(
            hidden, w1_packed, Y,
            routing["sorted_token_ids"], routing["sorted_expert_ids"],
            routing["num_valid_ids"], topk,
            0, 0,
            routing["sorted_weights"],
            x_scale, w1_scale,
            None,           # exp_bias
            act.value,
            block_m, split_k,
            "",
        )

    for _ in range(num_warmup):
        _run()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _run()
    torch.cuda.synchronize()
    print(f"test: done, shape={Y.shape}", flush=True)


def test_moe_cktile2stages_gemm2_ck(
    M, N, K, E, topk,
    in_dtype="bf16",
    w_dtype="fp4x2",
    out_dtype="bf16",
    scale_dtype="fp32",
    activation="silu",
    block_m=32,
    split_k=1,
    num_warmup=3,
    **_,
):
    """``aiter.moe_cktile2stages_gemm2_ck`` (CK-Tile MoE down-projection)."""
    import torch
    import aiter

    torch.set_default_device("cuda")
    in_t  = _resolve_dtype(in_dtype)
    w_t   = _resolve_dtype(w_dtype)
    out_t = _resolve_dtype(out_dtype)
    if w_dtype != "fp4x2":
        raise ValueError(
            f"moe_cktile2stages_gemm2_ck only supports fp4x2 weights, got {w_dtype}"
        )
    if out_t != torch.bfloat16:
        raise ValueError(
            f"moe_cktile2stages_gemm2_ck only supports bf16 output, got {out_dtype}"
        )
    act = _resolve_activation(activation)
    print(
        f"test: moe_cktile2stages_gemm2_ck M={M} K={K} N={N} E={E} topk={topk} "
        f"in={in_dtype} w={w_dtype} out={out_dtype} act={activation}",
        flush=True,
    )

    routing = _init_moe_routing(M, K, E, topk, dtype=out_t)
    # ``torch.randn`` does not implement ``normal_kernel_cuda`` for FP8;
    # always allocate the BF16 source and quantize down for FP8 input.
    # The cktile gemm2 kernel takes the 3D ``(token, topk, N)`` activation
    # tensor directly (it unrolls topk internally); flattening to 2D drops
    # the ``topk`` axis and the kernel falls back to reading off-by-one,
    # producing GPU memory access faults.
    inter_states = torch.randn(M, topk, N, dtype=torch.bfloat16)
    if in_t != torch.bfloat16:
        from aiter import pertoken_quant
        from aiter import dtypes
        inter_q, x_scale = pertoken_quant(
            inter_states.view(-1, N // 128, 128), quant_dtype=dtypes.fp8,
        )
        inter_states_in = inter_q.view(M, topk, N)
        x_scale = x_scale.squeeze(-1).to(_resolve_dtype(scale_dtype))
    else:
        inter_states_in = inter_states  # already (M, topk, N)
        x_scale = None

    w2_packed, w2_scale = _build_cktile_fp4_weights(
        E, K, N, in_t, scale_dtype,
    )
    Y = torch.empty(M, topk, K, dtype=out_t, device="cuda")

    def _run():
        aiter.moe_cktile2stages_gemm2_ck(
            inter_states_in, w2_packed, Y,
            routing["sorted_token_ids"], routing["sorted_expert_ids"],
            routing["num_valid_ids"], topk,
            0, 0,
            routing["sorted_weights"],
            x_scale, w2_scale,
            None,           # exp_bias
            act.value,
            block_m, split_k,
            "",
        )

    for _ in range(num_warmup):
        _run()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _run()
    torch.cuda.synchronize()
    print(f"test: done, shape={Y.shape}", flush=True)


# ---------------------------------------------------------------------------
# 4. ck_moe_stage1
# ---------------------------------------------------------------------------

def _build_ck_moe_inputs(M, N, K, E, topk, in_dtype, w_dtype, quant_type, block_k=128):
    """Build hidden + W1 + W2 + (optional) scales for the CK two-stage MoE.

    Returns ``(hidden, w1, w2, a_scale, w1_scale, w2_scale)`` with dtypes
    matching the requested ``in_dtype``/``w_dtype``/``quant_type``.
    """
    import torch
    from aiter import QuantType

    in_t = _resolve_dtype(in_dtype)
    w_t  = _resolve_dtype(w_dtype)
    qt   = _resolve_quant_type(quant_type)

    # Build BF16 reference tensors then cast / quantize.
    hidden_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w1_bf16     = torch.randn(E, N * 2, K, dtype=torch.bfloat16, device="cuda") / 10
    w2_bf16     = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda") / 10

    if qt == QuantType.No:
        return (
            hidden_bf16.to(in_t),
            w1_bf16.to(w_t),
            w2_bf16.to(w_t),
            None, None, None,
        )

    if qt == QuantType.per_1x128:
        # FP8 block-scaled (production gfx950 path).
        BLOCK_N, BLOCK_K = 128, 128
        w1_q, w1_scale = _quantize_weight_blockscale(w1_bf16, BLOCK_N, BLOCK_K, w_t)
        w2_q, w2_scale = _quantize_weight_blockscale(w2_bf16, BLOCK_N, BLOCK_K, w_t)
        a_q,  a_scale  = _quantize_act_blockscale(hidden_bf16, K, BLOCK_K, w_t)
        return a_q, w1_q, w2_q, a_scale, w1_scale, w2_scale

    raise NotImplementedError(
        f"ck_moe quant_type {quant_type!r} not implemented in this harness yet."
    )


def test_ck_moe_stage1(
    M, N, K, E, topk,
    in_dtype="bf16",
    w_dtype="bf16",
    out_dtype="bf16",
    quant_type="no",
    activation="silu",
    block_m=32,
    split_k=1,
    num_warmup=3,
    **_,
):
    """``aiter.ck_moe_stage1_fwd`` (CK MoE up-projection, fused activation)."""
    import torch
    from aiter.ops.moe_op import ck_moe_stage1_fwd

    torch.set_default_device("cuda")
    out_t = _resolve_dtype(out_dtype)
    qt    = _resolve_quant_type(quant_type)
    act   = _resolve_activation(activation)
    print(
        f"test: ck_moe_stage1 M={M} K={K} N={N} E={E} topk={topk} "
        f"in={in_dtype} w={w_dtype} out={out_dtype} quant={quant_type} "
        f"act={activation} block_m={block_m} splitk={split_k}",
        flush=True,
    )

    routing = _init_moe_routing(M, K, E, topk, dtype=out_t)
    hidden, w1, w2, a_scale, w1_scale, w2_scale = _build_ck_moe_inputs(
        M, N, K, E, topk, in_dtype, w_dtype, quant_type,
    )
    out = torch.empty(M, topk, N, dtype=out_t, device="cuda")

    def _run():
        ck_moe_stage1_fwd(
            hidden, w1, w2,
            routing["sorted_token_ids"], routing["sorted_expert_ids"],
            routing["num_valid_ids"], out, topk,
            "",
            w1_scale, a_scale,
            block_m,
            None,
            qt,
            act,
            split_k,
            False,
            out_t,
        )

    for _ in range(num_warmup):
        _run()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _run()
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_ck_moe_stage2(
    M, N, K, E, topk,
    in_dtype="bf16",
    w_dtype="bf16",
    out_dtype="bf16",
    quant_type="no",
    activation="silu",
    block_m=32,
    num_warmup=3,
    **_,
):
    """``aiter.ck_moe_stage2_fwd`` (CK MoE down-projection)."""
    import torch
    from aiter.ops.moe_op import ck_moe_stage2_fwd

    torch.set_default_device("cuda")
    in_t  = _resolve_dtype(in_dtype)
    out_t = _resolve_dtype(out_dtype)
    qt    = _resolve_quant_type(quant_type)
    act   = _resolve_activation(activation)
    print(
        f"test: ck_moe_stage2 M={M} K={K} N={N} E={E} topk={topk} "
        f"in={in_dtype} w={w_dtype} out={out_dtype} quant={quant_type} "
        f"act={activation} block_m={block_m}",
        flush=True,
    )

    routing = _init_moe_routing(M, K, E, topk, dtype=out_t)
    # FP8 has no `normal_kernel_cuda` implementation; allocate as a
    # higher-precision tensor and cast.
    if in_t in (torch.bfloat16, torch.float16, torch.float32):
        inter_states = torch.randn(M, topk, N, dtype=in_t)
    else:
        inter_states = torch.randn(M, topk, N, dtype=torch.bfloat16).to(in_t)
    _, w1, w2, a2_scale, _w1_scale, w2_scale = _build_ck_moe_inputs(
        M, N, K, E, topk, in_dtype, w_dtype, quant_type,
    )
    out = torch.empty(M, K, dtype=out_t, device="cuda")

    def _run():
        ck_moe_stage2_fwd(
            inter_states, w1, w2,
            routing["sorted_token_ids"], routing["sorted_expert_ids"],
            routing["num_valid_ids"], out, topk,
            "",
            w2_scale, a2_scale,
            block_m,
            None,
            qt,
            act,
            False,
        )

    for _ in range(num_warmup):
        _run()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _run()
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 6. SGLang Triton fused-MoE grouped GEMM (single invoke_fused_moe_kernel)
# ---------------------------------------------------------------------------

def test_sglang_fused_moe_triton_invoke(
    M, N, K, E, topk,
    in_dtype="bf16",
    w_dtype="fp8",
    out_dtype="bf16",
    num_warmup=3,
    **_,
):
    """A single SGLang Triton ``invoke_fused_moe_kernel`` (gate/up grouped GEMM).

    Drives exactly one ``fused_moe_kernel`` dispatch per iteration so the
    rocprofv3 counters line up with the per-event prediction of
    ``moe_triton_invoke_grouped_gemm`` (one grouped GEMM, FP8 weights).

    Shapes follow the gate/up pass: A = (M, K), B = (E, N, K) where N = 2*inter,
    K = hidden; the kernel writes C = (M*topk, N).

    Notes
    -----
    SGLang's low-level helpers (``moe_align_block_size``,
    ``try_get_optimal_moe_config``, ``invoke_fused_moe_kernel``) live under
    ``sglang.srt.layers.moe.fused_moe_triton.fused_moe``; their exact import
    path / signature can shift across SGLang releases. If an import or argument
    mismatch occurs, pin the harness to the target SGLang version.
    """
    import torch

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_kernels as _fmk,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        try_get_optimal_moe_config,
    )

    torch.set_default_device("cuda")
    out_t = _resolve_dtype(out_dtype)
    fp8_t = _resolve_dtype("fp8")
    if w_dtype != "fp8":
        raise ValueError(f"sglang fused_moe invoke harness requires fp8 weights, got {w_dtype}")

    # The fp8_w8a8 path (block_shape=None) quantizes the BF16 activations to fp8
    # *inside* invoke_fused_moe_kernel via scaled_fp8_quant; A must therefore be
    # passed as BF16 with A_scale=None (dynamic). The kernel reads
    # K = B.shape[-1] - padding_size, so the weight K dim is padded.
    pad = int(getattr(_fmk, "padding_size", 0) or 0)

    print(
        f"test: sglang_fused_moe_triton_invoke M={M} K={K} N={N} E={E} topk={topk} "
        f"in={in_dtype} w={w_dtype} out={out_dtype} pad={pad}",
        flush=True,
    )

    # Routing.
    hidden = torch.randn(M, K, dtype=out_t)
    gating = torch.randn(M, E, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    # FP8 expert weights with per-tensor (per-expert scalar) scales; K dim padded.
    w1 = (torch.randn(E, N, K + pad, dtype=out_t) / 10).to(fp8_t)
    w1_scale = torch.ones(E, dtype=torch.float32)
    out = torch.empty(M * topk, N, dtype=out_t)

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }
    try:
        get_config = try_get_optimal_moe_config(
            (E, N, K), (E, K, N), topk, "fp8_w8a8", M,
        )
        if isinstance(get_config, dict) and get_config:
            config = get_config
    except Exception as exc:  # noqa: BLE001 - fall back to the static config
        print(f"test: using fallback config ({exc})", flush=True)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E,
    )

    def _run():
        invoke_fused_moe_kernel(
            hidden, w1, None, out,
            None, w1_scale, None,
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            False, topk,
            config,
            tl_dtype_for(out_t),
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )

    for _ in range(num_warmup):
        _run()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _run()
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def tl_dtype_for(torch_dtype):
    """Map a torch dtype to the Triton compute_type expected by fused_moe_kernel."""
    import triton.language as tl
    import torch

    return {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }.get(torch_dtype, tl.bfloat16)


# ---------------------------------------------------------------------------
# OP_METADATA
# ---------------------------------------------------------------------------

OP_METADATA: dict = {
    "fmoe_fp8_blockscale_g1u1": {
        "fn":           test_fmoe_fp8_blockscale_g1u1,
        "category":     "MoE",
        "description":  "AITER FP8 fused MoE (ASM G1U1 kernel, FMoE)",
        "dtypes":       ["fp8"],
        "defaults":     {
            "M": 256, "N": 4096, "K": 7168, "E": 256, "topk": 8,
            "in_dtype": "fp8", "w_dtype": "fp8",
        },
        "required_args": ["M", "N", "K"],
    },
    "moe_cktile2stages_gemm1_ck": {
        "fn":           test_moe_cktile2stages_gemm1_ck,
        "category":     "MoE",
        "description":  "AITER CKTile 2-stage MoE GEMM-1 (up/gate projection)",
        "dtypes":       ["fp8", "i8"],
        "defaults":     {
            "M": 256, "N": 4096, "K": 7168, "E": 256, "topk": 8,
            "in_dtype": "fp8", "w_dtype": "fp8",
        },
        "required_args": ["M", "N", "K"],
    },
    "moe_cktile2stages_gemm2_ck": {
        "fn":           test_moe_cktile2stages_gemm2_ck,
        "category":     "MoE",
        "description":  "AITER CKTile 2-stage MoE GEMM-2 (down projection)",
        "dtypes":       ["fp8", "i8"],
        "defaults":     {
            "M": 256, "N": 7168, "K": 4096, "E": 256, "topk": 8,
            "in_dtype": "fp8", "w_dtype": "fp8",
        },
        "required_args": ["M", "N", "K"],
    },
    "ck_moe_stage1": {
        "fn":           test_ck_moe_stage1,
        "category":     "MoE",
        "description":  "AITER CK unfused MoE stage-1 (up+gate projection)",
        "dtypes":       ["fp8", "bf16"],
        "defaults":     {
            "M": 256, "N": 4096, "K": 7168, "E": 256, "topk": 8,
            "in_dtype": "fp8",
        },
        "required_args": ["M", "N", "K"],
    },
    "ck_moe_stage2": {
        "fn":           test_ck_moe_stage2,
        "category":     "MoE",
        "description":  "AITER CK unfused MoE stage-2 (down projection)",
        "dtypes":       ["fp8", "bf16"],
        "defaults":     {
            "M": 256, "N": 7168, "K": 4096, "E": 256, "topk": 8,
            "in_dtype": "fp8",
        },
        "required_args": ["M", "N", "K"],
    },
    "sglang_fused_moe_triton_invoke": {
        "fn":           test_sglang_fused_moe_triton_invoke,
        "category":     "MoE",
        "description":  "SGLang Triton fused-MoE grouped GEMM (invoke_fused_moe_kernel)",
        "dtypes":       ["fp8"],
        "defaults":     {
            "M": 15360, "N": 1536, "K": 2048, "E": 128, "topk": 8,
            "in_dtype": "bf16", "w_dtype": "fp8", "out_dtype": "bf16",
        },
        "required_args": ["M", "N", "K", "E", "topk"],
    },
}
