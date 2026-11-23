###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SDPA tile model: attention split into one Q·Kᵀ and one P·V tile GEMM on
one CU, scaled by the number of waves, plus softmax and memory terms.

Any GEMM model can time the tiles. It's passed as
``gemm_time(arch, M, N, K, B, dtype, force_to_l1, num_cus)``, for example
:func:`TraceLens.PerfModel.origami_helper.gemm_time_us`. Without one there is
no time.
"""

import math

from .perf_model import SDPA, Softmax
from .utils import name2bpe, torch_dtype_map

# Dense SDPA and flash attention. Sparse and linear attention are different
# algorithms; Origami's tile GEMMs do not describe them.
_NOT_STANDARD_ATTENTION = ("sparse", "linear", "gdn", "gated_delta", "mla", "indexer")


def is_standard_attention(name: str) -> bool:
    """True for SDPA and flash attention, false for sparse or linear attention."""
    lowered = name.lower()
    return not any(token in lowered for token in _NOT_STANDARD_ATTENTION)


def _attention_rectangles(params: dict) -> list[tuple[int, int]]:
    """``(N_Q, N_KV)`` rectangles this attention launch scores.

    Paged inference stores the current chunk in ``N_KV`` and the attended key
    length in ``c_sk`` / ``g_sk``. The tile model has to see the attended length.
    """
    rectangles = []
    for query_key, key_key in (("c_sq", "c_sk"), ("g_sq", "g_sk")):
        n_q = int(params.get(query_key) or 0)
        n_kv = int(params.get(key_key) or 0)
        if n_q > 0 and n_kv > 0:
            rectangles.append((n_q, n_kv))
    if rectangles:
        return rectangles
    n_q = int(params.get("N_Q") or 0)
    n_kv = int(params.get("N_KV") or 0)
    if n_q > 0 and n_kv > 0:
        return [(n_q, n_kv)]
    return []


def _attention_dtype(params: dict) -> tuple[str | None, str | None]:
    raw = params.get("dtype_Q")
    if raw is None:
        pair = params.get("dtype_A_B")
        raw = pair[0] if isinstance(pair, (list, tuple)) and pair else pair
    precision = params.get("simulation_dtype") or torch_dtype_map(raw)
    return precision, None if raw is None else str(raw)


def _attention_geometry(params: dict) -> tuple[int, int, int] | None:
    """``(heads, batch, head_dim)`` of this launch, or ``None`` if any is missing."""
    heads = int(params.get("H_Q") or 0)
    batch = int(params.get("B") or 1)
    head_dim = int(params.get("d_h") or params.get("d_h_qk") or 0)
    if heads <= 0 or batch <= 0 or head_dim <= 0:
        return None
    return heads, batch, head_dim


def standard_attention_time_us(name, params, arch, nbytes, gemm_time=None, bwd=False):
    """Origami tile-model time for one standard or flash attention launch.

    ``None`` for sparse attention, linear attention, or a shape the tile model
    cannot time. ``nbytes`` is that launch's bytes, split across the context
    and generation rectangles by their area.
    """
    if arch is None or not isinstance(params, dict) or params.get("_no_perf"):
        return None
    if not is_standard_attention(name):
        return None
    if bwd:
        return None
    rectangles = _attention_rectangles(params)
    if not rectangles:
        return None
    precision, dtype_a_b = _attention_dtype(params)
    if precision is None:
        return None
    geometry = _attention_geometry(params)
    if geometry is None:
        return None
    heads, batch, head_dim = geometry
    flash = bool(params.get("flash_impl")) or "flash" in name.lower()
    areas = [n_q * n_kv for n_q, n_kv in rectangles]
    total_area = sum(areas)
    total = 0.0
    for (n_q, n_kv), area in zip(rectangles, areas):
        part_bytes = float(nbytes or 0.0) * area / total_area
        part = sdpa_fwd_time_us(
            arch,
            precision,
            dtype_a_b or precision,
            part_bytes,
            batch,
            heads,
            n_q,
            n_kv,
            head_dim,
            fa=flash,
            gemm_time=gemm_time,
        )
        if part is None:
            return None
        total += float(part)
    return total


def sdpa_tile_time_us(perf_model, arch, gemm_time, bwd=False):
    """Tile-model time in µs for standard or flash attention, else None.

    SDPA and flash perf models are timed. So is dense inference attention
    (paged and varlen flash). Sparse and linear attention are not.
    """
    if arch is None:
        return None
    if not isinstance(perf_model, SDPA):
        from .extensions.attention_perf_model_extensions import InferenceAttention

        if not isinstance(perf_model, InferenceAttention):
            return None
        if bwd:
            return None
        # DeepSeek-V4 paged decode is sparse even when the class name is not.
        if any(cls.__name__ == "_V4PagedDecodeBase" for cls in type(perf_model).mro()):
            return None
        try:
            nbytes = perf_model.bytes()
        except TypeError:
            nbytes = perf_model.bytes(2)
        return standard_attention_time_us(
            type(perf_model).__name__,
            perf_model.param_details,
            arch,
            nbytes,
            gemm_time=gemm_time,
        )
    params = perf_model.param_details
    dtype_A_B = params["dtype_A_B"][0]
    dtype = params.get("simulation_dtype")
    if dtype is None:
        dtype = torch_dtype_map(dtype_A_B)
    name = type(perf_model).__name__
    if bwd:
        fa = name in ("flash_attention", "flash_attention_backward")
        bytes = perf_model.bytes_bwd()
        tile_time = sdpa_bwd_time_us
    else:
        fa = name == "flash_attention"
        bytes = perf_model.bytes()
        tile_time = sdpa_fwd_time_us
    return tile_time(
        arch,
        dtype,
        dtype_A_B,
        bytes,
        perf_model.B,
        perf_model.H_Q,
        perf_model.N_Q,
        perf_model.N_KV,
        perf_model.d_h,
        fa,
        gemm_time=gemm_time,
    )


def tile_gemm_time(arch, M, N, K, dtype, force_to_l1=False, gemm_time=None):
    """Time in µs of one tile GEMM on one CU, or None without a GEMM model.

    ``force_to_l1`` asks the model to treat the operands as cache resident,
    as in flash attention."""
    if gemm_time is None:
        return None
    return gemm_time(
        arch,
        M=M,
        N=N,
        K=K,
        B=1,
        dtype=dtype,
        force_to_l1=force_to_l1,
        num_cus=1,
    )


def sdpa_fwd_time_us(
    arch,
    dtype,
    dtype_A_B,
    bytes,
    B,
    H_Q,
    N_Q,
    N_KV,
    d_h,
    fa=True,
    gemm_time=None,
):
    """Forward tile-model time in µs, or None without a GEMM model."""
    force_to_l1 = False
    block_N_Q = N_Q
    block_N_KV = N_KV

    if fa:
        force_to_l1 = True
        # Every Q tile block goes through full K and V, so we keep block_N_KV same
        # and Q tile size is 128 for all the cases observed
        block_N_Q = min(128, N_Q)
        # block_N_KV = min(self.N_KV, self.N_KV)

    num_blocks_N_Q = math.ceil(N_Q / block_N_Q)
    # num_blocks_N_KV = math.ceil(N_KV / block_N_KV)
    total_num_blocks = num_blocks_N_Q * B * H_Q
    num_waves = math.ceil(total_num_blocks / arch["num_cus"])

    qkt_time = tile_gemm_time(
        arch,
        block_N_Q,
        block_N_KV,
        d_h,
        dtype,
        force_to_l1,
        gemm_time,
    )
    if qkt_time is None:
        return None
    qkt_time = num_waves * qkt_time

    # Flash softmax stays in L1 when the arch names that bandwidth. Without
    # it, the softmax term uses HBM; the tile GEMMs are still Origami's.
    softmax_in_l1 = force_to_l1 and bool(arch.get("l1_bw_gbps"))
    softmax_time = num_waves * Softmax.get_time(
        arch,
        block_N_Q,
        block_N_KV,
        name2bpe(dtype_A_B),
        1,
        force_to_l1=softmax_in_l1,
        num_cus=1,
    )
    pv_time = tile_gemm_time(
        arch,
        block_N_Q,
        d_h,
        block_N_KV,
        dtype,
        force_to_l1,
        gemm_time,
    )
    if pv_time is None:
        return None
    pv_time = num_waves * pv_time

    mem_time = (
        bytes
        / N_Q
        / N_KV
        * block_N_Q
        * block_N_KV
        / (arch["mem_bw_gbps"] * 1000)
        * num_waves
    )
    return qkt_time + softmax_time + pv_time + mem_time


def sdpa_bwd_time_us(
    arch,
    dtype,
    dtype_A_B,
    bytes,
    B,
    H_Q,
    N_Q,
    N_KV,
    d_h,
    fa=True,
    gemm_time=None,
):
    """Backward tile-model time in µs, or None without a GEMM model."""
    force_to_l1 = False
    block_N_Q = N_Q
    block_N_KV = N_KV
    qkt_time = 0
    pv_time = 0

    if fa:
        force_to_l1 = True
        # ∇Q is tiled — but it is not partitioned exclusively across thread blocks the same way ∇K and ∇V are.
        # Instead, multiple thread blocks may contribute to the same ∇Q tile, which is why atomics are needed on ∇Q
        block_N_Q = min(N_Q, N_Q)
        block_N_KV = min(128, N_KV)

    num_blocks_N_KV = math.ceil(N_KV / block_N_KV)
    # Partition happens on ∇K and ∇V and not ∇Q
    total_num_blocks = num_blocks_N_KV * B * H_Q
    num_waves = math.ceil(total_num_blocks / arch["num_cus"])

    qkt_fwd_time = tile_gemm_time(
        arch,
        block_N_Q,
        block_N_KV,
        d_h,
        dtype,
        force_to_l1,
        gemm_time,
    )
    if qkt_fwd_time is None:
        return None

    qkt_fwd_time = num_waves * qkt_fwd_time

    # B = B * H_Q, M = N_Q, N = d_H, K = N_KV
    pv_fwd_time = tile_gemm_time(
        arch,
        block_N_Q,
        d_h,
        block_N_KV,
        dtype,
        force_to_l1,
        gemm_time,
    )
    if pv_fwd_time is None:
        return None
    pv_fwd_time = num_waves * pv_fwd_time

    if fa:
        # In case of flash attention we have to recompute
        # B = B * H_Q, M = N_Q, N = N_KV, K = d_H
        qkt_time = qkt_fwd_time
        pv_time = pv_fwd_time

    # We don't need to model these GEMMs again,
    # as we already have the times
    p_grad_time = qkt_fwd_time
    v_grad_time = pv_fwd_time
    q_grad_time = pv_fwd_time
    k_grad_time = pv_fwd_time

    # p_grad_time = pv_fwd_time
    # v_grad_time = qkt_fwd_time
    # q_grad_time = qkt_fwd_time
    # k_grad_time = qkt_fwd_time

    softmax_time = num_waves * Softmax.get_time(
        arch,
        block_N_Q,
        block_N_KV,
        name2bpe(dtype_A_B),
        1,
        force_to_l1=force_to_l1,
        num_cus=1,
    )

    # We assume that we use atomics for adding up the gradients together
    atomic_latency_global_ns = 400  # ns for global memory
    atomic_latency_local_ns = 40  # ns for shared memory/ L1
    # This is the tile size for ∇K. For every tile of ∇Q, we need to accumulate the contributions
    # from all the ∇K blocks
    k_tile = block_N_KV
    warp_size = 64

    # Shared-memory tile reduction:
    # Each block uses atomics only once per (k_tile × d)
    # This optimization won't be there for now possibly?
    num_k_tiles = math.ceil(block_N_KV / k_tile)

    # Warp-level reduction:
    # Each warp atomics once per d vector
    # warps_per_block = (block_N_Q * self.d_h) // warp_size
    warp_reduction_updates_per_block_global = math.ceil(
        num_k_tiles * math.ceil(d_h / warp_size)
    )
    total_updates_global = warp_reduction_updates_per_block_global * num_waves

    warp_reduction_updates_per_block_local = math.ceil(
        k_tile * math.ceil(d_h / warp_size)
    )
    total_updates_local = warp_reduction_updates_per_block_local * num_waves

    # Total atomic time (serialized across all blocks)
    total_atomic_time_us = (
        atomic_latency_global_ns * total_updates_global
        + atomic_latency_local_ns * total_updates_local
    ) / 1e3

    # We have to read the first block and write the last block
    mem_time = (
        bytes
        / N_Q
        / N_KV
        * block_N_Q
        * block_N_KV
        / (arch["mem_bw_gbps"] * 1000)
        * num_waves
    )
    simulated_time = (
        qkt_time
        + pv_time
        + p_grad_time
        + v_grad_time
        + q_grad_time
        + k_grad_time
        + softmax_time
        + total_atomic_time_us
        + mem_time
    )
    return simulated_time
