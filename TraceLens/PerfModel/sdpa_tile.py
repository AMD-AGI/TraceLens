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


def sdpa_tile_time_us(perf_model, arch, gemm_time, bwd=False):
    """Tile-model time in µs for an SDPA perf model, or None for other ops."""
    if not isinstance(perf_model, SDPA) or arch is None:
        return None
    params = perf_model.param_details
    dtype_A_B = params["dtype_A_B"][0]
    dtype = params.get("simulation_dtype")
    if dtype is None:
        dtype = torch_dtype_map(dtype_A_B)
    name = type(perf_model).__name__
    if bwd:
        fa = name in ("flash_attention", "flash_attention_backward")
        bytes = perf_model.bytes_bwd(name2bpe(dtype_A_B))
        tile_time = sdpa_bwd_time_us
    else:
        fa = name == "flash_attention"
        bytes = perf_model.bytes(name2bpe(dtype_A_B))
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

    softmax_time = num_waves * Softmax.get_time(
        arch,
        block_N_Q,
        block_N_KV,
        name2bpe(dtype_A_B),
        1,
        force_to_l1=force_to_l1,
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
