###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pseudo ops for the expert-parallel mega-MoE path under ``aiter::moe_forward``.

Each ``aiter::moe_forward`` launches a routing plan, a token dispatch to peer
GPUs, the two grouped expert GEMMs (inside ``aiter::fused_moe_``), then the
combine barrier and the top-k reduce. The token count is recorded only on
``moe_forward``, and the expert geometry only on ``fused_moe_``, whose
activation is the fixed-size receive arena. Each kernel therefore gets its own
pseudo op that carries ``moe_forward``'s shapes plus the geometry as extra args.
"""

import logging
import re

from .pseudo_ops_utils import inject_pseudo_op

logger = logging.getLogger(__name__)

# Constants
MOE_FORWARD = "aiter::moe_forward"
FUSED_MOE = "aiter::fused_moe_"
EXPERT_GEMM_RE = re.compile(r"^a8w4_tdm_fp4_t\d+x\d+x\d+_.*?_K(\d+)_e\d+")
EXPERT_GEMM_UP = "pseudo_op::mega_moe_gemm_up"
EXPERT_GEMM_DOWN = "pseudo_op::mega_moe_gemm_down"
ROUTING_PSEUDO_OPS = (
    (re.compile(r"^tdm_compact_plan"), "pseudo_op::mega_moe_plan"),
    (re.compile(r"^ep_dispatch_tdm_"), "pseudo_op::mega_moe_dispatch"),
    (re.compile(r"^ep_combine_fused_sync"), "pseudo_op::mega_moe_combine_sync"),
    (re.compile(r"^ep_combine_fused_"), "pseudo_op::mega_moe_combine"),
)
# aiter::fused_moe_ argument positions on the mega-MoE path.
_RECV_PAYLOAD, _W1, _W2, _TOPK_WEIGHTS = 0, 1, 2, 3
_W1_SCALE, _W2_SCALE, _RECV_SCALE = 9, 10, 11


def create_pseudo_ops_moe_mega_ep(trace_tree):
    """Inject one pseudo op per mega-MoE kernel under each aiter::moe_forward."""

    injected = 0
    for uid in trace_tree.name2event_uids.get(MOE_FORWARD, []):
        moe_forward_evt = trace_tree.get_UID2event(uid)
        fused_moe_evt = _find_expert_gemm_op(trace_tree, moe_forward_evt)
        if fused_moe_evt is None:
            continue
        injected += _inject_layer(trace_tree, moe_forward_evt, fused_moe_evt)
    logger.info(f"Injected {injected} mega-MoE pseudo ops")


def _find_expert_gemm_op(trace_tree, evt):
    """The aiter::fused_moe_ descendant that carries the expert weights."""

    for child_evt in trace_tree.get_children_events(evt):
        if child_evt.get("name") == FUSED_MOE:
            dims = child_evt.get("args", {}).get("Input Dims") or []
            if len(dims) > _RECV_SCALE and dims[_W1] and dims[_W2]:
                return child_evt
        found = _find_expert_gemm_op(trace_tree, child_evt)
        if found is not None:
            return found
    return None


def _inject_layer(trace_tree, moe_forward_evt, fused_moe_evt):
    args = moe_forward_evt["args"]
    geometry = _expert_geometry(fused_moe_evt["args"]["Input Dims"])
    hidden_dim = args["Input Dims"][0][-1]
    seq_num = args.get("Sequence number", moe_forward_evt["UID"])
    injected = 0
    for kernel_evt in trace_tree.get_gpu_events(moe_forward_evt):
        name = _pseudo_op_name(kernel_evt.get("name", ""), hidden_dim)
        if name is None:
            continue
        inject_pseudo_op(
            trace_tree,
            kernel_evt,
            name,
            seq_num,
            dims=args.get("Input Dims"),
            types=args.get("Input type"),
            strides=args.get("Input Strides"),
            concrete_inputs=args.get("Concrete Inputs"),
            extra_args=geometry,
        )
        injected += 1
    return injected


def _expert_geometry(dims):
    w1, w2 = dims[_W1], dims[_W2]
    return {
        "mega_moe_topk": dims[_TOPK_WEIGHTS][-1],
        "mega_moe_local_experts": w1[0],
        "mega_moe_inter_dim": w1[1] // 2,
        "mega_moe_w1_shape": list(w1),
        "mega_moe_w2_shape": list(w2),
        "mega_moe_w1_scale_shape": list(dims[_W1_SCALE]),
        "mega_moe_w2_scale_shape": list(dims[_W2_SCALE]),
        "mega_moe_wire_row_bytes": dims[_RECV_PAYLOAD][-1],
        "mega_moe_wire_scale_bytes": dims[_RECV_SCALE][-1],
    }


def _pseudo_op_name(kernel_name, hidden_dim):
    gemm = EXPERT_GEMM_RE.match(kernel_name)
    if gemm:
        return EXPERT_GEMM_UP if int(gemm.group(1)) == hidden_dim else EXPERT_GEMM_DOWN
    for pattern, name in ROUTING_PSEUDO_OPS:
        if pattern.match(kernel_name):
            return name
    return None
