###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Map Flash MoE gemm2_a4w4 / moe_reduction Synthetic Op children to stage2.

Avoids putting FLOPs classes on bare Synthetic Op names: inject a stable
pseudo above each matching launcher/child under aiter::fused_moe_, inheriting
fused_moe_ Input Dims for the existing moe_flydsl_stage2 model.
"""

import logging

from .pseudo_ops_utils import inject_pseudo_op_above_event

logger = logging.getLogger(__name__)

FUSED_MOE_PARENT = "aiter::fused_moe_"
GEMM2_MARK = "gemm2_a4w4"
REDUCE_MARK = "moe_reduction_kernel"
PSEUDO_GEMM2 = "pseudo_op::moe_flydsl_gemm2_a4w4"
PSEUDO_REDUCE = "pseudo_op::moe_flydsl_moe_reduction"


def create_pseudo_ops_sglang_flash_moe_children(trace_tree):
    if FUSED_MOE_PARENT not in trace_tree.name2event_uids:
        return
    fused_uids = list(trace_tree.name2event_uids[FUSED_MOE_PARENT])
    for fused_uid in fused_uids:
        fused = trace_tree.get_UID2event(fused_uid)
        _walk_inject(trace_tree, fused, fused)


def _walk_inject(trace_tree, root, fused_moe_evt):
    stack = list(root.get("children") or [])
    while stack:
        uid = stack.pop()
        evt = trace_tree.get_UID2event(uid)
        if evt is None:
            continue
        name = evt.get("name") or ""
        if name in (PSEUDO_GEMM2, PSEUDO_REDUCE) or name.startswith(
            "pseudo_op::moe_flydsl"
        ):
            continue
        if GEMM2_MARK in name and "Synthetic Op" in name:
            inject_pseudo_op_above_event(
                trace_tree,
                evt,
                PSEUDO_GEMM2,
                shape_donor_evt=fused_moe_evt,
                extra_args={"MoE flash child": GEMM2_MARK},
            )
            logger.info("Injected %s above %s", PSEUDO_GEMM2, name[:80])
            continue
        if REDUCE_MARK in name and "Synthetic Op" in name:
            inject_pseudo_op_above_event(
                trace_tree,
                evt,
                PSEUDO_REDUCE,
                shape_donor_evt=fused_moe_evt,
                extra_args={"MoE flash child": REDUCE_MARK},
            )
            logger.info("Injected %s above %s", PSEUDO_REDUCE, name[:80])
            continue
        stack.extend(evt.get("children") or [])
