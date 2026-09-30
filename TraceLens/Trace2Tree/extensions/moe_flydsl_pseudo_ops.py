###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging

from .pseudo_ops_utils import inject_pseudo_op_above_event

logger = logging.getLogger(__name__)


STAGE_WRAPPERS = {
    "pseudo_op::moe_flydsl_stage1": "flydsl_moe_stage1",
    "pseudo_op::moe_flydsl_stage2": "flydsl_moe_stage2",
}

FUSED_MOE_PARENT = "aiter::fused_moe_"
_PYTHON_FUNC_CATS = {"python_func", "python_function"}


def create_pseudo_ops_moe_flydsl(trace_tree):
    """
    Create pseudo ops for the two flydsl MoE stages.

    Strategy: look up
    each stage marker in the tree's name index, then walk up the parent chain
    to confirm the stage event lives under an aiter::fused_moe_ ancestor. The
    pseudo op is injected as the new parent of the stage event, inheriting
    Input Dims / Input type / Input Strides / Concrete Inputs / Sequence number
    from the aiter::fused_moe_ donor.

    The extension is a no-op if aiter::fused_moe_ is not in the trace.
    """

    if FUSED_MOE_PARENT not in trace_tree.name2event_uids:
        return

    # Resolve full event names that contain each marker.
    marker_to_event_names = {marker: [] for marker in STAGE_WRAPPERS.values()}
    for name in trace_tree.name2event_uids:
        for marker in marker_to_event_names:
            if name.endswith(": " + marker):
                marker_to_event_names[marker].append(name)

    for pseudo_name, marker in STAGE_WRAPPERS.items():
        matched_names = marker_to_event_names[marker]
        if not matched_names:
            logger.debug(f"No events matching marker {marker!r}")
            continue

        injected = 0
        for ev_name in matched_names:
            for uid in trace_tree.name2event_uids[ev_name]:
                stage_evt = trace_tree.get_UID2event(uid)
                if stage_evt.get("cat") not in _PYTHON_FUNC_CATS:
                    continue
                fused_moe_evt = _find_fused_moe_ancestor(trace_tree, stage_evt)
                if fused_moe_evt is None:
                    continue
                seq_num = fused_moe_evt.get("args", {}).get(
                    "Sequence number", fused_moe_evt["UID"]
                )
                inject_pseudo_op_above_event(
                    trace_tree,
                    stage_evt,
                    pseudo_name,
                    shape_donor_evt=fused_moe_evt,
                    extra_args={
                        "Sequence number": seq_num,
                        "MoE flydsl stage": marker,
                    },
                )
                injected += 1

        logger.info(f"Injected {injected} {pseudo_name} pseudo ops")


def _find_fused_moe_ancestor(trace_tree, evt: dict):
    """
    Walk up the parent chain from evt and return the nearest aiter::fused_moe_
    ancestor, or None if no such ancestor exists.
    """

    current = trace_tree.get_parent_event(evt)
    while current is not None:
        if current.get("name") == FUSED_MOE_PARENT:
            return current
        current = trace_tree.get_parent_event(current)
    return None
