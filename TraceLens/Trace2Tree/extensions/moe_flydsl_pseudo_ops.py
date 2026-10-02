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
    # Opus a8w4 down projection; the decode wrapper is used rather than
    # opus_moe_stage2_a8w4_fwd so the sibling route-reduce kernel stays out of
    # the GEMM's roofline.
    "pseudo_op::moe_opus_stage2_a8w4": "opus_moe_stage2_a8w4_decode_fwd",
}

FUSED_MOE_PARENT = "aiter::fused_moe_"
_PYTHON_FUNC_CATS = {"python_func", "python_function"}
_SGLANG_PROFILER_PREFIX = "sglang_profiler::"


def create_pseudo_ops_moe_flydsl(trace_tree):
    """
    Create pseudo ops for the two-stage MoE implementations under
    aiter::fused_moe_ (flydsl stage1/stage2, opus a8w4 stage2).

    Strategy: look up
    each stage marker in the tree's name index, then walk up the parent chain
    to confirm the stage event lives under an aiter::fused_moe_ ancestor. The
    pseudo op is injected as the new parent of the stage event, inheriting
    Input Dims / Input type / Input Strides / Concrete Inputs / Sequence number
    from the aiter::fused_moe_ donor. The donor only carries the pre-quant
    activation dtype, so the quantized operand dtypes are read off the stage
    wrapper and attached alongside.

    The extension is a no-op if aiter::fused_moe_ is not in the trace.
    """

    if FUSED_MOE_PARENT not in trace_tree.name2event_uids:
        return

    for pseudo_name, marker in STAGE_WRAPPERS.items():
        matched_names, allowed_cats = _stage_event_names(trace_tree, marker)
        if not matched_names:
            logger.debug(f"No events matching marker {marker!r}")
            continue

        injected = 0
        quant_dtypes = None
        for ev_name in matched_names:
            for uid in trace_tree.name2event_uids[ev_name]:
                stage_evt = trace_tree.get_UID2event(uid)
                if stage_evt.get("cat") not in allowed_cats:
                    continue
                fused_moe_evt = _find_fused_moe_ancestor(trace_tree, stage_evt)
                if fused_moe_evt is None:
                    continue
                seq_num = fused_moe_evt.get("args", {}).get(
                    "Sequence number", fused_moe_evt["UID"]
                )
                quant_dtypes = _stage_quant_dtypes(trace_tree, stage_evt)
                extra_args = {
                    "Sequence number": seq_num,
                    "MoE stage": marker,
                }
                if quant_dtypes:
                    extra_args["MoE quant input type"] = quant_dtypes[0]
                    extra_args["MoE quant weight type"] = quant_dtypes[1]
                inject_pseudo_op_above_event(
                    trace_tree,
                    stage_evt,
                    pseudo_name,
                    shape_donor_evt=fused_moe_evt,
                    extra_args=extra_args,
                )
                injected += 1

        logger.info(f"Injected {injected} {pseudo_name} pseudo ops")
        if injected and quant_dtypes and quant_dtypes[0] != quant_dtypes[1]:
            logger.warning(
                f"{pseudo_name}: mixed-precision MoE GEMM "
                f"({quant_dtypes[0]} x {quant_dtypes[1]}). The roofline is "
                "taken against the activation-dtype peak and is an estimate; "
                "the achievable rate for this operand mix may differ."
            )


def _stage_event_names(trace_tree, marker: str):
    """
    Return the event names carrying a stage marker plus the event categories
    they are expected to have.

    The kernel-shape-profiler op ("sglang_profiler::<module>_<launcher>") is the
    outermost wrapper of a stage and carries its operand types, so it is
    preferred. A trace profiled without that tool only exposes the stage as a
    Python frame ("<file>(<line>): flydsl_moe_stage1"), which then anchors the
    injection instead. Returning a single kind keeps a trace carrying both from
    yielding two nested pseudo ops per stage.
    """

    profiler_ops = [
        n
        for n in trace_tree.name2event_uids
        if n.startswith(_SGLANG_PROFILER_PREFIX) and n.endswith(marker)
    ]
    if profiler_ops:
        return profiler_ops, {"cpu_op"}

    py_frames = [n for n in trace_tree.name2event_uids if n.endswith(": " + marker)]
    return py_frames, _PYTHON_FUNC_CATS


def _stage_quant_dtypes(trace_tree, stage_evt: dict):
    """
    Return (activation dtype, weight dtype) for a stage, or None.

    The kernel-shape-profiler op for a stage lists the operands the GEMM
    actually consumes, which for a quantized MoE are narrower than the
    aiter::fused_moe_ activations. It is the stage event itself in a stack-less
    trace and its first descendant otherwise.
    """

    stack = [stage_evt]
    while stack:
        evt = stack.pop(0)
        if evt.get("name", "").startswith(_SGLANG_PROFILER_PREFIX):
            types = (evt.get("args") or {}).get("Input type") or []
            if len(types) >= 2 and types[0] and types[1]:
                return types[0], types[1]
        stack.extend(trace_tree.get_children_events(evt) or [])
    return None


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
