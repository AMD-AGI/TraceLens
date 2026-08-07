###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
import sys

from tqdm import tqdm

from .pseudo_ops_utils import inject_pseudo_op

logger = logging.getLogger(__name__)

# When checking for cpu_op extensions, do not descend into GPU leaf categories:
# they are huge in python_func-augmented traces and do not contain cpu_ops.
_SKIP_CPUOP_DESCENT_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

_ATTR_DESC_HAS_CPU = "_pseudo_desc_has_cpu_op"


def _compute_desc_has_cpu_op(trace_tree, event: dict, memo: dict) -> bool:
    """Same predicate as legacy ``_has_cpu_op_descendant``; fills *memo* and ``_ATTR_DESC_HAS_CPU``."""
    uid = event["UID"]
    if uid in memo:
        return memo[uid]
    out = False
    for child_uid in event.get("children", []):
        child = trace_tree.get_UID2event(child_uid)
        if child.get("cat") == "cpu_op":
            out = True
            break
        if child.get("non_gpu_path", False):
            continue
        if child.get("cat") in _SKIP_CPUOP_DESCENT_CATS:
            continue
        if _compute_desc_has_cpu_op(trace_tree, child, memo):
            out = True
            break
    memo[uid] = out
    event[_ATTR_DESC_HAS_CPU] = out
    return out


def annotate_cpu_op_descendant_flags(trace_tree) -> int:
    """One O(N) pass: set ``_pseudo_desc_has_cpu_op`` on every event (MoE fused skip logic).

    Returns:
        Number of events annotated.
    """
    memo: dict = {}
    n = 0
    for ev in trace_tree.events:
        _compute_desc_has_cpu_op(trace_tree, ev, memo)
        n += 1
    return n


def create_pseudo_ops_moe_fused_aiter(trace_tree, *, show_progress: bool = False):
    """
    Create pseudo ops for vllm::rocm_aiter_fused_moe operations. Isolates MoE compute kernel from indexing/quantization kernels.
    """

    if "vllm::rocm_aiter_fused_moe" not in trace_tree.name2event_uids:
        return

    moe_op_events = [
        trace_tree.get_UID2event(uid)
        for uid in trace_tree.name2event_uids["vllm::rocm_aiter_fused_moe"]
    ]

    n_ann = annotate_cpu_op_descendant_flags(trace_tree)

    skipped_cpu_desc = 0
    skipped_no_gpu = 0
    skipped_kernel_count = 0
    injected = 0

    for moe_op_event in moe_op_events:
        r = _create_pseudo_op_moe_fused_aiter(trace_tree, moe_op_event)
        if r == "skip_cpu_desc":
            skipped_cpu_desc += 1
        elif r == "skip_no_gpu":
            skipped_no_gpu += 1
        elif r == "skip_kernel_count":
            skipped_kernel_count += 1
        elif r == "injected":
            injected += 1

    msg = (
        f"MoE_Fused (AITER): annotated_events={n_ann} moe_ops={len(moe_op_events)} "
        f"injected={injected} skipped_cpu_op_descendant={skipped_cpu_desc} "
        f"skipped_no_gpu_events={skipped_no_gpu} skipped_bad_kernel_count={skipped_kernel_count}"
    )
    logger.info("pseudo_ops %s", msg)
    if show_progress:
        tqdm.write(f"perf: pseudo_ops: {msg}", file=sys.stderr)


def is_aiter_fused_moe_kernel(kernel_event: dict) -> bool:
    """Check if kernel is MoE compute (not sorting/quantization)."""

    if kernel_event.get("cat") != "kernel":
        return False

    kernel_name = kernel_event["name"]
    is_moe_kernel_match = (
        "aiter::" in kernel_name
        and "fmoe" in kernel_name
        and "MoeSorting" not in kernel_name
        and "quant" not in kernel_name.lower()
    )

    return is_moe_kernel_match


def _create_pseudo_op_moe_fused_aiter(trace_tree, moe_op_event: dict) -> str:
    """Create single pseudo op for one MoE operation. Returns outcome tag for stats."""

    if moe_op_event.get("name") != "vllm::rocm_aiter_fused_moe":
        logger.warning(
            f"Expected vllm::rocm_aiter_fused_moe, found {moe_op_event['name']}"
        )
        return "skip_name"

    if moe_op_event.get(_ATTR_DESC_HAS_CPU, False):
        logger.info(
            f"Skipping pseudo op for UID {moe_op_event['UID']}: has cpu_op descendant"
        )
        return "skip_cpu_desc"

    gpu_event_ids = moe_op_event.get("gpu_events", [])
    if not gpu_event_ids:
        logger.warning(f"No GPU events for MoE UID {moe_op_event['UID']}")
        return "skip_no_gpu"

    gpu_events = [trace_tree.get_UID2event(uid) for uid in gpu_event_ids]
    moe_kernels = [e for e in gpu_events if is_aiter_fused_moe_kernel(e)]

    if len(moe_kernels) != 1:
        logger.warning(
            f"Expected 1 MoE kernel, found {len(moe_kernels)} for UID {moe_op_event['UID']}"
        )
        return "skip_kernel_count"

    moe_kernel = moe_kernels[0]
    seq_num = moe_op_event["args"].get("Sequence number", moe_op_event["UID"])

    inject_pseudo_op(
        trace_tree,
        moe_kernel,
        "pseudo_op::moe_aiter_fused_1stage",
        seq_num,
        dims=moe_op_event["args"].get("Input Dims"),
        types=moe_op_event["args"].get("Input type"),
        strides=moe_op_event["args"].get("Input Strides"),
        concrete_inputs=moe_op_event["args"].get("Concrete Inputs"),
    )
    return "injected"
