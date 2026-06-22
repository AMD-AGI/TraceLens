###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
import sys

from tqdm import tqdm

from .pseudo_ops_utils import inject_pseudo_ops_batch

logger = logging.getLogger(__name__)


def create_pseudo_ops_moe_unfused_triton(trace_tree, *, show_progress: bool = False):
    """
    Create pseudo ops for vllm::moe_forward unfused operations.
    Creates separate pseudo ops for each matmul_ogs GEMM kernel.

    Args:
        trace_tree: TraceToTree instance
    """

    if "vllm::moe_forward" not in trace_tree.name2event_uids:
        return

    moe_op_events = [
        trace_tree.get_UID2event(uid)
        for uid in trace_tree.name2event_uids["vllm::moe_forward"]
    ]

    logger.info(f"Processing {len(moe_op_events)} unfused MoE operations")

    total_specs = 0
    total_injected = 0
    skipped = 0

    for moe_op_event in moe_op_events:
        n_specs, n_ins, sk = _create_pseudo_op_moe_unfused_triton(
            trace_tree, moe_op_event
        )
        total_specs += n_specs
        total_injected += n_ins
        skipped += sk

    msg = (
        f"MoE_Unfused_Triton: moe_forwards={len(moe_op_events)} batch_specs={total_specs} "
        f"pseudo_ops_inserted={total_injected} moe_forwards_skipped={skipped} "
        f"(inject_pseudo_ops_batch: one children rewrite per parent per forward)"
    )
    logger.info("pseudo_ops %s", msg)
    if show_progress:
        tqdm.write(f"perf: pseudo_ops: {msg}", file=sys.stderr)


def is_matmul_ogs_kernel(kernel_event: dict) -> bool:
    """
    Check if kernel is matmul_ogs (unfused MoE GEMM).

    Args:
        kernel_event: Event dictionary

    Returns:
        True if kernel is a matmul_ogs kernel
    """

    return (
        kernel_event.get("cat") == "kernel"
        and "matmul_ogs" in kernel_event["name"].lower()
    )


def _is_gated_kernel(kernel_name: str) -> bool:
    """
    Check if kernel has gating activation (swiglu/glu).

    Args:
        kernel_name: Name of the matmul_ogs kernel

    Returns:
        True if kernel has swiglu or glu activation, False otherwise
    """
    kernel_lower = kernel_name.lower()
    return "_swiglu" in kernel_lower or "_glu" in kernel_lower


def _extract_topk_from_moe(trace_tree, moe_op_event: dict):
    """
    Extract topk (number of active experts per token) from TopK child operation.

    Uses TraceLens native TraceToTree methods for tree traversal.

    Args:
        trace_tree: TraceToTree instance
        moe_op_event: vllm::moe_forward event

    Returns:
        topk value as int, or 4 as default if not found

    Note:
        Returns default value of 4 if TopK operation not found (e.g., when python_func layer is inserted)
    """

    def search_for_topk(event, depth=0, max_depth=3):
        """Recursively search for TopK operation through Python function wrappers."""

        if depth > max_depth:
            return None

        # Check if this is a TopK operation
        if "TopK" in event.get("name", ""):
            concrete = event["args"].get("Concrete Inputs", [])
            # TopK's k parameter is the second concrete input
            if len(concrete) > 1 and concrete[1]:
                try:
                    return int(concrete[1])
                except (ValueError, TypeError):
                    pass

        # Recursively search children (for Python function wrappers)
        # Use native get_children_events() method
        for child_event in trace_tree.get_children_events(event):
            if child_event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            result = search_for_topk(child_event, depth + 1, max_depth)
            if result is not None:
                return result

        return None

    # Look for TopK child operation (with recursive search for Python wrappers)
    # Use native get_children_events() method
    for child_event in trace_tree.get_children_events(moe_op_event):
        topk_value = search_for_topk(child_event, max_depth=10)
        if topk_value is not None:
            return topk_value

    # TopK not found - return default value of 4 (common MoE configuration)
    logger.warning(
        f"TopK operation not found in children of vllm::moe_forward (UID={moe_op_event['UID']})."
    )

    return 4  # Default topk value


def _create_pseudo_op_moe_unfused_triton(trace_tree, moe_op_event: dict):
    """
    Create pseudo ops for each matmul_ogs kernel in unfused MoE.

    Returns:
        (num_specs, num_inserted, skipped_flag) where skipped_flag is 1 if this moe_forward did nothing.
    """

    if moe_op_event.get("name") != "vllm::moe_forward":
        logger.warning(f"Expected vllm::moe_forward, found {moe_op_event['name']}")
        return (0, 0, 1)

    gpu_event_ids = moe_op_event.get("gpu_events", [])
    if not gpu_event_ids:
        logger.warning(f"No GPU events for MoE UID {moe_op_event['UID']}")
        return (0, 0, 1)

    gpu_events = [trace_tree.get_UID2event(uid) for uid in gpu_event_ids]
    matmul_kernels = [e for e in gpu_events if is_matmul_ogs_kernel(e)]

    if len(matmul_kernels) == 0:
        logger.warning(f"No matmul_ogs kernels found for UID {moe_op_event['UID']}")
        return (0, 0, 1)

    # Extract topk from tree (raises ValueError if not found)
    topk = _extract_topk_from_moe(trace_tree, moe_op_event)

    seq_num = moe_op_event["args"].get("Sequence number", moe_op_event["UID"])

    # Sort by start time to ensure first GEMM is up, second is down
    matmul_kernels_sorted = sorted(matmul_kernels, key=lambda k: k["ts"])

    common_tail = (
        moe_op_event["args"].get("Input Dims"),
        moe_op_event["args"].get("Input type"),
        moe_op_event["args"].get("Input Strides"),
        moe_op_event["args"].get("Concrete Inputs"),
    )

    specs = []

    # Create pseudo op for each GEMM
    for idx, kernel in enumerate(matmul_kernels_sorted):

        # First GEMM is up projection, second is down projection
        gemm_type = "up" if idx == 0 else "down"
        gated = _is_gated_kernel(kernel["name"])

        pseudo_op_name = f"pseudo_op::moe_triton_unfused_{gemm_type}"

        extra_args = {
            "MoE GEMM type": gemm_type,
            "MoE GEMM index": idx,
            "MoE GEMM gated": gated,
            "MoE topk": topk,
        }

        specs.append((kernel, pseudo_op_name, seq_num) + common_tail + (extra_args,))

    finalize_matmul_scatter_kernel = [
        e for e in gpu_events if "matmul_scatter" in e["name"].lower()
    ]
    if len(finalize_matmul_scatter_kernel) > 1:
        logger.warning(
            f"Multiple matmul_scatter kernels found for UID {moe_op_event['UID']}"
        )
        return (0, 0, 1)
    if len(finalize_matmul_scatter_kernel) == 1:
        fin = finalize_matmul_scatter_kernel[0]
        specs.append(
            (
                fin,
                "pseudo_op::moe_triton_unfused_finalize_matmul_scatter",
                seq_num,
            )
            + common_tail
            + ({"MoE GEMM type": "finalize_matmul_scatter"},)
        )

    n_ins = inject_pseudo_ops_batch(trace_tree, specs)
    return (len(specs), n_ins, 0)
