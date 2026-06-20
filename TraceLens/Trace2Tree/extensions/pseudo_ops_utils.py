###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import logging
from typing import Any, Optional, List, Callable

logger = logging.getLogger(__name__)


_SGLANG_SUFFIX_RE = re.compile(r"^(sglang_profiler::.+?)_\d+$")


def normalize_sglang_profiler_op_names(tree):
    """Strip volatile trailing _<digits> from sglang_profiler cpu_op names."""
    for old in [n for n in tree.name2event_uids if n.startswith("sglang_profiler::")]:
        m = _SGLANG_SUFFIX_RE.match(old)
        if not m:
            continue
        uids = tree.name2event_uids[old]
        if not uids or tree.events_by_uid[uids[0]].get("cat") != "cpu_op":
            continue
        new = m.group(1)
        for uid in uids:
            tree.events_by_uid[uid]["name"] = new
        tree.name2event_uids.setdefault(new, []).extend(uids)
        del tree.name2event_uids[old]


def set_bookkeeping_attr(tree, event: dict):
    """Add bookkeeping attributes for a new pseudo event in the tree."""

    UID = len(tree.events)
    event["UID"] = UID
    tree.events.append(event)
    tree.events_by_uid[UID] = event

    seq_num = event["args"].get("Sequence number")
    if seq_num is not None:
        if seq_num not in tree.seq_num2event_uids_map:
            tree.seq_num2event_uids_map[seq_num] = []
        tree.seq_num2event_uids_map[seq_num].append(UID)


def inject_pseudo_op(
    tree,
    kernel_evt,
    name,
    seq_num,
    dims=None,
    types=None,
    strides=None,
    concrete_inputs=None,
    extra_args=None,
):
    """
    Create pseudo op between parent CPU op and kernel.
    Creates: Parent CPU Op → Pseudo Op → Launcher → Kernel

    Args:
        tree: TraceToTree instance
        kernel_evt: Kernel event to inject pseudo-op for
        name: Name of the pseudo-op
        seq_num: Sequence number
        dims: Input dimensions (uses parent if None)
        types: Input types (uses parent if None)
        strides: Input strides (uses parent if None)
        concrete_inputs: Concrete inputs (uses parent if None)
        extra_args: Additional custom args to add to pseudo-op (dict)
    """

    launcher_evt = tree.get_parent_event(kernel_evt)
    orig_cpu_evt = tree.get_parent_event(launcher_evt)

    pseudo_evt = {
        "ph": "X",
        "name": name,
        "cat": "cpu_op",
        "pid": orig_cpu_evt["pid"],
        "tid": orig_cpu_evt["tid"],
        "args": {
            "Input Dims": (
                orig_cpu_evt["args"].get("Input Dims") if dims is None else dims
            ),
            "Input type": (
                orig_cpu_evt["args"].get("Input type") if types is None else types
            ),
            "Input Strides": (
                orig_cpu_evt["args"].get("Input Strides")
                if strides is None
                else strides
            ),
            "Concrete Inputs": (
                orig_cpu_evt["args"].get("Concrete Inputs")
                if concrete_inputs is None
                else concrete_inputs
            ),
            "Sequence number": seq_num,
            "External id": kernel_evt["args"]["correlation"],
            "Pseudo op": True,
        },
        "children": [launcher_evt["UID"]],
        "gpu_events": [kernel_evt["UID"]],
    }

    # Add any extra custom args
    if extra_args:
        pseudo_evt["args"].update(extra_args)

    set_bookkeeping_attr(tree, pseudo_evt)

    pseudo_evt["parent"] = orig_cpu_evt["UID"]
    children = orig_cpu_evt["children"]
    children.remove(launcher_evt["UID"])
    children.append(pseudo_evt["UID"])


def inject_pseudo_op_wrap_children(
    tree,
    parent_evt,
    name,
    shape_donor_evt=None,
    extra_args=None,
):
    """
    Create pseudo op that wraps all children of a parent event.
    Creates: Parent → Pseudo Op → [all original children]

    Unlike inject_pseudo_op (which isolates a single kernel), this wraps
    the entire subtree under a parent into a single pseudo op.

    Args:
        tree: TraceToTree instance
        parent_evt: Parent event whose children will be wrapped
        name: Name of the pseudo-op
        shape_donor_evt: Event to inherit shapes from (uses parent if None)
        extra_args: Additional custom args to add to pseudo-op (dict)
    """

    children_uids = parent_evt.get("children", [])
    if not children_uids:
        return

    donor = shape_donor_evt if shape_donor_evt is not None else parent_evt
    donor_args = donor.get("args", {})

    pseudo_evt = {
        "ph": "X",
        "name": name,
        "cat": "cpu_op",
        "pid": parent_evt["pid"],
        "tid": parent_evt["tid"],
        "ts": parent_evt["ts"],
        "dur": parent_evt["dur"],
        "args": {
            "Input Dims": donor_args.get("Input Dims"),
            "Input type": donor_args.get("Input type"),
            "Input Strides": donor_args.get("Input Strides"),
            "Concrete Inputs": donor_args.get("Concrete Inputs"),
            "Sequence number": donor_args.get("Sequence number", parent_evt.get("UID")),
            "Pseudo op": True,
        },
        "children": list(children_uids),
        "gpu_events": list(parent_evt.get("gpu_events", [])),
    }

    if extra_args:
        pseudo_evt["args"].update(extra_args)

    set_bookkeeping_attr(tree, pseudo_evt)

    for child_uid in children_uids:
        child_evt = tree.get_UID2event(child_uid)
        child_evt["parent"] = pseudo_evt["UID"]

    parent_evt["children"] = [pseudo_evt["UID"]]
    pseudo_evt["parent"] = parent_evt["UID"]

    # Descendants that were cpu_root_nodes are no longer roots since they now
    # live under the pseudo op. Removals/additions are accumulated into per-tree
    # pending sets and callers MUST invoke ``finalize_pseudo_op_mutations(tree)`` after the
    # last call to commit the accumulated changes back to ``tree.cpu_root_nodes``.
    pending_removals = getattr(tree, "_cpu_root_nodes_pending_removals", None)
    if pending_removals is None:
        pending_removals = set()
        tree._cpu_root_nodes_pending_removals = pending_removals
        tree._cpu_root_nodes_pending_additions = []
        tree._cpu_root_nodes_set = set(tree.cpu_root_nodes)
    pending_additions = tree._cpu_root_nodes_pending_additions
    root_set = tree._cpu_root_nodes_set

    stack = list(children_uids)
    while stack:
        uid = stack.pop()
        if uid in root_set:
            pending_removals.add(uid)
            root_set.discard(uid)
        evt = tree.get_UID2event(uid)
        stack.extend(evt.get("children", []))
    pending_additions.append(pseudo_evt["UID"])


def finalize_pseudo_op_mutations(tree):
    """Commit any pending cpu_root_nodes mutations from
    ``inject_pseudo_op_wrap_children`` back to ``tree.cpu_root_nodes``
    """
    pending_removals = getattr(tree, "_cpu_root_nodes_pending_removals", None)
    if pending_removals is None:
        return
    pending_additions = tree._cpu_root_nodes_pending_additions
    if pending_removals:
        tree.cpu_root_nodes = [
            u for u in tree.cpu_root_nodes if u not in pending_removals
        ]
    if pending_additions:
        tree.cpu_root_nodes.extend(pending_additions)
    del tree._cpu_root_nodes_pending_removals
    del tree._cpu_root_nodes_pending_additions
    if hasattr(tree, "_cpu_root_nodes_set"):
        del tree._cpu_root_nodes_set


def inject_pseudo_op_above_event(
    tree,
    target_evt,
    name,
    shape_donor_evt=None,
    extra_args=None,
):
    """
    Insert a new pseudo cpu_op between target_evt and its current parent.

    Resulting layout: parent -> pseudo_evt -> target_evt (target's subtree unchanged).
    Pseudo args (Input Dims / Input type / Input Strides / Concrete Inputs /
    Sequence number) are inherited from shape_donor_evt; if None, falls back
    to the target's current parent.

    Args:
        tree: TraceToTree instance
        target_evt: Existing event the pseudo op should wrap (becomes its sole child)
        name: Name of the pseudo-op
        shape_donor_evt: Event to inherit shapes from (uses parent if None)
        extra_args: Additional custom args to add to pseudo-op (dict)

    Returns:
        The pseudo event dict, or None if target_evt has no parent.
    """

    parent_evt = tree.get_parent_event(target_evt)
    if parent_evt is None:
        logger.warning(
            f"inject_pseudo_op_above_event: target UID {target_evt.get('UID')} "
            f"has no parent; skipping injection of {name}"
        )
        return None

    donor = shape_donor_evt if shape_donor_evt is not None else parent_evt
    donor_args = donor.get("args", {})

    pseudo_evt = {
        "ph": "X",
        "name": name,
        "cat": "cpu_op",
        "pid": target_evt.get("pid", parent_evt.get("pid")),
        "tid": target_evt.get("tid", parent_evt.get("tid")),
        "ts": target_evt.get("ts"),
        "dur": target_evt.get("dur"),
        "args": {
            "Input Dims": donor_args.get("Input Dims"),
            "Input type": donor_args.get("Input type"),
            "Input Strides": donor_args.get("Input Strides"),
            "Concrete Inputs": donor_args.get("Concrete Inputs"),
            "Sequence number": donor_args.get("Sequence number", parent_evt.get("UID")),
            "Pseudo op": True,
        },
        "children": [target_evt["UID"]],
        "gpu_events": list(target_evt.get("gpu_events", [])),
    }

    if extra_args:
        pseudo_evt["args"].update(extra_args)

    set_bookkeeping_attr(tree, pseudo_evt)

    pseudo_evt["parent"] = parent_evt["UID"]
    parent_children = parent_evt["children"]
    idx = parent_children.index(target_evt["UID"])
    parent_children[idx] = pseudo_evt["UID"]
    target_evt["parent"] = pseudo_evt["UID"]

    return pseudo_evt


def apply_pseudo_op_extensions(tree, verbose: bool = False):
    """
    Apply all available pseudo-op extensions to trace tree.
    Extensions are automatically detected and applied.
    """

    normalize_sglang_profiler_op_names(tree)

    # Auto-detect and add all known pseudo-op extensions
    extensions = []

    if "vllm::moe_forward" in tree.name2event_uids:

        # MoE: AITER Fused Implementation
        if "vllm::rocm_aiter_fused_moe" in tree.name2event_uids:
            from .moe_aiter_pseudo_ops import create_pseudo_ops_moe_fused_aiter

            extensions.append(("MoE_Fused", create_pseudo_ops_moe_fused_aiter))
            if verbose:
                logger.info("Auto-detected fused MoE operations")

        # MoE: Triton Fused Implementation
        # TO DO: Update kernel detection approach (Look for gpt_oss_triton_kernels_moe.py)
        else:
            # Check if any kernel events contain matmul_ogs: Triton MoE kernel
            has_matmul_ogs = any(
                "matmul_ogs" in event.get("name", "").lower()
                for event in tree.events
                if event.get("cat") == "kernel"
            )

            if has_matmul_ogs:
                from .moe_unfused_triton_pseudo_ops import (
                    create_pseudo_ops_moe_unfused_triton,
                )

                extensions.append(
                    ("MoE_Unfused_Triton", create_pseudo_ops_moe_unfused_triton)
                )
                if verbose:
                    logger.info(
                        "Auto-detected GPT_OSS unfused MoE operations with Triton kernels"
                    )

    # MoE: GPTQ/AWQ quantized unfused implementation (vllm::outplace_fused_experts)
    if "vllm::outplace_fused_experts" in tree.name2event_uids:
        has_gptq_awq = any(
            "fused_moe_kernel_gptq_awq" in event.get("name", "")
            for event in tree.events
            if event.get("cat") == "kernel"
        )
        if has_gptq_awq:
            from .moe_gptq_awq_pseudo_ops import create_pseudo_ops_moe_gptq_awq

            extensions.append(("MoE_GPTQ_AWQ", create_pseudo_ops_moe_gptq_awq))
            if verbose:
                logger.info(
                    "Auto-detected GPTQ/AWQ MoE operations (outplace_fused_experts)"
                )

    # MoE: flydsl 2-stage implementation (gated on aiter::fused_moe_ parent op)
    if "aiter::fused_moe_" in tree.name2event_uids:
        from .moe_flydsl_pseudo_ops import create_pseudo_ops_moe_flydsl

        extensions.append(("MoE_Flydsl", create_pseudo_ops_moe_flydsl))
        if verbose:
            logger.info("Auto-detected flydsl MoE operations under aiter::fused_moe_")

    # MLA Decode: AITER implementation
    if "aiter::mla_decode_stage1_asm_fwd" in tree.name2event_uids:
        has_mla_python_func = any(
            re.search(r"aiter/mla.py\(\d+\): mla_decode_fwd", name)
            for name in tree.name2event_uids
        )
        if has_mla_python_func:
            from .mla_decode_pseudo_ops import create_pseudo_ops_mla_decode

            extensions.append(("MLA_Decode", create_pseudo_ops_mla_decode))
            if verbose:
                logger.info("Auto-detected MLA decode operations")

    # MLA Prefill: AITER fp8 implementation
    if "aiter::mla_prefill_ps_asm_fwd" in tree.name2event_uids:
        has_prefill_python_func = any(
            re.search(r":\s*mla_fp8_prefill_attn(\b|$)", name)
            for name in tree.name2event_uids
        )
        if has_prefill_python_func:
            from .mla_prefill_pseudo_ops import create_pseudo_ops_mla_prefill

            extensions.append(("MLA_Prefill", create_pseudo_ops_mla_prefill))
            if verbose:
                logger.info("Auto-detected MLA prefill operations")
    if "_rocm_C::paged_attention" in tree.name2event_uids:
        from .paged_attn_perf_meta import mark_rocm_paged_attn_kvcache_dtype

        extensions.append(
            ("RocmPagedAttn_KVCacheDtype", mark_rocm_paged_attn_kvcache_dtype)
        )
        if verbose:
            logger.info(
                "Auto-detected _rocm_C::paged_attention — will propagate "
                "perf_meta.KCache_dtype/VCache_dtype to cpu_op parents"
            )
    if "aiter::paged_attention_v1" in tree.name2event_uids:
        from .paged_attn_perf_meta import mark_aiter_paged_attn_kvcache_dtype

        extensions.append(
            ("AiterPagedAttn_KVCacheDtype", mark_aiter_paged_attn_kvcache_dtype)
        )
        if verbose:
            logger.info(
                "Auto-detected aiter::paged_attention_v1 — will propagate "
                "perf_meta.k_cache_dtype/v_cache_dtype to cpu_op parents"
            )
    # Apply extensions onto tree
    for ext_info in extensions:
        # ext_info tuple of (extension_name, extension_function)
        ext_name, ext_func = ext_info

        if verbose:
            logger.info(f"Applying pseudo-op extension: {ext_name}")

        try:
            ext_func(tree)
        except Exception as e:
            logger.warning(f"Failed to apply pseudo-op extension {ext_name}: {e}")

    # Commit any pending cpu_root_nodes mutations accumulated by
    # inject_pseudo_op_wrap_children (no-op if none were made).
    finalize_pseudo_op_mutations(tree)
