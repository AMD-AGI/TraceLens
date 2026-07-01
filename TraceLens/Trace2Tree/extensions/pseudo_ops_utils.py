###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import logging
import os
import sys
import time
from collections import defaultdict
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)


_SGLANG_SUFFIX_RE = re.compile(r"^(sglang_profiler::.+?)_\d+$")
_MLA_DECODE_FWD_NAME_RE = re.compile(r"aiter/mla.py\(\d+\): mla_decode_fwd")
_MLA_FP8_PREFILL_NAME_RE = re.compile(r":\s*mla_fp8_prefill_attn(\b|$)")


def _any_kernel_event_name_contains(tree, needle: str, *, lower: bool = True) -> bool:
    """True if some *kernel* event's name contains *needle* (via ``name2event_uids``).

    Avoids scanning all ``tree.events``; only unique names that match the
    substring are checked, then UIDs are filtered by ``cat == \"kernel\"``.
    """
    if lower:
        needle_l = needle.lower()

        def hit(n: str) -> bool:
            return needle_l in n.lower()

    else:

        def hit(n: str) -> bool:
            return needle in n

    for name in tree.name2event_uids:
        if not hit(name):
            continue
        for uid in tree.name2event_uids[name]:
            if tree.get_UID2event(uid).get("cat") == "kernel":
                return True
    return False


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


# One inject_pseudo_op spec: kernel, name, seq_num, optional shape/args kwargs.
PseudoOpInjectSpec = Tuple[
    dict,
    str,
    Any,
    Optional[Any],
    Optional[Any],
    Optional[Any],
    Optional[Any],
    Optional[dict],
]


def _build_pseudo_op_event_dict(
    kernel_evt: dict,
    orig_cpu_evt: dict,
    launcher_evt: dict,
    name: str,
    seq_num,
    dims=None,
    types=None,
    strides=None,
    concrete_inputs=None,
    extra_args=None,
) -> dict:
    """Build pseudo cpu_op dict (no bookkeeping / parent / children mutation)."""
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
    if extra_args:
        pseudo_evt["args"].update(extra_args)
    return pseudo_evt


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

    pseudo_evt = _build_pseudo_op_event_dict(
        kernel_evt,
        orig_cpu_evt,
        launcher_evt,
        name,
        seq_num,
        dims=dims,
        types=types,
        strides=strides,
        concrete_inputs=concrete_inputs,
        extra_args=extra_args,
    )

    set_bookkeeping_attr(tree, pseudo_evt)

    pseudo_evt["parent"] = orig_cpu_evt["UID"]
    children = orig_cpu_evt["children"]
    children.remove(launcher_evt["UID"])
    children.append(pseudo_evt["UID"])


def inject_pseudo_ops_batch(tree, specs: Sequence[PseudoOpInjectSpec]) -> int:
    """Create many pseudo ops; rewrite each parent's ``children`` at most once.

    Equivalent to calling :func:`inject_pseudo_op` repeatedly in *specs* order,
    but avoids repeated ``list.remove`` on large ``children`` lists (same parent).

    Each spec is
    ``(kernel_evt, name, seq_num, dims, types, strides, concrete_inputs, extra_args)``
    with the same semantics as :func:`inject_pseudo_op` (``None`` for optional
    shape fields means inherit from the resolved parent cpu op).

    Returns:
        Number of pseudo ops successfully inserted.
    """
    pending = defaultdict(list)
    inserted = 0

    for spec in specs:
        (
            kernel_evt,
            name,
            seq_num,
            dims,
            types,
            strides,
            concrete_inputs,
            extra_args,
        ) = spec
        launcher_evt = tree.get_parent_event(kernel_evt)
        if launcher_evt is None:
            logger.warning(
                "inject_pseudo_ops_batch: kernel UID %s has no parent launcher; skip %s",
                kernel_evt.get("UID"),
                name,
            )
            continue
        orig_cpu_evt = tree.get_parent_event(launcher_evt)
        if orig_cpu_evt is None:
            logger.warning(
                "inject_pseudo_ops_batch: launcher UID %s has no parent cpu op; skip %s",
                launcher_evt.get("UID"),
                name,
            )
            continue

        pseudo_evt = _build_pseudo_op_event_dict(
            kernel_evt,
            orig_cpu_evt,
            launcher_evt,
            name,
            seq_num,
            dims=dims,
            types=types,
            strides=strides,
            concrete_inputs=concrete_inputs,
            extra_args=extra_args,
        )
        set_bookkeeping_attr(tree, pseudo_evt)
        pseudo_evt["parent"] = orig_cpu_evt["UID"]
        pending[orig_cpu_evt["UID"]].append((launcher_evt["UID"], pseudo_evt["UID"]))
        inserted += 1

    for _orig_uid, pairs in pending.items():
        orig = tree.get_UID2event(_orig_uid)
        ch = orig["children"]
        launcher_set = {lu for lu, _ in pairs}
        new_ch = [c for c in ch if c not in launcher_set]
        new_ch.extend(pu for _, pu in pairs)
        ch[:] = new_ch

    return inserted


def inject_pseudo_op_wrap_children(
    tree,
    parent_evt,
    name,
    shape_donor_evt=None,
    extra_args=None,
    cpu_roots_acc: Optional[set] = None,
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
        cpu_roots_acc: If set, CPU root bookkeeping uses this mutable set
            (membership + ``-= roots_to_remove`` + add pseudo UID) and
            **does not** rewrite ``tree.cpu_root_nodes``. Callers that pass a
            shared accumulator across many wraps must assign
            ``tree.cpu_root_nodes`` once at the end (see MLA decode/prefill).
            Avoids O(|cpu_root_nodes|) per call when wrapping thousands of ops.
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

    # Descendants that were cpu_root_nodes are no longer roots since they
    # now live under the pseudo op. Remove them and promote the pseudo op.
    roots_to_remove = set()
    stack = list(children_uids)
    if cpu_roots_acc is not None:
        root_membership = cpu_roots_acc
    else:
        root_membership = frozenset(tree.cpu_root_nodes)
    while stack:
        uid = stack.pop()
        if uid in root_membership:
            roots_to_remove.add(uid)
        evt = tree.get_UID2event(uid)
        stack.extend(evt.get("children", []))
    if cpu_roots_acc is not None:
        if roots_to_remove:
            cpu_roots_acc.difference_update(roots_to_remove)
        cpu_roots_acc.add(pseudo_evt["UID"])
    else:
        if roots_to_remove:
            tree.cpu_root_nodes[:] = [
                u for u in tree.cpu_root_nodes if u not in roots_to_remove
            ]
        tree.cpu_root_nodes.append(pseudo_evt["UID"])


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


def apply_pseudo_op_extensions(
    tree, verbose: bool = False, show_progress: bool = False
):
    """
    Apply all available pseudo-op extensions to trace tree.
    Extensions are automatically detected and applied.

    When ``show_progress`` is True, emit ``tqdm`` milestone lines and a bar over
    the apply phase (stderr), mirroring perf-report progress.

    MoE fused vs unfused branch lines and per-extension timings are also written
    to stderr when ``show_progress`` or ``verbose`` is True, or when the
    environment variable ``TRACELENS_PSEUDO_OPS_LOG`` is set to a non-empty value
    other than ``0`` / ``false`` (case-insensitive).
    """

    pseudo_ops_emit = (
        show_progress
        or verbose
        or (
            (os.environ.get("TRACELENS_PSEUDO_OPS_LOG") or "").strip().lower()
            not in ("", "0", "false", "no", "off")
        )
    )

    if show_progress:
        tqdm.write(
            "perf: pseudo_ops: normalize_sglang_profiler_op_names …",
            file=sys.stderr,
        )
    normalize_sglang_profiler_op_names(tree)

    # Auto-detect and add all known pseudo-op extensions
    extensions = []

    moe_branch_msg: Optional[str] = None
    if "vllm::moe_forward" in tree.name2event_uids:

        # MoE: AITER Fused Implementation
        if "vllm::rocm_aiter_fused_moe" in tree.name2event_uids:
            from .moe_aiter_pseudo_ops import create_pseudo_ops_moe_fused_aiter

            extensions.append(
                (
                    "MoE_Fused",
                    partial(
                        create_pseudo_ops_moe_fused_aiter,
                        show_progress=pseudo_ops_emit,
                    ),
                )
            )
            moe_branch_msg = (
                "vllm::moe_forward + vllm::rocm_aiter_fused_moe → extension MoE_Fused "
                "(AITER fused pseudo ops)"
            )
            if verbose:
                logger.info("Auto-detected fused MoE operations")

        # MoE: Triton Fused Implementation
        # TO DO: Update kernel detection approach (Look for gpt_oss_triton_kernels_moe.py)
        else:
            # Check if any kernel events contain matmul_ogs: Triton MoE kernel
            has_matmul_ogs = _any_kernel_event_name_contains(tree, "matmul_ogs")

            if has_matmul_ogs:
                from .moe_unfused_triton_pseudo_ops import (
                    create_pseudo_ops_moe_unfused_triton,
                )

                extensions.append(
                    (
                        "MoE_Unfused_Triton",
                        partial(
                            create_pseudo_ops_moe_unfused_triton,
                            show_progress=pseudo_ops_emit,
                        ),
                    )
                )
                moe_branch_msg = (
                    "vllm::moe_forward + kernel names containing matmul_ogs → "
                    "extension MoE_Unfused_Triton (Triton unfused pseudo ops)"
                )
                if verbose:
                    logger.info(
                        "Auto-detected GPT_OSS unfused MoE operations with Triton kernels"
                    )
            else:
                moe_branch_msg = (
                    "vllm::moe_forward present but no vllm::rocm_aiter_fused_moe and "
                    "no matmul_ogs in kernel names → no MoE pseudo-op extension from this branch"
                )

    # MoE: GPTQ/AWQ quantized unfused implementation (vllm::outplace_fused_experts)
    if "vllm::outplace_fused_experts" in tree.name2event_uids:
        has_gptq_awq = _any_kernel_event_name_contains(
            tree, "fused_moe_kernel_gptq_awq", lower=False
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
        # Prefilter keys before regex: scanning every unique name is costly on
        # large traces (millions of events, hundreds of thousands of names).
        has_mla_python_func = any(
            _MLA_DECODE_FWD_NAME_RE.search(name)
            for name in tree.name2event_uids
            if "mla.py" in name and "mla_decode_fwd" in name
        )
        if has_mla_python_func:
            from .mla_decode_pseudo_ops import create_pseudo_ops_mla_decode

            extensions.append(("MLA_Decode", create_pseudo_ops_mla_decode))
            if verbose:
                logger.info("Auto-detected MLA decode operations")

    # MLA Prefill: AITER fp8 implementation
    if "aiter::mla_prefill_ps_asm_fwd" in tree.name2event_uids:
        has_prefill_python_func = any(
            _MLA_FP8_PREFILL_NAME_RE.search(name)
            for name in tree.name2event_uids
            if "mla_fp8_prefill_attn" in name
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
    # MoE pseudo-op branch (mutually exclusive fused vs unfused from vllm::moe_forward)
    if pseudo_ops_emit:
        if moe_branch_msg is not None:
            tqdm.write(
                f"perf: pseudo_ops: MoE branch: {moe_branch_msg}",
                file=sys.stderr,
            )
            logger.info("pseudo_ops MoE branch: %s", moe_branch_msg)
        else:
            tqdm.write(
                "perf: pseudo_ops: MoE branch: vllm::moe_forward not in trace "
                "(fused/unfused MoE pseudo ops from this detector are inactive)",
                file=sys.stderr,
            )
            logger.info(
                "pseudo_ops MoE branch: vllm::moe_forward not in trace "
                "(fused/unfused MoE pseudo ops from this detector are inactive)"
            )
        ext_names = [x[0] for x in extensions]
        tqdm.write(
            f"perf: pseudo_ops: extension order ({len(extensions)}): "
            f"{', '.join(ext_names) if ext_names else '(none)'}",
            file=sys.stderr,
        )
        logger.info("pseudo_ops extension order: %s", ext_names)

    # Apply extensions onto tree
    if show_progress:
        tqdm.write(
            f"perf: pseudo_ops: detected {len(extensions)} extension(s) to run …",
            file=sys.stderr,
        )
    _ext_iter = extensions
    if show_progress:
        _ext_iter = tqdm(
            extensions,
            desc="perf: pseudo-op extensions",
            unit="ext",
            file=sys.stderr,
            mininterval=0.3,
        )
    for ext_info in _ext_iter:
        # ext_info tuple of (extension_name, extension_function)
        ext_name, ext_func = ext_info

        if verbose:
            logger.info(f"Applying pseudo-op extension: {ext_name}")

        try:
            t0 = time.perf_counter()
            ext_func(tree)
            dt = time.perf_counter() - t0
            if pseudo_ops_emit:
                tqdm.write(
                    f"perf: pseudo_ops: extension {ext_name} finished in {dt:.2f}s",
                    file=sys.stderr,
                )
                logger.info("pseudo_ops extension %s finished in %.2fs", ext_name, dt)
        except Exception as e:
            logger.warning(f"Failed to apply pseudo-op extension {ext_name}: {e}")
