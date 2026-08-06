###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
import os
import re
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
