###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""SGLang Flash sparse paged-decode → existing pseudo_v4_paged_decode_* models.

Flash SGLang launches `_pa_decode_sparse_*` (and optional
`_aiter_sparse_decode_reduce`) under python owners in
`aiter/.../pa_decode_sparse.py` / `aiter_sparse_decode_reduce.py`, not under
ATOM `sparse_attn_v4_paged_decode`. This injector bridges those owners to the
mode-specific V4 paged-decode pseudo ops already modeled on main.
"""

import os
import re
import logging

from .pseudo_ops_utils import inject_pseudo_op_wrap_children

logger = logging.getLogger(__name__)

SPARSE_OWNER_RE = re.compile(
    r"pa_decode_sparse\.py\(\d+\):\s*(?:_pa_decode_sparse_gfx950_gluon|pa_decode_sparse)\b"
)
REDUCE_OWNER_RE = re.compile(
    r"aiter_sparse_decode_reduce\.py\(\d+\):\s*aiter_sparse_split_reduce\b"
)
# Also accept hip_flash_mla owners that wrap the decode forward.
FLASH_MLA_OWNER_RE = re.compile(r"hip_flash_mla\.py\(\d+\):\s*aiter_sparse_decode_fwd\b")

PARTIAL_KERNEL = "_pa_decode_sparse_"
REDUCE_KERNEL = "_aiter_sparse_decode_reduce"


def create_pseudo_ops_sglang_sparse_decode(trace_tree):
    """Inject pseudo_v4_paged_decode_{mode} under each Flash sparse-decode owner."""
    owners = _find_owners(trace_tree)
    if not owners:
        logger.warning("No SGLang Flash sparse-decode python owners found")
        return

    model_name = os.environ.get("TL_MODEL")
    tp = _safe_int(os.environ.get("TL_TP"), default=1)
    owners.sort(key=lambda e: (e.get("pid", 0), e.get("tid", 0), e.get("ts", 0)))
    logger.info(
        "Processing %d SGLang sparse-decode owners (model_name=%r, tp=%s)",
        len(owners),
        model_name,
        tp,
    )
    for call_idx, owner in enumerate(owners):
        _inject_owner(trace_tree, owner, model_name, tp, call_idx)


def _find_owners(trace_tree):
    matched = []
    for name, uids in trace_tree.name2event_uids.items():
        if not (
            SPARSE_OWNER_RE.search(name)
            or REDUCE_OWNER_RE.search(name)
            or FLASH_MLA_OWNER_RE.search(name)
        ):
            continue
        for uid in uids:
            evt = trace_tree.get_UID2event(uid)
            if evt.get("cat") in {"python_function", "python_func"}:
                matched.append(evt)
    # Prefer the gluon/pa_decode_sparse owners; keep reduce-only owners that
    # are not already under a sparse owner we will wrap.
    sparse_first = [
        e
        for e in matched
        if SPARSE_OWNER_RE.search(e.get("name", ""))
        or FLASH_MLA_OWNER_RE.search(e.get("name", ""))
    ]
    if sparse_first:
        return sparse_first
    return matched


def _kernel_names_under(trace_tree, evt):
    names = []
    for uid in evt.get("gpu_events", []):
        kevt = trace_tree.get_UID2event(uid)
        if kevt is not None:
            names.append(kevt.get("name", ""))
    # Also scan children launchers for Synthetic Op naming.
    stack = list(evt.get("children") or [])
    while stack:
        uid = stack.pop()
        child = trace_tree.get_UID2event(uid)
        if child is None:
            continue
        n = child.get("name") or ""
        if PARTIAL_KERNEL in n or REDUCE_KERNEL in n:
            names.append(n)
        stack.extend(child.get("children") or [])
    return names


def _detect_mode(kernel_names):
    joined = " ".join(kernel_names).lower()
    if "hca" in joined:
        return "hca"
    if "csa" in joined or "compress" in joined:
        return "csa"
    return "swa"


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _inject_owner(trace_tree, owner, model_name, tp, call_idx):
    if not owner.get("children") and not owner.get("gpu_events"):
        logger.warning(
            "SGLang sparse-decode owner UID %s has no children/gpu_events; skip",
            owner.get("UID"),
        )
        return
    # Avoid double-inject if a V4 pseudo is already the sole child.
    children = owner.get("children") or []
    if len(children) == 1:
        child = trace_tree.get_UID2event(children[0])
        if child and str(child.get("name", "")).startswith("pseudo_v4_paged_decode_"):
            return

    kernel_names = _kernel_names_under(trace_tree, owner)
    mode = _detect_mode(kernel_names)
    extra_args = {
        "v4_mode": mode,
        "v4_call_index": call_idx,
        "v4_model_name": model_name,
        "v4_tp": tp,
        "sglang_sparse_decode": True,
    }
    inject_pseudo_op_wrap_children(
        trace_tree,
        owner,
        f"pseudo_v4_paged_decode_{mode}",
        shape_donor_evt=owner,
        extra_args=extra_args,
    )
