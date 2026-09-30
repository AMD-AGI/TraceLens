###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pseudo ops for DeepSeek-V4 sparse paged-decode attention.

For each ``sparse_attn_v4_paged_decode`` python_function, wrap its children in a
``pseudo_v4_paged_decode_{mode}`` op: walk up to the enclosing
``aiter::v4_attention_with_output`` (shape donor), detect the mode (SWA / CSA /
HCA) and geometry (H_Q, d_h) from its kernels, and stamp variant config as
``extra_args`` for the perf model.

Works for both DeepSeek-V4-Flash (TP=1) and DeepSeek-V4-Pro (TP=4); mode mix and
head count are detected from the trace.
"""

import os
import re
import logging

from .pseudo_ops_utils import inject_pseudo_op_wrap_children

logger = logging.getLogger(__name__)

SPARSE_DECODE_PATTERN = re.compile(
    r"paged_decode\.py\(\d+\):\s*sparse_attn_v4_paged_decode\b"
)
V4_ATTN_OP_NAME = "aiter::v4_attention_with_output"
QK_NORM_ROPE_PATTERN = re.compile(r"qk_norm_rope_H(\d+)_D(\d+)_RD(\d+)")

_HCA_PREFIXES = ("hca_compress_forward", "hca_norm_rope_scatter", "hca_")
_CSA_PREFIXES = ("fused_compress_attn", "_update_compressor_states_kernel")


def create_pseudo_ops_v4_paged_decode(trace_tree):
    """Inject a mode-specific pseudo op under each sparse_attn_v4_paged_decode."""

    py_funcs = _find_sparse_decode_python_funcs(trace_tree)
    if not py_funcs:
        logger.warning(
            "No python_function events matching sparse_attn_v4_paged_decode found"
        )
        return

    model_name = os.environ.get("TL_MODEL")
    tp = _safe_int(os.environ.get("TL_TP"), default=1)
    logger.info(
        f"Processing {len(py_funcs)} V4 sparse paged-decode ops "
        f"(model_name={model_name!r}, tp={tp})"
    )

    # Stable program order for layer-id metadata.
    py_funcs.sort(key=lambda e: (e.get("pid", 0), e.get("tid", 0), e.get("ts", 0)))
    for call_idx, py_func_evt in enumerate(py_funcs):
        _create_pseudo_op(trace_tree, py_func_evt, model_name, tp, call_idx)


def _find_sparse_decode_python_funcs(trace_tree):
    matched = []
    for name, uids in trace_tree.name2event_uids.items():
        if SPARSE_DECODE_PATTERN.search(name):
            for uid in uids:
                evt = trace_tree.get_UID2event(uid)
                if evt.get("cat") == "python_function":
                    matched.append(evt)
    return matched


def _find_v4_attn_ancestor(trace_tree, evt):
    """Find the enclosing v4_attention_with_output ancestor."""
    cur = trace_tree.get_parent_event(evt)
    while cur is not None:
        if cur.get("name") == V4_ATTN_OP_NAME:
            return cur
        cur = trace_tree.get_parent_event(cur)
    return None


def _kernel_names_under(trace_tree, evt):
    """GPU kernel names in an event's subtree."""
    names = []
    for uid in evt.get("gpu_events", []):
        kevt = trace_tree.get_UID2event(uid)
        if kevt is not None:
            names.append(kevt.get("name", ""))
    return names


def _detect_v4_mode(kernel_names):
    """Classify SWA / CSA / HCA from kernel names."""
    if any(n.startswith(_HCA_PREFIXES) for n in kernel_names):
        return "hca"
    if any(n.startswith(_CSA_PREFIXES) for n in kernel_names):
        return "csa"
    return "swa"


def _parse_geometry(kernel_names):
    """Extract (H_Q, d_h) from the qk_norm_rope kernel name."""
    for n in kernel_names:
        m = QK_NORM_ROPE_PATTERN.search(n)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _create_pseudo_op(trace_tree, py_func_evt, model_name, tp, call_idx):
    v4_attn_evt = _find_v4_attn_ancestor(trace_tree, py_func_evt)
    if v4_attn_evt is None:
        logger.warning(
            f"No {V4_ATTN_OP_NAME} ancestor for '{py_func_evt.get('name')}' "
            f"(UID {py_func_evt.get('UID')}), skipping"
        )
        return

    if not py_func_evt.get("gpu_events"):
        logger.warning(
            f"No GPU events for sparse decode python_function "
            f"UID {py_func_evt.get('UID')}, skipping"
        )
        return

    kernel_names = _kernel_names_under(trace_tree, v4_attn_evt)
    mode = _detect_v4_mode(kernel_names)
    H_Q, d_h = _parse_geometry(kernel_names)

    extra_args = {
        "v4_mode": mode,
        "v4_call_index": call_idx,
        "v4_model_name": model_name,
        "v4_tp": tp,
    }
    if H_Q is not None:
        extra_args["v4_H_Q"] = H_Q
    if d_h is not None:
        extra_args["v4_d_h"] = d_h

    inject_pseudo_op_wrap_children(
        trace_tree,
        py_func_evt,
        f"pseudo_v4_paged_decode_{mode}",
        shape_donor_evt=v4_attn_evt,
        extra_args=extra_args,
    )
