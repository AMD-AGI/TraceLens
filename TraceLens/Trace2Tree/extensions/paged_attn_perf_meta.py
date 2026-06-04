###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Trace2Tree extensions that attach perf-model metadata to ``cpu_op`` ancestors
of paged-attention leaf ops.

All metadata is written to a brand-new top-level ``event["perf_meta"]`` dict
so existing ``event["args"]`` consumers are untouched.
"""

import logging

logger = logging.getLogger(__name__)


def mark_paged_attn_v1_parents(trace_tree):
    """Mark cpu_op ancestors of ``aiter::paged_attention_v1`` so they get
    skipped during unified perf-event collection — the leaf has its own perf
    model and should win over parents like ``vllm::unified_attention_with_output``.
    """
    leaf_uids = trace_tree.name2event_uids.get("aiter::paged_attention_v1", [])
    if not leaf_uids:
        return
    for uid in leaf_uids:
        leaf = trace_tree.get_UID2event(uid)
        parent = trace_tree.get_parent_event(leaf)
        while parent is not None:
            if parent.get("cat") == "cpu_op":
                parent.setdefault("perf_meta", {})["exclude_perf_model"] = True
            parent = trace_tree.get_parent_event(parent)


def mark_rocm_paged_attn_kvcache_dtype(trace_tree):
    """Propagate K/V cache dtypes from ``_rocm_C::paged_attention`` leaves up
    to their cpu_op ancestors so that the parent's perf model
    (``vllm::unified_attention_with_output``) can pick the FP8 KV-cache dtype.

    Op signature is ``(out, exp_sum, max_logits, tmp_out, query, key_cache,
    value_cache, ...)`` — K-cache dtype at ``Input type[5]``, V-cache at ``[6]``.
    """
    leaf_uids = trace_tree.name2event_uids.get("_rocm_C::paged_attention", [])
    if not leaf_uids:
        return
    for uid in leaf_uids:
        leaf = trace_tree.get_UID2event(uid)
        types = (leaf.get("args") or {}).get("Input type") or []
        if len(types) <= 6:
            continue
        k_dtype, v_dtype = types[5], types[6]
        parent = trace_tree.get_parent_event(leaf)
        while parent is not None:
            if parent.get("cat") == "cpu_op":
                meta = parent.setdefault("perf_meta", {})
                meta["KCache_dtype"] = k_dtype
                meta["VCache_dtype"] = v_dtype
            parent = trace_tree.get_parent_event(parent)
