###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inject stable Flash indexer pseudo ops off SGLang/aiter python owners.

Covers:
  - pa_mqa_logits_fp4 → pseudo_pa_mqa_logits_fp4
  - quantize_fp4_indexer_tensor → pseudo_quantize_fp4_indexer
  - aten::topk indexer parent (PD wall) → pseudo_flash_indexer_topk
  - fused_k_norm_rope flashmla → fused_k_norm_rope_q_flashmla
"""

import os
import re
import logging

from .pseudo_ops_utils import inject_pseudo_op_wrap_children, inject_pseudo_op_above_event

logger = logging.getLogger(__name__)

MQA_OWNER_RE = re.compile(
    r"pa_mqa_logits_fp4\.py\(\d+\):\s*flydsl_pa_mqa_logits_fp4\b"
)
QUANT_OWNER_RE = re.compile(
    r"fp4_indexer\.py\(\d+\):\s*quantize_fp4_indexer_tensor\b"
)
FLASHMLA_OWNER_RE = re.compile(
    r"(?:elementwise(?:_hip)?\.py\(\d+\):\s*fused_k_norm_rope_flashmla"
    r"(?:_with_q)?\b)"
)
ATEN_TOPK = "aten::topk"


def create_pseudo_ops_sglang_flash_indexer(trace_tree):
    model_name = os.environ.get("TL_MODEL")
    _inject_named_owners(
        trace_tree,
        MQA_OWNER_RE,
        "pseudo_pa_mqa_logits_fp4",
        extra={"v4_model_name": model_name},
    )
    _inject_named_owners(
        trace_tree,
        QUANT_OWNER_RE,
        "pseudo_quantize_fp4_indexer",
        extra={"v4_model_name": model_name},
    )
    _inject_named_owners(
        trace_tree,
        FLASHMLA_OWNER_RE,
        "fused_k_norm_rope_q_flashmla",
        extra={"v4_model_name": model_name, "d_h": 512, "rope_dim": 64},
    )
    _inject_aten_topk_parents(trace_tree, model_name)


def _inject_named_owners(trace_tree, pattern, pseudo_name, extra=None):
    owners = []
    for name, uids in trace_tree.name2event_uids.items():
        if not pattern.search(name):
            continue
        for uid in uids:
            evt = trace_tree.get_UID2event(uid)
            if evt.get("cat") in {"python_function", "python_func"}:
                owners.append(evt)
    for owner in owners:
        children = owner.get("children") or []
        if len(children) == 1:
            child = trace_tree.get_UID2event(children[0])
            if child and child.get("name") == pseudo_name:
                continue
        if not children and not owner.get("gpu_events"):
            continue
        inject_pseudo_op_wrap_children(
            trace_tree,
            owner,
            pseudo_name,
            shape_donor_evt=owner,
            extra_args=dict(extra or {}),
        )
        logger.info("Injected %s under %s", pseudo_name, owner.get("name"))


def _inject_aten_topk_parents(trace_tree, model_name):
    """Attach a Flash indexer TopK pseudo around qualifying aten::topk ops.

    Prefer injecting above the event when it has a parent; if it is a CPU root
    (common on Flash PD captures), wrap its children instead so kernel time
    still rolls under a FLOPs-capable pseudo name.
    """
    uids = list(trace_tree.name2event_uids.get(ATEN_TOPK, []))
    for uid in uids:
        evt = trace_tree.get_UID2event(uid)
        if evt.get("cat") != "cpu_op":
            continue
        parent = trace_tree.get_parent_event(evt)
        if parent is not None and parent.get("name") == "pseudo_flash_indexer_topk":
            continue
        children = evt.get("children") or []
        if len(children) == 1:
            child = trace_tree.get_UID2event(children[0])
            if child and child.get("name") == "pseudo_flash_indexer_topk":
                continue
        args = evt.get("args") or {}
        dims = args.get("Input Dims") or []
        types = args.get("Input type") or []
        if not dims or not dims[0] or len(dims[0]) < 2:
            continue
        rows, cols = dims[0][0], dims[0][1]
        if rows is None or cols is None or cols < 1024:
            continue
        dtype0 = str(types[0]) if types else ""
        if types and "int" in dtype0.lower() and "float" not in dtype0.lower():
            continue
        extra = {"v4_model_name": model_name, "indexer_topk_parent": True}
        if parent is None:
            if not children and not evt.get("gpu_events"):
                continue
            inject_pseudo_op_wrap_children(
                trace_tree,
                evt,
                "pseudo_flash_indexer_topk",
                shape_donor_evt=evt,
                extra_args=extra,
            )
        else:
            inject_pseudo_op_above_event(
                trace_tree,
                evt,
                "pseudo_flash_indexer_topk",
                shape_donor_evt=evt,
                extra_args=extra,
            )
