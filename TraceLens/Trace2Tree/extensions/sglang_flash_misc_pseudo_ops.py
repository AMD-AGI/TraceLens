###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inject stable Flash SGLang pseudo ops for mHC / MXFP8 / router / RMSNorm.

Python owners (from Flash decode captures) are wrapped so Synthetic Op launches
gain a FLOPs-capable cpu_op name. Geometry falls back to Flash defaults when
the owner has no Input Dims (common on graph-capture Synthetic Op paths).
"""

import os
import re
import logging

from .pseudo_ops_utils import inject_pseudo_op_wrap_children

logger = logging.getLogger(__name__)

HC_BOUNDARY_RE = re.compile(
    r"(?:mhc_boundary_hip\.py\(\d+\):\s*(?:hc_boundary_fused|_hc_boundary_partials)"
    r"|deepseek_v4_fused_mhc\.py\(\d+\):\s*hc_boundary)\b"
)
HC_SINKHORN_RE = re.compile(
    r"mhc_boundary_hip\.py\(\d+\):\s*hc_mix_reduce_sinkhorn_vec\b"
)
RMSNORM_FQ_RE = re.compile(
    r"rmsnorm_fake_quant_amd_gfx95\.py\(\d+\):\s*rmsnorm_fake_quant_fp8\b"
)
ROUTER_GEMV_RE = re.compile(
    r"rocm_router_gate\.py\(\d+\):\s*rocm_router_gemv_split_k\b"
)
ROUTER_GATE_RE = re.compile(r"rocm_router_gate\.py\(\d+\):\s*rocm_router_gate\b")
MXFP8_RE = re.compile(
    r"mxfp8_native_amd_gfx95\.py\(\d+\):\s*mxfp8_native_blockscaled_linear\b"
)

# Flash residual / hidden defaults (TP=1 Flash).
_FLASH_DEFAULTS = {
    "n": 4,
    "C": 5120,
    "hidden_dim": 5120,
    "num_experts": 384,
    "topk": 6,
}


def create_pseudo_ops_sglang_flash_misc(trace_tree):
    model_name = os.environ.get("TL_MODEL")
    tokens = _infer_tokens_from_annotation(trace_tree)
    base = {"v4_model_name": model_name, "num_tokens": tokens, "M": tokens}
    _wrap(
        trace_tree,
        HC_BOUNDARY_RE,
        "pseudo_hc_boundary",
        {**base, **_FLASH_DEFAULTS},
    )
    _wrap(
        trace_tree,
        HC_SINKHORN_RE,
        "pseudo_hc_mix_reduce_sinkhorn",
        {**base, "n": _FLASH_DEFAULTS["n"], "C": 1},
    )
    _wrap(
        trace_tree,
        RMSNORM_FQ_RE,
        "pseudo_rmsnorm_fake_quant_fp8",
        {**base, "N": _FLASH_DEFAULTS["hidden_dim"], "hidden_dim": _FLASH_DEFAULTS["hidden_dim"]},
    )
    _wrap(
        trace_tree,
        ROUTER_GEMV_RE,
        "pseudo_router_gemv_split_k",
        {
            **base,
            "E": _FLASH_DEFAULTS["num_experts"],
            "K": _FLASH_DEFAULTS["hidden_dim"],
            "num_experts": _FLASH_DEFAULTS["num_experts"],
            "hidden_dim": _FLASH_DEFAULTS["hidden_dim"],
        },
    )
    _wrap(
        trace_tree,
        ROUTER_GATE_RE,
        "pseudo_router_gate",
        {
            **base,
            "E": _FLASH_DEFAULTS["num_experts"],
            "topk": _FLASH_DEFAULTS["topk"],
            "num_experts": _FLASH_DEFAULTS["num_experts"],
        },
    )
    _wrap(
        trace_tree,
        MXFP8_RE,
        "pseudo_mxfp8_gemv",
        {
            **base,
            "N": _FLASH_DEFAULTS["hidden_dim"],
            "K": _FLASH_DEFAULTS["hidden_dim"],
        },
    )


def _infer_tokens_from_annotation(trace_tree):
    """Best-effort token count from any annotated attention-like event."""
    for evt in trace_tree.events:
        ann = evt.get("annotation")
        if not ann or ann == "NA":
            continue
        try:
            from TraceLens.TraceUtils.utils.annotation_utils import IterationAnnotation

            stats = IterationAnnotation(str(ann)).chunk_stats()
            tokens = stats.get("c_sq", 0) + stats.get("g_sq", 0)
            if tokens:
                return int(tokens)
        except Exception:
            continue
    return 1


def _wrap(trace_tree, pattern, pseudo_name, extra):
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
            extra_args=dict(extra),
        )
        logger.info("Injected %s under %s", pseudo_name, owner.get("name"))
