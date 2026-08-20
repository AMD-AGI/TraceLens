###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Multi-source metadata gathering for roofline analysis.

Gathers metadata from annotation, filename (isl/osl/conc), trace Input Dims,
and user input. Runs sanity checks and merges into a unified dict.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Filename regex: isl1024, osl8, conc4, tp1
FILENAME_ISL_RE = re.compile(r"isl(\d+)", re.I)
FILENAME_OSL_RE = re.compile(r"osl(\d+)", re.I)
FILENAME_CONC_RE = re.compile(r"conc(\d+)", re.I)
FILENAME_TP_RE = re.compile(r"tp(\d+)", re.I)


def parse_filename_metadata(filepath: str) -> Dict[str, Any]:
    """
    Parse isl, osl, conc, tp from trace filename.
    Example: mi355_tp1_isl1024_osl8_conc4_opt_asm64x256.pt.trace.json.gz
    """
    basename = filepath.split("/")[-1] if "/" in filepath else filepath
    result = {}
    m = FILENAME_ISL_RE.search(basename)
    if m:
        result["isl"] = int(m.group(1))
    m = FILENAME_OSL_RE.search(basename)
    if m:
        result["osl"] = int(m.group(1))
    m = FILENAME_CONC_RE.search(basename)
    if m:
        result["conc"] = int(m.group(1))
    m = FILENAME_TP_RE.search(basename)
    if m:
        result["tp"] = int(m.group(1))
    if "isl" in result and "conc" in result:
        result["num_tokens_prefill"] = result["isl"] * result["conc"]
    if "conc" in result:
        result["num_tokens_decode"] = result["conc"]
    return result


def parse_trace_input_dims(events: List[dict]) -> Dict[str, Any]:
    """
    Aggregate N_Q, N_KV from kernel Input Dims (for attention kernels).
    Returns block-level aggregates when available.
    For multi-layer models, each layer has one attention kernel with same N_Q/N_KV.
    """
    nq_values = []
    nkv_values = []
    for e in events:
        if e.get("cat") != "kernel":
            continue
        name = e.get("name", "")
        if "unified_attention" not in name and "attention" not in name.lower():
            continue
        dims = e.get("args", {}).get("Input Dims")
        if not dims or len(dims) < 2:
            continue
        try:
            q_shape = dims[0]
            k_shape = dims[1]
            if isinstance(q_shape, (list, tuple)) and len(q_shape) >= 3:
                nq = q_shape[0] if len(q_shape) == 3 else q_shape[-3]
                nq_values.append(nq)
            if isinstance(k_shape, (list, tuple)) and len(k_shape) >= 3:
                nkv = k_shape[-3] if len(k_shape) >= 3 else k_shape[0]
                nkv_values.append(nkv)
        except (IndexError, TypeError):
            continue
    result = {}
    if nq_values:
        result["trace_avg_nq"] = sum(nq_values) // len(nq_values)
        result["trace_attention_kernel_count"] = len(nq_values)
    if nkv_values:
        result["trace_avg_nkv"] = sum(nkv_values) // len(nkv_values)
    return result


def run_sanity_checks(
    annotation_meta: Dict[str, Any],
    filename_meta: Dict[str, Any],
    trace_meta: Dict[str, Any],
    user_meta: Dict[str, Any],
) -> List[str]:
    """Cross-validate metadata from different sources. Returns list of warning messages."""
    warnings = []
    batch_ann = annotation_meta.get("batch_size")
    batch_file = None
    if "isl" in filename_meta and "conc" in filename_meta:
        batch_file = filename_meta["isl"] * filename_meta["conc"]
    if batch_ann is not None and batch_file is not None:
        if abs(batch_ann - batch_file) > max(1, 0.1 * batch_ann):
            warnings.append(
                f"Batch size mismatch: annotation={batch_ann}, filename(isl*conc)={batch_file}"
            )
    num_tokens_user = user_meta.get("num_tokens")
    if num_tokens_user is not None and batch_ann is not None:
        if abs(num_tokens_user - batch_ann) > max(1, 0.1 * batch_ann):
            warnings.append(
                f"num_tokens mismatch: user={num_tokens_user}, annotation batch_size={batch_ann}"
            )
    ctx_ann = annotation_meta.get("context_sum")
    ctx_trace = trace_meta.get("trace_avg_nq")
    if ctx_ann is not None and ctx_trace is not None:
        if abs(ctx_ann - ctx_trace) > max(1, 0.2 * ctx_ann):
            warnings.append(
                f"Context sum mismatch: annotation={ctx_ann}, trace avg N_Q={ctx_trace}"
            )
    return warnings


def merge_metadata(
    annotation_meta: Optional[Dict[str, Any]] = None,
    filename_meta: Optional[Dict[str, Any]] = None,
    trace_meta: Optional[Dict[str, Any]] = None,
    user_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge metadata from all sources. Prefer annotation for per-iteration,
    filename for run config, trace for per-kernel accuracy, user for overrides.
    """
    merged = {}
    annotation_meta = annotation_meta or {}
    filename_meta = filename_meta or {}
    trace_meta = trace_meta or {}
    user_meta = user_meta or {}
    merged["annotation"] = annotation_meta
    merged["filename"] = filename_meta
    merged["trace"] = trace_meta
    merged["user"] = user_meta
    num_tokens = (
        user_meta.get("num_tokens")
        or annotation_meta.get("batch_size")
        or filename_meta.get("num_tokens_prefill")
        or filename_meta.get("num_tokens_decode")
    )
    context_length = (
        user_meta.get("context_length")
        or annotation_meta.get("context_sum")
        or annotation_meta.get("generation_sum")
        or filename_meta.get("isl")
        or num_tokens
    )
    merged["num_tokens"] = num_tokens
    merged["context_length"] = context_length
    merged["batch_size"] = annotation_meta.get("batch_size") or num_tokens
    merged["context_sum"] = annotation_meta.get("context_sum")
    merged["generation_sum"] = annotation_meta.get("generation_sum")
    warnings = run_sanity_checks(annotation_meta, filename_meta, trace_meta, user_meta)
    for w in warnings:
        logger.warning("Metadata sanity check: %s", w)
    merged["_warnings"] = warnings
    return merged


def gather_metadata(
    trace_path: str,
    events: Optional[List[dict]] = None,
    annotation_meta: Optional[Dict[str, Any]] = None,
    num_tokens: Optional[int] = None,
    context_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Gather metadata from all available sources.

    Args:
        trace_path: Path to trace file (for filename parsing)
        events: Trace events (for Input Dims); if None, trace Input Dims are skipped
        annotation_meta: Pre-parsed annotation metadata (e.g., from vllm_trace_split)
        num_tokens: User-provided num_tokens
        context_length: User-provided context_length

    Returns:
        Merged metadata dict with num_tokens, context_length, batch_size, etc.
    """
    filename_meta = parse_filename_metadata(trace_path)
    trace_meta = parse_trace_input_dims(events or [])
    user_meta = {}
    if num_tokens is not None:
        user_meta["num_tokens"] = num_tokens
    if context_length is not None:
        user_meta["context_length"] = context_length
    return merge_metadata(
        annotation_meta=annotation_meta,
        filename_meta=filename_meta,
        trace_meta=trace_meta,
        user_meta=user_meta,
    )
