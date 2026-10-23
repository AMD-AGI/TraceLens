###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Adapter for TraceUtils' canonical trace splitter.

Calls TraceLens.TraceUtils.split_trace.split_and_enumerate in-process and
converts its output to the format expected by extract_trace_data.py:
[(trace_dict, region_metadata), ...].

Usage:
    from trace_split_adapter import split_vllm_trace, get_steady_state_key
    result = split_vllm_trace("trace.json.gz")
    if result:
        for trace_dict, metadata in result:
            key = get_steady_state_key(metadata)
"""

import json
import os
import tempfile
from typing import List, Optional, Tuple

from TraceLens.TraceUtils.split_trace import split_and_enumerate
from TraceLens.util import DataLoader, get_filename


def get_steady_state_key(metadata: dict) -> str:
    """Return a key for grouping by steady-state region.

    For single-iteration data the key encodes (type, token_count) so that
    corresponding iterations from two traces match by directory name.
    """
    ctx = metadata.get("context_requests", 0)
    gen = metadata.get("generation_requests", 0)
    ctx_sum = metadata.get("context_sum", 0)
    gen_sum = metadata.get("generation_sum", 0)
    batch = metadata.get("batch_size", 0)

    if ctx > 0 and gen == 0:
        return f"prefill_only_{ctx_sum}"
    if ctx == 0 and gen > 0:
        return f"decode_only_{gen_sum}"
    if ctx > 0 and gen > 0:
        return f"prefill_decode_{ctx_sum}_{gen_sum}"
    return f"prefill_decode_{batch}_{batch}"


def _phase_to_region_meta(phase: dict) -> dict:
    """Map TraceUtils phase dict to region_meta expected by extract_trace_data."""
    num_prefill = phase.get("num_prefill", 0)
    num_prefilldecode = phase.get("num_prefilldecode", 0)
    num_decode = phase.get("num_decode", 0)
    avg_bs = phase.get("avg_bs", 0)
    avg_conc = phase.get("avg_conc", 0)

    if num_prefill > 0 and num_prefilldecode == 0 and num_decode == 0:
        # Prefill-only
        return {
            "context_requests": num_prefill,
            "generation_requests": 0,
            "context_sum": avg_bs,
            "generation_sum": 0,
            "batch_size": avg_bs,
            "num_requests": avg_conc,
        }
    if num_prefill == 0 and num_prefilldecode == 0 and num_decode > 0:
        # Decode-only
        return {
            "context_requests": 0,
            "generation_requests": num_decode,
            "context_sum": 0,
            "generation_sum": avg_bs,
            "batch_size": avg_bs,
            "num_requests": avg_conc,
        }
    # Prefill-decode (or combined)
    return {
        "context_requests": num_prefill + num_prefilldecode,
        "generation_requests": num_decode + num_prefilldecode,
        "context_sum": avg_bs,
        "generation_sum": avg_bs,
        "batch_size": avg_bs,
        "num_requests": avg_conc,
    }


def _is_single_iteration(phase: dict) -> bool:
    """True if phase represents a single annotation iteration."""
    total = (
        phase.get("num_prefill", 0)
        + phase.get("num_prefilldecode", 0)
        + phase.get("num_decode", 0)
    )
    return total <= 1


def _iter_type_key(phase: dict) -> str:
    """Group key for deduplicating single iterations by type + token count."""
    np = phase.get("num_prefill", 0)
    npd = phase.get("num_prefilldecode", 0)
    nd = phase.get("num_decode", 0)
    bs = phase.get("avg_bs", 0)
    if np > 0:
        return f"prefill_{bs}"
    if npd > 0:
        return f"prefilldecode_{bs}"
    if nd > 0:
        return f"decode_{bs}"
    return f"empty_{bs}"


def _stamp_and_rewrite_gpu_op_uid(trace_path: str, tmpdir: str) -> str:
    """Stamp a trace-unique ``gpu_op_uid`` on every event before splitting.

    ``split_and_enumerate`` stamps no ``gpu_op_uid``, so split regions come
    back with ``gpu_op_uid = None`` everywhere -- which collapses to all-NaN
    diffs and duplicate ``(source, gpu_op_uid)`` keys downstream.

    This wrapper loads the trace exactly as the splitter does, stamps
    ``gpu_op_uid = index`` (the event's position in the original, unfiltered
    ``traceEvents`` array) on every event, and writes a stamped copy that is
    fed to ``split_and_enumerate``. The slicer preserves extra event fields,
    so each region kernel keeps a globally-unique, source-aligned uid. The
    canonical splitter is left untouched.
    """
    trace_json = DataLoader.load_data(get_filename(trace_path))
    for i, e in enumerate(trace_json.get("traceEvents", [])):
        if isinstance(e, dict):
            e["gpu_op_uid"] = i
    stamped_path = os.path.join(tmpdir, "stamped_source.json")
    with open(stamped_path, "w") as f:
        json.dump(trace_json, f)
    return stamped_path


def split_vllm_trace(
    trace_path: str,
    emit_gpu_op_uid: bool = False,
) -> Optional[List[Tuple[dict, dict]]]:  # pragma: no cover
    """
    Split a vLLM/SGLang/ATOM trace using the canonical TraceUtils trace splitter.

    Uses split_and_enumerate(..., store_single_iteration=True) to get
    per-annotation-iteration data, then selects one representative iteration
    per unique (type, token_count) group. This matches the reference
    analysis style of showing per-step averages rather than multi-step
    totals.

    Returns None if no annotation iterations are found, or if the trace is
    not splittable (caller falls back to whole-trace extraction).

    When ``emit_gpu_op_uid`` is True (opt-in; off by default so non-semantic
    callers are byte-for-byte unaffected), each split-region kernel is tagged
    with a ``gpu_op_uid`` equal to its index in the source trace, which
    scoreable semantic diffs need.
    """
    with tempfile.TemporaryDirectory(prefix="trace_split_") as tmpdir:
        split_input = trace_path
        if emit_gpu_op_uid:
            # Stamp source-array indices onto events before splitting so
            # split-region kernels keep a unique, source-aligned gpu_op_uid
            # (see helper above).
            split_input = _stamp_and_rewrite_gpu_op_uid(trace_path, tmpdir)
        execution_details = split_and_enumerate(
            split_input,
            tmpdir,
            find_steady_state=True,
            store_single_iteration=True,
            iterations="all",
        )
        if not execution_details:
            return None

        # Collect single-iteration entries, grouped by type+token_count
        groups: dict = {}  # iter_type_key -> list of (entry, phase)
        for entry in execution_details:
            out_path = entry.get("output_path")
            if not out_path or not os.path.isfile(out_path):
                continue
            phase = entry.get("phase") or {}
            if not _is_single_iteration(phase):
                continue
            total_steps = (
                phase.get("num_prefill", 0)
                + phase.get("num_prefilldecode", 0)
                + phase.get("num_decode", 0)
            )
            if total_steps == 0:
                continue
            key = _iter_type_key(phase)
            groups.setdefault(key, []).append((entry, phase))

        # Pick the median-busy-time representative from each group
        output = []
        for key, entries in groups.items():
            entries.sort(key=lambda ep: ep[0].get("gpu_busy_duration", 0))
            mid = len(entries) // 2
            entry, phase = entries[mid]
            region_meta = _phase_to_region_meta(phase)
            if "gpu_duration" in entry:
                region_meta["traceutils_gpu_duration_us"] = entry["gpu_duration"]
            if "gpu_busy_duration" in entry:
                region_meta["traceutils_gpu_busy_duration_us"] = entry[
                    "gpu_busy_duration"
                ]
            trace_dict = DataLoader.load_data(entry["output_path"])
            output.append((trace_dict, region_meta))

        return output if output else None
