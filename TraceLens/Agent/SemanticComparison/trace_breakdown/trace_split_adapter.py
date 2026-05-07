###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Adapter for TraceUtils split_inference_trace_annotation.py.

Invokes TraceLens.TraceUtils.split_inference_trace_annotation as a subprocess
and converts its output to the format expected by extract_trace_data.py:
[(trace_dict, region_metadata), ...].

Usage:
    from trace_split_adapter import split_vllm_trace, get_steady_state_key
    result = split_vllm_trace("trace.json.gz")
    if result:
        for trace_dict, metadata in result:
            key = get_steady_state_key(metadata)
"""

import gzip
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple


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


def _get_tracelens_root() -> str:
    """Resolve TraceLens package root (tracelens_private/ or repo root)."""
    # trace_split_adapter.py is at TraceLens/AgenticMode/SemanticComparison/trace_breakdown/
    this_file = os.path.abspath(__file__)
    trace_breakdown = os.path.dirname(this_file)
    semantic = os.path.dirname(trace_breakdown)
    agentic = os.path.dirname(semantic)
    tracelens_pkg = os.path.dirname(agentic)
    root = os.path.dirname(tracelens_pkg)
    return root


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


def _load_trace(path: str) -> dict:
    """Load trace JSON from .json or .json.gz file."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path, "r") as f:
        return json.load(f)


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


def split_vllm_trace(trace_path: str) -> Optional[List[Tuple[dict, dict]]]:
    """
    Split a vLLM trace using TraceUtils split_inference_trace_annotation.

    Uses --store-single-iteration to get per-annotation-iteration data,
    then selects one representative iteration per unique (type, token_count)
    group.  This matches the reference analysis style of showing per-step
    averages rather than multi-step totals.

    Returns None if no annotation iterations are found (caller should fall back
    to full trace).
    """
    root = _get_tracelens_root()
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = root + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = root

    with tempfile.TemporaryDirectory(prefix="trace_split_") as tmpdir:
        cmd = [
            sys.executable,
            "-m",
            "TraceLens.TraceUtils.split_inference_trace_annotation",
            trace_path,
            "-o",
            tmpdir,
            "--find-steady-state",
            "--store-single-iteration",
            "--iterations",
            "all",
        ]
        result = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"TraceUtils split failed (exit {result.returncode}): {result.stderr}",
                file=sys.stderr,
            )
            return None

        details_path = os.path.join(tmpdir, "execution_details.json")
        if not os.path.isfile(details_path):
            return None

        with open(details_path, "r") as f:
            execution_details = json.load(f)

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
            trace_dict = _load_trace(entry["output_path"])
            output.append((trace_dict, region_meta))

        return output if output else None
