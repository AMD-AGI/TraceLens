#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TraceLens Agent - Orchestrator Preparation Script
Steps 2-5: GPU Utilization, Top Ops, Tree Data Pre-computation, Category Filtering
"""

import argparse
import ast
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from typing import Any
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from category_analyses.analysis_utils import parse_first_shape, shape_aware_lookup
from utils.arch_utils import list_platforms, load_arch
from TraceLens.TreePerf import TreePerfAnalyzer
from TraceLens.TreePerf.gpu_event_analyser import GPUEventAnalyser
from TraceLens.Agent.Analysis.utils.classify_kernels import (
    classify_kernel,
)

CATEGORY_SKILL_MAP = {
    "cpu_idle": "cpu-idle-analyzer",
    "gemm": "gemm-analyzer",
    "groupedgemm_fwd": "gemm-analyzer",
    "groupedgemm_bwd": "gemm-analyzer",
    "moe_fused": "moe-analyzer",
    "moe_unfused": "moe-analyzer",
    "sdpa_fwd": "sdpa-analyzer",
    "sdpa_bwd": "sdpa-analyzer",
    "inferenceattention": "sdpa-analyzer",
    "elementwise": "elementwise-analyzer",
    "reduce": "reduce-analyzer",
    "triton": "triton-analyzer",
    "norm": "norm-analyzer",
    "norm_fwd": "norm-analyzer",
    "norm_bwd": "norm-analyzer",
    "rmsnorm": "norm-analyzer",
    "convolution": "convolution-analyzer",
    "conv_fwd": "convolution-analyzer",
    "conv_bwd": "convolution-analyzer",
}

FUSION_EXCLUDED_KERNELS = ["nccl", "rccl", "memcpy", "memset"]
FUSION_ALREADY_FUSED = [
    "attn_fwd",
    "attn_bwd",
    "fmha",
    "unified_attention",
    "paged_attention",
    "flash_attn",
    "flash_fwd",
    "silu_and_mul",
    "SiluAndMul",
]
_NORM_KERNEL_PATTERNS = [
    "batchnorm",
    "layernorm",
    "groupnorm",
    "instancenorm",
    "rmsnorm",
]
_MODULE_INDEX_RE = re.compile(r"_(\d+)$")

# Layout-transform kernels emitted by BOTH backends (tile/autotuner choice, NOT fusion)
_LAYOUT_TRANSFORM_PATTERNS = [
    "batched_transpose",
    "nchw2nhwc",
    "nhwc2nchw",
    "transpose_nchw",
    "transpose_cnhw",
]

# Case-A gate constants (comparative mode)
_CASE_A_MAX_DELTA = 15  # G2: |n1 - n2| must be <= this
_CASE_A_MAX_N1 = 30     # G3: n1 must be <= this

# Event categories that are never fusion candidate containers
_SKIP_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"}


# ---------------------------------------------------------------------------
# Module-level helpers shared by standalone and comparative extraction
# ---------------------------------------------------------------------------

def _strip_module_index(name):
    prefix = (
        name[len("nn.Module: "):]
        if name.startswith("nn.Module: ")
        else name
    )
    return _MODULE_INDEX_RE.sub("", prefix)


def _has_fused_kernel(kernel_list):
    return any(
        any(p in k["name"].lower() for p in FUSION_ALREADY_FUSED)
        for k in kernel_list
    )


def _build_parent_chain(ev, tree) -> list:
    """Walk ancestors and return a list of cleaned parent names (outermost last)."""
    parent_chain = []
    ancestor = ev
    while tree.get_parent_event(ancestor) is not None:
        ancestor = tree.get_parent_event(ancestor)
        pname = ancestor.get("name", "")
        if pname:
            if pname.startswith("nn.Module: "):
                pname = pname[len("nn.Module: "):]
            elif "/" in pname:
                pname = pname.rsplit("/", 1)[-1]
            parent_chain.append(pname)
    return parent_chain


def _dedup_by_kernel_set(candidates: list, kernels_field: str, score_fn) -> list:
    """Deduplicate fusion candidates with identical kernel name tuples.

    When two candidates share the same ordered kernel names, keeps the one
    with the higher ``score_fn`` value.  Handles both ``name`` and
    ``kernel_name`` key variants in the kernel dicts.
    """
    seen: dict = {}
    for c in candidates:
        kernels = c.get(kernels_field, [])
        if not kernels:
            continue
        nk = "name" if "name" in kernels[0] else "kernel_name"
        kset = tuple(k.get(nk, "") for k in kernels)
        prev = seen.get(kset)
        if prev is None or score_fn(c) > score_fn(prev):
            seen[kset] = c
    return list(seen.values())


# ---------------------------------------------------------------------------
# Comparative fusion extraction helpers (used only by
# _extract_comparative_fusion_candidates)
# ---------------------------------------------------------------------------

def _is_case_a_fusion_gap(trace1_kernels: list, trace2_kernels: list) -> bool:
    """Return True if Case-A gates pass (trace1 has more kernels than trace2).

    Gates:
      G1: n1 > n2              — focus on trace1 inefficiencies only
      G2: |n1-n2| <= 15        — large deltas indicate structural divergence (Case B)
      G3: n1 <= 30             — oversized sub-trees are also Case B
      G4: n2 >= 1              — cannot do gap analysis with nothing on trace2
    """
    n1, n2 = len(trace1_kernels), len(trace2_kernels)
    if n1 <= n2:                          # G1
        return False
    if (n1 - n2) > _CASE_A_MAX_DELTA:    # G2
        return False
    if n1 > _CASE_A_MAX_N1:              # G3
        return False
    if n2 < 1:                           # G4
        return False
    return True


def _build_diff_stats_lookups(df):
    """Build UID→kernel and LCA→trace2-kernels mappings from diff_stats DataFrame."""
    uid_to_t1 = {}  # gpu_event_uid → {name, type, dur_us, lca_id}
    lca_to_t2: dict = defaultdict(list)  # lca_id → [kernel dicts]

    has_uid = "gpu_event_uid" in df.columns
    for _, row in df.iterrows():
        kname = row["name"]
        ktype, *_ = classify_kernel(kname)
        lca_id = row["lowest_common_ancestor_id"]
        dur_us = float(row.get("kernel_time", 0) or 0)

        if row["source"] == "trace1" and has_uid and pd.notna(row.get("gpu_event_uid")):
            uid = int(row["gpu_event_uid"])
            uid_to_t1[uid] = {
                "name": kname,
                "type": ktype,
                "dur_us": dur_us,
                "lca_id": lca_id,
                "gpu_event_uid": uid,
            }
        elif row["source"] == "trace2":
            lca_to_t2[lca_id].append({
                "name": kname,
                "type": ktype,
                "dur_us": dur_us,
            })

    return uid_to_t1, dict(lca_to_t2)


def _apply_comparative_gates(t1_kernels, t2_kernels):
    """Apply Case-A gates and delta filters.  Returns True if candidate should be kept."""
    if not _is_case_a_fusion_gap(t1_kernels, t2_kernels):
        return False

    n1, n2 = len(t1_kernels), len(t2_kernels)
    delta_kernels = sorted(t1_kernels, key=lambda k: k["dur_us"])[: n1 - n2]

    # Skip if all delta kernels are excluded infrastructure (nccl/rccl/memcpy).
    if delta_kernels and all(
        any(x in k["name"].lower() for x in FUSION_EXCLUDED_KERNELS)
        for k in delta_kernels
    ):
        return False

    # Skip if all delta kernels are layout-transforms (tile/autotuner choice).
    if delta_kernels and all(
        any(p in k["name"].lower() for p in _LAYOUT_TRANSFORM_PATTERNS)
        for k in delta_kernels
    ):
        return False

    return True


def _make_comparative_candidate(
    module_name, base_name, t1_kernels, t2_kernels, parent_chain=None, input_dims=None, lca_id=None,
):
    """Build a comparative candidate dict from trace1/trace2 kernel lists."""
    n1, n2 = len(t1_kernels), len(t2_kernels)
    type_sig = [k["type"] for k in t1_kernels]
    type_summary: dict = {}
    for k in t1_kernels:
        type_summary[k["type"]] = type_summary.get(k["type"], 0) + 1

    candidate = {
        "module_name": module_name,
        "base_name": base_name,
        "parent_chain": parent_chain or [],
        "instance_count": 1,
        "kernel_count_trace1": n1,
        "kernel_count_trace2": n2,
        "delta": n1 - n2,
        "kernels_trace1": t1_kernels,
        "kernels_trace2": t2_kernels,
        "kernel_type_signature": type_sig,
        "kernel_type_summary": type_summary,
        "has_fused_kernel": _has_fused_kernel(t1_kernels),
        "total_kernel_time_us_trace1": sum(k["dur_us"] for k in t1_kernels),
        "total_kernel_time_us_trace2": sum(k["dur_us"] for k in t2_kernels),
        "input_dims": input_dims,
    }
    if lca_id is not None:
        candidate["lca_id"] = int(lca_id)
    return candidate


def _extract_comparative_fusion_candidates(
    trace1_csv_dir: str, analyzer=None, tree=None,
) -> list:
    """Extract fusion candidates from diff_stats.csv (comparative mode).

    Walks trace1's event tree at every level (nn.Module, aten::, etc.),
    cross-references descendant GPU kernel UIDs against diff_stats.csv to
    find trace2 counterparts via LCA.  Deduplicates by ``base_name`` with
    instance_count accumulation (same as standalone).
    """
    diff_stats_path = os.path.join(trace1_csv_dir, "diff_stats.csv")
    if not os.path.exists(diff_stats_path):
        print(f"  ⚠️  diff_stats.csv not found at {diff_stats_path}")
        return []

    df = pd.read_csv(diff_stats_path)
    if df.empty or "lowest_common_ancestor_id" not in df.columns:
        print("  ⚠️  diff_stats.csv is empty or missing expected columns")
        return []

    df = df.dropna(subset=["lowest_common_ancestor_id"])

    if tree is None or analyzer is None:
        print("  ⚠️  tree/analyzer not provided; comparative extraction requires both")
        return []

    uid_to_t1, lca_to_t2 = _build_diff_stats_lookups(df)
    if not uid_to_t1:
        print("  ⚠️  No trace1 kernel UIDs found in diff_stats.csv")
        return []

    categorizer = analyzer.event_to_category
    seen_base: dict = {}

    for ev in tree.events:
        if categorizer(ev) in _SKIP_CATEGORIES:
            continue
        gpu_uids = list(dict.fromkeys(ev.get("gpu_events", [])))
        if len(gpu_uids) < 2:
            continue
        name = ev.get("name", "")
        base = _strip_module_index(name)
        if not base:
            continue
        
        # Gather trace1 kernels that appear in diff_stats
        t1_kernels = []
        matched_lca_ids = set()
        for uid in gpu_uids:
            info = uid_to_t1.get(uid)
            if info is not None:
                t1_kernels.append(info)
                matched_lca_ids.add(info["lca_id"])

        if len(t1_kernels) < 2:
            continue

        # Gather trace2 kernels from the matched LCAs
        t2_kernels = []
        for lca_id in matched_lca_ids:
            t2_kernels.extend(lca_to_t2.get(lca_id, []))

        if not _apply_comparative_gates(t1_kernels, t2_kernels):
            continue

        t1_time = sum(k["dur_us"] for k in t1_kernels)
        t2_time = sum(k["dur_us"] for k in t2_kernels)

        if base in seen_base:
            seen_base[base]["instance_count"] += 1
            seen_base[base]["total_kernel_time_us_trace1"] += t1_time
            seen_base[base]["total_kernel_time_us_trace2"] += t2_time
            continue

        candidate = _make_comparative_candidate(
            module_name=name,
            base_name=base,
            t1_kernels=t1_kernels,
            t2_kernels=t2_kernels,
            parent_chain=_build_parent_chain(ev, tree),
            input_dims=ev.get("args", {}).get("Input Dims"),
        )
        seen_base[base] = candidate

    candidates = [
        c
        for c in seen_base.values()
        if c.get("total_kernel_time_us_trace1", 0)
        >= c.get("total_kernel_time_us_trace2", 0)
    ]

    candidates = _dedup_by_kernel_set(
        candidates,
        kernels_field="kernels_trace1",
        score_fn=lambda c: (
            c.get("module_name", "").startswith("nn.Module:"),
            len(c.get("parent_chain", [])),
        ),
    )

    return candidates


def _is_fusion_eligible(name):
    lower = name.lower()
    return not any(x in lower for x in FUSION_EXCLUDED_KERNELS) and not any(
        x in lower for x in FUSION_ALREADY_FUSED
    )


def _extract_standalone_fusion_candidates(analyzer, tree, trace1_csv_dir: str) -> list:
    """Extract fusion candidates from the live trace tree (standalone mode).

    Walks tree.events to find modules that launch >= 2 GPU kernels, then runs a
    sibling-sequence pass and post-processing (attention narrowing, kernel-set
    dedup, data-movement enrichment).  Returns candidates in the standalone schema.
    """
    seen_base = {}
    base_order = []
    categorizer = analyzer.event_to_category

    for ev in tree.events:
        if categorizer(ev) in _SKIP_CATEGORIES:
            continue
        gpu_uids = ev.get("gpu_events", [])
        if len(gpu_uids) < 2:
            continue
        name = ev.get("name", "")
        base = _strip_module_index(name)
        if not base:
            continue

        if base in seen_base:
            seen_base[base]["instance_count"] += 1
            seen_base[base]["total_kernel_time_us"] += sum(
                tree.get_UID2event(u).get("dur", 0)
                for u in gpu_uids
                if categorizer(tree.get_UID2event(u)) == "kernel"
            )
            continue

        kernels = []
        for uid in gpu_uids:
            try:
                k = tree.get_UID2event(uid)
                if categorizer(k) == "kernel":
                    kname = k.get("name", "")
                    ktype, *_ = classify_kernel(kname)
                    kernels.append(
                        {
                            "name": kname,
                            "type": ktype,
                            "dur_us": k.get("dur", 0),
                            "eligible": _is_fusion_eligible(kname),
                            "gpu_event_uid": uid,
                        }
                    )
            except (KeyError, IndexError):
                pass
        if len(kernels) < 2:
            continue

        type_sig = [k["type"] for k in kernels]
        type_summary = {}
        for k in kernels:
            type_summary[k["type"]] = type_summary.get(k["type"], 0) + 1

        entry = {
            "module_name": name,
            "base_name": base,
            "parent_chain": _build_parent_chain(ev, tree),
            "instance_count": 1,
            "kernel_count": len(kernels),
            "eligible_kernel_count": sum(1 for k in kernels if k["eligible"]),
            "kernels": kernels,
            "kernel_type_signature": type_sig,
            "kernel_type_summary": type_summary,
            "has_fused_kernel": _has_fused_kernel(kernels),
            "total_kernel_time_us": sum(k["dur_us"] for k in kernels),
            "input_dims": ev.get("args", {}).get("Input Dims"),
        }
        seen_base[base] = entry
        base_order.append(base)

    # Sibling sequence extraction
    collected_events = analyzer.collect_unified_perf_events()
    parent_groups = defaultdict(list)
    for ev in collected_events:
        gpu_uids = ev.get("gpu_events", [])
        if len(gpu_uids) != 1:
            continue
        parent_uid = ev.get("parent")
        if parent_uid is None:
            continue
        try:
            k = tree.get_UID2event(gpu_uids[0])
            if categorizer(k) != "kernel":
                continue
            kname = k.get("name", "")
            ktype, *_ = classify_kernel(kname)
            parent_groups[parent_uid].append(
                {
                    "op": ev.get("name", ""),
                    "kernel_type": ktype,
                    "kernel_name": kname,
                    "dur_us": k.get("dur", 0),
                    "gpu_event_uid": gpu_uids[0],
                }
            )
        except (KeyError, IndexError):
            pass

    sibling_seqs = []
    seen_sibling_bases = {}
    for parent_uid, children in parent_groups.items():
        if len(children) < 2:
            continue
        try:
            parent_evt = tree.get_UID2event(parent_uid)
        except (KeyError, IndexError):
            continue
        pname = parent_evt.get("name", "")
        sbase = _strip_module_index(pname)
        if sbase in seen_sibling_bases:
            seen_sibling_bases[sbase]["instance_count"] += 1
            seen_sibling_bases[sbase]["total_time_us"] += sum(
                c["dur_us"] for c in children
            )
            continue
        entry = {
            "ancestor_name": pname,
            "base_name": sbase,
            "sequence": children,
            "kernel_type_signature": [c["kernel_type"] for c in children],
            "total_time_us": sum(c["dur_us"] for c in children),
            "instance_count": 1,
            "input_dims": parent_evt.get("args", {}).get("Input Dims"),
        }
        seen_sibling_bases[sbase] = entry
        sibling_seqs.append(entry)

    fusion_candidates = []
    for b in base_order:
        m = seen_base[b]
        if m["has_fused_kernel"] or m["eligible_kernel_count"] < 2:
            continue
        if _is_gemm_norm_only(m):
            continue
        fusion_candidates.append(m)

    for s in sibling_seqs:
        if len(s["sequence"]) < 2:
            continue
        fusion_candidates.append(
            {
                "module_name": s["ancestor_name"],
                "base_name": s["base_name"],
                "parent_chain": [],
                "instance_count": s["instance_count"],
                "kernel_count": len(s["sequence"]),
                "eligible_kernel_count": len(s["sequence"]),
                "kernels": s["sequence"],
                "kernel_type_signature": s["kernel_type_signature"],
                "kernel_type_summary": {},
                "has_fused_kernel": False,
                "total_kernel_time_us": s["total_time_us"],
                "source": "sibling_sequence",
                "input_dims": s.get("input_dims"),
            }
        )

    # Post-process: attention narrowing + data movement enrichment
    csv_path = os.path.join(trace1_csv_dir, "unified_perf_summary.csv")
    if os.path.exists(csv_path):
        perf_lookup = _build_kernel_perf_lookup(csv_path)

        for c in fusion_candidates:
            core = _extract_attention_core(c.get("kernels", []), perf_lookup)
            if core is not None:
                c["kernels"] = core
                c["kernel_count"] = len(core)
                c["eligible_kernel_count"] = len(core)
                c["kernel_type_signature"] = [
                    k.get("type", k.get("kernel_type", "Unknown")) for k in core
                ]
                c["total_kernel_time_us"] = sum(
                    k.get("dur_us", 0) for k in core
                ) * c.get("instance_count", 1)

        fusion_candidates = _dedup_by_kernel_set(
            fusion_candidates,
            kernels_field="kernels",
            score_fn=lambda c: (
                c.get("module_name", "").startswith("nn.Module:"),
                len(c.get("parent_chain", [])),
            ),
        )

        for c in fusion_candidates:
            for k in c.get("kernels", []):
                if k.get("data_in_mb") is not None:
                    continue
                kname = k.get("name", k.get("kernel_name", ""))
                entry = shape_aware_lookup(
                    perf_lookup, kname, c.get("input_dims")
                )
                if entry.get("data_in_mb") is not None:
                    k["data_in_mb"] = entry["data_in_mb"]
                    k["data_out_mb"] = entry["data_out_mb"]

    print(
        f"  ✓ Fusion candidates: {len(seen_base)} unique module types, "
        f"{len(fusion_candidates)} candidates"
    )
    return fusion_candidates


def _normalize_category(row):
    """Normalize upstream ``op category`` to a filesystem-safe key."""
    category = row.get("op category", "")
    if pd.isna(category) or category == "":
        return "other", "Other"
    category_name = category.replace(" ", "_").replace("/", "_").lower()
    return category_name, category


def _build_trace2_ops_summary_by_enhanced_category(trace2_csv_dir: str) -> list:
    """Build trace 2 ops summary grouped by enhanced category (op-name-based).

    Uses unified_perf_summary.csv and get_enhanced_category() so that trace 2
    categories match the trace 1 enhanced categories (e.g. CONV_fwd + CONV_bwd
    merge into 'convolution', NORM_fwd + NORM_bwd merge into 'norm').
    Falls back to ops_summary_by_category.csv if unified_perf_summary.csv is
    not available.
    """
    unified_path = os.path.join(trace2_csv_dir, "unified_perf_summary.csv")
    if not os.path.isfile(unified_path):
        return json.loads(
            pd.read_csv(
                os.path.join(trace2_csv_dir, "ops_summary_by_category.csv")
            ).to_json(orient="records")
        )

    df = pd.read_csv(unified_path)
    df["enhanced_category"], df["display_name"] = zip(
        *df.apply(_normalize_category, axis=1)
    )

    if "Kernel Time (µs)_sum" in df.columns:
        time_col = "Kernel Time (µs)_sum"
        time_scale = 1 / 1000  # µs → ms
    elif "total_direct_kernel_time_ms" in df.columns:
        time_col = "total_direct_kernel_time_ms"
        time_scale = 1.0
    else:
        return json.loads(
            pd.read_csv(
                os.path.join(trace2_csv_dir, "ops_summary_by_category.csv")
            ).to_json(orient="records")
        )

    count_col = (
        "operation_count" if "operation_count" in df.columns else "enhanced_category"
    )
    grouped = (
        df.groupby("enhanced_category")
        .agg(
            Count=(count_col, "sum" if count_col == "operation_count" else "size"),
            total_direct_kernel_time_ms=(time_col, "sum"),
        )
        .reset_index()
    )
    grouped["total_direct_kernel_time_ms"] *= time_scale
    total_time = grouped["total_direct_kernel_time_ms"].sum()
    grouped["Percentage (%)"] = (
        (grouped["total_direct_kernel_time_ms"] / total_time * 100)
        if total_time > 0
        else 0
    )
    grouped = grouped.sort_values("total_direct_kernel_time_ms", ascending=False)
    grouped["Cumulative Percentage (%)"] = grouped["Percentage (%)"].cumsum()
    grouped = grouped.rename(columns={"enhanced_category": "op category"})

    return json.loads(grouped.to_json(orient="records"))


def _gpu_utilization_metrics_from_gpu_timeline_df(gpu_timeline: pd.DataFrame) -> dict:
    """Build the same gpu_utilization dict used in category_manifest from gpu_timeline.csv rows."""
    gpu_data = {}
    for _, row in gpu_timeline.iterrows():
        gpu_data[row["type"]] = {"time_ms": row["time ms"], "percent": row["percent"]}
    return {
        "total_time_ms": gpu_data.get("total_time", {}).get("time_ms", 0),
        "computation_time_percent": gpu_data.get("computation_time", {}).get(
            "percent", 0
        ),
        "exposed_comm_time_percent": gpu_data.get("exposed_comm_time", {}).get(
            "percent", 0
        ),
        "exposed_memcpy_time_percent": gpu_data.get("exposed_memcpy_time", {}).get(
            "percent", 0
        ),
        "idle_time_percent": gpu_data.get("idle_time", {}).get("percent", 0),
    }




def _compute_data_in_out(op_category, perf_params_str, data_moved_mb):
    """Split Data Moved into (data_in_mb, data_out_mb) using op type and shapes."""
    half = (data_moved_mb / 2, data_moved_mb / 2) if data_moved_mb else (None, None)
    try:
        params = ast.literal_eval(str(perf_params_str)) if perf_params_str else {}
    except Exception:
        return half
    if data_moved_mb is None:
        return half
    if op_category == "GEMM":
        M, N, K = params.get("M"), params.get("N"), params.get("K")
        if all((M, N, K)):
            total = M * K + K * N + M * N
            return (
                data_moved_mb * (M * K + K * N) / total,
                data_moved_mb * M * N / total,
            )
    elif op_category == "reduce":
        return data_moved_mb, 0.0
    elif op_category == "elementwise" and "shape_in1" in params:
        return data_moved_mb * 2 / 3, data_moved_mb / 3
    return half


def _build_kernel_perf_lookup(csv_path):
    """GPU kernel name -> {shape -> {op_category, data_in_mb, data_out_mb}} from perf CSV."""
    df = pd.read_csv(csv_path)
    lookup = defaultdict(dict)
    for _, row in df.iterrows():
        kd = row.get("kernel_details_summary", "")
        if pd.isna(kd):
            continue
        cat = row.get("op category", "")
        dm = row.get("Data Moved (MB)")
        dm = float(dm) if dm is not None and not pd.isna(dm) else None
        pp = row.get("perf_params", "")
        if pd.isna(pp):
            pp = ""
        data_in, data_out = _compute_data_in_out(cat, pp, dm)
        shape = parse_first_shape(row.get("Input Dims"))
        for kn in re.findall(r"'name':\s*'([^']+)'", str(kd)):
            if shape not in lookup[kn]:
                lookup[kn][shape] = {
                    "op_category": cat,
                    "data_in_mb": data_in,
                    "data_out_mb": data_out,
                }
    return dict(lookup)


def _is_gemm_norm_only(entry):
    """Reject GEMM + norm-only candidates (BN folding, not kernel fusion)."""
    kernels = entry.get("kernels", [])
    has_gemm = any(
        k.get("type", "").upper() in ("GEMM", "GEMM (GENERIC)")
        or "conv" in k.get("name", "").lower()
        for k in kernels
    )
    if not has_gemm:
        return False
    non_gemm = [
        k
        for k in kernels
        if k.get("type", "").upper() not in ("GEMM", "GEMM (GENERIC)")
        and "conv" not in k.get("name", "").lower()
    ]
    if not non_gemm:
        return False
    return all(
        any(p in k.get("name", "").lower() for p in _NORM_KERNEL_PATTERNS)
        or k.get("type", "") in ("Elementwise Add", "Copy/Cast")
        for k in non_gemm
    )


def _prefix_lookup(lookup, kname):
    """Look up by exact key, falling back to prefix match for truncated CSV names."""
    result = lookup.get(kname)
    if result is not None:
        return result
    for csv_name in lookup:
        if kname.startswith(csv_name) or csv_name.startswith(kname):
            return lookup[csv_name]
    return None


def _extract_attention_core(kernels, perf_lookup):
    """If kernels contain unfused attention (softmax), return just QKt+softmax+PV."""
    name_key = "name" if "name" in (kernels[0] if kernels else {}) else "kernel_name"

    def is_gemm(k):
        entries = _prefix_lookup(perf_lookup, k.get(name_key, "")) or {}
        return any(e.get("op_category") == "GEMM" for e in entries.values())

    for i, k in enumerate(kernels):
        if "softmax" not in k.get(name_key, "").lower():
            continue
        qk = next((j for j in range(i - 1, -1, -1) if is_gemm(kernels[j])), None)
        pv = next((j for j in range(i + 1, len(kernels)) if is_gemm(kernels[j])), None)
        if qk is not None and pv is not None:
            return kernels[qk : pv + 1]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare category data for TraceLens analysis"
    )
    parser.add_argument("--trace-path", required=True, help="Path to trace file")
    parser.add_argument(
        "--platform",
        required=True,
        choices=list_platforms(),
        help="AMD platform (MI300X, MI325X, MI350X, MI355X, MI455X)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--comparison-scope",
        choices=("standalone", "comparative"),
        default="standalone",
        help=(
            "standalone: read <output-dir>/perf_report_csvs (single trace). "
            "comparative: trace 1 from perf_report_trace1_csvs, trace 2 from perf_report_trace2_csvs."
        ),
    )
    parser.add_argument(
        "--disable_pseudo_ops",
        action="store_false",
        dest="enable_pseudo_ops",
        default=True,
        help="Disable pseudo-op augmentation (enabled by default).",
    )

    args = parser.parse_args()

    trace_path = args.trace_path
    platform = args.platform
    output_dir = args.output_dir
    enable_pseudo_ops = args.enable_pseudo_ops
    comparison_scope = args.comparison_scope
    if comparison_scope == "comparative":
        trace1_csv_dir = os.path.join(output_dir, "perf_report_trace1_csvs")
        trace2_csv_dir = os.path.join(output_dir, "perf_report_trace2_csvs")
    else:
        trace1_csv_dir = os.path.join(output_dir, "perf_report_csvs")
        trace2_csv_dir = None

    print("=" * 80)
    print("TRACELENS AGENT - ORCHESTRATOR PREPARATION")
    print("=" * 80)
    print(f"Platform: {platform}")
    print(f"Trace: {trace_path}")
    print(f"Output: {output_dir}")
    print(f"Comparison scope: {comparison_scope}")
    print(f"Trace 1 perf CSVs: {trace1_csv_dir}")
    if trace2_csv_dir is not None:
        print(f"Trace 2 perf CSVs: {trace2_csv_dir}")
    print(f"Pseudo Ops: {'Enabled' if enable_pseudo_ops else 'Disabled'}")
    print("=" * 80)

    # Create directory structure (chmod 777 so host user can write when running in container as root)
    for d in [
        output_dir,
        f"{output_dir}/metadata",
        f"{output_dir}/category_data",
        f"{output_dir}/category_findings",
        f"{output_dir}/system_findings",
    ]:
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o777)

    platform_specs = load_arch(platform)

    # ============================================================================
    # STEP 2: Assess GPU Utilization
    # ============================================================================
    print("\n[STEP 2] Assessing GPU Utilization...")

    gpu_timeline = pd.read_csv(os.path.join(trace1_csv_dir, "gpu_timeline.csv"))
    gpu_utilization_metrics = _gpu_utilization_metrics_from_gpu_timeline_df(
        gpu_timeline
    )

    if comparison_scope == "comparative" and trace2_csv_dir is not None:
        trace2_gpu_utilization = _gpu_utilization_metrics_from_gpu_timeline_df(
            pd.read_csv(os.path.join(trace2_csv_dir, "gpu_timeline.csv"))
        )
        with open(
            os.path.join(output_dir, "metadata", "trace2_gpu_utilization.json"), "w"
        ) as f:
            json.dump(trace2_gpu_utilization, f, indent=2)
        trace2_ops_summary_by_category = _build_trace2_ops_summary_by_enhanced_category(
            trace2_csv_dir
        )

    print(f"\nGPU Utilization Metrics:")
    print(f"  Total Time: {gpu_utilization_metrics['total_time_ms']:.2f} ms")
    print(f"  Computation: {gpu_utilization_metrics['computation_time_percent']:.2f}%")
    print(
        f"  Communication: {gpu_utilization_metrics['exposed_comm_time_percent']:.4f}%"
    )
    print(f"  MemCpy: {gpu_utilization_metrics['exposed_memcpy_time_percent']:.2f}%")
    print(f"  Idle: {gpu_utilization_metrics['idle_time_percent']:.2f}%")

    if gpu_utilization_metrics["computation_time_percent"] < 95:
        print(f"  ⚠️  WARNING: Compute utilization < 95%")

    # ============================================================================
    # STEP 3: Identify Top Operations
    # ============================================================================
    print("\n[STEP 3] Identifying Top Operations...")

    ops_summary = pd.read_csv(os.path.join(trace1_csv_dir, "ops_summary.csv"))

    # Sort by total_direct_kernel_time_ms if available, else by time column
    if "total_direct_kernel_time_ms" in ops_summary.columns:
        ops_summary_sorted = ops_summary.sort_values(
            "total_direct_kernel_time_ms", ascending=False
        )
        time_col = "total_direct_kernel_time_ms"
    elif "Kernel Time (µs)_sum" in ops_summary.columns:
        ops_summary_sorted = ops_summary.sort_values(
            "Kernel Time (µs)_sum", ascending=False
        )
        time_col = "Kernel Time (µs)_sum"
    else:
        print(f"  ⚠️  Could not determine time column")
        print(f"  Available columns: {ops_summary.columns.tolist()}")
        ops_summary_sorted = ops_summary
        time_col = None

    print(f"\nTop 10 Operations by GPU Time:")
    print("=" * 80)
    if time_col:
        for idx, row in ops_summary_sorted.head(10).iterrows():
            op_name = row.get("name", "Unknown")
            time_val = row[time_col]
            category = row.get("op category", "N/A")
            print(f"  {op_name:50s} | {time_val:10.2f} | {category}")

    # ============================================================================
    # STEP 4: Pre-compute Tree Data (Optimization)
    # ============================================================================
    print("\n[STEP 4] Pre-computing Tree Data for Bottleneck Operations...")

    try:
        print(f"  Loading trace: {trace_path}")
        print(f"  Pseudo ops: {'enabled' if enable_pseudo_ops else 'disabled'}")
        analyzer = TreePerfAnalyzer.from_file(
            trace_path,
            add_python_func=True,
            enable_pseudo_ops=enable_pseudo_ops,
        )
        tree = analyzer.tree
        print(f"  ✓ Trace loaded successfully")
        print(f"  ✓ Tree has {len(tree.events)} events")

        # Read unified performance summary
        unified_df = pd.read_csv(
            os.path.join(trace1_csv_dir, "unified_perf_summary.csv")
        )

        # Get unique categories
        categories = unified_df["op category"].unique()
        print(f"\n  Found {len(categories)} categories")

        # For each category, pre-compute tree data for bottlenecks
        for category in categories:
            if pd.isna(category) or category == "":
                category_name = "other"
                display_name = "Other"
            else:
                category_name = category.replace(" ", "_").replace("/", "_").lower()
                display_name = category

            # Filter operations for this category
            if pd.isna(category) or category == "":
                category_df = unified_df[
                    unified_df["op category"].isna() | (unified_df["op category"] == "")
                ]
            else:
                category_df = unified_df[unified_df["op category"] == category]

            if len(category_df) == 0:
                continue

            # Identify bottlenecks: ops with >10% of category time
            if "Kernel Time (µs)_sum" in category_df.columns:
                category_total_time = category_df["Kernel Time (µs)_sum"].sum()
                category_df = category_df.copy()
                category_df["category_percent"] = (
                    (category_df["Kernel Time (µs)_sum"] / category_total_time) * 100
                    if category_total_time > 0
                    else 0
                )

                bottleneck_ops = category_df[category_df["category_percent"] > 10]

                # If no ops > 10%, take top 5 by time
                if len(bottleneck_ops) == 0:
                    bottleneck_ops = category_df.nlargest(
                        min(5, len(category_df)), "Kernel Time (µs)_sum"
                    )

                # Simplified tree data (no full tree traversal to avoid complexity)
                tree_data = {}
                for idx, row in bottleneck_ops.iterrows():
                    target_uid = row.get("ex_UID", row.get("UID", None))
                    if pd.isna(target_uid):
                        continue

                    tree_data[str(int(target_uid))] = {
                        "op_name": row.get("name", "Unknown"),
                        "ex_uid": int(target_uid),
                        "input_dims": str(row.get("Input Dims", "")),
                        "parent_chain": [],  # Simplified
                        "subtree": [],
                        "fusion_opportunity": False,
                        "notes": "Tree traversal simplified - using CSV data",
                    }

                # Save tree data
                tree_data_file = (
                    f"{output_dir}/category_data/{category_name}_tree_data.json"
                )
                with open(tree_data_file, "w") as f:
                    json.dump(tree_data, f, indent=2)

        print(f"  ✓ Pre-computed tree data for bottleneck operations")

        # ====================================================================
        # STEP 4.5: Pre-compute Multi-Kernel Issue Data
        # ====================================================================
        print("\n[STEP 4.5] Pre-computing Multi-Kernel Issue Data...")

        try:
            gpu_analyser = GPUEventAnalyser(tree.events)
            event_lists = gpu_analyser.get_gpu_event_lists()
            mk_gpu_events = event_lists[GPUEventAnalyser.all_gpu_key]
            mk_comm_events = event_lists[GPUEventAnalyser.communication_key]
            mk_memcpy_events = event_lists[GPUEventAnalyser.memcpy_key]

            # Sub-classify memcpy by direction (not provided by GPUEventAnalyser)
            memcpy_by_direction = {"D2H": [], "H2D": [], "D2D": [], "other": []}
            for event in mk_memcpy_events:
                name = event.get("name", "").lower()
                if "dtoh" in name or "device -> host" in name or "devicetohost" in name:
                    memcpy_by_direction["D2H"].append(event)
                elif (
                    "htod" in name or "host -> device" in name or "hosttodevice" in name
                ):
                    memcpy_by_direction["H2D"].append(event)
                elif (
                    "dtod" in name
                    or "device -> device" in name
                    or "devicetodevice" in name
                ):
                    memcpy_by_direction["D2D"].append(event)
                else:
                    memcpy_by_direction["other"].append(event)

            # Build memcpy summary
            memcpy_summary = {
                "total_count": len(mk_memcpy_events),
                "total_time_us": 0,
                "by_direction": {},
            }
            for direction, events in memcpy_by_direction.items():
                if not events:
                    continue
                durations = [e.get("dur", 0) for e in events]
                sizes = [e.get("args", {}).get("bytes", 0) for e in events]
                dir_summary = {
                    "count": len(events),
                    "total_time_us": round(sum(durations), 2),
                    "avg_time_us": round(sum(durations) / len(durations), 2),
                    "max_time_us": round(max(durations), 2),
                    "total_bytes": sum(s for s in sizes if s),
                    "avg_bytes": (
                        round(sum(s for s in sizes if s) / len(events), 2)
                        if any(sizes)
                        else 0
                    ),
                }
                memcpy_summary["by_direction"][direction] = dir_summary
                memcpy_summary["total_time_us"] += dir_summary["total_time_us"]
            memcpy_summary["total_time_us"] = round(memcpy_summary["total_time_us"], 2)

            # Build NCCL/communication summary
            nccl_summary = {"total_count": len(mk_comm_events), "total_time_us": 0}
            if mk_comm_events:
                nccl_durations = [e.get("dur", 0) for e in mk_comm_events]
                nccl_summary["total_time_us"] = round(sum(nccl_durations), 2)
                nccl_summary["avg_time_us"] = round(
                    sum(nccl_durations) / len(nccl_durations), 2
                )
                nccl_summary["max_time_us"] = round(max(nccl_durations), 2)

                # Top NCCL ops by duration
                sorted_nccl = sorted(
                    mk_comm_events, key=lambda e: e.get("dur", 0), reverse=True
                )
                nccl_summary["top_ops"] = [
                    {
                        "name": e.get("name", ""),
                        "duration_us": round(e.get("dur", 0), 2),
                        "stream": e.get("args", {}).get("stream", None),
                    }
                    for e in sorted_nccl[:10]
                ]

            # Compute overlap metrics using GPUEventAnalyser
            overlap_analysis = {}
            if mk_gpu_events:
                try:
                    GPUEventAnalyser.verify_dict_gpu_event_lists(event_lists)
                    metrics = GPUEventAnalyser.compute_metrics_dict(event_lists)

                    total_time = metrics.get("total_time", 0)
                    comp_time = metrics.get("computation_time", 0)
                    total_comm_time = metrics.get("total_comm_time", 0)
                    exposed_comm_time = metrics.get("exposed_comm_time", 0)
                    total_memcpy_time = metrics.get("total_memcpy_time", 0)
                    exposed_memcpy_time = metrics.get("exposed_memcpy_time", 0)

                    overlap_analysis = {
                        "total_time_us": round(total_time, 2),
                        "computation_time_us": round(comp_time, 2),
                        "total_comm_time_us": round(total_comm_time, 2),
                        "exposed_comm_time_us": round(exposed_comm_time, 2),
                        "total_memcpy_time_us": round(total_memcpy_time, 2),
                        "exposed_memcpy_time_us": round(exposed_memcpy_time, 2),
                        "comm_overlap_ratio": (
                            round(1 - (exposed_comm_time / total_comm_time), 4)
                            if total_comm_time > 0
                            else None
                        ),
                        "memcpy_overlap_ratio": (
                            round(1 - (exposed_memcpy_time / total_memcpy_time), 4)
                            if total_memcpy_time > 0
                            else None
                        ),
                        "comm_percent_of_total": (
                            round(exposed_comm_time / total_time * 100, 2)
                            if total_time > 0
                            else 0
                        ),
                        "memcpy_percent_of_total": (
                            round(exposed_memcpy_time / total_time * 100, 2)
                            if total_time > 0
                            else 0
                        ),
                    }
                except Exception as e:
                    print(f"    ⚠️  Could not compute overlap metrics: {e}")
                    overlap_analysis = {"error": str(e)}

            # Write multi_kernel_data.json (raw statistics only -- pattern
            # detection and recommendations are handled by multi_kernel_analysis.py
            # and the multi-kernel-analyzer sub-agent respectively)
            multi_kernel_data = {
                "memcpy_summary": memcpy_summary,
                "nccl_summary": nccl_summary,
                "overlap_analysis": overlap_analysis,
            }

            multi_kernel_data_file = (
                f"{output_dir}/category_data/multi_kernel_data.json"
            )
            with open(multi_kernel_data_file, "w") as f:
                json.dump(multi_kernel_data, f, indent=2)

            print(
                f"  ✓ Multi-kernel data: {len(mk_memcpy_events)} memcpy events, {len(mk_comm_events)} NCCL events"
            )

        except Exception as e:
            print(f"  ⚠️  Error during multi-kernel data pre-computation: {e}")
            traceback.print_exc()
            # Write empty data so downstream scripts don't fail
            multi_kernel_data = {
                "memcpy_summary": {
                    "total_count": 0,
                    "total_time_us": 0,
                    "by_direction": {},
                },
                "nccl_summary": {"total_count": 0, "total_time_us": 0},
                "overlap_analysis": {},
                "error": str(e),
            }
            multi_kernel_data_file = (
                f"{output_dir}/category_data/multi_kernel_data.json"
            )
            with open(multi_kernel_data_file, "w") as f:
                json.dump(multi_kernel_data, f, indent=2)

        # ====================================================================
        # STEP 4b: Extract Kernel Fusion Candidates (Experimental)
        # ====================================================================
        print("\n[STEP 4b] Extracting Kernel Fusion Candidates...")

        fusion_candidates_file = (
            f"{output_dir}/category_data/fusion_candidates.json"
        )

        try:
            if comparison_scope == "comparative":
                # Comparative mode: use LCA-aligned diff_stats.csv as the source
                # so candidates carry both kernels_trace1 and kernels_trace2 context.
                diff_stats_csv = os.path.join(trace1_csv_dir, "diff_stats.csv")
                if os.path.exists(diff_stats_csv):
                    fusion_candidates = _extract_comparative_fusion_candidates(
                        trace1_csv_dir, analyzer=analyzer, tree=tree,
                    )
                    print(
                        f"  ✓ Comparative fusion: {len(fusion_candidates)} candidates "
                        f"(from diff_stats.csv)"
                    )
                else:
                    print(
                        "  ⚠️  diff_stats.csv not found; "
                        "comparative fusion extraction skipped"
                    )
                    fusion_candidates = []
            else:
                fusion_candidates = _extract_standalone_fusion_candidates(
                    analyzer, tree, trace1_csv_dir
                )

            if comparison_scope == "comparative":
                fusion_candidates.sort(
                    key=lambda c: c.get("total_kernel_time_us_trace1", 0),
                    reverse=True,
                )
            else:
                fusion_candidates.sort(
                    key=lambda c: c.get("total_kernel_time_us", 0),
                    reverse=True,
                )

            with open(fusion_candidates_file, "w") as f:
                json.dump(fusion_candidates, f, indent=2, default=str)

            print(
                f"  ✓ Written to fusion_candidates.json "
                f"({os.path.getsize(fusion_candidates_file) / 1024:.1f} KB)"
            )

        except Exception as ex:
            print(f"  ⚠️  Error during fusion candidate extraction: {ex}")
            traceback.print_exc()
            fusion_candidates = []
            with open(fusion_candidates_file, "w") as f:
                json.dump([], f)

    except Exception as e:
        print(f"  ⚠️  Error during tree data pre-computation: {e}")
        traceback.print_exc()

    # ============================================================================
    # STEP 5: Filter and Export Category Data
    # ============================================================================
    print("\n[STEP 5] Filtering and Exporting Category Data...")

    unified_df = pd.read_csv(os.path.join(trace1_csv_dir, "unified_perf_summary.csv"))

    # Apply enhanced categorization
    unified_df["enhanced_category"], unified_df["display_name"] = zip(
        *unified_df.apply(_normalize_category, axis=1)
    )

    categories = unified_df["enhanced_category"].unique()
    exported_categories = []

    for category_name in categories:
        category_df = unified_df[unified_df["enhanced_category"] == category_name]
        display_name = category_df.iloc[0]["display_name"]

        print(f"\n  Category: {display_name} ({category_name})")

        if len(category_df) == 0:
            print(f"    No operations - skipping")
            continue

        # Export filtered CSV
        csv_file = f"{output_dir}/category_data/{category_name}_ops.csv"
        category_df.to_csv(csv_file, index=False)
        print(f"    ✓ Exported CSV: {len(category_df)} ops")

        # Create metadata JSON
        metadata = {
            "platform": platform,
            "peak_hbm_bw_tbs": platform_specs["mem_bw_gbps"] / 1000,
            "max_achievable_tflops": platform_specs["max_achievable_tflops"],
            "memory_gb": platform_specs["memory_gb"],
            "trace_path": trace_path,
            "output_dir": output_dir,
            "category": display_name,
            "category_name": category_name,
            "gpu_utilization": gpu_utilization_metrics,
            "trace_loading_policy": "DO_NOT_LOAD_TRACE_use_precomputed_tree_data",
        }

        metadata_file = f"{output_dir}/metadata/{category_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"    ✓ Exported metadata")

        # Resolve tree_data_file: check enhanced name first, then fall back
        # to original op category names from Step 4 which may differ
        tree_data_file = f"{output_dir}/category_data/{category_name}_tree_data.json"
        if not os.path.exists(tree_data_file):
            orig_categories = category_df["op category"].dropna().unique()
            for orig_cat in orig_categories:
                orig_name = orig_cat.replace(" ", "_").replace("/", "_").lower()
                candidate = f"{output_dir}/category_data/{orig_name}_tree_data.json"
                if os.path.exists(candidate):
                    tree_data_file = candidate
                    break
            else:
                tree_data_file = None

        exported_categories.append(
            {
                "name": category_name,
                "display_name": display_name,
                "skill": CATEGORY_SKILL_MAP.get(category_name, "generic-op-analyzer"),
                "tier": "compute_kernel",
                "ops_count": len(category_df),
                "csv_file": csv_file,
                "metadata_file": metadata_file,
                "tree_data_file": tree_data_file,
            }
        )

    # ============================================================================
    # CPU/Idle Category Creation (always created)
    # ============================================================================
    print(f"\n  Category: CPU/Idle Analysis (cpu_idle)")
    print(f"    Idle time: {gpu_utilization_metrics['idle_time_percent']:.1f}%")

    cpu_idle_metadata = {
        "platform": platform,
        "peak_hbm_bw_tbs": platform_specs["mem_bw_gbps"] / 1000,
        "max_achievable_tflops": platform_specs["max_achievable_tflops"],
        "memory_gb": platform_specs["memory_gb"],
        "trace_path": trace_path,
        "output_dir": output_dir,
        "category": "CPU/Idle Analysis",
        "category_name": "cpu_idle",
        "gpu_utilization": gpu_utilization_metrics,
    }

    cpu_idle_metadata_file = f"{output_dir}/metadata/cpu_idle_metadata.json"
    with open(cpu_idle_metadata_file, "w") as f:
        json.dump(cpu_idle_metadata, f, indent=2)

    cpu_idle_csv = f"{output_dir}/category_data/cpu_idle_ops.csv"
    pd.DataFrame().to_csv(cpu_idle_csv, index=False)

    print(f"    ✓ Exported metadata")

    exported_categories.insert(
        0,
        {
            "name": "cpu_idle",
            "display_name": "CPU/Idle Analysis",
            "skill": "cpu-idle-analyzer",
            "tier": "system",
            "ops_count": 0,
            "csv_file": cpu_idle_csv,
            "metadata_file": cpu_idle_metadata_file,
            "tree_data_file": None,
            "priority": 0,
        },
    )

    # ============================================================================
    # Multi-Kernel System-Level Category Creation
    # ============================================================================
    multi_kernel_data_file = f"{output_dir}/category_data/multi_kernel_data.json"
    has_multi_kernel_events = False
    if os.path.exists(multi_kernel_data_file):
        with open(multi_kernel_data_file, "r") as f:
            mk_data = json.load(f)
        has_multi_kernel_events = (
            mk_data.get("memcpy_summary", {}).get("total_count", 0) > 0
            or mk_data.get("nccl_summary", {}).get("total_count", 0) > 0
        )

    if has_multi_kernel_events:
        print(f"\n  Category: Multi-Kernel Issues (multi_kernel)")
        print(f"    ℹ️  Multi-kernel data available (memcpy/NCCL events present)")

        # Create multi-kernel metadata
        multi_kernel_metadata = {
            "platform": platform,
            "peak_hbm_bw_tbs": platform_specs["mem_bw_gbps"] / 1000,
            "max_achievable_tflops": platform_specs["max_achievable_tflops"],
            "memory_gb": platform_specs["memory_gb"],
            "trace_path": trace_path,
            "output_dir": output_dir,
            "category": "Multi-Kernel Issues",
            "category_name": "multi_kernel",
            "gpu_utilization": gpu_utilization_metrics,
            "tier": "system",
        }

        multi_kernel_metadata_file = f"{output_dir}/metadata/multi_kernel_metadata.json"
        with open(multi_kernel_metadata_file, "w") as f:
            json.dump(multi_kernel_metadata, f, indent=2)

        print(f"    ✓ Exported metadata")

        # Add to categories list (system tier)
        exported_categories.append(
            {
                "name": "multi_kernel",
                "display_name": "Multi-Kernel Issues",
                "skill": "multi-kernel-analyzer",
                "tier": "system",
                "ops_count": 0,
                "csv_file": None,
                "metadata_file": multi_kernel_metadata_file,
                "data_file": multi_kernel_data_file,
                "tree_data_file": None,
            }
        )

    # ============================================================================
    # Kernel Fusion System-Level Category Creation (Experimental)
    # ============================================================================
    fusion_candidates_file = f"{output_dir}/category_data/fusion_candidates.json"
    if os.path.exists(fusion_candidates_file):
        with open(fusion_candidates_file, "r") as f:
            _fc = json.load(f)
        if _fc:
            print(f"\n  Category: Kernel Fusion Opportunities (kernel_fusion)")
            print(f"    ℹ️  {len(_fc)} fusion candidates available")
            exported_categories.append(
                {
                    "name": "kernel_fusion",
                    "display_name": "Kernel Fusion Opportunities",
                    "skill": "kernel-fusion-analyzer",
                    "tier": "system",
                    "ops_count": len(_fc),
                    "csv_file": None,
                    "metadata_file": None,
                    "data_file": fusion_candidates_file,
                    "tree_data_file": None,
                }
            )

    # ============================================================================
    # STEP 5.5: Calculate Time Metric Breakdown per Category
    # ============================================================================
    print("\n[STEP 5.5] Calculating Time Metric Breakdown per Category...")

    # Calculate GPU kernel time vs CPU duration per category
    # GPU kernel time = actual GPU execution (use for bottleneck prioritization)
    # CPU duration = total operation time including sync/launch overhead
    # Sync time = operations where CPU duration >> GPU kernel time

    for cat_info in exported_categories:
        category_name = cat_info["name"]
        if category_name in ("cpu_idle", "multi_kernel"):
            continue  # Skip system-level categories - no ops CSV

        category_df = unified_df[unified_df["enhanced_category"] == category_name]

        # Calculate GPU kernel time (ms)
        if "Kernel Time (µs)_sum" in category_df.columns:
            gpu_kernel_time_ms = category_df["Kernel Time (µs)_sum"].sum() / 1000
        elif "total_direct_kernel_time_ms" in category_df.columns:
            gpu_kernel_time_ms = category_df["total_direct_kernel_time_ms"].sum()
        else:
            gpu_kernel_time_ms = 0

        # Calculate CPU duration (ms) - total_duration_us if available
        if "total_duration_us" in category_df.columns:
            cpu_duration_ms = category_df["total_duration_us"].sum() / 1000
        elif "Duration (µs)_sum" in category_df.columns:
            cpu_duration_ms = category_df["Duration (µs)_sum"].sum() / 1000
        else:
            cpu_duration_ms = gpu_kernel_time_ms  # Fallback to kernel time

        # Calculate sync time (ops where CPU duration >> GPU kernel time)
        # Sync bottleneck = CPU duration - GPU kernel time when ratio > 5x
        sync_time_ms = 0
        sync_ops_count = 0

        for _, row in category_df.iterrows():
            if "Kernel Time (µs)_sum" in row and "total_duration_us" in row:
                kernel_us = row.get("Kernel Time (µs)_sum", 0) or 0
                duration_us = row.get("total_duration_us", 0) or 0
                if kernel_us > 0 and duration_us > kernel_us * 5:
                    sync_time_ms += (duration_us - kernel_us) / 1000
                    sync_ops_count += 1

        # Add time metrics to category info
        cat_info["gpu_kernel_time_ms"] = round(gpu_kernel_time_ms, 3)
        cat_info["cpu_duration_ms"] = round(cpu_duration_ms, 3)
        cat_info["sync_time_ms"] = round(sync_time_ms, 3)
        cat_info["sync_ops_count"] = sync_ops_count

        # Flag sync bottleneck if significant
        if sync_time_ms > 0.1 * gpu_kernel_time_ms and sync_time_ms > 1:
            cat_info["has_sync_bottleneck"] = True
            print(
                f"    ⚠️  {category_name}: Sync bottleneck detected ({sync_time_ms:.2f}ms sync time)"
            )
        else:
            cat_info["has_sync_bottleneck"] = False

        # Also update the metadata file with time breakdown
        metadata_file = cat_info.get("metadata_file")
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            metadata["time_breakdown"] = {
                "gpu_kernel_time_ms": cat_info["gpu_kernel_time_ms"],
                "cpu_duration_ms": cat_info["cpu_duration_ms"],
                "sync_time_ms": cat_info["sync_time_ms"],
                "sync_ops_count": cat_info["sync_ops_count"],
                "has_sync_bottleneck": cat_info["has_sync_bottleneck"],
                "note": "Use gpu_kernel_time_ms for bottleneck prioritization. sync_time_ms indicates host-device sync overhead.",
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

    print(f"  ✓ Time metrics calculated for all categories")

    # Save category manifest
    manifest = {
        "platform": platform,
        "trace_path": trace_path,
        "output_dir": output_dir,
        "comparison_scope": comparison_scope,
        "gpu_utilization": gpu_utilization_metrics,
        "categories": exported_categories,
        "time_metric_note": "Use gpu_kernel_time_ms for bottleneck prioritization. cpu_duration_ms includes sync/launch overhead.",
    }
    if comparison_scope == "comparative" and trace2_gpu_utilization is not None:
        manifest["trace2_gpu_utilization"] = trace2_gpu_utilization
        manifest["trace2_ops_summary_by_category"] = trace2_ops_summary_by_category

    manifest_file = f"{output_dir}/category_data/category_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    model_info_path = os.path.join(output_dir, "metadata", "model_info.json")
    default_info = {
        "model": "Cannot be inferred from trace",
        "architecture": "Cannot be inferred from trace",
        "scale": "Cannot be inferred from trace",
        "precision": "Cannot be inferred from trace",
    }
    if os.path.exists(model_info_path):
        with open(model_info_path) as f:
            existing = json.load(f)
        model_info = {**default_info, **existing}
    else:
        model_info = default_info
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Orchestrator Preparation Complete (Steps 2-5)")
    print(f"✓ Exported {len(exported_categories)} categories")
    print(f"✓ Manifest saved: {manifest_file}")
    if comparison_scope == "comparative":
        print(f"✓ metadata/trace2_gpu_utilization.json + manifest trace2_* fields")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
