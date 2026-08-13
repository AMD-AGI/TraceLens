###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the TraceLens Agent Analysis per-category analyzers.

Covers every module under category_analyses/ plus the shared kernel classifier:

- utils/classify_kernels: regex-driven kernel classifier.
- Per-category classifiers and extract_category_specific for convolution,
  cpu_idle, elementwise, gemm, moe, norm, other, reduce, sdpa, triton.
- kernel_fusion_analysis: pure, deterministic fusion helpers + filter/dedup.
- multi_kernel_analysis: severity classifiers + timeline cross-validation.
- sdpa_analysis: perf-param and kernel-breakdown parsing plus the flash /
  paged attention detectors.
- The per-module main() drivers, exercised end-to-end over tmp_path fixtures.
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(ANALYSIS_DIR, "category_analyses"))

from TraceLens.Agent.Analysis.utils.classify_kernels import (
    MIN_CONFIDENCE,
    classify_kernel,
    get_perf_category,
    run_assertions,
)
from TraceLens.Agent.Analysis.category_analyses.kernel_fusion_analysis import (
    MAX_FUSION_KERNEL_COUNT,
    MIN_IMPACT_SCORE,
    OVERLAP_EFFICIENCY,
    _classify_confidence,
    _comparative_estimate,
    _filter_and_dedup,
    _is_matrix_op,
    _is_norm_kernel,
    _roofline_savings_us,
    _split_into_subgroups,
    _standalone_estimate,
    compute_fusion_impact_estimates,
)
from TraceLens.Agent.Analysis.category_analyses.multi_kernel_analysis import (
    classify_memcpy_severity,
    classify_nccl_blocking_severity,
    classify_overlap_severity,
    cross_validate_with_timeline,
)
from TraceLens.Agent.Analysis.category_analyses.sdpa_analysis import (
    classify_sdpa_operation,
    detect_flash_attention,
    detect_paged_attention,
    extract_category_specific as sdpa_extract,
    parse_kernel_breakdown,
    parse_perf_params,
)
from TraceLens.Agent.Analysis.category_analyses.convolution_analysis import (
    extract_category_specific as conv_extract,
)
from TraceLens.Agent.Analysis.category_analyses.cpu_idle_analysis import (
    analyze_kernel_patterns,
    load_gpu_timeline,
    load_ops_summary,
)
from TraceLens.Agent.Analysis.category_analyses.elementwise_analysis import (
    extract_category_specific as elementwise_extract,
)
from TraceLens.Agent.Analysis.category_analyses.norm_analysis import (
    extract_category_specific as norm_extract,
)
from TraceLens.Agent.Analysis.category_analyses.moe_analysis import (
    _check_moe_data,
    extract_category_specific as moe_extract,
)
from TraceLens.Agent.Analysis.category_analyses.triton_analysis import (
    classify_triton_operation,
    extract_category_specific as triton_extract,
)
from TraceLens.Agent.Analysis.category_analyses.gemm_analysis import (
    classify_gemm_operation,
    detect_quantized_gemm,
    extract_category_specific as gemm_extract,
)
from TraceLens.Agent.Analysis.category_analyses.reduce_analysis import (
    detect_softmax,
    extract_category_specific as reduce_extract,
)
from TraceLens.Agent.Analysis.category_analyses.other_analysis import (
    _classify_other_op,
    classify_other_operation,
    extract_category_specific as other_extract,
)
from TraceLens.Agent.Analysis.category_analyses import (
    analysis_utils as au,
    convolution_analysis,
    cpu_idle_analysis,
    elementwise_analysis,
    gemm_analysis,
    kernel_fusion_analysis,
    moe_analysis,
    multi_kernel_analysis,
    norm_analysis,
    other_analysis,
    reduce_analysis,
    sdpa_analysis,
    triton_analysis,
)
from TraceLens.Agent.Analysis.utils import arch_utils
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import rmsnorm_perf_model_extensions as rms_ext
from TraceLens.Trace2Tree.extensions.moe_aiter_pseudo_ops import (
    _create_pseudo_op_moe_fused_aiter,
    is_aiter_fused_moe_kernel,
)
from TraceLens.Trace2Tree.extensions.moe_flydsl_pseudo_ops import (
    FUSED_MOE_PARENT,
    create_pseudo_ops_moe_flydsl,
)
from TraceLens.Trace2Tree.extensions.moe_gptq_awq_pseudo_ops import (
    _extract_topk_from_outplace,
    create_pseudo_ops_moe_gptq_awq,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from tests.test_conv_backward_bytes import (
    _conv_bias_bwd_event,
    _conv_bias_fwd_event,
    _conv_bias_relu_bwd_event,
    _conv_bias_relu_fwd_event,
)
from tests.test_dit_fused_ln_modulate import _fused_ln_fwd_event
from tests.test_evoformer_attention_ops import _event as _evoformer_event
from tests.test_trace2tree import _add_gpu_chain, _mk_event

# ----- classify_kernel: representative rule hits -----


@pytest.mark.parametrize(
    "name,expected_type,expected_cat",
    [
        ("memcpy_dtoh", "MemCpy", "MemCpy"),
        ("Cijk_Alik_Bljk", "GEMM", "GEMM"),
        ("rmsnorm2d", "RMSNorm", "Others"),
        ("rotary_embed", "Rotary Embedding", "Elementwise"),
        ("vectorized_elementwise_kernel FillFunctor", "Elementwise", "Elementwise"),
    ],
)
def test_classify_kernel_representative(name, expected_type, expected_cat):
    ktype, perf_cat, conf = classify_kernel(name)
    assert ktype == expected_type
    assert perf_cat == expected_cat
    assert conf >= MIN_CONFIDENCE


def test_classify_kernel_priority_tiebreak():
    """A name matching both a high-prio (Cijk_, 12) and low-prio (gemm, 5) rule
    resolves to the higher-priority kernel_type and reports that priority."""
    ktype, _, conf = classify_kernel("Cijk_gemm_kernel")
    assert ktype == "GEMM"
    assert conf == 12


def test_classify_kernel_no_match_is_unknown():
    assert classify_kernel("zzz_totally_unrecognized") == ("Unknown", "Others", 0)


def test_classify_kernel_low_priority_downgraded_to_unknown():
    """A generic catch-all rule (quant, prio 4 < MIN_CONFIDENCE) is not confident
    enough, so classify_kernel downgrades to Unknown/Others."""
    assert MIN_CONFIDENCE == 5
    ktype, perf_cat, conf = classify_kernel("myquantthing")
    assert ktype == "Unknown"
    assert perf_cat == "Others"
    assert conf == 0


# ----- get_perf_category -----


def test_get_perf_category_known():
    assert get_perf_category("Attention") == "SDPA"
    assert get_perf_category("GEMM") == "GEMM"
    assert get_perf_category("Unknown") == "Others"


def test_get_perf_category_unmapped_defaults_to_others():
    assert get_perf_category("NotARealType") == "Others"


# ----- run_assertions -----


def _classified(name, ktype, confidence, index=0):
    return {
        "index": index,
        "name": name,
        "kernel_type": ktype,
        "confidence": confidence,
    }


def test_run_assertions_clean():
    classified = [_classified("Cijk_x", "GEMM", 12, i) for i in range(5)]
    assert run_assertions(classified) == []


def test_run_assertions_flags_unknown():
    classified = [
        _classified("mystery", "Unknown", 0, 0),
        _classified("Cijk_x", "GEMM", 12, 1),
    ]
    errors = run_assertions(classified)
    assert any("Unknown" in e for e in errors)


def test_run_assertions_flags_low_confidence_majority():
    """> 20% of kernels with confidence <= 2 (and not Unknown) triggers a warning."""
    classified = [
        _classified("a", "Elementwise", 1, 0),
        _classified("b", "Elementwise", 2, 1),
        _classified("c", "GEMM", 12, 2),
    ]
    errors = run_assertions(classified)
    assert any("low classification confidence" in e for e in errors)


def test_run_assertions_low_confidence_below_threshold_no_warning():
    """A single low-confidence kernel (< 20% of total) does not warn."""
    classified = [_classified("a", "Elementwise", 1, 0)] + [
        _classified(f"g{i}", "GEMM", 12, i + 1) for i in range(9)
    ]
    errors = run_assertions(classified)
    assert not any("low classification confidence" in e for e in errors)


# ----- _classify_confidence -----


def test_classify_confidence_high_attention():
    """Name hint (attention) AND composition (2 GEMM + softmax) -> high."""
    candidate = {"base_name": "attention", "module_name": ""}
    enriched = [
        {"name": "gemm_qk", "type": "GEMM"},
        {"name": "softmax_kernel", "type": "Elementwise"},
        {"name": "gemm_pv", "type": "GEMM"},
    ]
    assert _classify_confidence(candidate, enriched) == "high"


def test_classify_confidence_high_norm():
    """Name hint (rmsnorm) AND composition (rsqrt) -> high."""
    candidate = {"base_name": "rmsnorm", "module_name": ""}
    enriched = [{"name": "rsqrt_kernel", "type": "Elementwise"}]
    assert _classify_confidence(candidate, enriched) == "high"


def test_classify_confidence_medium_composition_only_norm():
    """No name hint but rsqrt composition -> medium."""
    candidate = {"base_name": "unnamed_block", "module_name": ""}
    enriched = [{"name": "rsqrt_kernel", "type": "Elementwise"}]
    assert _classify_confidence(candidate, enriched) == "medium"


def test_classify_confidence_medium_composition_only_rope():
    """neg + catarray composition (no name hint) -> rope comp signal -> medium."""
    candidate = {"base_name": "unnamed_block", "module_name": ""}
    enriched = [
        {"name": "neg_kernel", "type": "Elementwise"},
        {"name": "catarraybatchedcopy", "type": "Elementwise"},
    ]
    assert _classify_confidence(candidate, enriched) == "medium"


def test_classify_confidence_medium_name_only():
    """Name hint (rope) but no confirming composition -> medium."""
    candidate = {"base_name": "rope", "module_name": ""}
    enriched = [{"name": "some_kernel", "type": "Elementwise"}]
    assert _classify_confidence(candidate, enriched) == "medium"


def test_classify_confidence_low():
    """Neither name nor composition signal -> low."""
    candidate = {"base_name": "opaque_block", "module_name": ""}
    enriched = [{"name": "some_kernel", "type": "Elementwise"}]
    assert _classify_confidence(candidate, enriched) == "low"


# ----- _split_into_subgroups -----


def test_split_gemm_epilogue():
    kernels = [{"type": "GEMM"}, {"type": "Elementwise"}]
    typed = _split_into_subgroups(kernels)
    assert len(typed) == 1
    assert typed[0][0] == "gemm_epilogue"


def test_split_gemm_only():
    typed = _split_into_subgroups([{"type": "GEMM"}])
    assert len(typed) == 1
    assert typed[0][0] == "gemm_only"


def test_split_elementwise_only():
    typed = _split_into_subgroups([{"type": "Elementwise"}, {"type": "Elementwise"}])
    assert len(typed) == 1
    assert typed[0][0] == "elementwise"


def test_split_multiple_gemm_boundaries():
    """E, GEMM, E, GEMM splits at each GEMM into three typed sub-groups."""
    kernels = [
        {"type": "Elementwise"},
        {"type": "GEMM"},
        {"type": "Elementwise"},
        {"type": "GEMM"},
    ]
    typed = _split_into_subgroups(kernels)
    assert [t for t, _ in typed] == ["elementwise", "gemm_epilogue", "gemm_only"]


# ----- _roofline_savings_us -----


def _ew(name, dur_us, data_in=None, data_out=None, data_moved=None, gflops=None):
    return {
        "name": name,
        "type": "Elementwise",
        "dur_us": dur_us,
        "data_moved_mb": data_moved,
        "data_in_mb": data_in,
        "data_out_mb": data_out,
        "gflops": gflops,
        "compute_spec": None,
        "has_perf_data": True,
    }


def test_roofline_savings_memory_only():
    """Single memory-bound kernel: fused == memory_time, savings = dur - memory_time."""
    enriched = [_ew("ew", dur_us=10.0, data_in=1.0, data_out=1.0, gflops=None)]
    # bytes = 2e6; peak = 1e12 B/s; frac=1.0 -> memory_time = 2.0 us; savings = 10 - 2
    savings = _roofline_savings_us(enriched, 1e12, 1.0, 1.0, 100.0)
    assert savings == pytest.approx(8.0)


def test_roofline_savings_no_modeled_returns_zero():
    enriched = [dict(_ew("ew", 10.0, 1.0, 1.0), has_perf_data=False)]
    assert _roofline_savings_us(enriched, 1e12, 1.0, 1.0, 100.0) == 0.0


def test_roofline_savings_zero_external_data_returns_zero():
    enriched = [_ew("ew", 10.0, data_in=0.0, data_out=0.0)]
    assert _roofline_savings_us(enriched, 1e12, 1.0, 1.0, 100.0) == 0.0


def test_roofline_savings_clamped_at_zero():
    """When current time is below fused time, savings clamp to 0."""
    enriched = [_ew("ew", dur_us=1.0, data_in=1.0, data_out=1.0)]
    # memory_time = 2.0 us > current 1.0 us -> max(0, 1 - 2) = 0
    assert _roofline_savings_us(enriched, 1e12, 1.0, 1.0, 100.0) == 0.0


def test_roofline_savings_unmodeled_subtracted():
    """Unmodeled kernel time is subtracted from savings, not fused."""
    enriched = [
        _ew("ew", dur_us=10.0, data_in=1.0, data_out=1.0),
        dict(_ew("unmodeled", dur_us=5.0), has_perf_data=False),
    ]
    # current = 15, memory_time = 2, unmodeled = 5 -> max(0, 15 - 2 - 5) = 8
    assert _roofline_savings_us(enriched, 1e12, 1.0, 1.0, 100.0) == pytest.approx(8.0)


def test_roofline_savings_overlap_blend():
    """Blends max (perfect overlap) and sum (no overlap) via OVERLAP_EFFICIENCY."""
    enriched = [
        _ew("ew1", dur_us=10.0, data_in=1.0, gflops=0.0005),
        _ew("ew2", dur_us=10.0, data_out=1.0, gflops=0.0005),
    ]
    # memory_time = 2.0; vector_time = (0.001e9)/(1e12)*1e6 = 1.0; matrix = 0
    # optimal = 2, sum = 3, fused = 2 + (1-0.85)*(3-2) = 2.15; current 20 -> 17.85
    savings = _roofline_savings_us(enriched, 1e12, 1.0, 1.0, 100.0)
    expected = 20.0 - (2.0 + (1.0 - OVERLAP_EFFICIENCY) * (3.0 - 2.0))
    assert savings == pytest.approx(expected)
    assert savings == pytest.approx(17.85)


# ----- _standalone_estimate -----


def _lookup(**entries):
    """Build a kernel_lookup keyed by name with a single shape=None entry each."""
    table = {}
    for name, (dm, gf, spec) in entries.items():
        table[name] = {
            None: {"Data Moved (MB)": dm, "GFLOPS": gf, "Compute Spec": spec}
        }
    return table


STD_ARGS = dict(peak_bw_bytes_s=1e12, vector_maf=1.0, matrix_maf=700.0)


def _standalone(candidate, lookup, min_savings_ms=0.01, baseline_ms=1.0):
    return _standalone_estimate(
        candidate,
        lookup,
        STD_ARGS["peak_bw_bytes_s"],
        STD_ARGS["vector_maf"],
        STD_ARGS["matrix_maf"],
        min_savings_ms,
        baseline_ms,
    )


def test_standalone_skip_too_few_kernels():
    candidate = {"base_name": "x", "kernels": [{"name": "k", "dur_us": 10}]}
    assert _standalone(candidate, {}) is None


def test_standalone_skip_has_fused_kernel():
    candidate = {
        "base_name": "x",
        "has_fused_kernel": True,
        "kernels": [{"name": "a", "dur_us": 10}, {"name": "b", "dur_us": 10}],
    }
    assert _standalone(candidate, {}) is None


def test_standalone_skip_no_perf_models():
    candidate = {
        "base_name": "x",
        "kernels": [{"name": "a", "dur_us": 10}, {"name": "b", "dur_us": 10}],
    }
    assert _standalone(candidate, {}) is None


def test_standalone_skip_all_matrix():
    candidate = {
        "base_name": "x",
        "kernels": [
            {"name": "g1", "type": "GEMM", "dur_us": 10},
            {"name": "g2", "type": "GEMM", "dur_us": 10},
        ],
    }
    lookup = _lookup(g1=(1.0, None, "matrix_bf16"), g2=(1.0, None, "matrix_bf16"))
    assert _standalone(candidate, lookup) is None


def test_standalone_skip_triton():
    candidate = {
        "base_name": "x",
        "kernels": [
            {"name": "triton_poi_fused_add", "type": "Elementwise", "dur_us": 10},
            {"name": "ew", "type": "Elementwise", "dur_us": 10},
        ],
    }
    lookup = _lookup(triton_poi_fused_add=(1.0, None, None), ew=(1.0, None, None))
    assert _standalone(candidate, lookup) is None


def test_standalone_skip_norm_only_epilogue():
    """GEMM + norm-only epilogue is not a fusion candidate."""
    candidate = {
        "base_name": "x",
        "kernels": [
            {"name": "Cijk_gemm", "type": "GEMM", "dur_us": 100},
            {"name": "rmsnorm_kernel", "type": "Elementwise", "dur_us": 100},
        ],
    }
    lookup = _lookup(
        Cijk_gemm=(1.0, None, "matrix_bf16"), rmsnorm_kernel=(1.0, None, None)
    )
    assert _standalone(candidate, lookup) is None


def test_standalone_skip_below_modeled_frac():
    """3 kernels, only 1 modeled -> modeled_frac 0.33 < 0.5 -> skip."""
    candidate = {
        "base_name": "x",
        "kernels": [
            {"name": "ew_modeled", "type": "Elementwise", "dur_us": 10},
            {"name": "unmodeled_a", "type": "Elementwise", "dur_us": 10},
            {"name": "unmodeled_b", "type": "Elementwise", "dur_us": 10},
        ],
    }
    lookup = _lookup(ew_modeled=(1.0, None, None))
    assert _standalone(candidate, lookup) is None


def _golden_candidate():
    return {
        "base_name": "mlp_block",
        "instance_count": 1,
        "kernels": [
            {"name": "Cijk_gemm", "type": "GEMM", "dur_us": 100},
            {"name": "ew_add", "type": "Elementwise", "dur_us": 100},
        ],
    }


def _golden_lookup():
    return _lookup(Cijk_gemm=(1.0, None, "matrix_bf16"), ew_add=(1.0, None, None))


def test_standalone_gemm_epilogue_full_estimate():
    """GEMM+elementwise: savings = elementwise dur; impact bands via TARGET_*."""
    est = _standalone(_golden_candidate(), _golden_lookup(), baseline_ms=1.0)
    assert est is not None
    # gap_high = 100/200 = 0.5; time_ms = 0.2; impact_high = 0.5*0.2/1*100 = 10.0
    assert est["impact_score_high"] == 10.0
    assert est["impact_score"] == 8.75
    assert est["impact_score_low"] == 7.5
    assert est["bound_type"] == "compute"
    assert est["estimation"] == "full"
    assert est["category"] == "kernel_fusion"
    assert est["type"] == "kernel_fusion"
    assert est["affected_gpu_kernels"] == ["Cijk_gemm", "ew_add"]


def test_standalone_skip_below_min_impact_score():
    """Same shape but a large baseline pushes impact_score_high below MIN_IMPACT_SCORE."""
    assert (
        _standalone(_golden_candidate(), _golden_lookup(), baseline_ms=1000.0) is None
    )


def test_standalone_skip_below_min_savings_ms():
    """min_savings_ms above the 0.1 ms projected savings drops the candidate."""
    assert (
        _standalone(_golden_candidate(), _golden_lookup(), min_savings_ms=1.0) is None
    )


# ----- _comparative_estimate -----


def test_comparative_estimate_measured_gap():
    candidate = {
        "base_name": "attn",
        "instance_count": 1,
        "kernels_trace1": [{"name": "k1", "type": "GEMM"}],
        "total_kernel_time_us_trace1": 10_000,
        "total_kernel_time_us_trace2": 5_000,
    }
    est = _comparative_estimate(candidate, min_savings_ms=0.1, baseline_ms=100.0)
    assert est is not None
    # gap = 5000 us -> savings_high = 5.0 ms; impact_high = 5/100*100 = 5.0
    assert est["impact_score_high"] == 5.0
    assert est["impact_score"] == 4.38  # (87.5/100)*5 = 4.375 -> 4.38
    assert est["impact_score_low"] == 3.75
    assert est["bound_type"] == "compute"
    assert est["estimation"] == "measured"
    assert est["type"] == "kernel_fusion"


def test_comparative_estimate_skip_no_gap():
    """trace2 not faster than trace1 -> no savings -> skip."""
    candidate = {
        "base_name": "attn",
        "kernels_trace1": [{"name": "k1", "type": "GEMM"}],
        "total_kernel_time_us_trace1": 5_000,
        "total_kernel_time_us_trace2": 6_000,
    }
    assert _comparative_estimate(candidate, 0.1, 100.0) is None


def test_comparative_estimate_skip_no_trace1_kernels():
    candidate = {"base_name": "attn", "kernels_trace1": []}
    assert _comparative_estimate(candidate, 0.1, 100.0) is None


# ----- compute_fusion_impact_estimates -----


def test_compute_fusion_invalid_baseline_returns_empty(capsys):
    out = compute_fusion_impact_estimates(
        [{}], {}, 5.3, {"matrix_bf16": 700}, baseline_ms=0
    )
    assert out == []
    assert "invalid" in capsys.readouterr().err


def test_compute_fusion_standalone_dispatch():
    out = compute_fusion_impact_estimates(
        [_golden_candidate()],
        _golden_lookup(),
        5.3,
        {"matrix_bf16": 700},
        baseline_ms=1.0,
    )
    assert len(out) == 1
    assert out[0]["impact_score_high"] == 10.0


def test_compute_fusion_comparative_dispatch_and_sort():
    """Comparative mode dispatches to _comparative_estimate; results sort by impact desc."""
    small = {
        "base_name": "small",
        "kernels_trace1": [{"name": "k", "type": "Elementwise"}],
        "total_kernel_time_us_trace1": 6_000,
        "total_kernel_time_us_trace2": 3_000,
    }
    large = {
        "base_name": "large",
        "kernels_trace1": [{"name": "k", "type": "GEMM"}],
        "total_kernel_time_us_trace1": 20_000,
        "total_kernel_time_us_trace2": 5_000,
    }
    out = compute_fusion_impact_estimates(
        [small, large],
        {},
        5.3,
        {"matrix_bf16": 700},
        baseline_ms=100.0,
        is_comparative=True,
    )
    assert len(out) == 2
    assert out[0]["operation"] == "large"
    assert out[0]["impact_score"] >= out[1]["impact_score"]


# ----- classify_memcpy_severity -----


def test_memcpy_no_timeline_guard():
    """total_time_ms <= 0 short-circuits to not-flagged."""
    r = classify_memcpy_severity({"total_time_us": 5000, "total_count": 3}, 0)
    assert r["flagged"] is False
    assert "No timeline" in r["details"]


def test_memcpy_flagged_by_percent():
    """D2H > 5% of total time is flagged."""
    summary = {
        "total_time_us": 100_000,
        "total_count": 5,
        "by_direction": {
            "D2H": {"count": 3, "total_time_us": 100_000, "avg_bytes": 1024},
        },
    }
    r = classify_memcpy_severity(summary, total_time_ms=1000.0)
    # D2H time = 100 ms of 1000 ms = 10% > 5% -> flagged
    assert r["flagged"] is True
    assert len(r["issues"]) == 1
    assert r["issues"][0]["direction"] == "D2H"


def test_memcpy_flagged_by_count():
    """H2D count > 10 flags even when the time percentage is tiny."""
    summary = {
        "total_time_us": 1_000,
        "total_count": 11,
        "by_direction": {
            "H2D": {"count": 11, "total_time_us": 1_000, "avg_bytes": 64},
        },
    }
    r = classify_memcpy_severity(summary, total_time_ms=1_000_000.0)
    assert r["flagged"] is True
    assert r["issues"][0]["direction"] == "H2D"


def test_memcpy_not_flagged_below_thresholds():
    summary = {
        "total_time_us": 1_000,
        "total_count": 2,
        "by_direction": {
            "D2H": {"count": 2, "total_time_us": 1_000, "avg_bytes": 64},
        },
    }
    r = classify_memcpy_severity(summary, total_time_ms=1_000.0)
    # 1 ms of 1000 ms = 0.1% and count 2 -> not flagged
    assert r["flagged"] is False
    assert r["issues"] == []


def test_memcpy_zero_count_direction_skipped():
    summary = {
        "total_time_us": 0,
        "total_count": 0,
        "by_direction": {"D2H": {"count": 0, "total_time_us": 0}},
    }
    r = classify_memcpy_severity(summary, total_time_ms=1_000.0)
    assert r["flagged"] is False


# ----- classify_nccl_blocking_severity -----


def test_nccl_blocking_no_comm_guard():
    r = classify_nccl_blocking_severity({"total_comm_time_us": 0})
    assert r["flagged"] is False
    assert "No communication" in r["details"]


def test_nccl_blocking_flagged_above_5pct():
    r = classify_nccl_blocking_severity(
        {
            "total_comm_time_us": 10_000,
            "exposed_comm_time_us": 8_000,
            "comm_percent_of_total": 8.0,
        }
    )
    assert r["flagged"] is True
    assert r["exposed_percent_of_total"] == 8.0


def test_nccl_blocking_not_flagged_at_5pct():
    r = classify_nccl_blocking_severity(
        {
            "total_comm_time_us": 10_000,
            "exposed_comm_time_us": 5_000,
            "comm_percent_of_total": 5.0,
        }
    )
    # strictly > 5 required
    assert r["flagged"] is False


# ----- classify_overlap_severity -----


def test_overlap_insufficient_data_guard():
    r = classify_overlap_severity({"comm_overlap_ratio": None, "total_comm_time_us": 0})
    assert r["flagged"] is False
    assert "Insufficient" in r["details"]


def test_overlap_low_comm_time_guard():
    """< 100 us total comm is treated as insufficient data."""
    r = classify_overlap_severity({"comm_overlap_ratio": 0.1, "total_comm_time_us": 50})
    assert r["flagged"] is False


def test_overlap_flagged_below_70pct():
    r = classify_overlap_severity(
        {"comm_overlap_ratio": 0.5, "total_comm_time_us": 10_000}
    )
    assert r["flagged"] is True
    assert r["overlap_percent"] == 50.0
    assert r["target_percent"] == 70


def test_overlap_not_flagged_at_70pct():
    r = classify_overlap_severity(
        {"comm_overlap_ratio": 0.7, "total_comm_time_us": 10_000}
    )
    assert r["flagged"] is False


# ----- cross_validate_with_timeline -----


def _timeline(rows):
    return pd.DataFrame(rows)


def test_cross_validate_pass_within_tolerance():
    overlap = {
        "computation_time_us": 100_000,
        "exposed_comm_time_us": 5_000,
        "exposed_memcpy_time_us": 1_000,
        "total_time_us": 200_000,
    }
    tl = _timeline(
        [
            {"type": "computation_time", "time ms": 100.0, "percent": 50.0},
            {"type": "exposed_comm_time", "time ms": 5.0, "percent": 2.5},
            {"type": "exposed_memcpy_time", "time ms": 1.0, "percent": 0.5},
            {"type": "total_time", "time ms": 200.0, "percent": 100.0},
        ]
    )
    v = cross_validate_with_timeline(overlap, tl, tolerance_pct=2.0)
    assert v["status"] == "PASS"
    assert all(c["status"] == "PASS" for c in v["checks"])
    assert v["warnings"] == []


def test_cross_validate_warn_beyond_tolerance():
    overlap = {"total_time_us": 200_000}
    tl = _timeline([{"type": "total_time", "time ms": 100.0, "percent": 100.0}])
    v = cross_validate_with_timeline(overlap, tl, tolerance_pct=2.0)
    # mk=200 ms vs tl=100 ms -> 50% diff > 2% -> WARN
    assert v["status"] == "WARN"
    total_check = next(c for c in v["checks"] if c["metric"] == "Total GPU time")
    assert total_check["status"] == "WARN"
    assert len(v["warnings"]) == 1


def test_cross_validate_missing_type_skips():
    overlap = {"computation_time_us": 100_000}
    tl = _timeline([{"type": "total_time", "time ms": 100.0, "percent": 100.0}])
    v = cross_validate_with_timeline(overlap, tl)
    comp_check = next(c for c in v["checks"] if c["metric"] == "Computation time")
    assert comp_check["status"] == "SKIP"


def test_cross_validate_both_zero_pass():
    overlap = {"exposed_comm_time_us": 0}
    tl = _timeline([{"type": "exposed_comm_time", "time ms": 0, "percent": 0}])
    v = cross_validate_with_timeline(overlap, tl)
    comm_check = next(
        c for c in v["checks"] if c["metric"] == "Exposed communication time"
    )
    assert comm_check["status"] == "PASS"
    assert comm_check["diff_ms"] == 0


# ----- Unit tests: sdpa_analysis.parse_perf_params -----


@pytest.mark.parametrize(
    "ctx,gen,expected",
    [
        (900, 100, "prefill_heavy"),  # ctx_ratio 0.9 > 0.8
        (500, 500, "mixed"),  # ctx_ratio 0.5 within [0.2, 0.8]
        (100, 900, "decode_heavy"),  # ctx_ratio 0.1 < 0.2
    ],
)
def test_parse_perf_params_workload_type(ctx, gen, expected):
    params = {"sum_ctx_tokens": ctx, "sum_gen_tokens": gen}
    r = parse_perf_params(str(params))
    assert r["workload_type"] == expected


def test_parse_perf_params_workload_type_boundaries():
    """Boundary values 0.8 and 0.2 fall into the 'mixed' band (strict >/<)."""
    at_high = parse_perf_params(str({"sum_ctx_tokens": 800, "sum_gen_tokens": 200}))
    assert at_high["ctx_ratio"] == 0.8
    assert at_high["workload_type"] == "mixed"
    at_low = parse_perf_params(str({"sum_ctx_tokens": 200, "sum_gen_tokens": 800}))
    assert at_low["ctx_ratio"] == 0.2
    assert at_low["workload_type"] == "mixed"


def test_parse_perf_params_no_tokens_unknown():
    r = parse_perf_params(str({"B": 2}))
    assert r["workload_type"] == "unknown"
    assert r["ctx_ratio"] is None


def test_parse_perf_params_gqa():
    r = parse_perf_params(str({"H_Q": 8, "H_KV": 2}))
    assert r["attention_pattern"] == "GQA"
    assert r["gqa_ratio"] == 4


def test_parse_perf_params_mha():
    r = parse_perf_params(str({"H_Q": 8, "H_KV": 8}))
    assert r["attention_pattern"] == "MHA"
    assert "gqa_ratio" not in r


def test_parse_perf_params_empty_returns_defaults():
    r = parse_perf_params("")
    assert r["workload_type"] == "unknown"
    assert r["batch_size"] is None


def test_parse_perf_params_malformed_returns_defaults():
    r = parse_perf_params("not a dict literal {")
    assert r["workload_type"] == "unknown"


# ----- Unit tests: sdpa_analysis.parse_kernel_breakdown -----


def test_parse_kernel_breakdown_regex_and_percent():
    kd = str(
        [
            {"name": "paged_attention_2d", "mean_duration_us": 10.0},
            {"name": "reshape_and_cache", "mean_duration_us": 30.0},
        ]
    )
    r = parse_kernel_breakdown(kd)
    assert r["total_kernel_time_us"] == 40.0
    assert r["has_paged_attention"] is True
    assert r["has_reshape_cache"] is True
    pcts = {k["name"]: k["percent"] for k in r["kernels"]}
    assert pcts["paged_attention_2d"] == 25.0
    assert pcts["reshape_and_cache"] == 75.0


def test_parse_kernel_breakdown_fwd_kernel_type():
    kd = str([{"name": "attn_fwd_kernel", "mean_duration_us": 5.0}])
    r = parse_kernel_breakdown(kd)
    assert r["has_fwd_kernel"] is True
    assert r["kernels"][0]["kernel_type"] == "fwd_kernel"
    assert r["kernels"][0]["percent"] == 100.0


def test_parse_kernel_breakdown_strips_np_float64():
    kd = "[{'name': 'k', 'mean_duration_us': np.float64(12.5)}]"
    r = parse_kernel_breakdown(kd)
    assert r["total_kernel_time_us"] == 12.5


def test_parse_kernel_breakdown_empty():
    r = parse_kernel_breakdown("")
    assert r["kernels"] == []
    assert r["total_kernel_time_us"] == 0


# ----- Unit tests: sdpa_analysis detect_* helpers -----


def test_detect_flash_attention():
    assert detect_flash_attention("flash_attn::_flash_attn_forward") is True
    assert detect_flash_attention("aten::mm") is False


def test_detect_paged_attention():
    assert detect_paged_attention("paged_attention_2d") is True
    assert detect_paged_attention("aten::mm") is False


# ----- shared metadata fixture for get_peak_specs consumers -----


def _metadata(peak_hbm_bw_tbs=5.3, matrix_bf16=1300.0):
    """Metadata dict as produced by the orchestrator; get_peak_specs reads it."""
    return {
        "max_achievable_tflops": {"matrix_bf16": matrix_bf16},
        "peak_hbm_bw_tbs": peak_hbm_bw_tbs,
    }


# ----- convolution_analysis.extract_category_specific -----


def test_conv_extract_with_transpose():
    df = pd.DataFrame(
        {
            "name": ["conv2d", "aten::transpose", "TransPose_kernel"],
            "Kernel Time (µs)_sum": [800.0, 100.0, 100.0],
        }
    )
    r = conv_extract(df, _metadata())
    assert r["transpose_count"] == 2
    assert r["transpose_time_ms"] == pytest.approx(0.2)
    # 200 us of 1000 us total = 20%
    assert r["transpose_overhead_percent"] == pytest.approx(20.0)
    assert r["peak_hbm_bw_tbs"] == 5.3
    assert r["peak_maf_tflops"] == 1300.0


def test_conv_extract_no_transpose():
    df = pd.DataFrame(
        {"name": ["conv2d", "relu"], "Kernel Time (µs)_sum": [500.0, 500.0]}
    )
    r = conv_extract(df, _metadata())
    assert r["transpose_count"] == 0
    assert r["transpose_time_ms"] == 0
    assert r["transpose_overhead_percent"] == 0


def test_conv_extract_missing_kernel_time_column():
    """No 'Kernel Time (µs)_sum' column -> total 0 -> zero-div guard, 0% overhead."""
    df = pd.DataFrame({"name": ["conv2d", "relu"]})
    r = conv_extract(df, _metadata())
    assert r["transpose_count"] == 0
    assert r["transpose_overhead_percent"] == 0


# ----- cpu_idle_analysis.analyze_kernel_patterns -----


def test_analyze_kernel_patterns_none():
    r = analyze_kernel_patterns(None)
    assert r == {
        "short_kernel_count": 0,
        "total_kernel_count": 0,
        "avg_kernel_time_us": 0,
        "kernel_count_by_category": {},
    }


def test_analyze_kernel_patterns_empty_df():
    r = analyze_kernel_patterns(pd.DataFrame())
    assert r["total_kernel_count"] == 0
    assert r["kernel_count_by_category"] == {}


def test_analyze_kernel_patterns_sum_and_count_no_mean():
    df = pd.DataFrame({"Kernel Time (µs)_sum": [100.0, 300.0], "Count": [2, 2]})
    r = analyze_kernel_patterns(df)
    assert r["total_kernel_count"] == 4
    assert r["avg_kernel_time_us"] == pytest.approx(100.0)
    # no mean column -> short count stays zero
    assert r["short_kernel_count"] == 0


def test_analyze_kernel_patterns_with_mean_short_kernels():
    df = pd.DataFrame(
        {
            "Kernel Time (µs)_sum": [10.0, 500.0],
            "Count": [5, 1],
            "Kernel Time (µs)_mean": [2.0, 500.0],
        }
    )
    r = analyze_kernel_patterns(df)
    assert r["total_kernel_count"] == 6
    # only the 2.0us-mean row (<10) counts its 5 kernels as short
    assert r["short_kernel_count"] == 5


def test_analyze_kernel_patterns_by_category():
    df = pd.DataFrame(
        {
            "Kernel Time (µs)_sum": [10.0, 20.0, 30.0],
            "Count": [1, 2, 3],
            "op category": ["GEMM", "GEMM", "Elementwise"],
        }
    )
    r = analyze_kernel_patterns(df)
    assert r["kernel_count_by_category"] == {"GEMM": 3, "Elementwise": 3}


# ----- cpu_idle_analysis load helpers -----


def test_load_ops_summary_missing_returns_none(tmp_path):
    assert load_ops_summary(str(tmp_path)) is None


def test_load_ops_summary_reads_csv(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    pd.DataFrame({"Count": [1, 2]}).to_csv(csv_dir / "ops_summary.csv", index=False)
    df = load_ops_summary(str(tmp_path))
    assert list(df["Count"]) == [1, 2]


def test_load_gpu_timeline_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gpu_timeline(str(tmp_path))


def test_load_gpu_timeline_reads_rows(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "type": ["idle_time", "total_time"],
            "time ms": [5.0, 100.0],
            "percent": [5.0, 100.0],
        }
    ).to_csv(csv_dir / "gpu_timeline.csv", index=False)
    tl = load_gpu_timeline(str(tmp_path))
    assert tl["idle_time"]["percent"] == 5.0
    assert tl["total_time"]["time_ms"] == 100.0


# ----- elementwise / norm one-liner extractors -----


def test_elementwise_extract():
    assert elementwise_extract(pd.DataFrame(), _metadata())["peak_hbm_bw_tbs"] == 5.3


def test_norm_extract():
    assert (
        norm_extract(pd.DataFrame(), _metadata(peak_hbm_bw_tbs=8.0))["peak_hbm_bw_tbs"]
        == 8.0
    )


# ----- moe_analysis -----


def test_moe_extract_delegates_to_peak_specs():
    r = moe_extract(pd.DataFrame(), _metadata())
    assert r["peak_hbm_bw_tbs"] == 5.3
    assert r["peak_maf_tflops"] == 1300.0


def test_check_moe_data_absent_returns_no_data(tmp_path):
    r = _check_moe_data(str(tmp_path), "moe_fused", "standalone")
    assert r["status"] == "NO_DATA"
    assert r["operation_count"] == 0
    assert r["category"] == "moe_fused"


def test_check_moe_data_present_returns_none(tmp_path):
    cat_dir = tmp_path / "category_data"
    cat_dir.mkdir()
    (cat_dir / "moe_fused_ops.csv").write_text("name\n")
    assert _check_moe_data(str(tmp_path), "moe_fused", "standalone") is None


# ----- triton_analysis -----


@pytest.mark.parametrize(
    "name,expected",
    [
        ("triton_poi_fused_add_0", "pointwise"),
        ("triton_red_sum_1", "reduction"),
        ("triton_per_fused_2", "persistent"),
        ("some_other_kernel", "other"),
        ("TRITON_POI_UPPER", "pointwise"),
    ],
)
def test_classify_triton_operation(name, expected):
    assert classify_triton_operation(name, None)["kernel_type"] == expected


def test_triton_extract_tallies_and_merges_specs():
    df = pd.DataFrame(
        {
            "name": [
                "triton_poi_a",
                "triton_poi_b",
                "triton_red_c",
                "triton_per_d",
                "misc_e",
            ]
        }
    )
    r = triton_extract(df, _metadata())
    assert r["pointwise_count"] == 2
    assert r["reduction_count"] == 1
    assert r["persistent_count"] == 1
    assert r["other_count"] == 1
    assert r["peak_hbm_bw_tbs"] == 5.3


# ----- gemm_analysis -----


@pytest.mark.parametrize(
    "name,quantized",
    [
        ("gemm_w8a8", True),
        ("fp8_matmul", True),
        ("mxfp4_gemm", True),
        ("Cijk_regular", False),
    ],
)
def test_detect_quantized_gemm(name, quantized):
    assert detect_quantized_gemm(name) is quantized


def test_classify_gemm_operation():
    q = classify_gemm_operation("gemm_int8", None)
    assert q == {"is_quantized": True, "gemm_type": "quantized"}
    r = classify_gemm_operation("Cijk_plain", None)
    assert r == {"is_quantized": False, "gemm_type": "regular"}


def test_gemm_extract_counts_quant_and_missing_perf():
    df = pd.DataFrame(
        {
            "name": ["gemm_fp8", "Cijk_plain", "gemm_w4a16"],
            "TFLOPS/s_mean": [100.0, float("nan"), float("nan")],
        }
    )
    r = gemm_extract(df, _metadata())
    assert r["quantized_count"] == 2
    assert r["missing_perf_model_count"] == 2


def test_gemm_extract_missing_tflops_column():
    df = pd.DataFrame({"name": ["Cijk_plain"]})
    r = gemm_extract(df, _metadata())
    assert r["quantized_count"] == 0
    assert r["missing_perf_model_count"] == 0


# ----- reduce_analysis -----


def test_detect_softmax():
    assert detect_softmax("aten::_softmax") is True
    assert detect_softmax("aten::sum") is False


def test_reduce_extract_softmax_count():
    df = pd.DataFrame({"name": ["softmax_kernel", "aten::sum", "log_softmax"]})
    r = reduce_extract(df, _metadata())
    assert r["softmax_count"] == 2
    assert r["peak_hbm_bw_tbs"] == 5.3


def test_reduce_extract_empty():
    r = reduce_extract(pd.DataFrame({"name": []}), _metadata())
    assert r["softmax_count"] == 0


# ----- other_analysis -----


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ncclKernel_AllReduce", "communication"),
        ("rccl_broadcast", "communication"),
        ("hipGraphLaunch", "graph"),
        ("aten::some_misc_op", "miscellaneous"),
    ],
)
def test_classify_other_operation(name, expected):
    assert classify_other_operation(name) == expected


def test_classify_other_op_callback():
    assert _classify_other_op("hipgraph_exec", None) == {"sub_category": "graph"}


def test_other_extract_counts_graph_and_misc():
    df = pd.DataFrame({"name": ["cudagraph_launch", "aten::foo", "aten::bar"]})
    r = other_extract(df, _metadata())
    assert r["graph_count"] == 1
    assert r["miscellaneous_count"] == 2
    assert r["communication_count"] == 0
    assert "communication_ops_skipped" not in r


def test_other_extract_with_skipped_comm_ops():
    df = pd.DataFrame({"name": ["aten::foo"]})
    skipped = {"count": 3, "op_names": ["a", "b", "c"], "message": "x"}
    r = other_extract(df, _metadata(), skipped_comm_ops=skipped)
    assert r["communication_ops_skipped"] == skipped


def test_other_extract_empty_skipped_comm_ops_omitted():
    df = pd.DataFrame({"name": ["aten::foo"]})
    r = other_extract(df, _metadata(), skipped_comm_ops={"count": 0})
    assert "communication_ops_skipped" not in r


# ----- sdpa_analysis.classify_sdpa_operation -----


def test_classify_sdpa_flash():
    row = pd.Series({"has_perf_model": True})
    r = classify_sdpa_operation("flash_attn::fwd", row)
    assert r["is_flash_attention"] is True
    assert r["is_paged_attention"] is False
    assert r["attention_type"] == "flash"
    assert r["has_perf_model"] is True


def test_classify_sdpa_standard():
    r = classify_sdpa_operation(
        "aten::scaled_dot_product_attention", pd.Series(dtype=object)
    )
    assert r["attention_type"] == "standard"
    assert r["has_perf_model"] is False


def test_classify_sdpa_paged_with_kernel_breakdown():
    kd = str(
        [
            {"name": "paged_attention_2d", "mean_duration_us": 30.0},
            {"name": "reshape_and_cache", "mean_duration_us": 10.0},
        ]
    )
    row = pd.Series({"kernel_details_summary": kd})
    r = classify_sdpa_operation("paged_attention_2d", row)
    assert r["is_paged_attention"] is True
    assert r["attention_type"] == "paged"
    assert r["kernel_breakdown"]["has_paged_attention_kernel"] is True
    assert r["kernel_breakdown"]["has_reshape_cache"] is True


def test_classify_sdpa_with_perf_params_gqa():
    row = pd.Series({"perf_params": str({"H_Q": 8, "H_KV": 2})})
    r = classify_sdpa_operation("flash_attn::fwd", row)
    assert r["workload_profile"]["attention_pattern"] == "GQA"
    assert r["workload_profile"]["gqa_ratio"] == 4


# ----- sdpa_analysis.extract_category_specific -----


def test_sdpa_extract_all_flash():
    df = pd.DataFrame(
        {"name": ["flash_attn::fwd", "flash_attn::bwd"], "has_perf_model": [True, True]}
    )
    r = sdpa_extract(df, _metadata())
    assert r["flash_attention_count"] == 2
    assert r["paged_attention_count"] == 0
    assert r["flash_attention_detected"] is True
    assert r["paged_attention_detected"] is False
    assert r["has_perf_model_count"] == 2
    assert "workload_profile" not in r


def test_sdpa_extract_paged_breakdown_and_workload():
    kd = str(
        [
            {"name": "paged_attention_2d", "mean_duration_us": 80.0},
            {"name": "reshape_and_cache", "mean_duration_us": 20.0},
        ]
    )
    df = pd.DataFrame(
        {
            "name": ["paged_attention_2d"],
            "kernel_details_summary": [kd],
            "perf_params": [str({"sum_ctx_tokens": 100, "sum_gen_tokens": 900})],
        }
    )
    r = sdpa_extract(df, _metadata())
    assert r["paged_attention_count"] == 1
    assert r["kernel_breakdown_avg"]["avg_paged_attention_percent"] == pytest.approx(
        80.0
    )
    assert r["kernel_breakdown_avg"]["avg_reshape_cache_percent"] == pytest.approx(20.0)
    # ctx_ratio 0.1 < 0.2 -> decode_heavy
    assert r["workload_profile"]["profile_type"] == "decode_heavy"
    assert r["workload_profile"]["total_gen_tokens"] == 900


def test_sdpa_extract_mixed_workload():
    df = pd.DataFrame(
        {
            "name": ["flash_attn::fwd"],
            "perf_params": [str({"sum_ctx_tokens": 500, "sum_gen_tokens": 500})],
        }
    )
    r = sdpa_extract(df, _metadata())
    assert r["workload_profile"]["profile_type"] == "mixed"


def test_sdpa_extract_empty():
    r = sdpa_extract(pd.DataFrame({"name": []}), _metadata())
    assert r["flash_attention_count"] == 0
    assert r["paged_attention_count"] == 0
    assert "workload_profile" not in r


# ----- kernel_fusion_analysis._is_norm_kernel / _is_matrix_op -----


@pytest.mark.parametrize(
    "info,expected",
    [
        ({"type": "LayerNorm"}, True),
        ({"type": "Elementwise", "name": "rmsnorm_kernel"}, True),
        ({"kernel_name": "miopenBatchNorm"}, True),
        ({"type": "Elementwise", "name": "add_kernel"}, False),
        ({}, False),
    ],
)
def test_is_norm_kernel(info, expected):
    assert _is_norm_kernel(info) is expected


@pytest.mark.parametrize(
    "info,expected",
    [
        ({"compute_spec": "matrix_bf16"}, True),
        ({"type": "GEMM"}, True),
        ({"type": "conv2d"}, True),
        ({"compute_spec": "vector_fp32"}, False),
        ({"type": "Elementwise"}, False),
    ],
)
def test_is_matrix_op(info, expected):
    assert _is_matrix_op(info) is expected


# ----- kernel_fusion_analysis._filter_and_dedup -----


def _cand(name, insts, kcount, time_us, comparative=False):
    """Build a fusion candidate for either standalone or comparative keys."""
    if comparative:
        return {
            "module_name": name,
            "instance_count": insts,
            "kernel_count_trace1": kcount,
            "total_kernel_time_us_trace1": time_us,
        }
    return {
        "module_name": name,
        "instance_count": insts,
        "kernel_count": kcount,
        "total_kernel_time_us": time_us,
    }


def test_filter_drops_over_kernel_cap():
    big = _cand("big", 1, MAX_FUSION_KERNEL_COUNT + 1, 1000.0)
    ok = _cand("ok", 1, 3, 1000.0)
    out = _filter_and_dedup([big, ok], baseline_ms=0)
    assert [c["module_name"] for c in out] == ["ok"]


def test_filter_drops_below_baseline_floor():
    """baseline_ms>0 sets a min total time of baseline*10*MIN_IMPACT_SCORE us."""
    baseline_ms = 1.0
    floor = baseline_ms * 10 * MIN_IMPACT_SCORE  # 20 us
    tiny = _cand("tiny", 1, 2, floor - 1)
    keep = _cand("keep", 1, 2, floor + 100)
    out = _filter_and_dedup([tiny, keep], baseline_ms=baseline_ms)
    assert [c["module_name"] for c in out] == ["keep"]


def test_filter_dedup_keeps_shorter_name():
    a = _cand("a_short", 2, 3, 100.04)
    b = _cand("a_much_longer_name", 2, 3, 100.04)
    out = _filter_and_dedup([b, a], baseline_ms=0)
    assert len(out) == 1
    assert out[0]["module_name"] == "a_short"


def test_filter_dedup_distinct_signatures_kept():
    a = _cand("a", 1, 3, 100.0)
    b = _cand("b", 2, 3, 100.0)
    out = _filter_and_dedup([a, b], baseline_ms=0)
    assert len(out) == 2


def test_filter_comparative_uses_trace1_keys():
    big = _cand("big", 1, MAX_FUSION_KERNEL_COUNT + 1, 5000.0, comparative=True)
    ok = _cand("ok", 1, 3, 5000.0, comparative=True)
    out = _filter_and_dedup([big, ok], baseline_ms=0, is_comparative=True)
    assert [c["module_name"] for c in out] == ["ok"]


def test_filter_comparative_dedup():
    a = _cand("short", 1, 2, 200.0, comparative=True)
    b = _cand("longer_name", 1, 2, 200.0, comparative=True)
    out = _filter_and_dedup([a, b], baseline_ms=0, is_comparative=True)
    assert len(out) == 1
    assert out[0]["module_name"] == "short"


# ----- driver main() end-to-end smoke tests -----
#
# These exercise each module's argparse main() over a minimal fixture tree so the
# run_category_analysis wiring (load -> build -> extract -> write) is covered.


def _write_category_inputs(base, category, ops_df):
    """Lay out the {category}_ops.csv + {category}_metadata.json that
    load_category_data expects under an output dir."""
    cat_dir = os.path.join(base, "category_data")
    meta_dir = os.path.join(base, "metadata")
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    ops_df.to_csv(os.path.join(cat_dir, f"{category}_ops.csv"), index=False)
    metadata = {
        "max_achievable_tflops": {"matrix_bf16": 1300.0},
        "peak_hbm_bw_tbs": 5.3,
        "gpu_utilization": {"total_time_ms": 100.0},
    }
    with open(os.path.join(meta_dir, f"{category}_metadata.json"), "w") as f:
        json.dump(metadata, f)


def _read_metrics(base, category):
    with open(os.path.join(base, "category_data", f"{category}_metrics.json")) as f:
        return json.load(f)


def _simple_ops():
    return pd.DataFrame(
        {
            "name": ["op_a", "op_b"],
            "count": [1, 2],
            "Kernel Time (µs)_sum": [500.0, 300.0],
        }
    )


@pytest.mark.parametrize(
    "mod,category",
    [
        (convolution_analysis, "convolution"),
        (elementwise_analysis, "elementwise"),
        (norm_analysis, "norm"),
        (reduce_analysis, "reduce"),
        (triton_analysis, "triton"),
        (gemm_analysis, "gemm"),
        (sdpa_analysis, "sdpa_fwd"),
    ],
)
def test_driver_main_writes_ok_metrics(mod, category, tmp_path, monkeypatch):
    base = str(tmp_path)
    _write_category_inputs(base, category, _simple_ops())
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    mod.main()
    metrics = _read_metrics(base, category)
    assert metrics["status"] == "OK"
    assert metrics["category"] == category
    assert metrics["operation_count"] == 2


def test_driver_other_main_writes_ok(tmp_path, monkeypatch):
    base = str(tmp_path)
    df = pd.DataFrame(
        {
            "name": ["ncclKernel_x", "aten::misc", "hipgraph_launch"],
            "count": [1, 1, 1],
            "Kernel Time (µs)_sum": [100.0, 200.0, 300.0],
        }
    )
    _write_category_inputs(base, "other", df)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    other_analysis.main()
    metrics = _read_metrics(base, "other")
    assert metrics["status"] == "OK"
    # the nccl op is stripped into skipped_comm_ops
    assert metrics["category_specific"]["communication_ops_skipped"]["count"] == 1


def test_driver_other_main_missing_csv_errors(tmp_path, monkeypatch):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    with pytest.raises(SystemExit):
        other_analysis.main()
    metrics = _read_metrics(base, "other")
    assert metrics["status"] == "ERROR"


def test_driver_moe_main_no_data(tmp_path, monkeypatch):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    moe_analysis.main()
    metrics = _read_metrics(base, "moe_fused")
    assert metrics["status"] == "NO_DATA"


def test_driver_moe_main_with_data(tmp_path, monkeypatch):
    base = str(tmp_path)
    _write_category_inputs(base, "moe_fused", _simple_ops())
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    moe_analysis.main()
    metrics = _read_metrics(base, "moe_fused")
    assert metrics["status"] == "OK"


def test_driver_cpu_idle_main(tmp_path, monkeypatch):
    base = str(tmp_path)
    csv_dir = os.path.join(base, "perf_report_csvs")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    pd.DataFrame(
        {
            "type": ["idle_time", "computation_time", "total_time"],
            "time ms": [20.0, 80.0, 100.0],
            "percent": [20.0, 80.0, 100.0],
        }
    ).to_csv(os.path.join(csv_dir, "gpu_timeline.csv"), index=False)
    pd.DataFrame({"Kernel Time (µs)_sum": [50.0], "Count": [5]}).to_csv(
        os.path.join(csv_dir, "ops_summary.csv"), index=False
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    cpu_idle_analysis.main()
    metrics = _read_metrics(base, "cpu_idle")
    assert metrics["status"] == "OK"
    # idle 20% > 15% threshold
    assert metrics["idle_flagged"] is True


# ----- kernel_fusion_analysis.main() end-to-end -----


def _write_fusion_inputs(base, candidates):
    cat_dir = os.path.join(base, "category_data")
    meta_dir = os.path.join(base, "metadata")
    csv_dir = os.path.join(base, "perf_report_csvs")
    for d in (cat_dir, meta_dir, csv_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(cat_dir, "fusion_candidates.json"), "w") as f:
        json.dump(candidates, f)
    with open(os.path.join(cat_dir, "category_manifest.json"), "w") as f:
        json.dump(
            {"platform": "MI300X", "gpu_utilization": {"total_time_ms": 100.0}}, f
        )
    with open(os.path.join(meta_dir, "gemm_metadata.json"), "w") as f:
        json.dump(
            {"peak_hbm_bw_tbs": 5.3, "max_achievable_tflops": {"matrix_bf16": 1300.0}},
            f,
        )
    pd.DataFrame(
        {
            "kernel_details_summary": [
                "[{'name': 'Cijk_gemm', 'mean_duration_us': 100.0}]"
            ],
            "Input Dims": ["[[1, 2]]"],
            "Data Moved (MB)": [1.0],
            "GFLOPS": [None],
            "FLOPS/Byte": [None],
            "Compute Spec": ["matrix_bf16"],
        }
    ).to_csv(os.path.join(csv_dir, "unified_perf_summary.csv"), index=False)


def test_driver_fusion_main_missing_files_errors(tmp_path, monkeypatch):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    with pytest.raises(SystemExit):
        kernel_fusion_analysis.main()
    metrics = _read_metrics(base, "kernel_fusion")
    assert metrics["status"] == "ERROR"


def test_driver_fusion_main_no_candidates(tmp_path, monkeypatch):
    base = str(tmp_path)
    _write_fusion_inputs(base, [])
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    kernel_fusion_analysis.main()
    metrics = _read_metrics(base, "kernel_fusion")
    assert metrics["status"] == "NO_DATA"
    assert metrics["candidate_count"] == 0


def test_driver_fusion_main_ok_with_estimate(tmp_path, monkeypatch):
    base = str(tmp_path)
    candidate = {
        "module_name": "mlp_block",
        "base_name": "mlp_block",
        "instance_count": 1,
        "kernel_count": 2,
        "total_kernel_time_us": 3000.0,
        "kernels": [
            {"name": "Cijk_gemm", "type": "GEMM", "dur_us": 1500},
            {"name": "ew_add", "type": "Elementwise", "dur_us": 1500},
        ],
    }
    _write_fusion_inputs(base, [candidate])
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    kernel_fusion_analysis.main()
    metrics = _read_metrics(base, "kernel_fusion")
    assert metrics["status"] == "OK"
    assert metrics["candidate_count"] == 1
    assert metrics["platform"] == "MI300X"
    assert "high_confidence_kernel_map" in metrics


# ----- multi_kernel_analysis.main() end-to-end -----


def test_driver_multi_kernel_main_missing_data(tmp_path, monkeypatch):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    multi_kernel_analysis.main()
    metrics = _read_metrics(base, "multi_kernel")
    assert metrics["status"] == "ERROR"
    assert metrics["memcpy_assessment"]["flagged"] is False


def test_driver_multi_kernel_main_success(tmp_path, monkeypatch):
    base = str(tmp_path)
    cat_dir = os.path.join(base, "category_data")
    csv_dir = os.path.join(base, "perf_report_csvs")
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    mk_data = {
        "memcpy_summary": {
            "total_time_us": 100_000,
            "total_count": 5,
            "by_direction": {
                "D2H": {"count": 3, "total_time_us": 100_000, "avg_bytes": 1024}
            },
        },
        "overlap_analysis": {
            "total_comm_time_us": 10_000,
            "exposed_comm_time_us": 8_000,
            "comm_percent_of_total": 8.0,
            "comm_overlap_ratio": 0.2,
            "total_time_us": 200_000,
        },
        "nccl_summary": {"total_count": 2, "total_time_us": 10_000, "top_ops": []},
    }
    with open(os.path.join(cat_dir, "multi_kernel_data.json"), "w") as f:
        json.dump(mk_data, f)
    pd.DataFrame(
        {
            "type": ["total_time", "exposed_comm_time"],
            "time ms": [200.0, 8.0],
            "percent": [100.0, 4.0],
        }
    ).to_csv(os.path.join(csv_dir, "gpu_timeline.csv"), index=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    multi_kernel_analysis.main()
    metrics = _read_metrics(base, "multi_kernel")
    assert metrics["status"] == "SUCCESS"
    assert metrics["memcpy_assessment"]["flagged"] is True
    assert metrics["overlap_assessment"]["flagged"] is True
    assert len(metrics["patterns_detected"]) >= 1


class TestArchUtils:
    def test_list_and_load_platform(self):
        platforms = arch_utils.list_platforms()
        assert platforms
        for name in platforms[:3]:
            arch = arch_utils.load_arch(name)
            assert "name" in arch or "mem_bw_gbps" in arch

    def test_tl_extension_override(self, tmp_path, monkeypatch):
        ext_pkg = tmp_path / "fake_ext_pkg"
        arch_dir = ext_pkg / "Agent" / "Analysis" / "utils" / "arch"
        arch_dir.mkdir(parents=True)
        custom = {"name": "CUSTOM", "mem_bw_gbps": 1000, "memory_gb": 80}
        (arch_dir / "CUSTOM.json").write_text(json.dumps(custom))

        init_py = ext_pkg / "__init__.py"
        init_py.write_text("")

        fake_mod = ModuleType("fake_ext_pkg")
        fake_mod.__file__ = str(init_py)
        monkeypatch.setitem(sys.modules, "fake_ext_pkg", fake_mod)
        monkeypatch.setenv("TL_EXTENSION", "fake_ext_pkg")

        mapping = arch_utils._collect_arch_jsons()
        assert "CUSTOM" in mapping
        assert arch_utils.load_arch("CUSTOM")["mem_bw_gbps"] == 1000


class TestCategoryAnalysisHelpers:
    _META = {"peak_hbm_bw_tbs": 5.3, "peak_maf_tflops": {"matrix_fp16": 654}}

    def test_gemm_classifiers(self):
        assert gemm_analysis.detect_quantized_gemm("aten::w8a8_mm")
        info = gemm_analysis.classify_gemm_operation("aten::mm", None)
        assert info["gemm_type"] == "regular"
        qinfo = gemm_analysis.classify_gemm_operation("aten::fp8_mm", None)
        assert qinfo["is_quantized"] is True
        ops = pd.DataFrame(
            {
                "name": ["aten::mm", "aten::fp8_mm"],
                "TFLOPS/s_mean": [100.0, None],
            }
        )
        extra = gemm_analysis.extract_category_specific(ops, self._META)
        assert extra["quantized_count"] == 1
        assert extra["missing_perf_model_count"] == 1

    def test_elementwise_extract(self):
        out = elementwise_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::add"]}), self._META
        )
        assert out["peak_hbm_bw_tbs"] == 5.3

    def test_norm_extract(self):
        out = norm_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::layer_norm"]}), self._META
        )
        assert out["peak_hbm_bw_tbs"] == 5.3

    def test_reduce_extract(self):
        out = reduce_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::sum"]}), self._META
        )
        assert "peak_hbm_bw_tbs" in out

    def test_convolution_extract_with_transpose(self):
        ops = pd.DataFrame(
            {
                "name": ["aten::conv2d", "aten::conv_transpose2d"],
                "Kernel Time (µs)_sum": [1000.0, 500.0],
            }
        )
        out = convolution_analysis.extract_category_specific(ops, self._META)
        assert out["transpose_count"] == 1
        assert out["transpose_time_ms"] == pytest.approx(0.5)

    def test_triton_classifiers(self):
        assert (
            triton_analysis.classify_triton_operation("triton_poi_add", None)[
                "kernel_type"
            ]
            == "pointwise"
        )
        assert (
            triton_analysis.classify_triton_operation("triton_red_sum", None)[
                "kernel_type"
            ]
            == "reduction"
        )
        assert (
            triton_analysis.classify_triton_operation("triton_per_mm", None)[
                "kernel_type"
            ]
            == "persistent"
        )
        assert (
            triton_analysis.classify_triton_operation("other", None)["kernel_type"]
            == "other"
        )
        out = triton_analysis.extract_category_specific(
            pd.DataFrame(
                {
                    "name": [
                        "triton_poi_a",
                        "triton_red_b",
                        "triton_per_c",
                        "other",
                    ]
                }
            ),
            self._META,
        )
        assert out["pointwise_count"] == 1
        assert out["reduction_count"] == 1

    def test_reduce_softmax_detect(self):
        assert reduce_analysis.detect_softmax("aten::_softmax")
        out = reduce_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::_softmax", "aten::sum"]}), self._META
        )
        assert out["softmax_count"] == 1

    def test_moe_extract_and_no_data_check(self, tmp_path):
        out = moe_analysis.extract_category_specific(
            pd.DataFrame({"name": ["moe_dispatch"]}), self._META
        )
        assert "peak_hbm_bw_tbs" in out
        missing = moe_analysis._check_moe_data(str(tmp_path), "moe_fused", "standalone")
        assert missing["status"] == "NO_DATA"


class TestPerfModelNormAndConvDeep:
    def test_batch_norm_bwd_full(self):
        event = {
            "name": "aten::miopen_batch_norm_backward",
            "args": {
                "Input Dims": [
                    (8, 16, 32, 32),
                    (8, 16, 32, 32),
                    (16,),
                    (16,),
                    (16,),
                    (16,),
                    (16,),
                    (),
                ],
                "Input type": ["float"] * 7 + ["Scalar"],
                "Input Strides": [(16384, 1024, 32, 1)] * 2 + [(1,)] * 5 + [()],
                "Concrete Inputs": ["", "", "", "", "", "", "", "1e-5"],
            },
        }
        model = perf_model.BatchNormBwd(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_group_norm_bwd(self):
        event = {
            "args": {
                "Input Dims": [
                    None,
                    (4, 8, 32, 32),
                    (8,),
                    (8,),
                    (8,),
                    (8,),
                    (4, 8, 32, 32),
                    (),
                ],
                "Input type": ["c10::BFloat16"] * 7 + ["Scalar"],
                "Input Strides": [(), (8192, 1024, 32, 1), (1,)] * 2
                + [(8192, 1024, 32, 1)] * 2
                + [(), ()],
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "8",
                    "8",
                    "[True, True]",
                ],
            }
        }
        model = perf_model.GroupNormBwd(event)
        assert model.flops() > 0

    def test_conv_bias_bwd_with_forward_cache(self):
        fwd = perf_model.ConvBias_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        bwd = perf_model.ConvBias_Backward(_conv_bias_bwd_event())
        assert bwd.flops_bwd() > 0


def _build_moe_tree(events: List[Dict], add_python_func: bool = False) -> TraceToTree:
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return tree


def _setup_gemm_output_dir(tmp_path):
    out = tmp_path / "analysis_output"
    (out / "category_data").mkdir(parents=True)
    (out / "metadata").mkdir()
    df = pd.DataFrame(
        {
            "name": ["aten::mm", "aten::addmm"],
            "count": [12, 3],
            "Kernel Time (µs)_sum": [100_000.0, 50_000.0],
            "Kernel Time (µs)_mean": [8000.0, 16000.0],
            "Kernel Time (µs)_std": [500.0, 100.0],
            "TFLOPS/s_mean": [400.0, 350.0],
            "TB/s_mean": [0.5, 0.4],
            "FLOPS/Byte": [2000.0, 1800.0],
            "Roofline Bound": ["COMPUTE_BOUND", "COMPUTE_BOUND"],
            "Compute Spec": ["matrix_bf16", "matrix_bf16"],
            "kernel_details_summary": ["[{'name': 'Cijk_a'}]", "[{'name': 'Cijk_b'}]"],
            "call_stack_full": ["['aten::mm', 'Linear']", "['aten::addmm']"],
            "Input Dims": ["[[32, 64], [64, 128]]", "[[32, 64], [64, 128], [32, 128]]"],
            "Input type": ["['fp16', 'fp16']", "['fp16', 'fp16', 'fp16']"],
        }
    )
    df.to_csv(out / "category_data" / "gemm_ops.csv", index=False)
    meta = {
        "platform": "MI300X",
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
        "output_dir": str(out),
    }
    (out / "metadata" / "gemm_metadata.json").write_text(json.dumps(meta))
    return str(out)


class TestAnalysisUtilsRunCategoryPhase9:
    def test_run_category_analysis_success(self, tmp_path):
        out = _setup_gemm_output_dir(tmp_path)
        au.run_category_analysis("gemm", out, {}, lambda ops_df, _m: {"n": len(ops_df)})
        metrics = json.loads(
            Path(out, "category_data", "gemm_metrics.json").read_text()
        )
        assert metrics["status"] == "OK"

    def test_run_category_analysis_no_data(self, tmp_path):
        out = tmp_path / "empty"
        (out / "category_data").mkdir(parents=True)
        au.run_category_analysis(
            "gemm",
            str(out),
            {},
            lambda _o, _m: {},
            no_data_check_fn=lambda _o, c, _s: {"category": c, "status": "NO_DATA"},
        )
        assert (
            json.loads((out / "category_data" / "gemm_metrics.json").read_text())[
                "status"
            ]
            == "NO_DATA"
        )

    def test_run_category_analysis_missing_csv_exits(self, tmp_path):
        out = tmp_path / "missing"
        (out / "category_data").mkdir(parents=True)
        with pytest.raises(SystemExit):
            au.run_category_analysis("gemm", str(out), {}, lambda _o, _m: {})


class TestArchAndMoePhase9:
    def test_arch_utils(self, monkeypatch):
        assert arch_utils.list_platforms()
        monkeypatch.setenv("TL_EXTENSION", "not_a_real_package_xyz")
        assert isinstance(arch_utils._collect_arch_jsons(), dict)

    def test_moe_pseudo_op_edges(self):
        assert not is_aiter_fused_moe_kernel(
            {"cat": "kernel", "name": "aiter::quant_fmoe"}
        )
        tree = _build_moe_tree([])
        _create_pseudo_op_moe_fused_aiter(tree, {"name": "wrong", "UID": 0})
        assert _extract_topk_from_outplace({"UID": 1, "args": {}}) == 8

        events = []
        moe = _mk_event(
            "cpu_op",
            "vllm::outplace_fused_experts",
            100,
            200,
            1,
            1,
            {
                "Input Dims": [
                    [128, 4096],
                    [8, 4096, 512],
                    [8, 4096, 512],
                    [128, 6],
                    [128, 6],
                ],
                "Sequence number": 2,
            },
        )
        events.append(moe)
        _add_gpu_chain(events, moe, 20, "fused_moe_kernel_gptq_awq_up", 110, 150)
        _add_gpu_chain(events, moe, 21, "fused_moe_kernel_gptq_awq_down", 160, 190)
        tree2 = _build_moe_tree(events)
        create_pseudo_ops_moe_gptq_awq(tree2)
        assert any(e.get("args", {}).get("Pseudo op") for e in tree2.events)

        fly_events = [
            _mk_event("cpu_op", FUSED_MOE_PARENT, 0, 500, 1, 1, {"Sequence number": 9}),
            _mk_event(
                "python_function", "flydsl.py(10): flydsl_moe_stage1", 50, 100, 1, 1, {}
            ),
        ]
        create_pseudo_ops_moe_flydsl(_build_moe_tree(fly_events, add_python_func=True))


class TestPerfModelConvAndNormBoost:
    @pytest.mark.parametrize(
        "cls,fwd_factory,bwd_cls,bwd_factory",
        [
            (
                perf_model.ConvBias_,
                _conv_bias_fwd_event,
                perf_model.ConvBias_Backward,
                _conv_bias_bwd_event,
            ),
            (
                perf_model.ConvBiasReLU_,
                _conv_bias_relu_fwd_event,
                perf_model.ConvBiasReLU_Backward,
                _conv_bias_relu_bwd_event,
            ),
        ],
    )
    def test_conv_bias_family(self, cls, fwd_factory, bwd_cls, bwd_factory):
        fwd = cls(fwd_factory())
        assert fwd.flops() > 0
        assert fwd.bytes() > 0
        bwd = bwd_cls(bwd_factory())
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0

    def test_fused_ln_modulate(self):
        fwd = perf_model.FusedLnModulate(_fused_ln_fwd_event())
        assert fwd.flops() > 0
        assert fwd.bytes() > 0

    def test_evoformer_attention(self):
        evo = perf_model.evoformer_attention(_evoformer_event())
        assert evo.flops() > 0
        assert evo.bytes() > 0

    def test_reduce_and_grouped_gemm(self):
        reduce_evt = {
            "name": "aten::mean",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "[1]", "True"],
            },
        }
        model = perf_model.aten_reduce(reduce_evt)
        assert model.flops() > 0
        gg_event = {
            "args": {
                "Input Dims": [
                    [4, 128],
                    [8, 256, 128],
                    [8, 256],
                    [8],
                    [8],
                    [8],
                    [8],
                    [8],
                    [4, 4],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float",
                    "c10::Int",
                ]
                + ["c10::Int"] * 5,
            }
        }
        g = perf_model.primus_turbo_grouped_gemm(gg_event)
        assert g.flops() > 0


class TestRmsNormExtensionsBytes:
    def test_rmsnorm_family_bytes(self):
        base = {
            "args": {
                "Input Dims": [(4, 256), (256,), (256,)],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(256, 1), (1,), (1,)],
            }
        }
        for cls in (rms_ext.aiter_rmsnorm,):
            model = cls(base)
            b = model.bytes()
            assert b is None or b > 0
            assert model.get_compute_precision() in (None, "bf16", "fp8")
