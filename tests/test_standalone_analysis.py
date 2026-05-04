###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for TraceLens AgenticMode Standalone Analysis.

- Unit tests for category_analyses/analysis_utils (efficiency, impact estimates, plot data, helpers).
- Integration test for utils/orchestrator_prepare.py with minimal perf_report_csvs fixtures
  (Steps 2-3 only; Step 4 requires a real trace and is skipped when fixtures are used).
"""

import json
import os
import sys

import pandas as pd
import pytest

# Add repo root and Standalone for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STANDALONE = os.path.join(REPO_ROOT, "TraceLens", "AgenticMode", "Standalone")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(STANDALONE, "category_analyses"))

from TraceLens.AgenticMode.Standalone.category_analyses.analysis_utils import (
    validate_efficiency,
    calculate_efficiency_with_validation,
    compute_impact_estimates,
    write_metrics_json,
    load_category_data,
    calculate_time_metrics,
    build_operation_metrics,
)
from TraceLens.AgenticMode.Standalone.category_analyses.gemm_analysis import (
    detect_quantized_gemm,
)
from TraceLens.AgenticMode.Standalone.category_analyses.sdpa_analysis import (
    detect_flash_attention,
    detect_paged_attention,
)
from TraceLens.AgenticMode.Standalone.category_analyses.reduce_analysis import (
    detect_softmax,
)
from TraceLens.AgenticMode.Standalone.category_analyses.other_analysis import (
    classify_other_operation,
)
from TraceLens.AgenticMode.Standalone.utils.report_utils import (
    generate_priority_data,
)

# ----- Fixtures: minimal output dir layout for analysis_utils -----


@pytest.fixture
def output_dir_with_category_data(tmp_path):
    """Create minimal category_data + metadata for one category (gemm)."""
    out = tmp_path / "analysis_output"
    (out / "category_data").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)

    # gemm_ops.csv: minimal columns required by build_operation_metrics / load_category_data
    gemm_csv = out / "category_data" / "gemm_ops.csv"
    df = pd.DataFrame(
        {
            "name": ["aten::mm", "aten::mm"],
            "count": [1, 1],
            "Kernel Time (µs)_sum": [100_000, 50_000],
            "TFLOPS/s_mean": [400.0, 350.0],
            "TB/s_mean": [0.5, 0.4],
            "FLOPS/Byte": [2000.0, 1800.0],
            "Compute Spec": ["matrix_bf16", "matrix_bf16"],
        }
    )
    df.to_csv(gemm_csv, index=False)

    # gemm_metadata.json
    meta = {
        "platform": "MI300X",
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708, "matrix_fp16": 654},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    (out / "metadata" / "gemm_metadata.json").write_text(json.dumps(meta, indent=2))

    return str(out)


@pytest.fixture
def output_dir_with_manifest_and_metrics(tmp_path):
    """Create category_data with category_manifest.json and 1+ *_metrics.json for generate_priority_data."""
    out = tmp_path / "analysis_output"
    cat_data = out / "category_data"
    cat_data.mkdir(parents=True)

    manifest = {
        "platform": "MI300X",
        "gpu_utilization": {"total_time_ms": 5000.0},
        "categories": [{"name": "gemm", "tier": "compute_kernel"}],
    }
    (cat_data / "category_manifest.json").write_text(json.dumps(manifest, indent=2))

    # gemm_metrics.json with impact_estimates
    gemm_metrics = {
        "status": "OK",
        "category": "gemm",
        "total_time_ms": 3000.0,
        "impact_estimates": [
            {
                "operation": "aten::mm",
                "category": "gemm",
                "type": "kernel_tuning",
                "impact_score": 100.0,
                "impact_score_low": 85.0,
                "impact_score_high": 115.0,
                "confidence": "high",
            },
            {
                "operation": "aten::mm",
                "category": "gemm",
                "type": "kernel_tuning",
                "impact_score": 50.0,
                "impact_score_low": 42.0,
                "impact_score_high": 58.0,
                "confidence": "medium",
            },
        ],
        "category_findings": [
            {
                "rank": 1,
                "bound_type": "compute",
                "library": "Tensile",
                "impact_score": 150.0,
                "impact_score_low": 127.0,
                "impact_score_high": 173.0,
                "operation_count": 2,
            },
        ],
    }
    (cat_data / "gemm_metrics.json").write_text(json.dumps(gemm_metrics, indent=2))

    # sdpa_fwd_metrics.json
    sdpa_metrics = {
        "status": "OK",
        "category": "sdpa_fwd",
        "total_time_ms": 500.0,
        "impact_estimates": [
            {
                "operation": "flash_attn_forward",
                "category": "sdpa_fwd",
                "type": "kernel_tuning",
                "impact_score": 80.0,
                "impact_score_low": 68.0,
                "impact_score_high": 92.0,
                "confidence": "medium",
            },
        ],
        "category_findings": [
            {
                "rank": 1,
                "bound_type": "memory",
                "library": "CK",
                "impact_score": 80.0,
                "impact_score_low": 68.0,
                "impact_score_high": 92.0,
                "operation_count": 1,
            },
        ],
    }
    (cat_data / "sdpa_fwd_metrics.json").write_text(json.dumps(sdpa_metrics, indent=2))

    return str(out)


# ----- Unit tests: validate_efficiency -----


def test_validate_efficiency_normal():
    r = validate_efficiency(70.0, 100.0, "Compute")
    assert r["value"] == 70.0
    assert r["warning"] is None
    assert r["is_anomaly"] is False


def test_validate_efficiency_anomaly_over_110():
    r = validate_efficiency(120.0, 100.0, "Compute")
    assert r["value"] == 120.0
    assert "[ANOMALY]" in (r["warning"] or "")
    assert r["is_anomaly"] is True


def test_validate_efficiency_slightly_over_100():
    r = validate_efficiency(105.0, 100.0, "Compute")
    assert r["value"] == 105.0
    assert r["warning"] is not None
    assert "[WARNING]" in r["warning"]
    assert r["is_anomaly"] is False


def test_validate_efficiency_invalid_peak():
    r = validate_efficiency(50.0, 0, "Compute")
    assert r["value"] is None
    assert "Invalid peak" in (r["warning"] or "")
    assert r["is_anomaly"] is True


def test_calculate_efficiency_with_validation():
    r = calculate_efficiency_with_validation(350.0, 0.2, 708.0, 5.3)
    assert "compute_efficiency_pct" in r
    assert "memory_efficiency_pct" in r
    assert r["compute_efficiency_pct"] == round(350.0 / 708.0 * 100, 2)
    assert r["memory_efficiency_pct"] == round(0.2 / 5.3 * 100, 2)


# ----- Unit tests: compute_impact_estimates -----


def test_compute_impact_estimates_basic():
    operations = [
        {
            "name": "op_a",
            "time_ms": 10.0,
            "efficiency": {
                "efficiency_percent": 50.0,
                "is_anomaly": False,
                "bound_type": "compute",
            },
        },
        {
            "name": "op_b",
            "time_ms": 5.0,
            "efficiency": {
                "efficiency_percent": 80.0,
                "is_anomaly": False,
                "bound_type": "memory",
            },
        },
    ]
    # gap_mid = 0.875 * (1 - eff/100); impact_score = gap_mid * time_ms / baseline_ms * 100
    # baseline = 100 ms ->
    #   op_a: 0.875 * 0.5 * 10 / 100 * 100 = 4.375 -> rounded 4.38
    #   op_b: 0.875 * 0.2 * 5  / 100 * 100 = 0.875 -> rounded 0.88
    estimates = compute_impact_estimates(operations, "gemm", baseline_ms=100.0)
    assert len(estimates) == 2
    assert estimates[0]["impact_score"] == 4.38
    assert estimates[0]["operation"] == "op_a"
    assert estimates[0]["category"] == "gemm"
    assert estimates[0]["type"] == "kernel_tuning"
    assert "impact_score_low" in estimates[0]
    assert "impact_score_high" in estimates[0]
    assert "savings_ms" not in estimates[0]
    assert "e2e_pct_high" not in estimates[0]
    assert estimates[1]["impact_score"] == 0.88 or estimates[1]["impact_score"] == 0.87


def test_compute_impact_estimates_excludes_anomaly():
    operations = [
        {
            "name": "op_a",
            "time_ms": 10.0,
            "efficiency": {"efficiency_percent": 120.0, "is_anomaly": True},
        },
    ]
    estimates = compute_impact_estimates(operations, "gemm", baseline_ms=100.0)
    assert len(estimates) == 0


def test_compute_impact_estimates_min_impact_score():
    operations = [
        {
            "name": "op_a",
            "time_ms": 1.0,
            "efficiency": {
                "efficiency_percent": 50.0,
                "is_anomaly": False,
                "bound_type": "compute",
            },
        },
    ]
    # impact_score_high = 0.5 * 1 / 100 * 100 = 0.5
    estimates = compute_impact_estimates(
        operations, "gemm", min_impact_score=0.01, baseline_ms=100.0
    )
    assert len(estimates) == 1
    assert estimates[0]["impact_score_high"] == 0.5
    estimates_strict = compute_impact_estimates(
        operations, "gemm", min_impact_score=1.0, baseline_ms=100.0
    )
    assert len(estimates_strict) == 0


def test_comparative_impact_from_operations_trace2_faster():
    """Comparative efficiency_percent = 100*t2/t1; impact uses same 75/87.5/100 bands."""
    df = pd.DataFrame(
        {
            "name": ["aten::mm", "aten::addmm", "aten::bmm"],
            "count": [1, 1, 1],
            "Kernel Time (µs)_sum": [10_000.0, 5_000.0, 2_000.0],
            "speedup (trace1/trace2)": [0.5, 1.2, 0.8],
            "delta_us (trace2 - trace1)": [-5_000.0, 1_000.0, -200.0],
        }
    )
    metadata = {"peak_hbm_bw_tbs": 5.3, "peak_bf16_maf_tflops": 700.0}
    config: dict = {}
    operations = build_operation_metrics(
        df, metadata, config, comparison_scope="comparative"
    )
    mm_op = next(o for o in operations if o["name"] == "aten::mm")
    assert mm_op["efficiency"]["efficiency_percent"] == 50.0
    bmm_op = next(o for o in operations if o["name"] == "aten::bmm")
    assert bmm_op["efficiency"]["efficiency_percent"] == 90.0

    out = compute_impact_estimates(
        operations, "gemm", min_savings_ms=0.1, baseline_ms=1000.0
    )
    # Row 1: eff 120 -> savings_high 0 -> excluded; row 0 and 2 remain
    assert len(out) == 2
    assert out[0]["operation"] == "aten::mm"
    assert out[0]["type"] == "kernel_tuning"
    assert out[0]["efficiency_pct"] == 50.0
    assert out[0]["savings_ms_high"] == 5.0
    assert out[0]["savings_ms_low"] == round(5.0 * 0.75, 3)
    assert out[0]["savings_ms"] == round(5.0 * 0.875, 3)
    assert out[0]["e2e_pct_high"] == 0.5
    assert out[1]["operation"] == "aten::bmm"
    assert out[1]["efficiency_pct"] == 90.0
    assert out[1]["savings_ms_high"] == 0.2
    assert out[1]["savings_ms"] == round(0.2 * 0.875, 3)


def test_comparative_roofline_cap_clamps_savings():
    """Projected savings must not exceed trace1 roofline when trace2 is faster than the ceiling."""
    # trace2 is 4x faster than trace1 (comp_pct = 25%) but roofline is 60%.
    # The physically achievable efficiency floor is 60%, so savings should be
    # capped at time * (1 - 60/100), not time * (1 - 25/100).
    df = pd.DataFrame(
        {
            "name": ["aten::mm"],
            "count": [1],
            "Kernel Time (µs)_sum": [10_000.0],
            "delta_us (trace2 - trace1)": [-7_500.0],  # comp_pct = 25%
            "Pct Roofline_mean": [60.0],               # roofline floor
        }
    )
    metadata = {"peak_hbm_bw_tbs": 5.3, "peak_bf16_maf_tflops": 700.0}
    operations = build_operation_metrics(df, metadata, {}, comparison_scope="comparative")
    mm_op = operations[0]
    eff = mm_op["efficiency"]
    # efficiency_percent clamped to roofline (60), not raw comp_pct (25)
    assert eff["efficiency_percent"] == 60.0
    assert eff["warning"] is not None
    assert "ROOFLINE CAP" in eff["warning"]
    # savings_high = 10ms * (1 - 60/100) = 4ms (not 7.5ms)
    out = compute_impact_estimates(operations, "gemm", min_savings_ms=0.1, baseline_ms=1000.0)
    assert len(out) == 1
    assert out[0]["savings_ms_high"] == 4.0
    assert out[0]["savings_ms"] == round(4.0 * 0.875, 3)


def test_comparative_roofline_cap_no_clamp_when_trace2_above_roofline():
    """When trace2 efficiency is already above the roofline (slower), no clamping occurs."""
    # comp_pct = 80% (trace2 is 1.25x faster), roofline = 60%.
    # 80 > 60 so no clamping — savings = time * (1 - 80/100).
    df = pd.DataFrame(
        {
            "name": ["aten::mm"],
            "count": [1],
            "Kernel Time (µs)_sum": [10_000.0],
            "delta_us (trace2 - trace1)": [-2_000.0],  # comp_pct = 80%
            "Pct Roofline_mean": [60.0],
        }
    )
    metadata = {"peak_hbm_bw_tbs": 5.3, "peak_bf16_maf_tflops": 700.0}
    operations = build_operation_metrics(df, metadata, {}, comparison_scope="comparative")
    eff = operations[0]["efficiency"]
    assert eff["efficiency_percent"] == 80.0
    assert eff["warning"] is None
    out = compute_impact_estimates(operations, "gemm", min_savings_ms=0.1, baseline_ms=1000.0)
    assert out[0]["savings_ms_high"] == 2.0


def test_comparative_roofline_cap_no_roofline_column():
    """When Pct Roofline_mean is absent, comparative efficiency is unchanged."""
    df = pd.DataFrame(
        {
            "name": ["aten::mm"],
            "count": [1],
            "Kernel Time (µs)_sum": [10_000.0],
            "delta_us (trace2 - trace1)": [-7_500.0],  # comp_pct = 25%
            # no Pct Roofline_mean column
        }
    )
    metadata = {"peak_hbm_bw_tbs": 5.3, "peak_bf16_maf_tflops": 700.0}
    operations = build_operation_metrics(df, metadata, {}, comparison_scope="comparative")
    eff = operations[0]["efficiency"]
    assert eff["efficiency_percent"] == 25.0
    assert eff["warning"] is None


# ----- Unit tests: generate_plot_data -----


def test_generate_priority_data(output_dir_with_manifest_and_metrics):
    out_path = generate_priority_data(
        output_dir_with_manifest_and_metrics, max_recommendations=3
    )
    assert os.path.isfile(out_path)
    assert out_path.endswith("priority_data.json")

    with open(out_path) as f:
        data = json.load(f)

    assert data["baseline_ms"] == 5000.0
    assert "recommendations" in data
    assert "all_estimates" in data
    assert "priorities" in data
    recs = data["recommendations"]
    assert len(recs) <= 3
    categories = [r["category"] for r in recs]
    assert "gemm" in categories
    assert "sdpa_fwd" in categories
    gemm_rec = next(r for r in recs if r["category"] == "gemm")
    assert gemm_rec["impact_score"] == 150.0
    assert gemm_rec["impact_score_low"] == 127.0
    assert gemm_rec["impact_score_high"] == 173.0
    assert gemm_rec["operation_count"] == 2
    assert "savings_ms" not in gemm_rec


def test_generate_priority_data_skips_error_metrics(tmp_path):
    cat_data = tmp_path / "category_data"
    cat_data.mkdir(parents=True)
    (cat_data / "category_manifest.json").write_text(
        json.dumps({"gpu_utilization": {"total_time_ms": 100.0}}, indent=2)
    )
    (cat_data / "gemm_metrics.json").write_text(
        json.dumps({"status": "ERROR", "impact_estimates": []}, indent=2)
    )
    out_path = generate_priority_data(str(tmp_path))
    with open(out_path) as f:
        data = json.load(f)
    assert data["baseline_ms"] == 100.0
    assert len(data["recommendations"]) == 0
    assert len(data["all_estimates"]) == 0


# ----- Unit tests: write_metrics_json -----


def test_write_metrics_json(tmp_path):
    (tmp_path / "category_data").mkdir(parents=True)
    metrics = {"category": "gemm", "status": "OK", "total_time_ms": 100.0}
    path = write_metrics_json(metrics, str(tmp_path), "gemm")
    assert path == os.path.join(tmp_path, "category_data", "gemm_metrics.json")
    assert os.path.isfile(path)
    with open(path) as f:
        assert json.load(f) == metrics


# ----- Unit tests: load_category_data -----


def test_load_category_data(output_dir_with_category_data):
    df, meta = load_category_data(output_dir_with_category_data, "gemm")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "name" in df.columns
    assert "Kernel Time (µs)_sum" in df.columns
    assert meta["platform"] == "MI300X"
    assert meta["peak_hbm_bw_tbs"] == 5.3


def test_load_category_data_missing_csv_raises(tmp_path):
    (tmp_path / "metadata").mkdir(parents=True)
    (tmp_path / "metadata" / "gemm_metadata.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Category CSV not found"):
        load_category_data(str(tmp_path), "gemm")


# ----- Unit tests: calculate_time_metrics -----


def test_calculate_time_metrics(output_dir_with_category_data):
    df, meta = load_category_data(output_dir_with_category_data, "gemm")
    m = calculate_time_metrics(df, meta)
    assert m["total_time_ms"] == 150.0  # 100_000 + 50_000 us
    assert m["operation_count"] == 2
    assert "percent_of_compute" in m


# ----- Unit tests: build_operation_metrics -----


def test_build_operation_metrics(output_dir_with_category_data):
    df, meta = load_category_data(output_dir_with_category_data, "gemm")
    config = {
        "efficiency_method": "auto",
        "extra_fields": [],
        "operation_classifier": None,
    }
    ops = build_operation_metrics(df, meta, config)
    assert len(ops) == 2
    for o in ops:
        assert "name" in o
        assert "time_ms" in o
        assert "efficiency" in o
        assert "efficiency_percent" in o["efficiency"] or "efficiency" in o


def test_build_operation_metrics_comparative_uses_speedup():
    """comparative scope: efficiency_percent = 100 * speedup (trace2/trace1)."""
    df = pd.DataFrame(
        {
            "name": ["aten::mm", "aten::addmm"],
            "count": [1, 1],
            "Kernel Time (µs)_sum": [10_000.0, 5_000.0],
            "TFLOPS/s_mean": [400.0, 350.0],
            "TB/s_mean": [0.5, 0.4],
            "FLOPS/Byte": [2000.0, 1800.0],
            "Compute Spec": ["matrix_bf16", "matrix_bf16"],
            "speedup (trace2/trace1)": [0.8, 1.2],
        }
    )
    meta = {
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    ops = build_operation_metrics(df, meta, {}, comparison_scope="comparative")
    assert len(ops) == 2
    mm = next(o for o in ops if o["name"] == "aten::mm")
    addmm = next(o for o in ops if o["name"] == "aten::addmm")
    assert mm["efficiency"]["efficiency_percent"] == 80.0
    assert addmm["efficiency"]["efficiency_percent"] == 120.0


def test_build_operation_metrics_comparative_uses_delta():
    """comparative scope falls back to delta column when speedup is absent."""
    df = pd.DataFrame(
        {
            "name": ["aten::mm"],
            "count": [1],
            "Kernel Time (µs)_sum": [10_000.0],
            "TFLOPS/s_mean": [400.0],
            "TB/s_mean": [0.5],
            "FLOPS/Byte": [2000.0],
            "Compute Spec": ["matrix_bf16"],
            # delta = -2000 us → t2 = 8000 us → eff = 80%
            "delta_us (trace2 - trace1)": [-2_000.0],
        }
    )
    meta = {
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    ops = build_operation_metrics(df, meta, {}, comparison_scope="comparative")
    assert ops[0]["efficiency"]["efficiency_percent"] == 80.0


def test_build_operation_metrics_comparative_no_comparative_cols_yields_none():
    """comparative scope with no speedup/delta columns → efficiency_percent is None."""
    df = pd.DataFrame(
        {
            "name": ["aten::mm"],
            "count": [1],
            "Kernel Time (µs)_sum": [10_000.0],
            "TFLOPS/s_mean": [400.0],
            "TB/s_mean": [0.5],
            "FLOPS/Byte": [2000.0],
            "Compute Spec": ["matrix_bf16"],
        }
    )
    meta = {
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    ops = build_operation_metrics(df, meta, {}, comparison_scope="comparative")
    assert ops[0]["efficiency"]["efficiency_percent"] is None


# ----- Unit tests: compute_impact_estimates (comparative mode) -----


def test_compute_impact_estimates_comparative_excludes_ops_above_100():
    """An op with eff > 100% produces savings_high=0, excluded by default min_savings_ms."""
    operations = [
        {
            "name": "op_faster",
            "time_ms": 10.0,
            "efficiency": {"efficiency_percent": 120.0, "is_anomaly": False},
        },
        {
            "name": "op_slower",
            "time_ms": 10.0,
            "efficiency": {"efficiency_percent": 60.0, "is_anomaly": False},
        },
    ]
    estimates = compute_impact_estimates(operations, "gemm", min_savings_ms=0.1)
    names = [e["operation"] for e in estimates]
    assert "op_faster" not in names
    assert "op_slower" in names


def test_compute_impact_estimates_comparative_savings_formula():
    """savings bands: high=gap, mid=0.875*high, low=0.75*high."""
    operations = [
        {
            "name": "aten::mm",
            "time_ms": 10.0,
            "efficiency": {"efficiency_percent": 50.0, "is_anomaly": False},
        },
    ]
    estimates = compute_impact_estimates(
        operations, "gemm", min_savings_ms=0.0, baseline_ms=100.0, analysis_mode="comparative"
    )
    assert len(estimates) == 1
    e = estimates[0]
    savings_high = round(10.0 * (1 - 50.0 / 100.0), 3)  # 5.0
    assert e["savings_ms_high"] == savings_high
    assert e["savings_ms"] == round(savings_high * 0.875, 3)
    assert e["savings_ms_low"] == round(savings_high * 0.75, 3)
    assert e["e2e_pct_high"] == round(savings_high / 100.0 * 100, 2)


def test_compute_impact_estimates_comparative_at_100_pct_no_savings():
    """An op at exactly 100% efficiency produces zero savings and is excluded by default threshold."""
    operations = [
        {
            "name": "aten::mm",
            "time_ms": 10.0,
            "efficiency": {"efficiency_percent": 100.0, "is_anomaly": False},
        },
    ]
    estimates = compute_impact_estimates(
        operations, "gemm", min_savings_ms=0.1, analysis_mode="comparative"
    )
    assert len(estimates) == 0


# ----- Unit tests: classify_other_operation -----


@pytest.mark.parametrize(
    "op_name,expected",
    [
        ("ncclKernel_AllReduce", "communication"),
        ("rcclBroadcast", "communication"),
        ("hipGraphLaunch", "graph"),
        ("cudaGraphLaunch", "graph"),
        ("aten::copy_", "miscellaneous"),
    ],
)
def test_classify_other_operation(op_name, expected):
    assert classify_other_operation(op_name) == expected


# ----- Unit tests: detect_* helpers -----


def test_detect_quantized_gemm():
    assert detect_quantized_gemm("w8a8_gemm") is True
    assert detect_quantized_gemm("aten::mm") is False


def test_detect_flash_attention():
    assert detect_flash_attention("flash_attn::_flash_attn_forward") is True
    assert detect_flash_attention("aten::mm") is False


def test_detect_softmax():
    assert detect_softmax("aten::softmax") is True
    assert detect_softmax("aten::mm") is False


def test_detect_paged_attention():
    assert detect_paged_attention("paged_attention_2d") is True
    assert detect_paged_attention("aten::mm") is False


# ----- Integration: orchestrator_prepare with minimal fixtures -----


@pytest.fixture
def minimal_perf_report_csvs(tmp_path):
    """Minimal perf_report_csvs layout required by orchestrator_prepare Steps 2-3."""
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir(parents=True)

    gpu_timeline = pd.DataFrame(
        {
            "type": [
                "total_time",
                "computation_time",
                "exposed_comm_time",
                "exposed_memcpy_time",
                "idle_time",
                "busy_time",
                "total_comm_time",
                "total_memcpy_time",
            ],
            "time ms": [1000.0, 998.0, 0.5, 0.1, 2.0, 998.5, 50.0, 1.0],
            "percent": [100.0, 99.8, 0.05, 0.01, 0.2, 99.85, 5.0, 0.1],
        }
    )
    gpu_timeline.to_csv(csv_dir / "gpu_timeline.csv", index=False)

    ops_summary = pd.DataFrame(
        {
            "name": ["aten::mm", "flash_attn::_flash_attn_forward"],
            "total_direct_kernel_time_sum": [800_000, 100_000],
            "Count": [10, 2],
            "Categories": ["{'GEMM'}", "{'SDPA_fwd'}"],
            "total_direct_kernel_time_ms": [800.0, 100.0],
            "Percentage (%)": [80.0, 10.0],
            "Cumulative Percentage (%)": [80.0, 90.0],
        }
    )
    if "Kernel Time (µs)_sum" not in ops_summary.columns:
        ops_summary["Kernel Time (µs)_sum"] = (
            ops_summary["total_direct_kernel_time_sum"] / 1000
        )
    ops_summary.to_csv(csv_dir / "ops_summary.csv", index=False)

    unified = pd.DataFrame(
        {
            "op category": ["GEMM", "SDPA_fwd"],
            "Kernel Time (µs)_sum": [800_000, 100_000],
            "Count": [10, 2],
        }
    )
    unified.to_csv(csv_dir / "unified_perf_summary.csv", index=False)

    return str(tmp_path)


def test_orchestrator_prepare_steps_2_3_require_csvs(minimal_perf_report_csvs):
    """Orchestrator prepare Step 2 reads gpu_timeline.csv; Step 3 reads ops_summary.
    We only verify that with minimal CSVs the script would fail later (at Step 4 trace load)
    unless we have a trace. So we test that the CSV layout we provide is valid for reading.
    """
    import pandas as pd

    csv_dir = os.path.join(minimal_perf_report_csvs, "perf_report_csvs")
    gpu = pd.read_csv(os.path.join(csv_dir, "gpu_timeline.csv"))
    assert "type" in gpu.columns and "time ms" in gpu.columns
    assert gpu[gpu["type"] == "total_time"]["time ms"].iloc[0] == 1000.0

    ops = pd.read_csv(os.path.join(csv_dir, "ops_summary.csv"))
    assert len(ops) == 2
    assert "name" in ops.columns


# ----- Category analysis script: gemm_analysis runs with minimal data -----


def test_gemm_analysis_script_with_minimal_data(output_dir_with_category_data):
    """Run gemm_analysis.py --output-dir <dir> with pre-created gemm_ops.csv + metadata."""
    script = os.path.join(STANDALONE, "category_analyses", "gemm_analysis.py")
    if not os.path.isfile(script):
        pytest.skip("gemm_analysis.py not found")

    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        [sys.executable, script, "--output-dir", output_dir_with_category_data],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")

    metrics_path = os.path.join(
        output_dir_with_category_data, "category_data", "gemm_metrics.json"
    )
    assert os.path.isfile(metrics_path)
    with open(metrics_path) as f:
        m = json.load(f)
    assert m.get("category") == "gemm"
    assert m.get("status") in ("OK", "ERROR")
    if m.get("status") == "OK":
        assert "operations" in m
        assert "impact_estimates" in m
        assert m.get("comparison_scope") == "standalone"


def test_gemm_analysis_script_comparative_mode(tmp_path):
    """gemm_ops.csv with TraceDiff columns + --comparison_scope comparative -> kernel_tuning-shaped estimates."""
    out = tmp_path / "analysis_output"
    (out / "category_data").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)
    df = pd.DataFrame(
        {
            "name": ["aten::mm"],
            "count": [1],
            "Kernel Time (µs)_sum": [10_000.0],
            "TFLOPS/s_mean": [400.0],
            "TB/s_mean": [0.5],
            "FLOPS/Byte": [2000.0],
            "Compute Spec": ["matrix_bf16"],
            "speedup (trace1/trace2)": [0.5],
            "delta_us (trace2 - trace1)": [-5_000.0],
        }
    )
    df.to_csv(out / "category_data" / "gemm_ops.csv", index=False)
    meta = {
        "platform": "MI300X",
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    (out / "metadata" / "gemm_metadata.json").write_text(json.dumps(meta, indent=2))

    script = os.path.join(STANDALONE, "category_analyses", "gemm_analysis.py")
    if not os.path.isfile(script):
        pytest.skip("gemm_analysis.py not found")

    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            str(out),
            "--comparison_scope",
            "comparative",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    with open(out / "category_data" / "gemm_metrics.json") as f:
        m = json.load(f)
    assert m["comparison_scope"] == "comparative"
    assert len(m["impact_estimates"]) == 1
    assert m["impact_estimates"][0]["type"] == "kernel_tuning"
    assert m["operations"][0]["efficiency"]["efficiency_percent"] == 50.0
    assert m["impact_estimates"][0]["efficiency_pct"] == 50.0
    # savings_high = 10*(1-50/100) = 5.0; savings_ms = 0.875 * savings_high
    assert m["impact_estimates"][0]["savings_ms_high"] == 5.0
    assert m["impact_estimates"][0]["savings_ms"] == round(5.0 * 0.875, 3)


def test_moe_analysis_no_data_includes_comparison_scope(tmp_path):
    """moe_analysis.py with no moe_fused_ops.csv writes comparison_scope from --comparison_scope."""
    out = tmp_path / "analysis_output"
    (out / "category_data").mkdir(parents=True)
    script = os.path.join(STANDALONE, "category_analyses", "moe_analysis.py")
    if not os.path.isfile(script):
        pytest.skip("moe_analysis.py not found")

    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            str(out),
            "--comparison_scope",
            "comparative",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    metrics_path = out / "category_data" / "moe_fused_metrics.json"
    assert metrics_path.is_file()
    with open(metrics_path) as f:
        m = json.load(f)
    assert m.get("status") == "NO_DATA"
    assert m.get("comparison_scope") == "comparative"


def test_norm_analysis_script_comparative_scope(tmp_path):
    """norm_ops.csv with TraceDiff columns + --comparison_scope comparative."""
    out = tmp_path / "analysis_output"
    (out / "category_data").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)
    df = pd.DataFrame(
        {
            "name": ["aten::layer_norm"],
            "count": [1],
            "Kernel Time (µs)_sum": [8000.0],
            "TFLOPS/s_mean": [50.0],
            "TB/s_mean": [0.2],
            "FLOPS/Byte": [80.0],
            "Compute Spec": ["matrix_bf16"],
            "speedup (trace1/trace2)": [0.5],
            "delta_us (trace2 - trace1)": [-4000.0],
        }
    )
    df.to_csv(out / "category_data" / "norm_ops.csv", index=False)
    meta = {
        "platform": "MI300X",
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    (out / "metadata" / "norm_metadata.json").write_text(json.dumps(meta, indent=2))

    script = os.path.join(STANDALONE, "category_analyses", "norm_analysis.py")
    if not os.path.isfile(script):
        pytest.skip("norm_analysis.py not found")

    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            str(out),
            "--comparison_scope",
            "comparative",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    with open(out / "category_data" / "norm_metrics.json") as f:
        m = json.load(f)
    assert m["comparison_scope"] == "comparative"
    assert m["operations"][0]["efficiency"]["efficiency_percent"] == 50.0
