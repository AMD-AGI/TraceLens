###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for TraceLens Agent Analysis shared utils.

- Unit tests for category_analyses/analysis_utils (efficiency, impact estimates, plot data, helpers).
- Unit tests for the pure helpers in utils/orchestrator_prepare (per-category data
  prep and fusion-candidate extraction).
- Integration test that drives utils/orchestrator_prepare.py end-to-end over a
  bundled trace, skipped when the trace or generator is unavailable.
- Unit tests for utils/validation_utils (pure markdown/JSON validation heuristics).
"""

import json, os, subprocess, sys, pandas as pd, pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, os.path.join(ANALYSIS_DIR, "category_analyses"))
from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    _eff_bucket,
    _extract_call_chain,
    _extract_kernel_names,
    _extract_module_chain,
    _match_fusion_op,
    _parse_call_stack,
    _resolve_peak_maf,
    build_category_findings,
    build_operation_metrics,
    calculate_efficiency,
    calculate_efficiency_with_validation,
    calculate_time_metrics,
    classify_kernel_library,
    comparative_efficiency,
    compute_impact_estimates,
    format_args,
    get_peak_specs,
    load_category_data,
    parse_first_shape,
    shape_aware_lookup,
    standalone_efficiency,
    validate_efficiency,
    write_metrics_json,
)
from TraceLens.Agent.Analysis.utils.report_utils import (
    generate_priority_data,
    load_findings,
    load_manifest,
    load_manifest_categories,
    prepare_model_identification_data,
)
from TraceLens.Agent.Analysis.utils.validation_utils import (
    MarkerValidator,
    _category_findings_empty,
    _check_coverage,
    _check_priority_consistency,
    _check_time_sanity,
    _extract_detailed_analysis_subsection,
    _load_valid_args,
    _metrics_json_for_findings,
    _scan_args_cells,
    _validate_compute_data_tables,
    _validate_report_args_column,
    _validate_report_comparison_scope_diffs,
    _validate_report_priority_consistency,
    _validate_report_reasoning_candidates,
    validate_findings_file,
    validate_report,
    validate_subagent_outputs,
)
from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _apply_comparative_gates,
    _build_diff_stats_lookups,
    _build_kernel_perf_lookup,
    _build_parent_chain,
    _build_trace2_ops_summary_by_enhanced_category,
    _compute_data_in_out,
    _dedup_by_kernel_set,
    _extract_attention_core,
    _extract_comparative_fusion_candidates,
    _extract_standalone_fusion_candidates,
    _gpu_utilization_metrics_from_gpu_timeline_df,
    _has_fused_kernel,
    _is_case_a_fusion_gap,
    _is_fusion_eligible,
    _is_gemm_norm_only,
    _make_comparative_candidate,
    _normalize_category,
    _prefix_lookup,
    _strip_module_index,
)
from tests.fixtures.traces import NORM_TRACE, RESNET_TRACE
from tests.fixtures.agent import (
    _StubAnalyzer,
    _StubTree,
    _kernel_event,
    _write_minimal_orchestrator_csvs,
)
from TraceLens.Agent.Analysis.utils import orchestrator_prepare as op, plot_utils
from TraceLens.Agent.Analysis.category_analyses import (
    analysis_utils as au,
    kernel_fusion_analysis as kfa,
)
from TraceLens.Reporting import tracediff_comparison_extension as tde
from TraceLens.TreePerf import GPUEventAnalyser
from TraceLens.Agent.Analysis.utils.classify_kernels import (
    classify_all,
    main as classify_main,
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
            },
            {
                "operation": "aten::mm",
                "category": "gemm",
                "type": "kernel_tuning",
                "impact_score": 50.0,
                "impact_score_low": 42.0,
                "impact_score_high": 58.0,
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


@pytest.fixture
def output_dir_other_and_customcollective_nccl(tmp_path):
    """Minimal ops CSV + metadata for ``other`` and ``customcollective`` (NCCL-style name).

    ``classify_other_operation`` treats this op as communication; ``customcollective``
    analysis must still retain it, while ``other`` strips it for NCCL Analyzer routing.
    """
    out = tmp_path / "analysis_output"
    (out / "category_data").mkdir(parents=True)
    (out / "metadata").mkdir(parents=True)

    df = pd.DataFrame(
        {
            "name": ["ncclKernel_AllReduce_RING_LL"],
            "count": [1],
            "Kernel Time (µs)_sum": [100_000],
            "TFLOPS/s_mean": [10.0],
            "TB/s_mean": [0.1],
            "FLOPS/Byte": [100.0],
            "Compute Spec": ["matrix_bf16"],
        }
    )
    for cat in ("customcollective", "other"):
        df.to_csv(out / "category_data" / f"{cat}_ops.csv", index=False)

    meta = {
        "platform": "MI300X",
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    meta_text = json.dumps(meta, indent=2)
    for cat in ("customcollective", "other"):
        (out / "metadata" / f"{cat}_metadata.json").write_text(meta_text)

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
        operations, "gemm", min_impact_score=0.01, baseline_ms=1000.0
    )
    # Row 1: eff 120 -> gap_high 0 -> excluded; row 0 and 2 remain
    assert len(out) == 2
    assert out[0]["operation"] == "aten::mm"
    assert out[0]["type"] == "kernel_tuning"
    assert out[0]["efficiency_pct"] == 50.0
    assert out[0]["impact_score"] == 0.44
    assert out[0]["impact_score_high"] == 0.5
    assert out[0]["impact_score_low"] == 0.38
    assert out[1]["operation"] == "aten::bmm"
    assert out[1]["efficiency_pct"] == 90.0
    assert out[1]["impact_score_high"] == 0.02


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
            "Pct Roofline_mean": [60.0],  # roofline floor
        }
    )
    metadata = {"peak_hbm_bw_tbs": 5.3, "peak_bf16_maf_tflops": 700.0}
    operations = build_operation_metrics(
        df, metadata, {}, comparison_scope="comparative"
    )
    mm_op = operations[0]
    eff = mm_op["efficiency"]
    # efficiency_percent clamped to roofline (60), not raw comp_pct (25)
    assert eff["efficiency_percent"] == 60.0
    assert eff["warning"] is not None
    assert "ROOFLINE CAP" in eff["warning"]
    # gap_high = 1 - 60/100 = 0.4; impact_high = 0.4 * 10/1000 * 100 = 0.4 (not 0.75 from unclamped 25%)
    out = compute_impact_estimates(
        operations, "gemm", min_impact_score=0.01, baseline_ms=1000.0
    )
    assert len(out) == 1
    assert out[0]["impact_score_high"] == 0.4
    assert out[0]["impact_score"] == 0.35


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
    operations = build_operation_metrics(
        df, metadata, {}, comparison_scope="comparative"
    )
    eff = operations[0]["efficiency"]
    assert eff["efficiency_percent"] == 80.0
    assert eff["warning"] is None
    out = compute_impact_estimates(
        operations, "gemm", min_impact_score=0.01, baseline_ms=1000.0
    )
    assert out[0]["impact_score_high"] == 0.2


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
    operations = build_operation_metrics(
        df, metadata, {}, comparison_scope="comparative"
    )
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
        operations, "gemm", min_impact_score=0.01, baseline_ms=100.0
    )
    assert len(estimates) == 0


def test_other_analysis_customcollective_keeps_communication_classified_ops(
    output_dir_other_and_customcollective_nccl,
):
    """Regression: do not apply the communication pre-filter when category != other."""
    script = os.path.join(ANALYSIS_DIR, "category_analyses", "other_analysis.py")
    if not os.path.isfile(script):
        pytest.skip("other_analysis.py not found")

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            output_dir_other_and_customcollective_nccl,
            "--category",
            "customcollective",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")

    metrics_path = os.path.join(
        output_dir_other_and_customcollective_nccl,
        "category_data",
        "customcollective_metrics.json",
    )
    assert os.path.isfile(metrics_path)
    with open(metrics_path) as f:
        m = json.load(f)
    assert m.get("status") == "OK"
    assert m.get("category") == "customcollective"
    assert len(m.get("operations", [])) == 1
    assert m["operations"][0]["name"] == "ncclKernel_AllReduce_RING_LL"
    assert "communication_ops_skipped" not in m.get("category_specific", {})


def test_other_analysis_other_category_still_skips_communication_ops(
    output_dir_other_and_customcollective_nccl,
):
    script = os.path.join(ANALYSIS_DIR, "category_analyses", "other_analysis.py")
    if not os.path.isfile(script):
        pytest.skip("other_analysis.py not found")

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            output_dir_other_and_customcollective_nccl,
            "--category",
            "other",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")

    metrics_path = os.path.join(
        output_dir_other_and_customcollective_nccl,
        "category_data",
        "other_metrics.json",
    )
    assert os.path.isfile(metrics_path)
    with open(metrics_path) as f:
        m = json.load(f)
    assert m.get("status") == "OK"
    assert len(m.get("operations", [])) == 0
    skipped = m.get("category_specific", {}).get("communication_ops_skipped", {})
    assert skipped.get("count") == 1
    assert "ncclKernel_AllReduce_RING_LL" in (skipped.get("op_names") or [])


# ----- Category analysis script: gemm_analysis runs with minimal data -----


def test_gemm_analysis_script_with_minimal_data(output_dir_with_category_data):
    """Run gemm_analysis.py --output-dir <dir> with pre-created gemm_ops.csv + metadata."""
    script = os.path.join(ANALYSIS_DIR, "category_analyses", "gemm_analysis.py")
    if not os.path.isfile(script):
        pytest.skip("gemm_analysis.py not found")

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


# ----- Unit tests: analysis_utils.build_category_findings -----


def test_build_category_findings_quantified_grouping_standalone():
    """Quantified estimates group by (bound, library, eff_bucket); sums impact."""
    estimates = [
        {
            "operation": "a",
            "bound_type": "compute",
            "library": "Tensile",
            "efficiency_pct": 20.0,
            "impact_score": 3.0,
            "impact_score_low": 2.0,
            "impact_score_high": 4.0,
        },
        {
            "operation": "b",
            "bound_type": "compute",
            "library": "Tensile",
            "efficiency_pct": 25.0,
            "impact_score": 1.0,
            "impact_score_low": 0.5,
            "impact_score_high": 1.5,
        },
    ]
    findings = build_category_findings(estimates, "standalone")
    assert len(findings) == 1
    f = findings[0]
    assert f["bound_type"] == "compute"
    assert f["library"] == "Tensile"
    assert f["eff_bucket"] == "0-30"
    assert f["impact_score"] == 4.0
    assert f["member_count"] == 2
    assert f["rank"] == 1


def test_build_category_findings_drops_below_min_pitem():
    """A group summing below MIN_PITEM_IMPACT_SCORE (0.5) is dropped."""
    estimates = [
        {
            "operation": "tiny",
            "bound_type": "memory",
            "library": "CK",
            "efficiency_pct": 70.0,
            "impact_score": 0.2,
            "impact_score_low": 0.1,
            "impact_score_high": 0.3,
        },
    ]
    assert build_category_findings(estimates, "standalone") == []


def test_build_category_findings_rank_assignment():
    """Findings sort by impact desc and receive contiguous ranks."""
    estimates = [
        {
            "operation": "small",
            "bound_type": "memory",
            "library": "CK",
            "efficiency_pct": 50.0,
            "impact_score": 1.0,
            "impact_score_low": 0.5,
            "impact_score_high": 1.5,
        },
        {
            "operation": "big",
            "bound_type": "compute",
            "library": "Tensile",
            "efficiency_pct": 10.0,
            "impact_score": 5.0,
            "impact_score_low": 4.0,
            "impact_score_high": 6.0,
        },
    ]
    findings = build_category_findings(estimates, "standalone")
    assert [f["rank"] for f in findings] == [1, 2]
    assert findings[0]["library"] == "Tensile"
    assert findings[1]["library"] == "CK"


def test_build_category_findings_heuristic_grouping():
    """Heuristic estimates group by op-name and land in an 'unmodeled' finding."""
    estimates = [
        {
            "operation": "triton_kernel",
            "estimate_method": "heuristic",
            "library": "Triton",
            "impact_score": 0.4,
            "impact_score_low": 0.2,
            "impact_score_high": 0.6,
            "percent_of_total": 2.0,
        },
        {
            "operation": "triton_kernel",
            "estimate_method": "heuristic",
            "library": "Triton",
            "impact_score": 0.3,
            "impact_score_low": 0.15,
            "impact_score_high": 0.45,
            "percent_of_total": 1.5,
        },
    ]
    findings = build_category_findings(estimates, "standalone")
    assert len(findings) == 1
    f = findings[0]
    assert f["estimate_method"] == "heuristic"
    assert f["bound_type"] == "unmodeled"
    assert f["impact_score"] == 0.7
    assert f["percent_of_total"] == 3.5
    assert f["member_count"] == 2


def test_build_category_findings_comparative_ignores_eff_bucket():
    """Comparative scope groups quantified estimates without an efficiency bucket."""
    estimates = [
        {
            "operation": "a",
            "bound_type": "compute",
            "library": "Tensile",
            "efficiency_pct": 20.0,
            "impact_score": 2.0,
            "impact_score_low": 1.0,
            "impact_score_high": 3.0,
        },
        {
            "operation": "b",
            "bound_type": "compute",
            "library": "Tensile",
            "efficiency_pct": 90.0,
            "impact_score": 1.0,
            "impact_score_low": 0.5,
            "impact_score_high": 1.5,
        },
    ]
    findings = build_category_findings(estimates, "comparative")
    assert len(findings) == 1
    assert findings[0]["eff_bucket"] == "all"
    assert findings[0]["member_count"] == 2


# ----- MarkerValidator.scan -----


def test_marker_scan_valid_p_item():
    text = (
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "body\n"
        "<!-- impact-end -->\n"
    )
    errors, seen = MarkerValidator.scan(text, "f.md")
    assert errors == []
    assert seen == {"p_item"}


def test_marker_scan_pairing_mismatch():
    text = (
        "<!-- impact-begin kind=p_item low=1 mid=2 high=3 -->\n"
        "<!-- impact-end -->\n"
        "<!-- impact-begin kind=top_ops -->\n"
    )
    errors, _ = MarkerValidator.scan(text, "f.md")
    assert any("marker pairing mismatch" in e for e in errors)


def test_marker_scan_missing_kind_attr():
    text = "<!-- impact-begin low=1 mid=2 high=3 -->\n<!-- impact-end -->\n"
    errors, seen = MarkerValidator.scan(text, "f.md")
    assert any("missing kind= attribute" in e for e in errors)
    assert seen == set()


def test_marker_scan_unknown_kind():
    text = "<!-- impact-begin kind=bogus low=1 mid=2 high=3 -->\n<!-- impact-end -->\n"
    errors, _ = MarkerValidator.scan(text, "f.md")
    assert any("unknown kind=bogus" in e for e in errors)


def test_marker_scan_missing_required_attr():
    text = "<!-- impact-begin kind=p_item low=1 mid=2 -->\n<!-- impact-end -->\n"
    errors, _ = MarkerValidator.scan(text, "f.md")
    assert any("kind=p_item missing required attr high" in e for e in errors)


def test_marker_scan_detail_estimate_required_attrs():
    # detail_estimate requires low + high (not mid).
    text = "<!-- impact-begin kind=detail_estimate low=1 -->\n<!-- impact-end -->\n"
    errors, _ = MarkerValidator.scan(text, "f.md")
    assert any("kind=detail_estimate missing required attr high" in e for e in errors)


def test_marker_scan_mixed_null_numeric():
    text = (
        "<!-- impact-begin kind=p_item low=null mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )
    errors, _ = MarkerValidator.scan(text, "f.md")
    assert any("mixes null and numeric" in e for e in errors)


def test_marker_scan_all_null_is_allowed():
    text = (
        "<!-- impact-begin kind=p_item low=null mid=null high=null -->\n"
        "<!-- impact-end -->\n"
    )
    errors, seen = MarkerValidator.scan(text, "f.md")
    assert errors == []
    assert seen == {"p_item"}


# ----- MarkerValidator.check_findings_file -----


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _valid_category_findings(rank=1):
    return (
        f"<!-- reasoning-candidate tier=compute rank={rank} -->\n"
        f"### P{rank}: Do the thing\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )


def test_check_findings_file_valid_category(tmp_path):
    p = tmp_path / "gemm_findings.md"
    _write(str(p), _valid_category_findings())
    errors = MarkerValidator.check_findings_file(str(p), "category_findings")
    assert errors == []


def test_check_findings_file_missing_p_item(tmp_path):
    # A reasoning-candidate with only a detail_estimate marker (no p_item).
    text = (
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )
    p = tmp_path / "gemm_findings.md"
    _write(str(p), text)
    errors = MarkerValidator.check_findings_file(str(p), "category_findings")
    assert any("missing required kind=p_item" in e for e in errors)


def test_check_findings_file_skip_p_item_required(tmp_path):
    # skip_p_item_required suppresses the p_item requirement (empty category).
    text = (
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "not quantifiable from trace data\n"
    )
    p = tmp_path / "gemm_findings.md"
    _write(str(p), text)
    errors = MarkerValidator.check_findings_file(
        str(p), "category_findings", skip_p_item_required=True
    )
    assert errors == []


def test_check_findings_file_detail_estimate_missing(tmp_path):
    # reasoning-candidate with a p_item but no detail_estimate / sentinel.
    text = (
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "### P1: Do the thing\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )
    p = tmp_path / "gemm_findings.md"
    _write(str(p), text)
    errors = MarkerValidator.check_findings_file(str(p), "category_findings")
    assert any("missing" in e and "detail_estimate" in e for e in errors)


# ----- MarkerValidator.check_report -----


def test_check_report_requires_top_ops(tmp_path):
    text = (
        "<!-- impact-begin kind=p_item low=1 mid=2 high=3 -->\n" "<!-- impact-end -->\n"
    )
    p = tmp_path / "analysis.md"
    _write(str(p), text)
    errors = MarkerValidator.check_report(str(p))
    assert any("missing required kind=top_ops" in e for e in errors)


def test_check_report_with_top_ops_ok(tmp_path):
    text = "<!-- impact-begin kind=top_ops -->\n| a |\n<!-- impact-end -->\n"
    p = tmp_path / "analysis.md"
    _write(str(p), text)
    errors = MarkerValidator.check_report(str(p))
    assert errors == []


# ----- validate_report + _validate_report_priority_consistency (R1-R4) -----


def _passing_report():
    return """# Analysis Report

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Time | 1000 ms |
| Compute % | 99.8% |
| Idle % | 0.2% |
| Exposed Communication % | 0.05% |
| Top Bottleneck Category | gemm |

## Compute Kernel Optimizations

### P1: Optimize gemm
<!-- impact-begin kind=p_item category=gemm low=85.0 mid=100.0 high=115.0 -->
Some analysis text goes here to describe the gemm bottleneck in detail.
<!-- impact-end -->

### P2: Optimize sdpa
<!-- impact-begin kind=p_item category=sdpa_fwd low=42.0 mid=50.0 high=58.0 -->
Some analysis text goes here to describe the sdpa bottleneck in detail.
<!-- impact-end -->

## Kernel Fusion Opportunities (Experimental)

No fusion opportunities identified in this trace.

## System-Level Optimizations

Top Operations:

<!-- impact-begin kind=top_ops -->

| Op | Time |
|----|------|
| gemm | 100 |
| sdpa_fwd | 50 |

<!-- impact-end -->

## Detailed Analysis

Detailed narrative for each bottleneck lives here.

## Appendix

Supplementary reference material for the analysis run.
"""


def _priority_data_two():
    return {
        "findings": [
            {
                "category": "gemm",
                "global_rank": 1,
                "impact_score": 100.0,
                "impact_score_low": 85.0,
                "impact_score_high": 115.0,
            },
            {
                "category": "sdpa_fwd",
                "global_rank": 2,
                "impact_score": 50.0,
                "impact_score_low": 42.0,
                "impact_score_high": 58.0,
            },
        ],
        "priorities": [
            {"rank": 1, "category": "gemm"},
            {"rank": 2, "category": "sdpa_fwd"},
        ],
    }


def test_validate_report_passes(tmp_path):
    _write(str(tmp_path / "analysis.md"), _passing_report())
    (tmp_path / "priority_data.json").write_text(json.dumps(_priority_data_two()))
    passed, missing = validate_report(str(tmp_path))
    assert passed, missing
    assert missing == []


def test_validate_report_missing_file(tmp_path):
    passed, missing = validate_report(str(tmp_path))
    assert not passed
    assert missing == ["<file not found>"]


def test_validate_report_missing_section(tmp_path):
    # Drop the Appendix header.
    content = _passing_report().replace("## Appendix", "## Notes")
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(json.dumps(_priority_data_two()))
    passed, missing = validate_report(str(tmp_path))
    assert not passed
    assert any("Missing section: Appendix" in m for m in missing)


def test_validate_report_placeholder_metrics(tmp_path):
    content = _passing_report().replace(
        "| Total Time | 1000 ms |", "| Total Time | X ms |"
    )
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(json.dumps(_priority_data_two()))
    passed, missing = validate_report(str(tmp_path))
    assert not passed
    assert any("Placeholder value 'X ms'" in m for m in missing)


def test_validate_report_r3_marker_value_mismatch(tmp_path):
    # Corrupt the first p_item mid so it disagrees with findings[0].impact_score.
    content = _passing_report().replace("mid=100.0", "mid=999.0")
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(json.dumps(_priority_data_two()))
    passed, missing = validate_report(str(tmp_path))
    assert not passed
    assert any(m.startswith("R3:") for m in missing)


def test_priority_consistency_r1_heading_count_mismatch(tmp_path):
    # Report has 2 P-items but priority_data declares 3 findings -> R1.
    content = _passing_report()
    pd_data = _priority_data_two()
    pd_data["findings"].append(
        {
            "category": "reduce",
            "global_rank": 3,
            "impact_score": 10.0,
            "impact_score_low": 8.0,
            "impact_score_high": 12.0,
        }
    )
    (tmp_path / "priority_data.json").write_text(json.dumps(pd_data))
    errors = _validate_report_priority_consistency(content, str(tmp_path))
    assert any(e.startswith("R1:") for e in errors)


def test_priority_consistency_r2_category_mismatch(tmp_path):
    content = _passing_report().replace("category=gemm", "category=WRONG")
    (tmp_path / "priority_data.json").write_text(json.dumps(_priority_data_two()))
    errors = _validate_report_priority_consistency(content, str(tmp_path))
    assert any(e.startswith("R2:") for e in errors)


def test_priority_consistency_r4_top_ops_row_count(tmp_path):
    # Only one priority but the top_ops table has two data rows -> R4.
    content = _passing_report()
    pd_data = {
        "findings": _priority_data_two()["findings"],
        "priorities": [{"rank": 1, "category": "gemm"}],
    }
    (tmp_path / "priority_data.json").write_text(json.dumps(pd_data))
    errors = _validate_report_priority_consistency(content, str(tmp_path))
    assert any(e.startswith("R4:") for e in errors)


def test_priority_consistency_absent_json_is_silent(tmp_path):
    errors = _validate_report_priority_consistency(_passing_report(), str(tmp_path))
    assert errors == []


# ----- _check_priority_consistency (INV1 / INV2 / INV3 / INV7') -----


def _write_priority_data(output_dir, data):
    with open(os.path.join(output_dir, "priority_data.json"), "w") as f:
        json.dump(data, f)


def test_check_priority_consistency_pass(tmp_path):
    data = {
        "findings": [
            {"category": "gemm", "global_rank": 1, "impact_score": 100.0},
            {"category": "gemm", "global_rank": 2, "impact_score": 30.0},
        ],
        "priorities": [
            {
                "rank": 1,
                "category": "gemm",
                "source": "findings_rollup",
                "impact_score": 130.0,
            },
        ],
    }
    _write_priority_data(str(tmp_path), data)
    result = _check_priority_consistency(str(tmp_path), {})
    assert result["status"] == "PASS"
    assert result["messages"] == []


def test_check_priority_consistency_inv1_unsorted(tmp_path):
    data = {
        "findings": [
            {"category": "gemm", "global_rank": 1, "impact_score": 30.0},
            {"category": "gemm", "global_rank": 2, "impact_score": 100.0},
        ],
        "priorities": [],
    }
    _write_priority_data(str(tmp_path), data)
    result = _check_priority_consistency(str(tmp_path), {})
    assert result["status"] == "WARN"
    assert any("INV1" in m for m in result["messages"])


def test_check_priority_consistency_inv2_rank_gap(tmp_path):
    data = {
        "findings": [
            {"category": "gemm", "global_rank": 1, "impact_score": 100.0},
            {"category": "gemm", "global_rank": 5, "impact_score": 30.0},
        ],
        "priorities": [],
    }
    _write_priority_data(str(tmp_path), data)
    result = _check_priority_consistency(str(tmp_path), {})
    assert any("INV2" in m for m in result["messages"])


def test_check_priority_consistency_inv3_priority_rank_gap(tmp_path):
    data = {
        "findings": [],
        "priorities": [
            {"rank": 1, "category": "gemm"},
            {"rank": 3, "category": "sdpa_fwd"},
        ],
    }
    _write_priority_data(str(tmp_path), data)
    result = _check_priority_consistency(str(tmp_path), {})
    assert any("INV3" in m for m in result["messages"])


def test_check_priority_consistency_inv7_rollup_mismatch(tmp_path):
    data = {
        "findings": [
            {"category": "gemm", "global_rank": 1, "impact_score": 100.0},
            {"category": "gemm", "global_rank": 2, "impact_score": 30.0},
        ],
        "priorities": [
            {
                "rank": 1,
                "category": "gemm",
                "source": "findings_rollup",
                "impact_score": 999.0,
            },
        ],
    }
    _write_priority_data(str(tmp_path), data)
    result = _check_priority_consistency(str(tmp_path), {})
    assert any("INV7'" in m for m in result["messages"])


def test_check_priority_consistency_inv7_excludes_heuristic(tmp_path):
    # Rollup sum must exclude heuristic-estimated findings.
    data = {
        "findings": [
            {
                "category": "gemm",
                "global_rank": 1,
                "impact_score": 100.0,
                "estimate_method": "quantified",
            },
            {
                "category": "gemm",
                "global_rank": 2,
                "impact_score": 50.0,
                "estimate_method": "heuristic",
            },
        ],
        "priorities": [
            {
                "rank": 1,
                "category": "gemm",
                "source": "findings_rollup",
                "impact_score": 100.0,
            },
        ],
    }
    _write_priority_data(str(tmp_path), data)
    result = _check_priority_consistency(str(tmp_path), {})
    assert result["status"] == "PASS"


def test_check_priority_consistency_missing_file(tmp_path):
    result = _check_priority_consistency(str(tmp_path), {})
    assert result["status"] == "WARN"
    assert any("not found" in m for m in result["messages"])


# ----- _validate_compute_data_tables -----


def _setup_compute_findings(tmp_path, content, operations):
    output_dir = tmp_path
    (output_dir / "category_findings").mkdir(parents=True)
    (output_dir / "category_data").mkdir(parents=True)
    findings_path = output_dir / "category_findings" / "gemm_findings.md"
    findings_path.write_text(content)
    metrics = {"operations": operations}
    (output_dir / "category_data" / "gemm_metrics.json").write_text(json.dumps(metrics))
    return str(findings_path)


_STANDALONE_HEADER = (
    "| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | "
    "Count | FLOPS/Byte | Efficiency | Bound |"
)
_SEP = "|---|---|---|---|---|---|---|---|---|---|"


def _compute_block(row):
    return (
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "**Data:**\n\n"
        f"{_STANDALONE_HEADER}\n"
        f"{_SEP}\n"
        f"{row}\n"
    )


def test_compute_data_tables_valid(tmp_path):
    row = (
        "| aten::mm | M=2,N=3 | path/to/launch | mm_kernel | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |"
    )
    ops = [
        {
            "args": "M=2,N=3",
            "launcher_path": "path/to/launch",
            "kernel_name_trunc": "mm_kernel",
        }
    ]
    fp = _setup_compute_findings(tmp_path, _compute_block(row), ops)
    with open(fp) as f:
        content = f.read()
    errors = _validate_compute_data_tables(content, fp)
    assert errors == []


def test_compute_data_tables_args_mismatch(tmp_path):
    row = (
        "| aten::mm | WRONG_ARGS | path/to/launch | mm_kernel | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |"
    )
    ops = [
        {
            "args": "M=2,N=3",
            "launcher_path": "path/to/launch",
            "kernel_name_trunc": "mm_kernel",
        }
    ]
    fp = _setup_compute_findings(tmp_path, _compute_block(row), ops)
    with open(fp) as f:
        content = f.read()
    errors = _validate_compute_data_tables(content, fp)
    assert any("Args cell" in e for e in errors)


def test_compute_data_tables_kernel_path_mismatch(tmp_path):
    row = (
        "| aten::mm | M=2,N=3 | WRONG_PATH | mm_kernel | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |"
    )
    ops = [
        {
            "args": "M=2,N=3",
            "launcher_path": "path/to/launch",
            "kernel_name_trunc": "mm_kernel",
        }
    ]
    fp = _setup_compute_findings(tmp_path, _compute_block(row), ops)
    with open(fp) as f:
        content = f.read()
    errors = _validate_compute_data_tables(content, fp)
    assert any("Kernel Path cell" in e for e in errors)


def test_compute_data_tables_header_order_wrong(tmp_path):
    # Swap Args and Kernel Path columns in the header.
    bad_header = (
        "| Operation | Kernel Path | Args | Kernel Name | Time (ms) | %E2E | "
        "Count | FLOPS/Byte | Efficiency | Bound |"
    )
    content = (
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "**Data:**\n\n"
        f"{bad_header}\n"
        f"{_SEP}\n"
        "| aten::mm | path/to/launch | M=2,N=3 | mm_kernel | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |\n"
    )
    ops = [{"args": "M=2,N=3", "launcher_path": "path/to/launch"}]
    fp = _setup_compute_findings(tmp_path, content, ops)
    with open(fp) as f:
        text = f.read()
    errors = _validate_compute_data_tables(text, fp)
    assert any("canonical columns" in e for e in errors)


def test_compute_data_tables_missing_data_table(tmp_path):
    content = (
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "No data table in this block.\n"
    )
    ops = [{"args": "M=2,N=3"}]
    fp = _setup_compute_findings(tmp_path, content, ops)
    with open(fp) as f:
        text = f.read()
    errors = _validate_compute_data_tables(text, fp)
    assert any("no **Data:** table found" in e for e in errors)


# ----- _compute_data_in_out -----


def test_compute_data_in_out_gemm_split():
    # M=2, K=3, N=4 -> total = M*K + K*N + M*N = 6 + 12 + 8 = 26
    # data_in = dm*(M*K + K*N)/total = 10*18/26; data_out = dm*M*N/total = 10*8/26
    din, dout = _compute_data_in_out("GEMM", "{'M': 2, 'N': 4, 'K': 3}", 10.0)
    assert din == pytest.approx(10.0 * 18 / 26)
    assert dout == pytest.approx(10.0 * 8 / 26)
    assert din + dout == pytest.approx(10.0)


def test_compute_data_in_out_gemm_missing_dim_falls_back_to_half():
    # A missing/zero dim means not all(M, N, K) -> half split fallback.
    din, dout = _compute_data_in_out("GEMM", "{'M': 2, 'N': 4}", 10.0)
    assert din == pytest.approx(5.0)
    assert dout == pytest.approx(5.0)


def test_compute_data_in_out_reduce_all_in():
    din, dout = _compute_data_in_out("reduce", "{}", 8.0)
    assert din == 8.0
    assert dout == 0.0


def test_compute_data_in_out_elementwise_two_thirds_one_third():
    din, dout = _compute_data_in_out("elementwise", "{'shape_in1': [4, 4]}", 9.0)
    assert din == pytest.approx(6.0)
    assert dout == pytest.approx(3.0)


def test_compute_data_in_out_elementwise_without_shape_in1_is_half():
    din, dout = _compute_data_in_out("elementwise", "{'other': 1}", 9.0)
    assert din == pytest.approx(4.5)
    assert dout == pytest.approx(4.5)


def test_compute_data_in_out_literal_eval_failure_returns_half():
    din, dout = _compute_data_in_out("GEMM", "not-a-dict {{", 10.0)
    assert din == pytest.approx(5.0)
    assert dout == pytest.approx(5.0)


def test_compute_data_in_out_none_data_moved_is_none():
    din, dout = _compute_data_in_out("GEMM", "{'M': 2, 'N': 4, 'K': 3}", None)
    assert din is None
    assert dout is None


def test_compute_data_in_out_unknown_category_is_half():
    din, dout = _compute_data_in_out("mystery", "{}", 4.0)
    assert din == pytest.approx(2.0)
    assert dout == pytest.approx(2.0)


# ----- _is_case_a_fusion_gap -----


def _kernels(n):
    return [{"name": f"k{i}", "dur_us": 1.0} for i in range(n)]


def test_case_a_gap_pass():
    assert _is_case_a_fusion_gap(_kernels(5), _kernels(3)) is True


def test_case_a_gap_g1_n1_not_greater():
    # n1 <= n2 fails G1
    assert _is_case_a_fusion_gap(_kernels(3), _kernels(3)) is False
    assert _is_case_a_fusion_gap(_kernels(2), _kernels(3)) is False


def test_case_a_gap_g2_delta_too_large():
    # delta = 17 > 15 (n1=18, n2=1) fails G2
    assert _is_case_a_fusion_gap(_kernels(18), _kernels(1)) is False


def test_case_a_gap_g3_n1_too_large():
    # n1 = 31 > 30 fails G3 (delta small: 31 - 20 = 11 <= 15)
    assert _is_case_a_fusion_gap(_kernels(31), _kernels(20)) is False


def test_case_a_gap_g4_n2_zero():
    # n2 = 0 fails G4
    assert _is_case_a_fusion_gap(_kernels(3), _kernels(0)) is False


def test_case_a_gap_boundaries_inclusive():
    # delta exactly 15 (n1=16,n2=1) and n1 exactly 30 both pass.
    assert _is_case_a_fusion_gap(_kernels(16), _kernels(1)) is True
    assert _is_case_a_fusion_gap(_kernels(30), _kernels(20)) is True


# ----- _gpu_utilization_metrics_from_gpu_timeline_df -----


def test_gpu_utilization_metrics_full_rows():
    df = pd.DataFrame(
        {
            "type": [
                "total_time",
                "computation_time",
                "exposed_comm_time",
                "exposed_memcpy_time",
                "idle_time",
            ],
            "time ms": [1000.0, 998.0, 0.5, 0.1, 2.0],
            "percent": [100.0, 99.8, 0.05, 0.01, 0.2],
        }
    )
    m = _gpu_utilization_metrics_from_gpu_timeline_df(df)
    assert m["total_time_ms"] == 1000.0
    assert m["computation_time_percent"] == 99.8
    assert m["exposed_comm_time_percent"] == 0.05
    assert m["exposed_memcpy_time_percent"] == 0.01
    assert m["idle_time_percent"] == 0.2


def test_gpu_utilization_metrics_missing_rows_default_zero():
    df = pd.DataFrame(
        {
            "type": ["total_time"],
            "time ms": [500.0],
            "percent": [100.0],
        }
    )
    m = _gpu_utilization_metrics_from_gpu_timeline_df(df)
    assert m["total_time_ms"] == 500.0
    assert m["computation_time_percent"] == 0
    assert m["exposed_comm_time_percent"] == 0
    assert m["exposed_memcpy_time_percent"] == 0
    assert m["idle_time_percent"] == 0


# ----- _normalize_category -----


def test_normalize_category_plain():
    assert _normalize_category({"op category": "GEMM"}) == ("gemm", "GEMM")


def test_normalize_category_spaces_and_slashes():
    key, display = _normalize_category({"op category": "SDPA Fwd/Bwd"})
    assert key == "sdpa_fwd_bwd"
    assert display == "SDPA Fwd/Bwd"


def test_normalize_category_empty_is_other():
    assert _normalize_category({"op category": ""}) == ("other", "Other")


def test_normalize_category_missing_key_is_other():
    assert _normalize_category({}) == ("other", "Other")


# ----- _strip_module_index -----


def test_strip_module_index_removes_nn_module_prefix_and_index():
    assert _strip_module_index("nn.Module: LlamaMLP_12") == "LlamaMLP"


def test_strip_module_index_plain_name_with_index():
    assert _strip_module_index("some_op_3") == "some_op"


def test_strip_module_index_no_trailing_index():
    assert _strip_module_index("aten::mm") == "aten::mm"


# ----- _has_fused_kernel -----


def test_has_fused_kernel_true():
    kernels = [{"name": "regular_mm"}, {"name": "flash_attn_fwd"}]
    assert _has_fused_kernel(kernels) is True


def test_has_fused_kernel_false():
    kernels = [{"name": "regular_mm"}, {"name": "elementwise_add"}]
    assert _has_fused_kernel(kernels) is False


def test_has_fused_kernel_case_insensitive():
    assert _has_fused_kernel([{"name": "Some_FMHA_Kernel"}]) is True


# ----- _dedup_by_kernel_set -----


def test_dedup_by_kernel_set_keeps_higher_score():
    c_low = {"kernels": [{"name": "a"}, {"name": "b"}], "score": 1}
    c_high = {"kernels": [{"name": "a"}, {"name": "b"}], "score": 5}
    out = _dedup_by_kernel_set(
        [c_low, c_high], kernels_field="kernels", score_fn=lambda c: c["score"]
    )
    assert len(out) == 1
    assert out[0]["score"] == 5


def test_dedup_by_kernel_set_distinct_sets_both_kept():
    c1 = {"kernels": [{"name": "a"}, {"name": "b"}], "score": 1}
    c2 = {"kernels": [{"name": "a"}, {"name": "c"}], "score": 1}
    out = _dedup_by_kernel_set(
        [c1, c2], kernels_field="kernels", score_fn=lambda c: c["score"]
    )
    assert len(out) == 2


def test_dedup_by_kernel_set_skips_empty_kernels():
    c_empty = {"kernels": [], "score": 9}
    out = _dedup_by_kernel_set(
        [c_empty], kernels_field="kernels", score_fn=lambda c: c["score"]
    )
    assert out == []


def test_dedup_by_kernel_set_kernel_name_variant():
    # Kernel dicts using the "kernel_name" key instead of "name".
    c1 = {"seq": [{"kernel_name": "a"}, {"kernel_name": "b"}], "score": 2}
    c2 = {"seq": [{"kernel_name": "a"}, {"kernel_name": "b"}], "score": 7}
    out = _dedup_by_kernel_set(
        [c1, c2], kernels_field="seq", score_fn=lambda c: c["score"]
    )
    assert len(out) == 1
    assert out[0]["score"] == 7


# ----- _prefix_lookup -----


def test_prefix_lookup_exact_match():
    lookup = {"my_kernel": {"data": 1}}
    assert _prefix_lookup(lookup, "my_kernel") == {"data": 1}


def test_prefix_lookup_query_startswith_csv_name():
    # kname longer than the (truncated) csv key.
    lookup = {"my_kernel": {"data": 1}}
    assert _prefix_lookup(lookup, "my_kernel_specialization") == {"data": 1}


def test_prefix_lookup_csv_name_startswith_query():
    lookup = {"my_kernel_full_name": {"data": 1}}
    assert _prefix_lookup(lookup, "my_kernel") == {"data": 1}


def test_prefix_lookup_no_match_returns_none():
    lookup = {"my_kernel": {"data": 1}}
    assert _prefix_lookup(lookup, "unrelated") is None


# ----- _extract_attention_core -----


def test_extract_attention_core_narrows_to_qk_softmax_pv():
    kernels = [
        {"name": "pre_kernel"},
        {"name": "qk_gemm"},
        {"name": "softmax_kernel"},
        {"name": "pv_gemm"},
        {"name": "post_kernel"},
    ]
    perf_lookup = {
        "qk_gemm": {"s": {"op_category": "GEMM"}},
        "pv_gemm": {"s": {"op_category": "GEMM"}},
    }
    core = _extract_attention_core(kernels, perf_lookup)
    assert [k["name"] for k in core] == ["qk_gemm", "softmax_kernel", "pv_gemm"]


def test_extract_attention_core_no_softmax_returns_none():
    kernels = [{"name": "qk_gemm"}, {"name": "pv_gemm"}]
    perf_lookup = {
        "qk_gemm": {"s": {"op_category": "GEMM"}},
        "pv_gemm": {"s": {"op_category": "GEMM"}},
    }
    assert _extract_attention_core(kernels, perf_lookup) is None


def test_extract_attention_core_softmax_without_flanking_gemms_returns_none():
    kernels = [{"name": "softmax_kernel"}, {"name": "elementwise_add"}]
    perf_lookup = {}
    assert _extract_attention_core(kernels, perf_lookup) is None


# ----- orchestrator_prepare: _build_diff_stats_lookups -----


def _diff_stats_df():
    return pd.DataFrame(
        {
            "name": ["Cijk_A", "Cijk_B", "some_op", "Cijk_A"],
            "source": ["trace1", "trace1", "trace2", "trace2"],
            "lowest_common_ancestor_id": [10, 11, 10, 11],
            "kernel_time": [5.0, 3.0, 2.0, 1.0],
            "gpu_op_uid": [100, 101, None, None],
        }
    )


def test_build_diff_stats_lookups_splits_trace1_and_trace2():
    uid_to_t1, lca_to_t2 = _build_diff_stats_lookups(_diff_stats_df())
    assert set(uid_to_t1) == {100, 101}
    assert uid_to_t1[100]["type"] == "GEMM"
    assert uid_to_t1[100]["dur_us"] == 5.0
    assert uid_to_t1[100]["lca_id"] == 10
    assert set(lca_to_t2) == {10, 11}
    assert len(lca_to_t2[10]) == 1
    assert lca_to_t2[10][0]["name"] == "some_op"


def test_build_diff_stats_lookups_missing_uid_col_skips_trace1():
    df = _diff_stats_df().drop(columns=["gpu_op_uid"])
    uid_to_t1, lca_to_t2 = _build_diff_stats_lookups(df)
    # No gpu_op_uid column -> no trace1 kernels captured, trace2 still mapped.
    assert uid_to_t1 == {}
    assert set(lca_to_t2) == {10, 11}


def test_build_diff_stats_lookups_nan_uid_row_skipped():
    df = pd.DataFrame(
        {
            "name": ["Cijk_A", "Cijk_B"],
            "source": ["trace1", "trace1"],
            "lowest_common_ancestor_id": [10, 11],
            "kernel_time": [5.0, 3.0],
            "gpu_op_uid": [100, None],
        }
    )
    uid_to_t1, _ = _build_diff_stats_lookups(df)
    assert set(uid_to_t1) == {100}


def test_build_diff_stats_lookups_missing_kernel_time_defaults_zero():
    df = pd.DataFrame(
        {
            "name": ["Cijk_A"],
            "source": ["trace1"],
            "lowest_common_ancestor_id": [10],
            "kernel_time": [None],
            "gpu_op_uid": [100],
        }
    )
    uid_to_t1, _ = _build_diff_stats_lookups(df)
    assert uid_to_t1[100]["dur_us"] == 0.0


def test_build_diff_stats_lookups_multiple_trace2_same_lca():
    df = pd.DataFrame(
        {
            "name": ["k1", "k2"],
            "source": ["trace2", "trace2"],
            "lowest_common_ancestor_id": [10, 10],
            "kernel_time": [1.0, 2.0],
            "gpu_op_uid": [None, None],
        }
    )
    uid_to_t1, lca_to_t2 = _build_diff_stats_lookups(df)
    assert uid_to_t1 == {}
    assert len(lca_to_t2[10]) == 2


# ----- orchestrator_prepare: _apply_comparative_gates -----


def _typed_kernels(specs):
    return [{"name": n, "type": t, "dur_us": d} for (n, t, d) in specs]


def test_apply_comparative_gates_pass_with_eligible_delta():
    t1 = _typed_kernels(
        [("gemm_a", "GEMM", 1), ("gemm_b", "GEMM", 2), ("elemw", "X", 3)]
    )
    t2 = _typed_kernels([("gemm_a", "GEMM", 1)])
    assert _apply_comparative_gates(t1, t2) is True


def test_apply_comparative_gates_all_nccl_delta_false():
    t1 = _typed_kernels([("ncclX", "C", 1), ("ncclY", "C", 2), ("gemm", "GEMM", 3)])
    t2 = _typed_kernels([("gemm", "GEMM", 3)])
    assert _apply_comparative_gates(t1, t2) is False


def test_apply_comparative_gates_all_layout_transform_false():
    t1 = _typed_kernels([("batched_transpose_x", "X", 1), ("gemm", "GEMM", 2)])
    t2 = _typed_kernels([("gemm", "GEMM", 2)])
    assert _apply_comparative_gates(t1, t2) is False


def test_apply_comparative_gates_empty_delta_false():
    # n1 == n2 fails G1, so delta is empty and gates reject.
    t1 = _typed_kernels([("gemm_a", "GEMM", 1), ("gemm_b", "GEMM", 2)])
    assert _apply_comparative_gates(t1, t1) is False


# ----- orchestrator_prepare: _make_comparative_candidate -----


def test_make_comparative_candidate_full():
    t1 = _typed_kernels(
        [("gemm_a", "GEMM", 1), ("gemm_b", "GEMM", 2), ("elemw", "X", 3)]
    )
    t2 = _typed_kernels([("gemm_a", "GEMM", 1)])
    cand = _make_comparative_candidate("nn.Module: Foo_1", "Foo", t1, t2, lca_id=7)
    assert cand["kernel_count_trace1"] == 3
    assert cand["kernel_count_trace2"] == 1
    assert cand["delta"] == 2
    assert cand["lca_id"] == 7
    assert cand["kernel_type_summary"] == {"GEMM": 2, "X": 1}
    assert cand["parent_chain"] == []
    assert cand["total_kernel_time_us_trace1"] == 6
    assert cand["total_kernel_time_us_trace2"] == 1


def test_make_comparative_candidate_empty_t2_and_no_lca():
    t1 = _typed_kernels([("gemm_a", "GEMM", 1)])
    cand = _make_comparative_candidate("m", "b", t1, [])
    assert cand["total_kernel_time_us_trace2"] == 0
    assert "lca_id" not in cand
    assert cand["parent_chain"] == []


# ----- orchestrator_prepare: _is_gemm_norm_only -----


def test_is_gemm_norm_only_gemm_plus_layernorm_true():
    entry = {
        "kernels": [
            {"type": "GEMM", "name": "g"},
            {"type": "X", "name": "layernorm_k"},
        ]
    }
    assert _is_gemm_norm_only(entry) is True


def test_is_gemm_norm_only_gemm_only_false():
    assert _is_gemm_norm_only({"kernels": [{"type": "GEMM", "name": "g"}]}) is False


def test_is_gemm_norm_only_no_gemm_false():
    assert (
        _is_gemm_norm_only({"kernels": [{"type": "X", "name": "layernorm"}]}) is False
    )


def test_is_gemm_norm_only_gemm_plus_other_false():
    entry = {
        "kernels": [
            {"type": "GEMM", "name": "g"},
            {"type": "X", "name": "random"},
        ]
    }
    assert _is_gemm_norm_only(entry) is False


def test_is_gemm_norm_only_gemm_plus_elementwise_add_true():
    entry = {
        "kernels": [
            {"type": "GEMM", "name": "g"},
            {"type": "Elementwise Add", "name": "add"},
        ]
    }
    assert _is_gemm_norm_only(entry) is True


def test_is_gemm_norm_only_conv_counts_as_gemm():
    entry = {
        "kernels": [
            {"type": "X", "name": "conv_fwd"},
            {"type": "Y", "name": "layernorm"},
        ]
    }
    assert _is_gemm_norm_only(entry) is True


def test_is_gemm_norm_only_empty_false():
    assert _is_gemm_norm_only({"kernels": []}) is False


# ----- orchestrator_prepare: _is_fusion_eligible -----


@pytest.mark.parametrize(
    "name,expected",
    [
        ("regular_compute", True),
        ("some_nccl_kernel", False),
        ("flash_fwd_kernel", False),
        ("", True),
        ("FLASH_FWD", False),
    ],
)
def test_is_fusion_eligible(name, expected):
    assert _is_fusion_eligible(name) is expected


# ----- orchestrator_prepare: _build_parent_chain -----


class _ParentChainStubTree:
    """Minimal tree exposing get_parent_event for _build_parent_chain."""

    def __init__(self, parents):
        # parents maps id(event dict) -> parent dict (or None)
        self._parents = parents

    def get_parent_event(self, ev):
        return self._parents.get(id(ev))


def test_build_parent_chain_root_is_empty():
    ev = {"name": "leaf"}
    tree = _ParentChainStubTree({id(ev): None})
    assert _build_parent_chain(ev, tree) == []


def test_build_parent_chain_multi_level_cleans_names():
    ev = {"name": "leaf_ev"}
    leaf = {"name": "a/mm"}
    mid = {"name": "nn.Module: Foo_2"}
    root = {"name": "root"}
    tree = _ParentChainStubTree(
        {id(ev): leaf, id(leaf): mid, id(mid): root, id(root): None}
    )
    # Slash path keeps last segment, nn.Module prefix stripped, plain name kept.
    assert _build_parent_chain(ev, tree) == ["mm", "Foo_2", "root"]


# ----- orchestrator_prepare: _build_kernel_perf_lookup -----


def test_build_kernel_perf_lookup(tmp_path):
    csv = tmp_path / "unified_perf_summary.csv"
    pd.DataFrame(
        {
            "kernel_details_summary": ["[{'name': 'kA'}]", "[{'name': 'kB'}]"],
            "op category": ["GEMM", "reduce"],
            "Data Moved (MB)": [26.0, 8.0],
            "perf_params": ["{'M':2,'N':4,'K':3}", "{}"],
            "Input Dims": ["[[2,3]]", "[[4]]"],
        }
    ).to_csv(csv, index=False)
    lookup = _build_kernel_perf_lookup(str(csv))
    assert set(lookup) == {"kA", "kB"}
    a_entry = lookup["kA"][(2, 3)]
    assert a_entry["op_category"] == "GEMM"
    assert a_entry["data_in_mb"] == pytest.approx(18.0)
    assert a_entry["data_out_mb"] == pytest.approx(8.0)
    b_entry = lookup["kB"][(4,)]
    assert b_entry["data_in_mb"] == pytest.approx(8.0)
    assert b_entry["data_out_mb"] == 0.0


def test_build_kernel_perf_lookup_skips_nan_kernel_details(tmp_path):
    csv = tmp_path / "unified_perf_summary.csv"
    df = pd.DataFrame(
        {
            "kernel_details_summary": ["[{'name': 'kA'}]", None],
            "op category": ["GEMM", "reduce"],
            "Data Moved (MB)": [26.0, None],
            "perf_params": ["{'M':2,'N':4,'K':3}", None],
            "Input Dims": ["[[2,3]]", None],
        }
    )
    df.to_csv(csv, index=False)
    lookup = _build_kernel_perf_lookup(str(csv))
    assert set(lookup) == {"kA"}


# ----- orchestrator_prepare: _build_trace2_ops_summary_by_enhanced_category -----


def test_trace2_ops_summary_kernel_time_col(tmp_path):
    pd.DataFrame(
        {
            "op category": ["GEMM", "GEMM", "SDPA Fwd"],
            "Kernel Time (µs)_sum": [1000.0, 2000.0, 500.0],
            "operation_count": [1, 2, 3],
        }
    ).to_csv(tmp_path / "unified_perf_summary.csv", index=False)
    rows = _build_trace2_ops_summary_by_enhanced_category(str(tmp_path))
    gemm = next(r for r in rows if r["op category"] == "gemm")
    # µs -> ms and operation_count summed (1 + 2 = 3).
    assert gemm["total_direct_kernel_time_ms"] == pytest.approx(3.0)
    assert gemm["Count"] == 3
    sdpa = next(r for r in rows if r["op category"] == "sdpa_fwd")
    assert sdpa["total_direct_kernel_time_ms"] == pytest.approx(0.5)


def test_trace2_ops_summary_direct_ms_col_no_op_count(tmp_path):
    pd.DataFrame(
        {
            "op category": ["Norm Fwd", "Norm Bwd"],
            "total_direct_kernel_time_ms": [3.0, 1.0],
        }
    ).to_csv(tmp_path / "unified_perf_summary.csv", index=False)
    rows = _build_trace2_ops_summary_by_enhanced_category(str(tmp_path))
    fwd = next(r for r in rows if r["op category"] == "norm_fwd")
    assert fwd["total_direct_kernel_time_ms"] == pytest.approx(3.0)
    # No operation_count column -> Count falls back to group size (1).
    assert fwd["Count"] == 1
    assert fwd["Percentage (%)"] == pytest.approx(75.0)


def test_trace2_ops_summary_no_time_col_falls_back(tmp_path):
    pd.DataFrame({"op category": ["GEMM"], "foo": [1]}).to_csv(
        tmp_path / "unified_perf_summary.csv", index=False
    )
    pd.DataFrame({"op category": ["Fallback"], "Count": [9]}).to_csv(
        tmp_path / "ops_summary_by_category.csv", index=False
    )
    rows = _build_trace2_ops_summary_by_enhanced_category(str(tmp_path))
    assert rows == [{"op category": "Fallback", "Count": 9}]


def test_trace2_ops_summary_no_unified_falls_back(tmp_path):
    pd.DataFrame({"op category": ["OnlyFallback"], "Count": [7]}).to_csv(
        tmp_path / "ops_summary_by_category.csv", index=False
    )
    rows = _build_trace2_ops_summary_by_enhanced_category(str(tmp_path))
    assert rows == [{"op category": "OnlyFallback", "Count": 7}]


# ----- analysis_utils: calculate_efficiency / comparative / standalone -----


def test_calculate_efficiency_standalone_dict_maf_precision_aware():
    row = pd.Series(
        {
            "TFLOPS/s_mean": 400.0,
            "TB/s_mean": 0.5,
            "FLOPS/Byte": 2000.0,
            "Compute Spec": "matrix_fp16",
            "Roofline Bound": "COMPUTE_BOUND",
            "Pct Roofline_mean": 55.0,
        }
    )
    r = calculate_efficiency(
        row, 5.3, {"matrix_bf16": 708, "matrix_fp16": 654}, "standalone"
    )
    assert r["resolved_peak_maf"] == 654
    assert r["bound_type"] == "compute"
    assert r["efficiency_percent"] == 55.0
    assert r["compute_spec"] == "matrix_fp16"
    assert r["tflops_achieved"] == 400.0


def test_calculate_efficiency_standalone_scalar_maf_and_memory_bound():
    row = pd.Series(
        {
            "TFLOPS/s_mean": 300.0,
            "TB/s_mean": 4.0,
            "FLOPS/Byte": 50.0,
            "Roofline Bound": "MEMORY_BOUND",
            "Pct Roofline_mean": 70.0,
        }
    )
    r = calculate_efficiency(row, 5.3, 700.0, "standalone")
    assert r["resolved_peak_maf"] == 700.0
    assert r["bound_type"] == "memory"
    assert r["efficiency_percent"] == 70.0


def test_calculate_efficiency_standalone_missing_roofline_is_none():
    row = pd.Series(
        {
            "TFLOPS/s_mean": 300.0,
            "TB/s_mean": 4.0,
            "FLOPS/Byte": 50.0,
            "Pct Roofline_mean": float("nan"),
        }
    )
    r = calculate_efficiency(row, 5.3, 700.0, "standalone")
    assert r["efficiency_percent"] is None


def test_calculate_efficiency_standalone_roofline_anomaly():
    row = pd.Series(
        {
            "TFLOPS/s_mean": 300.0,
            "TB/s_mean": 4.0,
            "FLOPS/Byte": 50.0,
            "Pct Roofline_mean": 120.0,
        }
    )
    r = calculate_efficiency(row, 5.3, 700.0, "standalone")
    assert r["efficiency_percent"] == 120.0
    assert r["is_anomaly"] is True
    assert "[ANOMALY]" in r["warning"]


def test_calculate_efficiency_comparative_delta_col():
    row = pd.Series(
        {
            "TFLOPS/s_mean": 300.0,
            "TB/s_mean": 4.0,
            "FLOPS/Byte": 50.0,
            "Kernel Time (µs)_sum": 10_000.0,
            "delta_us (trace2 - trace1)": -2_000.0,
        }
    )
    r = calculate_efficiency(row, 5.3, 700.0, "comparative")
    assert r["efficiency_percent"] == 80.0
    assert r["warning"] is None


def test_comparative_efficiency_speedup_col():
    result = {"efficiency_percent": None, "is_anomaly": False, "warning": None}
    row = pd.Series({"Kernel Time (µs)_sum": 10_000.0, "speedup (trace2/trace1)": 0.9})
    comparative_efficiency(result, row)
    assert result["efficiency_percent"] == 90.0


def test_comparative_efficiency_roofline_clamp_sets_warning():
    result = {"efficiency_percent": None, "is_anomaly": False, "warning": None}
    row = pd.Series(
        {
            "Kernel Time (µs)_sum": 10_000.0,
            "delta_us (trace2 - trace1)": -7_500.0,
            "Pct Roofline_mean": 60.0,
        }
    )
    comparative_efficiency(result, row)
    assert result["efficiency_percent"] == 60.0
    assert "ROOFLINE CAP" in result["warning"]


def test_comparative_efficiency_missing_kernel_time_is_noop():
    result = {"efficiency_percent": None, "is_anomaly": False, "warning": None}
    comparative_efficiency(result, pd.Series({"other": 1}))
    assert result["efficiency_percent"] is None


def test_standalone_efficiency_missing_roofline_is_noop():
    result = {"efficiency_percent": None, "is_anomaly": False, "warning": None}
    standalone_efficiency(result, pd.Series({"other": 1}))
    assert result["efficiency_percent"] is None


# ----- analysis_utils: _resolve_peak_maf -----


def test_resolve_peak_maf_precision_hit():
    row = pd.Series({"Compute Spec": "matrix_fp16"})
    assert _resolve_peak_maf(row, {"matrix_bf16": 708, "matrix_fp16": 654}, 708) == 654


def test_resolve_peak_maf_unknown_spec_falls_back():
    row = pd.Series({"Compute Spec": "weird"})
    assert _resolve_peak_maf(row, {"matrix_bf16": 708}, 708) == 708


# ----- analysis_utils: get_peak_specs -----


def test_get_peak_specs_dict_form():
    specs = get_peak_specs(
        {"max_achievable_tflops": {"matrix_bf16": 708}, "peak_hbm_bw_tbs": 5.3}
    )
    assert specs == {"peak_maf_tflops": 708, "peak_hbm_bw_tbs": 5.3}


def test_get_peak_specs_scalar_form():
    specs = get_peak_specs({"peak_bf16_maf_tflops": 700.0, "peak_hbm_bw_tbs": 5.3})
    assert specs == {"peak_maf_tflops": 700.0, "peak_hbm_bw_tbs": 5.3}


# ----- analysis_utils: _eff_bucket -----


@pytest.mark.parametrize(
    "pct,expected",
    [
        (None, "unknown"),
        (10.0, "0-30"),
        (45.0, "30-60"),
        (80.0, "60-100"),
    ],
)
def test_eff_bucket(pct, expected):
    assert _eff_bucket(pct) == expected


# ----- analysis_utils: _extract_kernel_names -----


def test_extract_kernel_names_empty_and_nan():
    assert _extract_kernel_names("") == ("", "")
    assert _extract_kernel_names("nan") == ("", "")


def test_extract_kernel_names_flat_list_last_elem():
    assert _extract_kernel_names("['a', 'b', 'kernel_x']") == (
        "kernel_x",
        "kernel_x",
    )


def test_extract_kernel_names_sublists_joined():
    full, trunc = _extract_kernel_names("[['a', 'k1'], ['b', 'k2']]")
    assert full == "Kernel 1: k1<br>Kernel 2: k2"
    assert trunc == "Kernel 1: k1<br>Kernel 2: k2"


def test_extract_kernel_names_truncation():
    long_name = "x" * 100
    full, trunc = _extract_kernel_names(repr(["short", long_name]))
    assert full == long_name
    assert trunc == long_name[:75] + "..."


# ----- analysis_utils: format_args -----


def test_format_args_drops_scalar_and_normalizes_dtype():
    assert format_args("[[2, 3], []]", "['c10::BFloat16', 'Scalar']") == "(2,3) bf16"


def test_format_args_single_dim_keeps_trailing_comma():
    assert format_args("[[128]]", "['c10::BFloat16']") == "(128,) bf16"


def test_format_args_all_dropped_returns_none():
    assert format_args("[[]]", "['Scalar']") is None


def test_format_args_missing_inputs_returns_none():
    assert format_args(None, None) is None
    assert format_args(float("nan"), "['x']") is None


def test_format_args_unparseable_returns_none():
    assert format_args("bad{{", "['x']") is None


# ----- analysis_utils: classify_kernel_library -----


@pytest.mark.parametrize(
    "op_name,kernel_details,expected",
    [
        ("aiter::gemm", "", "AITER"),
        ("triton_kernel_op", "", "Triton"),
        ("aten::mm", "[{'name': 'Cijk_foo'}]", "Tensile"),
        ("aten::mm", "void at::native::x", "PyTorch Native"),
        ("aten::mm", "plain", None),
    ],
)
def test_classify_kernel_library(op_name, kernel_details, expected):
    assert classify_kernel_library(op_name, kernel_details) == expected


# ----- analysis_utils: shape_aware_lookup / parse_first_shape -----


def test_shape_aware_lookup_exact_and_prefix_and_fallback():
    table = {"my_kernel": {(2, 3): {"a": 1}, (9, 9): {"b": 2}}}
    assert shape_aware_lookup(table, "my_kernel", "[[2,3]]") == {"a": 1}
    # No shape hint -> first value.
    assert shape_aware_lookup(table, "my_kernel", None) == {"a": 1}
    # Prefix fallback: query longer than stored (truncated) csv name.
    assert shape_aware_lookup(table, "my_kernel_full", "[[2,3]]") == {"a": 1}
    # Unknown kernel -> empty dict.
    assert shape_aware_lookup(table, "unrelated") == {}


def test_parse_first_shape_variants():
    assert parse_first_shape("[[2,3],[3,4]]") == (2, 3)
    assert parse_first_shape("[5]") is None
    assert parse_first_shape("[]") is None
    assert parse_first_shape(None) is None
    assert parse_first_shape("bad{{") is None


# ----- analysis_utils: _parse_call_stack / _extract_module_chain / _extract_call_chain -----


def test_parse_call_stack_variants():
    assert _parse_call_stack("['a', 'b']") == ["a", "b"]
    assert _parse_call_stack("nan") == []
    assert _parse_call_stack("") == []
    assert _parse_call_stack("notalist") == []


def test_extract_module_chain_strips_prefix():
    chain = _extract_module_chain("['nn.Module: Foo', 'other', 'nn.Module: Bar']")
    assert chain == ["Foo", "Bar"]


def test_extract_call_chain_filters_dispatch_internals():
    stack = (
        "['nn.Module: Foo', 'torch/nn/modules/module.py', "
        "'myfile.py', 'aten::mm', 'plainword']"
    )
    # nn.Module kept, torch dispatch internal skipped, .py/:: kept, plain dropped.
    assert _extract_call_chain(stack) == [
        "nn.Module: Foo",
        "myfile.py",
        "aten::mm",
    ]


# ----- analysis_utils: _match_fusion_op -----


def test_match_fusion_op_exact_and_prefix():
    assert _match_fusion_op("[{'name': 'k1'}]", {"k1": "base"}) == "base"
    # Prefix fallback: trace kernel name longer than the map key.
    assert _match_fusion_op("[{'name': 'k1_long'}]", {"k1": "base"}) == "base"


def test_match_fusion_op_no_match_returns_none():
    assert _match_fusion_op("[{'name': 'other'}]", {"k1": "base"}) is None


# ----- analysis_utils: build_operation_metrics call-chain cap -----


def _many_ops_df(n):
    return pd.DataFrame(
        {
            "name": [f"op{i}" for i in range(n)],
            "count": [1] * n,
            "Kernel Time (µs)_sum": [float(1000 - i) for i in range(n)],
            "call_stack_full": ["['nn.Module: Foo', 'myfile.py']"] * n,
        }
    )


def test_build_operation_metrics_call_chain_present_under_cap():
    meta = {
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    ops = build_operation_metrics(_many_ops_df(50), meta, {})
    assert "call_chain" in ops[0]
    assert ops[0]["call_chain"] == ["nn.Module: Foo", "myfile.py"]
    assert "_raw_call_stack" not in ops[0]


def test_build_operation_metrics_call_chain_dropped_over_cap():
    meta = {
        "peak_hbm_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
    }
    ops = build_operation_metrics(_many_ops_df(101), meta, {})
    assert len(ops) == 101
    # Over _CALL_CHAIN_MAX_OPS -> call_chain omitted, raw scratch key removed.
    assert "call_chain" not in ops[0]
    assert "_raw_call_stack" not in ops[0]


# ----- orchestrator_prepare.main(): end-to-end standalone drive -----


_STANDALONE_TRACE = os.path.join(
    REPO_ROOT,
    "tests",
    "traces",
    "mi300",
    "gaunernst_bert-small-uncased__1016001.json.gz",
)


@pytest.fixture(scope="module")
def orchestrator_prepared_output(tmp_path_factory):
    """Run generate_perf_report + orchestrator_prepare.main() on a small real trace.

    Exercises Steps 2-5 of orchestrator_prepare (GPU utilization, top ops, tree
    pre-compute, standalone fusion extraction, category export) end-to-end.
    Skipped when the demo trace or the perf-report generator is unavailable.
    """
    if not os.path.isfile(_STANDALONE_TRACE):
        pytest.skip("demo trace not available")

    out = str(tmp_path_factory.mktemp("orch_prep"))
    csv_dir = os.path.join(out, "perf_report_csvs")

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    gen = subprocess.run(
        [
            sys.executable,
            "-m",
            "TraceLens.Reporting.generate_perf_report_pytorch",
            "--profile_json_path",
            _STANDALONE_TRACE,
            "--output_csvs_dir",
            csv_dir,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=300,
    )
    if gen.returncode != 0 or not os.path.isfile(
        os.path.join(csv_dir, "gpu_timeline.csv")
    ):
        pytest.skip("perf report generation failed: " + (gen.stderr or "")[-500:])

    old_argv = sys.argv
    sys.argv = [
        "orchestrator_prepare",
        "--trace-path",
        _STANDALONE_TRACE,
        "--platform",
        "MI300X",
        "--output-dir",
        out,
    ]
    try:
        op.main()
    finally:
        sys.argv = old_argv

    return out


def test_orchestrator_main_writes_manifest(orchestrator_prepared_output):
    manifest_path = os.path.join(
        orchestrator_prepared_output, "category_data", "category_manifest.json"
    )
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["platform"] == "MI300X"
    assert manifest["comparison_scope"] == "standalone"
    names = [c["name"] for c in manifest["categories"]]
    # cpu_idle is always inserted first as a system-tier category.
    assert "cpu_idle" in names
    assert manifest["gpu_utilization"]["total_time_ms"] > 0


def test_orchestrator_main_exports_category_data(orchestrator_prepared_output):
    cat_data = os.path.join(orchestrator_prepared_output, "category_data")
    # A real bert trace has GEMM ops; the standalone pipeline exports its CSV.
    assert os.path.isfile(os.path.join(cat_data, "gemm_ops.csv"))
    assert os.path.isfile(os.path.join(cat_data, "multi_kernel_data.json"))
    with open(os.path.join(cat_data, "fusion_candidates.json")) as f:
        fusion = json.load(f)
    assert isinstance(fusion, list)


def test_orchestrator_main_writes_metadata_with_time_breakdown(
    orchestrator_prepared_output,
):
    meta_path = os.path.join(
        orchestrator_prepared_output, "metadata", "gemm_metadata.json"
    )
    assert os.path.isfile(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["platform"] == "MI300X"
    assert meta["category_name"] == "gemm"
    # Step 5.5 augments each category metadata with a time_breakdown block.
    assert "time_breakdown" in meta
    assert "gpu_kernel_time_ms" in meta["time_breakdown"]


# ----- Fixtures -----


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _valid_compute_findings(rank=1, row=None):
    row = row or (
        "| aten::mm | M=2,N=3 | path/to/launch | mm_kernel | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |"
    )
    header = (
        "| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | "
        "Count | FLOPS/Byte | Efficiency | Bound |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    return (
        "## Recommendations\n\n"
        f"### P{rank}: Optimize gemm\n"
        "**Insight**\n slow\n**Action**\n tune\n**Impact**\n high\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n\n"
        "## Detailed Analysis\n\n"
        f"<!-- reasoning-candidate tier=compute rank={rank} -->\n"
        "**Data:**\n\n"
        f"{header}\n{sep}\n{row}\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )


def _valid_system_findings():
    return (
        "## Recommendations\n\n"
        "### P1: Fix idle\n"
        "**Insight**\n idle high\n**Action**\n overlap\n**Impact**\n medium\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n\n"
        "## Detailed Analysis\n\n"
        "<!-- reasoning-candidate tier=system rank=1 -->\n"
        "System detail block.\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )


@pytest.fixture
def output_dir_with_manifest(tmp_path):
    out = tmp_path / "analysis_output"
    cat_data = out / "category_data"
    cat_data.mkdir(parents=True)
    manifest = {
        "platform": "MI300X",
        "gpu_utilization": {
            "total_time_ms": 1000.0,
            "computation_time_percent": 80.0,
        },
        "categories": [
            {
                "name": "gemm",
                "display_name": "GEMM",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 600.0,
            },
            {
                "name": "sdpa_fwd",
                "display_name": "SDPA Fwd",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 200.0,
            },
            {"name": "cpu_idle", "tier": "system", "gpu_kernel_time_ms": 0},
        ],
    }
    (cat_data / "category_manifest.json").write_text(json.dumps(manifest, indent=2))
    return str(out)


@pytest.fixture
def perf_csv_dir(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "name": ["aten::mm", "aten::add"],
            "Input type": ["['c10::BFloat16']", "['c10::BFloat16']"],
            "Input Dims": ["[[2,3]]", "[[4,4]]"],
            "call_stack_full": [
                "['nn.Module: LlamaMLP', 'myfile.py']",
                "['nn.Module: Attention', 'other.py']",
            ],
        }
    ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
    return str(csv_dir)


# ----- validation_utils: validate_findings_file -----


def test_validate_findings_file_compute_pass(tmp_path):
    content = _valid_compute_findings()
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "args": "M=2,N=3",
                        "launcher_path": "path/to/launch",
                        "kernel_name_trunc": "mm_kernel",
                    }
                ]
            }
        )
    )
    passed, errors = validate_findings_file(str(fp), "compute")
    assert passed, errors
    assert errors == []


def test_validate_findings_file_missing_file(tmp_path):
    passed, errors = validate_findings_file(str(tmp_path / "nope.md"), "compute")
    assert not passed
    assert any("File not found" in e for e in errors)


def test_validate_findings_file_wrong_header_order(tmp_path):
    content = "## Detailed Analysis\n\n## Recommendations\n"
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), content)
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("Recommendations must appear before" in e for e in errors)


def test_validate_findings_file_relaxed_empty_category(tmp_path):
    content = (
        "## Recommendations\n\n## Detailed Analysis\n"
        "not quantifiable from trace data\n"
    )
    findings_dir = tmp_path / "category_findings"
    findings_dir.mkdir(parents=True)
    fp = findings_dir / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps({"category_findings": []})
    )
    passed, errors = validate_findings_file(str(fp), "compute")
    assert passed, errors


def test_validate_findings_file_tier_mismatch(tmp_path):
    content = _valid_compute_findings().replace("tier=compute", "tier=system")
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text("{}")
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("tier mismatch" in e for e in errors)


def test_validate_findings_file_p_item_count_mismatch(tmp_path):
    content = _valid_compute_findings()
    content += "\n### P2: Extra item\n"
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text("{}")
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("P-item count" in e or "headings but" in e for e in errors)


def test_validate_findings_file_system_pass(tmp_path):
    fp = tmp_path / "cpu_idle_findings.md"
    _write(str(fp), _valid_system_findings())
    passed, errors = validate_findings_file(str(fp), "system")
    assert passed, errors


def test_metrics_json_for_findings_and_empty_category(tmp_path):
    fp = tmp_path / "category_findings" / "gemm_findings.md"
    fp.parent.mkdir(parents=True)
    fp.write_text("x")
    mp = _metrics_json_for_findings(str(fp))
    assert mp.endswith("category_data/gemm_metrics.json")
    assert _category_findings_empty(str(fp)) is False
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps({"category_findings": []})
    )
    assert _category_findings_empty(str(fp)) is True


def test_scan_args_cells_and_load_valid_args(tmp_path):
    text = "| Operation | Args | Time |\n" "|---|---|---|\n" "| op | M=2,N=3 | 1 |\n"
    cells = list(_scan_args_cells(text))
    assert cells == [(3, "M=2,N=3")]
    (tmp_path / "gemm_metrics.json").write_text(
        json.dumps({"operations": [{"args": "M=2,N=3"}, {"args": None}]})
    )
    valid = _load_valid_args(str(tmp_path / "gemm_metrics.json"), "/nonexistent.json")
    assert valid == {"M=2,N=3"}


def test_validate_findings_file_missing_recommendations_section(tmp_path):
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), "## Detailed Analysis\n\nbody\n")
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("Missing required section" in e for e in errors)


def test_validate_findings_file_comparative_data_table(tmp_path):
    header = (
        "| Operation | Args (T1) | Kernel Path | Kernel Name | Trace 1 Time (ms) | "
        "Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |"
    )
    sep = "|---|" * 10 + "---|"
    row = (
        "| aten::mm | M=2,N=3 | path/to/launch | mm_kernel | 1.0 | 0.8 | 5 | -0.2 | "
        "2000 | compute |"
    )
    content = (
        "## Recommendations\n\n"
        "### P1: Optimize gemm\n"
        "**Insight**\n x\n**Action**\n y\n**Impact**\n z\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n\n"
        "## Detailed Analysis\n\n"
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "**Data:**\n\n"
        f"{header}\n{sep}\n{row}\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )
    findings_dir = tmp_path / "category_findings"
    findings_dir.mkdir()
    fp = findings_dir / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "args": "M=2,N=3",
                        "launcher_path": "path/to/launch",
                        "kernel_name_trunc": "mm_kernel",
                    }
                ]
            }
        )
    )
    passed, errors = validate_findings_file(str(fp), "compute", "comparative")
    assert passed, errors


def test_validate_compute_data_kernel_name_mismatch(tmp_path):
    row = (
        "| aten::mm | M=2,N=3 | path/to/launch | WRONG_KERNEL | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |"
    )
    content = _valid_compute_findings(row=row)
    findings_dir = tmp_path / "category_findings"
    findings_dir.mkdir()
    fp = findings_dir / "gemm_findings.md"
    _write(str(fp), content)
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "args": "M=2,N=3",
                        "launcher_path": "path/to/launch",
                        "kernel_name_trunc": "mm_kernel",
                    }
                ]
            }
        )
    )
    passed, errors = validate_findings_file(str(fp), "compute")
    assert not passed
    assert any("Kernel Name cell" in e for e in errors)


def test_check_priority_consistency_unreadable_json(tmp_path):

    (tmp_path / "priority_data.json").write_text("{not json")
    result = _check_priority_consistency(str(tmp_path), {})
    assert result["status"] == "WARN"
    assert any("unreadable" in m for m in result["messages"])


def test_validate_report_args_column_no_metrics_dir(tmp_path):
    content = "| Operation | Args | Time |\n|---|---|---|\n| op | x | 1 |\n"
    assert _validate_report_args_column(content, str(tmp_path)) == []


def test_extract_detailed_analysis_no_section():
    assert _extract_detailed_analysis_subsection("no detailed section", "### X") is None


def test_load_findings_failed_compute(output_dir_with_manifest):
    out = output_dir_with_manifest
    os.makedirs(os.path.join(out, "category_findings"))
    _write(os.path.join(out, "category_findings", "gemm_findings.md"), "Status: ERROR")
    data = load_findings(out)
    assert data["failed_compute"][0]["category"] == "gemm"


def test_generate_and_embed_plot_priority_data_failure(tmp_path, monkeypatch):
    def boom(_output_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(plot_utils, "generate_priority_data", boom)
    results = plot_utils.generate_and_embed_plot(str(tmp_path), "Title")
    assert results["plot_data"] is False


# ----- validation_utils: subagent batch checks -----


def test_validate_subagent_outputs_summary(output_dir_with_manifest, tmp_path):
    out = output_dir_with_manifest
    os.makedirs(os.path.join(out, "category_findings"), exist_ok=True)
    os.makedirs(os.path.join(out, "system_findings"), exist_ok=True)
    _write(os.path.join(out, "category_findings", "gemm_findings.md"), "ok")
    _write(os.path.join(out, "system_findings", "cpu_idle_findings.md"), "ok")
    generate_priority_data(out)
    results = validate_subagent_outputs(out)
    assert "time_check" in results
    assert "coverage_check" in results
    assert "priority_check" in results


def test_check_time_sanity_pass_and_warn():
    manifest = {
        "gpu_utilization": {"total_time_ms": 1000.0, "computation_time_percent": 80.0},
        "categories": [
            {"tier": "compute_kernel", "gpu_kernel_time_ms": 790.0},
        ],
    }
    assert _check_time_sanity(manifest)["status"] == "PASS"

    no_comp = {"gpu_utilization": {"total_time_ms": 0}, "categories": []}
    assert _check_time_sanity(no_comp)["status"] == "WARN"

    bad_sum = {
        "gpu_utilization": {"total_time_ms": 1000.0, "computation_time_percent": 80.0},
        "categories": [{"tier": "compute_kernel", "gpu_kernel_time_ms": 100.0}],
    }
    assert _check_time_sanity(bad_sum)["status"] == "WARN"


def test_check_coverage_missing_findings(output_dir_with_manifest):
    result = _check_coverage(
        output_dir_with_manifest,
        json.loads(
            open(
                os.path.join(
                    output_dir_with_manifest,
                    "category_data",
                    "category_manifest.json",
                )
            ).read()
        ),
    )
    assert result["status"] == "WARN"
    assert any("Missing compute findings" in m for m in result["messages"])


# ----- validation_utils: validate_report extras -----


def _full_report(extra_kf_impact=""):
    kf = "## Kernel Fusion Opportunities (Experimental)\n\n"
    if extra_kf_impact:
        kf += extra_kf_impact + "\n"
    return f"""# Analysis Report

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Time | 1000 ms |
| Compute % | 99.8% |
| Idle % | 0.2% |
| Exposed Communication % | 0.05% |
| Top Bottleneck Category | gemm |

## Compute Kernel Optimizations

### P1: Optimize gemm
<!-- impact-begin kind=p_item category=gemm low=85.0 mid=100.0 high=115.0 -->
Detail text for gemm optimization goes here with enough content.
<!-- impact-end -->

## {kf.split('## ')[1]}

## System-Level Optimizations

Top Operations:

<!-- impact-begin kind=top_ops -->

| Op | Time |
|----|------|
| gemm | 100 |

<!-- impact-end -->

## Detailed Analysis

### Compute Kernel Insights

#### P1: gemm detail
<!-- reasoning-candidate tier=compute rank=1 -->
More detail here.

### System-Level Insights

#### P1: idle detail
<!-- reasoning-candidate tier=system rank=1 -->
System detail.

## Appendix

Reference material.
"""


def test_validate_report_empty_and_placeholders(tmp_path):
    _write(str(tmp_path / "analysis.md"), "short")
    passed, missing = validate_report(str(tmp_path))
    assert not passed
    assert any("empty or too short" in m for m in missing)

    content = _full_report().replace(
        "Detail text for gemm optimization goes here with enough content.",
        "Unfilled template uses <Library> placeholder here.",
    )
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(
        json.dumps({"findings": [], "priorities": []})
    )
    passed, missing = validate_report(str(tmp_path))
    assert any("Unfilled placeholders" in m for m in missing)


def test_validate_report_missing_metrics_row(tmp_path):
    content = _full_report().replace("| Compute % | 99.8% |", "")
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "category": "gemm",
                        "global_rank": 1,
                        "impact_score": 100.0,
                        "impact_score_low": 85.0,
                        "impact_score_high": 115.0,
                    }
                ],
                "priorities": [{"rank": 1, "category": "gemm"}],
            }
        )
    )
    passed, missing = validate_report(str(tmp_path))
    assert not passed
    assert any("Missing metrics row" in m for m in missing)


def test_validate_report_no_exec_table(tmp_path):
    content = _full_report()
    exec_start = content.find("## Executive Summary")
    exec_end = content.find("## Compute Kernel Optimizations")
    content = (
        content[:exec_start]
        + "## Executive Summary\n\nNo table here.\n\n"
        + content[exec_end:]
    )
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(
        json.dumps({"findings": [], "priorities": []})
    )
    passed, missing = validate_report(str(tmp_path))
    assert any("No metrics table" in m for m in missing)


def test_validate_report_kf_section_absent_skips_scope_check(tmp_path):
    content = _full_report().replace(
        "## Kernel Fusion Opportunities (Experimental)\n\n", ""
    )
    errs = _validate_report_comparison_scope_diffs(content, str(tmp_path), "standalone")
    assert errs == []


def test_validate_report_args_column_mismatch(tmp_path):
    (tmp_path / "category_data").mkdir()
    (tmp_path / "category_data" / "gemm_metrics.json").write_text(
        json.dumps({"operations": [{"args": "M=2,N=3"}]})
    )
    content = _full_report()
    content += "\n| Operation | Args | Time |\n|---|---|---|\n| op | WRONG | 1 |\n"
    _write(str(tmp_path / "analysis.md"), content)
    (tmp_path / "priority_data.json").write_text(
        json.dumps({"findings": [], "priorities": []})
    )
    errors = _validate_report_args_column(content, str(tmp_path))
    assert any("Args cell" in e for e in errors)


def test_validate_report_kf_impact_standalone(tmp_path):
    good = "**Impact**: impact_score: 1.5 (perf-model coverage 2/3 kernels)"
    bad = "**Impact**: impact_score: 1.5"
    content = _full_report(extra_kf_impact=good)
    assert (
        _validate_report_comparison_scope_diffs(content, str(tmp_path), "standalone")
        == []
    )
    content_bad = _full_report(extra_kf_impact=bad)
    errs = _validate_report_comparison_scope_diffs(
        content_bad, str(tmp_path), "standalone"
    )
    assert any("standalone mode" in e for e in errs)


def test_validate_report_kf_impact_comparative(tmp_path):
    good = "**Impact**: impact_score: 1.5"
    bad_paren = "**Impact**: impact_score: 1.5 (perf-model coverage 2/3 kernels)"
    content = _full_report(extra_kf_impact=good)
    assert (
        _validate_report_comparison_scope_diffs(content, str(tmp_path), "comparative")
        == []
    )
    content_bad = _full_report(extra_kf_impact=bad_paren)
    errs = _validate_report_comparison_scope_diffs(
        content_bad, str(tmp_path), "comparative"
    )
    assert any("comparative mode" in e for e in errs)


def test_validate_report_reasoning_candidates_r5(tmp_path):
    content = _full_report()
    content = content.replace(
        "<!-- reasoning-candidate tier=compute rank=1 -->",
        "<!-- missing marker -->",
        1,
    )
    errs = _validate_report_reasoning_candidates(content)
    assert any("R5:" in e for e in errs)


def test_extract_detailed_analysis_subsection():
    content = "## Detailed Analysis\n\n### Compute Kernel Insights\n\n#### P1: x\nbody\n### Other\n"
    sub = _extract_detailed_analysis_subsection(content, "### Compute Kernel Insights")
    assert "#### P1: x" in sub
    assert _extract_detailed_analysis_subsection(content, "### Missing") is None


def test_marker_validator_system_findings_requires_p_item(tmp_path):
    fp = tmp_path / "cpu_idle_findings.md"
    _write(str(fp), "## Recommendations\n\n## Detailed Analysis\n")
    errors = MarkerValidator.check_findings_file(str(fp), "system_findings")
    assert any("missing required kind=p_item" in e for e in errors)


def test_marker_validator_p_item_heading_mismatch(tmp_path):
    text = (
        "### P1: One\n### P2: Two\n"
        "<!-- impact-begin kind=p_item low=1 mid=2 high=3 -->\n"
        "<!-- impact-end -->\n"
    )
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), text)
    errors = MarkerValidator.check_findings_file(str(fp), "category_findings")
    assert any("headings but" in e for e in errors)


def test_marker_validator_detail_estimate_sentinel(tmp_path):
    text = (
        "<!-- reasoning-candidate tier=compute rank=1 -->\n"
        "not quantifiable from trace data\n"
    )
    fp = tmp_path / "gemm_findings.md"
    _write(str(fp), text)
    errors = MarkerValidator._check_detail_estimate_per_candidate(
        text, "gemm_findings.md"
    )
    assert errors == []


# ----- report_utils -----


def test_load_manifest(output_dir_with_manifest):
    manifest = load_manifest(output_dir_with_manifest)
    assert manifest["platform"] == "MI300X"


def test_prepare_model_identification_standalone(tmp_path, perf_csv_dir):
    out = str(tmp_path)
    os.makedirs(os.path.join(out, "perf_report_csvs"), exist_ok=True)
    src = os.path.join(perf_csv_dir, "unified_perf_summary.csv")
    dst = os.path.join(out, "perf_report_csvs", "unified_perf_summary.csv")
    pd.read_csv(src).to_csv(dst, index=False)
    assert prepare_model_identification_data(out, "standalone") is True
    assert os.path.isfile(os.path.join(out, "metadata", "condensed_op_info.csv"))
    nn = open(os.path.join(out, "metadata", "nn_modules.txt")).read()
    assert "LlamaMLP" in nn


def test_prepare_model_identification_comparative(tmp_path, perf_csv_dir):
    out = str(tmp_path)
    t1 = os.path.join(out, "perf_report_trace1_csvs")
    os.makedirs(t1)
    pd.read_csv(os.path.join(perf_csv_dir, "unified_perf_summary.csv")).to_csv(
        os.path.join(t1, "unified_perf_summary.csv"), index=False
    )
    assert prepare_model_identification_data(out, "comparative") is True


def test_prepare_model_identification_missing_csv(tmp_path):
    assert prepare_model_identification_data(str(tmp_path)) is False


def test_prepare_model_identification_bad_columns(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    pd.DataFrame({"name": ["op"]}).to_csv(
        csv_dir / "unified_perf_summary.csv", index=False
    )
    assert prepare_model_identification_data(str(tmp_path)) is False


def test_load_manifest_categories(capsys, output_dir_with_manifest):
    data = load_manifest_categories(output_dir_with_manifest)
    assert "gemm" in [c["name"] for c in data["compute_categories"]]
    assert "cpu_idle" in [c["name"] for c in data["system_categories"]]
    captured = capsys.readouterr()
    assert "compute_categories" in captured.out


def test_load_findings(capsys, output_dir_with_manifest):
    out = output_dir_with_manifest
    os.makedirs(os.path.join(out, "category_findings"))
    os.makedirs(os.path.join(out, "system_findings"))
    _write(os.path.join(out, "category_findings", "gemm_findings.md"), "ok")
    _write(
        os.path.join(out, "system_findings", "cpu_idle_findings.md"), "Status: ERROR"
    )
    _write(os.path.join(out, "system_findings", "multi_kernel_findings.md"), "ok")
    manifest = json.loads(
        open(os.path.join(out, "category_data", "category_manifest.json")).read()
    )
    manifest["top_operations"] = [{"name": "aten::mm"}]
    open(os.path.join(out, "category_data", "category_manifest.json"), "w").write(
        json.dumps(manifest)
    )
    data = load_findings(out)
    assert "gemm" in data["compute_findings"]
    assert data["failed_system"][0]["category"] == "cpu_idle"
    assert len(data["top_ops"]) == 1


def test_generate_priority_data_unmodeled_and_heuristic(tmp_path):
    cat_data = tmp_path / "category_data"
    cat_data.mkdir()
    manifest = {
        "gpu_utilization": {
            "total_time_ms": 1000.0,
            "computation_time_percent": 100.0,
        },
        "categories": [
            {
                "name": "gemm",
                "display_name": "GEMM",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 100.0,
            },
            {
                "name": "norm",
                "display_name": "Norm",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 80.0,
            },
        ],
    }
    (cat_data / "category_manifest.json").write_text(json.dumps(manifest))
    (cat_data / "gemm_metrics.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "category": "gemm",
                "impact_estimates": [],
                "category_findings": [
                    {
                        "rank": 1,
                        "impact_score": 10.0,
                        "impact_score_low": 8.0,
                        "impact_score_high": 12.0,
                        "operation_count": 1,
                        "estimate_method": "heuristic",
                    }
                ],
            }
        )
    )
    out_path = generate_priority_data(str(tmp_path))
    with open(out_path) as f:
        data = json.load(f)
    assert data["baseline_ms"] == 1000.0
    assert any(p["source"] == "manifest_fallback" for p in data["priorities"])
    assert len(data["recommendations"]) == 0


def test_generate_priority_data_exception_fallback(tmp_path):
    out_path = generate_priority_data(str(tmp_path))
    with open(out_path) as f:
        data = json.load(f)
    assert data["baseline_ms"] == 0
    assert data["findings"] == []


# ----- plot_utils -----


def test_generate_perf_plot_success(output_dir_with_manifest):
    ok = plot_utils.generate_perf_plot(
        output_dir_with_manifest, "Test Breakdown", write_base64=True
    )
    assert ok is True
    png = os.path.join(output_dir_with_manifest, "perf_improvement.png")
    b64 = os.path.join(output_dir_with_manifest, "perf_improvement_base64.txt")
    assert os.path.isfile(png)
    assert os.path.isfile(b64)


def test_generate_perf_plot_missing_manifest(tmp_path):
    assert plot_utils.generate_perf_plot(str(tmp_path), "Title") is False


def test_generate_perf_plot_invalid_baseline(tmp_path):
    cat_data = tmp_path / "category_data"
    cat_data.mkdir()
    (cat_data / "category_manifest.json").write_text(
        json.dumps({"gpu_utilization": {"total_time_ms": 0}, "categories": []})
    )
    assert plot_utils.generate_perf_plot(str(tmp_path), "Title") is False


def test_generate_perf_plot_rest_only_segment(tmp_path):
    """No compute_kernel tiers -> plot shows only Non-computing rest segment."""
    cat_data = tmp_path / "category_data"
    cat_data.mkdir()
    (cat_data / "category_manifest.json").write_text(
        json.dumps(
            {
                "gpu_utilization": {"total_time_ms": 100.0},
                "categories": [{"name": "cpu_idle", "tier": "system"}],
            }
        )
    )
    assert plot_utils.generate_perf_plot(str(tmp_path), "Title") is True


def test_embed_plot_in_report(tmp_path):
    _write(str(tmp_path / "analysis.md"), "Before {{PERF_PLOT}} After")
    _write(str(tmp_path / "perf_improvement_base64.txt"), "abc123")
    assert plot_utils.embed_plot_in_report(str(tmp_path)) is True
    text = open(tmp_path / "analysis.md").read()
    assert "data:image/png;base64,abc123" in text


def test_embed_plot_missing_report(tmp_path):
    assert plot_utils.embed_plot_in_report(str(tmp_path)) is False


def test_embed_plot_removes_placeholder_without_b64(tmp_path):
    _write(str(tmp_path / "analysis.md"), "Before {{PERF_PLOT}} After")
    assert plot_utils.embed_plot_in_report(str(tmp_path)) is False
    assert "{{PERF_PLOT}}" not in open(tmp_path / "analysis.md").read()


def test_generate_and_embed_plot(output_dir_with_manifest):
    _write(os.path.join(output_dir_with_manifest, "analysis.md"), "{{PERF_PLOT}}")
    open(
        os.path.join(output_dir_with_manifest, "category_data", "gemm_metrics.json"),
        "w",
    ).write(
        json.dumps(
            {
                "status": "OK",
                "category": "gemm",
                "impact_estimates": [],
                "category_findings": [],
            }
        )
    )
    results = plot_utils.generate_and_embed_plot(output_dir_with_manifest, "Title")
    assert results["plot_data"] is True
    assert results["plot"] is True
    assert results["embed"] is True


def test_short_name():
    assert plot_utils._short_name("gemm") == "Gemm"
    assert plot_utils._short_name("verylongcategory") == "Verylon…"


# ----- classify_kernels CLI -----


def test_classify_all_and_main(tmp_path):
    extracted = {
        "source_file": "trace.json",
        "kernels": [
            {"name": "Cijk_Alik_Bljk", "dur": 100},
            {"name": "rmsnorm2d", "dur": 50},
        ],
    }
    inp = tmp_path / "extracted.json"
    out = tmp_path / "classified.json"
    inp.write_text(json.dumps(extracted))
    classified = classify_all(extracted)
    assert classified[0]["kernel_type"] == "GEMM"
    assert classified[1]["kernel_type"] == "RMSNorm"

    old_argv = sys.argv
    sys.argv = ["classify_kernels", str(inp), "-o", str(out)]
    try:
        classify_main()
    finally:
        sys.argv = old_argv
    result = json.loads(out.read_text())
    assert result["total_kernels"] == 2
    assert "type_summary" in result


def test_classify_kernels_main_stdout(tmp_path, capsys):
    extracted = {"kernels": [{"name": "Cijk_x", "dur": 1}]}
    inp = tmp_path / "extracted.json"
    inp.write_text(json.dumps(extracted))
    old_argv = sys.argv
    sys.argv = ["classify_kernels", str(inp)]
    try:
        classify_main()
    finally:
        sys.argv = old_argv
    assert "classified_kernels" in capsys.readouterr().out


# ----- orchestrator_prepare fusion extraction (mocked tree) -----


class _StubTree:
    def __init__(self, events, uid_map, parent_map=None):
        self.events = events
        self._uid_map = uid_map
        self._parent_map = parent_map or {}

    def get_UID2event(self, uid):
        return self._uid_map[uid]

    def get_parent_event(self, ev):
        return self._parent_map.get(id(ev))


class _StubAnalyzer:
    def __init__(self, tree, unified_events=None):
        self.tree = tree
        self._unified = unified_events or []

    def event_to_category(self, ev):
        return ev.get("_category", "aten")

    def collect_unified_perf_events(self):
        return self._unified


def _kernel_event(uid, name, dur=1000):
    return {"name": name, "dur": dur, "_category": "kernel", "gpu_events": []}


def test_extract_standalone_fusion_candidates(tmp_path):
    k1 = _kernel_event(10, "Cijk_gemm_a")
    k2 = _kernel_event(11, "vectorized_elementwise_kernel add")
    module = {
        "name": "nn.Module: MLP_0",
        "_category": "aten",
        "gpu_events": [10, 11],
        "args": {"Input Dims": "[[2,3]]"},
    }
    uid_map = {10: k1, 11: k2}
    tree = _StubTree([module], uid_map)
    analyzer = _StubAnalyzer(tree)

    csv_dir = tmp_path / "perf_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "kernel_details_summary": [
                "[{'name': 'Cijk_gemm_a'}]",
                "[{'name': 'vectorized_elementwise_kernel add'}]",
            ],
            "op category": ["GEMM", "elementwise"],
            "Data Moved (MB)": [10.0, 4.0],
            "perf_params": ["{'M':2,'N':4,'K':3}", "{'shape_in1': [4, 4]}"],
            "Input Dims": ["[[2,3]]", "[[4,4]]"],
        }
    ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

    cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
    assert isinstance(cands, list)
    assert any(c.get("base_name") == "MLP" for c in cands)


def test_extract_standalone_skips_fused_and_gemm_norm(tmp_path):
    k1 = _kernel_event(10, "Cijk_gemm")
    k2 = _kernel_event(11, "rmsnorm_kernel")
    module = {
        "name": "nn.Module: Block_0",
        "_category": "aten",
        "gpu_events": [10, 11],
    }
    tree = _StubTree([module], {10: k1, 11: k2})
    analyzer = _StubAnalyzer(tree)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "kernel_details_summary": ["[{'name': 'Cijk_gemm'}]"],
            "op category": ["GEMM"],
            "Data Moved (MB)": [1.0],
            "perf_params": ["{}"],
            "Input Dims": ["[[1,1]]"],
        }
    ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
    cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
    assert cands == []


def test_extract_comparative_fusion_candidates(tmp_path):
    csv_dir = tmp_path / "trace1_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "name": ["Cijk_A", "Cijk_B", "some_op"],
            "source": ["trace1", "trace1", "trace2"],
            "lowest_common_ancestor_id": [100, 100, 100],
            "kernel_time": [5000.0, 3000.0, 1000.0],
            "gpu_op_uid": [10, 11, None],
        }
    ).to_csv(csv_dir / "diff_stats.csv", index=False)

    uid_map = {
        10: {"name": "Cijk_A", "dur": 5000, "_category": "kernel", "gpu_events": []},
        11: {"name": "Cijk_B", "dur": 3000, "_category": "kernel", "gpu_events": []},
    }
    module = {
        "name": "nn.Module: Attn_0",
        "_category": "aten",
        "gpu_events": [10, 11],
        "args": {},
    }
    tree = _StubTree([module], uid_map)
    analyzer = _StubAnalyzer(tree)

    cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
    assert len(cands) >= 1
    assert cands[0]["kernel_count_trace1"] >= 2


def test_extract_comparative_missing_inputs(tmp_path):
    assert _extract_comparative_fusion_candidates(str(tmp_path)) == []
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    pd.DataFrame({"name": []}).to_csv(csv_dir / "diff_stats.csv", index=False)
    assert _extract_comparative_fusion_candidates(str(csv_dir)) == []


def _write_minimal_orchestrator_csvs(base, comparative=False):
    t1 = os.path.join(
        base, "perf_report_trace1_csvs" if comparative else "perf_report_csvs"
    )
    os.makedirs(t1)
    pd.DataFrame(
        {
            "type": ["total_time", "computation_time", "idle_time"],
            "time ms": [1000.0, 900.0, 100.0],
            "percent": [100.0, 90.0, 10.0],
        }
    ).to_csv(os.path.join(t1, "gpu_timeline.csv"), index=False)
    pd.DataFrame(
        {
            "name": ["aten::mm"],
            "total_direct_kernel_time_ms": [800.0],
            "op category": ["GEMM"],
        }
    ).to_csv(os.path.join(t1, "ops_summary.csv"), index=False)
    pd.DataFrame(
        {
            "name": ["aten::mm"],
            "op category": ["GEMM"],
            "Kernel Time (µs)_sum": [800_000.0],
            "total_duration_us": [900_000.0],
        }
    ).to_csv(os.path.join(t1, "unified_perf_summary.csv"), index=False)
    if comparative:
        t2 = os.path.join(base, "perf_report_trace2_csvs")
        os.makedirs(t2)
        pd.DataFrame(
            {
                "type": ["total_time", "computation_time", "idle_time"],
                "time ms": [900.0, 810.0, 90.0],
                "percent": [100.0, 90.0, 10.0],
            }
        ).to_csv(os.path.join(t2, "gpu_timeline.csv"), index=False)
        pd.DataFrame(
            {
                "op category": ["GEMM"],
                "Kernel Time (µs)_sum": [700_000.0],
                "operation_count": [1],
            }
        ).to_csv(os.path.join(t2, "unified_perf_summary.csv"), index=False)


def test_orchestrator_main_missing_csv_exits(tmp_path):
    script = os.path.join(ANALYSIS_DIR, "utils", "orchestrator_prepare.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--trace-path",
            "/fake/trace.json",
            "--platform",
            "MI300X",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    assert result.returncode == 1
    assert "STEP1_FAIL" in result.stderr + result.stdout


def test_orchestrator_main_comparative_mocked(tmp_path, monkeypatch):

    out = str(tmp_path)
    _write_minimal_orchestrator_csvs(out, comparative=True)

    k1 = _kernel_event(10, "Cijk_a")
    k2 = _kernel_event(11, "ew_add")
    module = {"name": "nn.Module: MLP_0", "_category": "aten", "gpu_events": [10, 11]}
    tree = _StubTree([module], {10: k1, 11: k2})
    analyzer = _StubAnalyzer(tree)

    class _FakeTreePerfAnalyzer:
        @classmethod
        def from_file(cls, *args, **kwargs):
            return analyzer

    monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
    monkeypatch.setattr(
        op,
        "_extract_comparative_fusion_candidates",
        lambda *a, **k: [{"module_name": "m", "total_kernel_time_us_trace1": 100}],
    )

    old_argv = sys.argv
    sys.argv = [
        "orchestrator_prepare",
        "--trace-path",
        "/fake/trace.json",
        "--platform",
        "MI300X",
        "--output-dir",
        out,
        "--comparison-scope",
        "comparative",
    ]
    try:
        op.main()
    finally:
        sys.argv = old_argv

    manifest = json.loads(
        open(os.path.join(out, "category_data", "category_manifest.json")).read()
    )
    assert manifest["comparison_scope"] == "comparative"
    assert "trace2_gpu_utilization" in manifest
    assert os.path.isfile(os.path.join(out, "metadata", "trace2_gpu_utilization.json"))


def test_orchestrator_main_standalone_mocked(tmp_path, monkeypatch):

    out = str(tmp_path)
    _write_minimal_orchestrator_csvs(out, comparative=False)

    k1 = _kernel_event(10, "Cijk_a")
    k2 = _kernel_event(11, "ew_add")
    module = {"name": "nn.Module: MLP_0", "_category": "aten", "gpu_events": [10, 11]}
    tree = _StubTree([module], {10: k1, 11: k2})
    analyzer = _StubAnalyzer(tree)

    class _FakeTreePerfAnalyzer:
        @classmethod
        def from_file(cls, *args, **kwargs):
            return analyzer

    monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
    monkeypatch.setattr(
        op,
        "_extract_standalone_fusion_candidates",
        lambda *a, **k: [
            {
                "module_name": "nn.Module: MLP_0",
                "base_name": "MLP",
                "total_kernel_time_us": 5000,
                "kernels": [{"name": "a"}, {"name": "b"}],
            }
        ],
    )

    old_argv = sys.argv
    sys.argv = [
        "orchestrator_prepare",
        "--trace-path",
        "/fake/trace.json",
        "--platform",
        "MI300X",
        "--output-dir",
        out,
    ]
    try:
        op.main()
    finally:
        sys.argv = old_argv

    manifest = json.loads(
        open(os.path.join(out, "category_data", "category_manifest.json")).read()
    )
    assert manifest["comparison_scope"] == "standalone"
    names = [c["name"] for c in manifest["categories"]]
    assert "gemm" in names
    assert "cpu_idle" in names
    assert "kernel_fusion" in names


class TestOrchestratorComparativeFusion:
    def test_comparative_fusion_two_kernel_match(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_A", dur=500)
        k2 = _kernel_event(11, "ew_add", dur=300)
        mod = {
            "name": "nn.Module: Block_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        tree = _StubTree([mod], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)

        csv_dir = tmp_path / "trace1_csvs"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "name": ["Cijk_A", "ew_add", "Cijk_A", "ew_add"],
                "source": ["trace1", "trace1", "trace2", "trace2"],
                "lowest_common_ancestor_id": [100, 100, 100, 100],
                "kernel_time": [5000.0, 3000.0, 4000.0, 2500.0],
                "gpu_op_uid": [10, 11, 10, 11],
            }
        ).to_csv(csv_dir / "diff_stats.csv", index=False)

        cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
        assert isinstance(cands, list)


class TestOrchestratorPhase11:
    def test_sibling_sequence_and_enrichment_skip(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_gemm_a", dur=500)
        k2 = _kernel_event(11, "Cijk_gemm_b", dur=400)
        k3 = _kernel_event(20, "Cijk_sib_a", dur=300)
        k4 = _kernel_event(21, "Cijk_sib_b", dur=200)
        mod = {
            "name": "nn.Module: MLP_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        parent = {"name": "nn.Module: Block_0", "_category": "aten", "UID": 100}
        child1 = {
            "name": "aten::mm",
            "_category": "aten",
            "gpu_events": [20],
            "parent": 100,
        }
        child2 = {
            "name": "aten::add",
            "_category": "aten",
            "gpu_events": [21],
            "parent": 100,
        }
        mod_dup = {
            "name": "nn.Module: MLP_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree(
            [mod, mod_dup, parent, child1, child2],
            {10: k1, 11: k2, 20: k3, 21: k4, 100: parent},
        )
        analyzer = _StubAnalyzer(tree, unified_events=[child1, child2])

        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_gemm_a'}]",
                    "[{'name': 'Cijk_gemm_b'}]",
                ],
                "op category": ["GEMM", "GEMM"],
                "Data Moved (MB)": [10.0, 8.0],
                "perf_params": ["{}", "{}"],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)
        sibling = [c for c in cands if c.get("source") == "sibling_sequence"]
        assert sibling

        for c in cands:
            for k in c.get("kernels", []):
                k["data_in_mb"] = 1.0

        cands2 = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert cands2

    def test_orchestrator_main_empty_csv_exit(self, tmp_path, monkeypatch):

        out = str(tmp_path / "empty_out")
        os.makedirs(out)
        empty_csv = os.path.join(out, "perf_report_csvs")
        os.makedirs(empty_csv)

        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            NORM_TRACE,
            "--platform",
            "MI300X",
            "--output-dir",
            out,
            "--comparison-scope",
            "standalone",
        ]
        try:
            with pytest.raises(SystemExit) as exc:
                op.main()
            assert exc.value.code == 1
        finally:
            sys.argv = old_argv

    def test_orchestrator_bottleneck_top5_fallback(self, tmp_path, monkeypatch):

        out = str(tmp_path / "orch_out")
        _write_minimal_orchestrator_csvs(out, comparative=False)
        csv_dir = os.path.join(out, "perf_report_csvs")
        rows = []
        for i in range(8):
            rows.append(
                {
                    "name": f"aten::op_{i}",
                    "op category": "GEMM",
                    "Kernel Time (µs)_sum": float(100 + i),
                    "total_duration_us": 60000.0,
                    "kernel_details_summary": f"[{{'name': 'k{i}'}}]",
                    "Data Moved (MB)": 1.0,
                    "perf_params": "{}",
                    "Input Dims": "[[2,3]]",
                }
            )
        pd.DataFrame(rows).to_csv(
            os.path.join(csv_dir, "unified_perf_summary.csv"), index=False
        )
        pd.DataFrame({"name": ["aten::op_0"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary.csv"), index=False
        )

        class _FakeTPA:
            @classmethod
            def from_file(cls, *args, **kwargs):
                k = _kernel_event(0, "k0", dur=100)
                mod = {
                    "name": "aten::mm",
                    "_category": "aten",
                    "gpu_events": [0],
                    "ts": 0,
                }
                tree = _StubTree([mod, k], {0: k})
                return _StubAnalyzer(tree)

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTPA)
        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            NORM_TRACE,
            "--platform",
            "MI300X",
            "--output-dir",
            out,
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        assert os.path.isdir(os.path.join(out, "category_data"))


class TestOrchestratorPhase12:
    def test_comparative_fusion_multi_kernel_module(self, tmp_path):
        csv_dir = tmp_path / "t1"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "name": ["Cijk_A", "Cijk_B"],
                "source": ["trace1", "trace1"],
                "lowest_common_ancestor_id": [100, 100],
                "kernel_time": [5000.0, 3000.0],
                "gpu_op_uid": [10, 11],
            }
        ).to_csv(csv_dir / "diff_stats.csv", index=False)

        k1 = _kernel_event(10, "Cijk_A", dur=500)
        k2 = _kernel_event(11, "Cijk_B", dur=300)
        mod = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,8,128,64]]"},
        }
        mod2 = {
            "name": "nn.Module: Attn_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([mod, mod2], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)
        cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
        assert isinstance(cands, list)


class TestOrchestratorPhase12B:
    def _run_orch(self, tmp_path, monkeypatch, unified_rows, tree_events=None):

        out = str(tmp_path / "orch")

        _write_minimal_orchestrator_csvs(out, comparative=False)
        csv_dir = os.path.join(out, "perf_report_csvs")
        pd.DataFrame(unified_rows).to_csv(
            os.path.join(csv_dir, "unified_perf_summary.csv"), index=False
        )
        pd.DataFrame({"name": ["aten::op_0"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary.csv"), index=False
        )

        k = _kernel_event(0, "k0", dur=100)
        evts = tree_events or [
            {"name": "aten::mm", "_category": "aten", "gpu_events": [0], "ts": 0}
        ]
        tree = _StubTree(evts + [k], {0: k})
        analyzer = _StubAnalyzer(tree)

        class _FakeTPA:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTPA)
        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            RESNET_TRACE if os.path.isfile(RESNET_TRACE) else NORM_TRACE,
            "--platform",
            "MI300X",
            "--output-dir",
            out,
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        return out

    def test_bottleneck_nlargest_and_empty_category(self, tmp_path, monkeypatch):
        rows = []
        for i in range(20):
            rows.append(
                {
                    "name": f"aten::op_{i}",
                    "op category": "GEMM" if i > 0 else "",
                    "Kernel Time (µs)_sum": 100.0,
                    "total_duration_us": 60000.0,
                    "kernel_details_summary": f"[{{'name': 'k{i}'}}]",
                    "Data Moved (MB)": 1.0,
                    "perf_params": "{}",
                    "Input Dims": "[[2,3]]",
                }
            )
        out = self._run_orch(tmp_path, monkeypatch, rows)
        assert os.path.isfile(os.path.join(out, "category_data", "gemm_ops.csv"))

    def test_overlap_metrics_exception_path(self, tmp_path, monkeypatch):

        def boom(*args, **kwargs):
            raise RuntimeError("overlap boom")

        monkeypatch.setattr(GPUEventAnalyser, "compute_metrics_dict", boom)
        rows = [
            {
                "name": "aten::mm",
                "op category": "GEMM",
                "Kernel Time (µs)_sum": 1000.0,
                "total_duration_us": 60000.0,
                "kernel_details_summary": "[{'name': 'k0'}]",
                "Data Moved (MB)": 1.0,
                "perf_params": "{}",
                "Input Dims": "[[2,3]]",
            }
        ]
        out = self._run_orch(
            tmp_path,
            monkeypatch,
            rows,
            tree_events=[
                {
                    "name": "gemm_kernel",
                    "dur": 100,
                    "ts": 1000,
                    "UID": 0,
                    "_category": "kernel",
                    "cat": "kernel",
                    "args": {"stream": 0},
                }
            ],
        )
        data = json.loads(
            open(os.path.join(out, "category_data", "multi_kernel_data.json")).read()
        )
        assert "overlap_analysis" in data

    def test_sync_bottleneck_detection(self, tmp_path, monkeypatch):
        rows = [
            {
                "name": "aten::slow_sync",
                "op category": "GEMM",
                "Kernel Time (µs)_sum": 100.0,
                "total_duration_us": 5000000.0,
                "kernel_details_summary": "[{'name': 'k0'}]",
                "Data Moved (MB)": 1.0,
                "perf_params": "{}",
                "Input Dims": "[[2,3]]",
            }
        ]
        out = self._run_orch(tmp_path, monkeypatch, rows)
        meta_dir = os.path.join(out, "metadata")
        gemm_meta = os.path.join(meta_dir, "gemm_metadata.json")
        assert os.path.isfile(gemm_meta)


class TestKernelFusionMain:
    def test_kernel_fusion_standalone_main(self, tmp_path):

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        cat_dir = os.path.join(out, "category_data")
        os.makedirs(cat_dir, exist_ok=True)
        candidates = [
            {
                "module_name": "nn.Module: MLP_0",
                "base_name": "MLP",
                "instance_count": 1,
                "kernel_count": 2,
                "total_kernel_time_us": 800000.0,
                "kernels": [
                    {"name": "Cijk_a", "type": "GEMM", "dur_us": 500000},
                    {"name": "ew_add", "type": "elementwise", "dur_us": 300000},
                ],
            }
        ]
        with open(os.path.join(cat_dir, "fusion_candidates.json"), "w") as f:
            json.dump(candidates, f)
        with open(os.path.join(cat_dir, "category_manifest.json"), "w") as f:
            json.dump(
                {
                    "platform": "MI300X",
                    "comparison_scope": "standalone",
                    "gpu_utilization": {"total_time_ms": 1000.0},
                },
                f,
            )
        meta_dir = os.path.join(out, "metadata")
        os.makedirs(meta_dir)
        json.dump(
            {"peak_hbm_bw_tbs": 5.3, "max_achievable_tflops": {"matrix_fp32": 100}},
            open(os.path.join(meta_dir, "gemm_metadata.json"), "w"),
        )

        old_argv = sys.argv
        sys.argv = ["kernel_fusion_analysis", "--output-dir", out]
        try:
            kfa.main()
        finally:
            sys.argv = old_argv
        assert os.path.isfile(
            os.path.join(out, "category_data", "kernel_fusion_metrics.json")
        )


class TestOrchestratorPhase4:
    def test_orchestrator_comparative_with_fusion(self, tmp_path, monkeypatch):

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=True)
        csv_dir = os.path.join(out, "perf_report_trace1_csvs")
        pd.DataFrame(
            {
                "name": ["Cijk_A", "ew_add"],
                "source": ["trace1", "trace1"],
                "lowest_common_ancestor_id": [100, 100],
                "kernel_time": [5000.0, 3000.0],
                "gpu_op_uid": [10, 11],
            }
        ).to_csv(os.path.join(csv_dir, "diff_stats.csv"), index=False)

        k1 = _kernel_event(10, "Cijk_A", dur=500)
        k2 = _kernel_event(11, "ew_add", dur=300)
        mod = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        tree = _StubTree([mod], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)

        class _FakeTreePerfAnalyzer:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)

        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            "/fake/trace.json",
            "--platform",
            "MI300X",
            "--output-dir",
            out,
            "--comparison-scope",
            "comparative",
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        assert manifest["comparison_scope"] == "comparative"


class TestOrchestratorPhase6:
    def _run_main(self, tmp_path, monkeypatch, fusion_extract_raises=False):

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        csv_dir = os.path.join(out, "perf_report_csvs")
        pd.DataFrame(
            {
                "name": ["aten::mm", "aten::add"],
                "op category": ["GEMM", ""],
                "Kernel Time (µs)_sum": [1000.0, 500.0],
                "total_duration_us": [60000.0, 1000.0],
                "kernel_details_summary": [
                    "[{'name': 'Cijk_a'}]",
                    "[{'name': 'ew_add'}]",
                ],
                "Data Moved (MB)": [10.0, 1.0],
                "perf_params": ["{}", "{}"],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
            }
        ).to_csv(os.path.join(csv_dir, "unified_perf_summary.csv"), index=False)
        pd.DataFrame({"name": ["aten::mm"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary.csv"), index=False
        )

        events = [
            {
                "name": "gemm_kernel",
                "dur": 100,
                "ts": 1000,
                "UID": 0,
                "_category": "kernel",
                "cat": "kernel",
                "args": {"stream": 0},
            },
            {
                "name": "MemcpyHtoD",
                "dur": 20,
                "ts": 1100,
                "UID": 1,
                "_category": "gpu_memcpy",
                "cat": "gpu_memcpy",
                "args": {"bytes": 4096, "stream": 1},
            },
            {
                "name": "MemcpyDtoH",
                "dur": 25,
                "ts": 1120,
                "UID": 2,
                "_category": "gpu_memcpy",
                "cat": "gpu_memcpy",
                "args": {"bytes": 8192, "stream": 1},
            },
            {
                "name": "MemcpyDtoD",
                "dur": 15,
                "ts": 1150,
                "UID": 3,
                "_category": "gpu_memcpy",
                "cat": "gpu_memcpy",
                "args": {"bytes": 2048, "stream": 1},
            },
            {
                "name": "MemcpyCustom",
                "dur": 10,
                "ts": 1170,
                "UID": 4,
                "_category": "gpu_memcpy",
                "cat": "gpu_memcpy",
                "args": {"bytes": 1024, "stream": 1},
            },
            {
                "name": "ncclKernel_AllReduce",
                "dur": 40,
                "ts": 1200,
                "UID": 5,
                "_category": "kernel",
                "cat": "kernel",
                "args": {"stream": 2},
            },
        ]
        tree = _StubTree(events, {i: e for i, e in enumerate(events)})
        analyzer = _StubAnalyzer(tree)

        class _FakeTreePerfAnalyzer:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
        if fusion_extract_raises:
            monkeypatch.setattr(
                op,
                "_extract_standalone_fusion_candidates",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fusion fail")),
            )
        else:
            k1 = _kernel_event(10, "Cijk_a", dur=500)
            k2 = _kernel_event(11, "vectorized_elementwise_kernel add", dur=300)
            module = {
                "name": "nn.Module: MLP_0",
                "_category": "aten",
                "gpu_events": [10, 11],
                "args": {"Input Dims": "[[2,3]]"},
            }
            tree2 = _StubTree([module], {10: k1, 11: k2})
            monkeypatch.setattr(
                op,
                "_extract_standalone_fusion_candidates",
                lambda a, t, d: _extract_standalone_fusion_candidates(
                    _StubAnalyzer(tree2), tree2, d
                ),
            )

        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            "/fake/trace.json",
            "--platform",
            "MI300X",
            "--output-dir",
            out,
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        return out

    def test_main_no_time_column_sync_and_memcpy_dirs(self, tmp_path, monkeypatch):
        out = self._run_main(tmp_path, monkeypatch)
        mk = json.loads(
            open(os.path.join(out, "category_data", "multi_kernel_data.json")).read()
        )
        dirs = mk["memcpy_summary"]["by_direction"]
        assert "H2D" in dirs and "D2H" in dirs and "D2D" in dirs and "other" in dirs
        assert mk["nccl_summary"]["total_count"] >= 1
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        cats = {c["name"]: c for c in manifest["categories"]}
        assert "gemm" in cats
        assert cats["gemm"].get("has_sync_bottleneck") is True

    def test_main_fusion_extraction_failure_writes_empty(self, tmp_path, monkeypatch):
        out = self._run_main(tmp_path, monkeypatch, fusion_extract_raises=True)
        fusion = json.loads(
            open(os.path.join(out, "category_data", "fusion_candidates.json")).read()
        )
        assert fusion == []

    def test_standalone_enrichment_without_data_mb(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_gemm_a", dur=500)
        k2 = _kernel_event(11, "vectorized_elementwise_kernel add", dur=300)
        module = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        tree = _StubTree([module], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_gemm_a'}]",
                    "[{'name': 'vectorized_elementwise_kernel add'}]",
                ],
                "op category": ["GEMM", "elementwise"],
                "Data Moved (MB)": [10.0, 4.0],
                "perf_params": ["{'M':2,'N':4,'K':3}", "{'shape_in1': [4, 4]}"],
                "Input Dims": ["[[2,3]]", "[[4,4]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)


class TestKernelFusionMainPhase6:
    def test_kernel_fusion_main_end_to_end(self, tmp_path):

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        cat_dir = os.path.join(out, "category_data")
        os.makedirs(cat_dir, exist_ok=True)
        fusion_path = os.path.join(cat_dir, "fusion_candidates.json")
        with open(fusion_path, "w") as f:
            json.dump(
                [
                    {
                        "module_name": "nn.Module: MLP_0",
                        "base_name": "MLP",
                        "kernels": [
                            {"name": "Cijk_a", "type": "GEMM", "dur_us": 500},
                            {"name": "ew_add", "type": "elementwise", "dur_us": 300},
                        ],
                        "total_kernel_time_us": 800,
                        "instance_count": 1,
                    }
                ],
                f,
            )
        old_argv = sys.argv
        sys.argv = ["kernel_fusion_analysis", "--output-dir", out]
        try:
            kfa.main()
        finally:
            sys.argv = old_argv
        cat_dir = os.path.join(out, "category_data")
        assert os.path.isfile(os.path.join(cat_dir, "kernel_fusion_metrics.json"))


class TestOrchestratorPush95:
    def test_comparative_without_tree(self, tmp_path, capsys):
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame({"lowest_common_ancestor_id": [1], "gpu_op_uid": [10]}).to_csv(
            csv_dir / "diff_stats.csv", index=False
        )
        assert _extract_comparative_fusion_candidates(str(csv_dir)) == []
        assert "tree/analyzer not provided" in capsys.readouterr().out

    def test_comparative_empty_uid_lookup(self, tmp_path, capsys):
        csv_dir = tmp_path / "csv2"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "lowest_common_ancestor_id": [1],
                "name": ["k"],
                "source": ["trace2"],
            }
        ).to_csv(csv_dir / "diff_stats.csv", index=False)
        tree = _StubTree([], {})
        analyzer = _StubAnalyzer(tree)
        assert (
            _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree) == []
        )
        assert "No trace1 kernel UIDs" in capsys.readouterr().out

    def test_standalone_gemm_norm_only_skipped(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_gemm_a")
        k2 = _kernel_event(11, "rmsnorm2d")
        mod = {
            "name": "nn.Module: Block_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([mod], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)
        csv_dir = tmp_path / "standalone_csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_gemm_a'}]",
                    "[{'name': 'rmsnorm2d'}]",
                ],
                "op category": ["GEMM", "NORM_fwd"],
                "Data Moved (MB)": [1.0, 1.0],
                "perf_params": ["{}", "{}"],
                "Input Dims": ["[[2,3]]", "[[2,3]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)


def test_comparative_fusion_full_path(tmp_path):
    csv_dir = tmp_path / "trace1_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "name": ["Cijk_A", "Cijk_B", "some_op"],
            "source": ["trace1", "trace1", "trace2"],
            "lowest_common_ancestor_id": [100, 100, 100],
            "kernel_time": [5000.0, 3000.0, 1000.0],
            "gpu_op_uid": [10, 11, None],
        }
    ).to_csv(csv_dir / "diff_stats.csv", index=False)
    uid_map = {
        10: {
            "name": "Cijk_A",
            "dur": 5000,
            "_category": "kernel",
            "gpu_events": [],
        },
        11: {
            "name": "Cijk_B",
            "dur": 3000,
            "_category": "kernel",
            "gpu_events": [],
        },
    }
    module = {
        "name": "nn.Module: Attn_0",
        "_category": "aten",
        "gpu_events": [10, 11],
        "args": {},
    }
    tree = _StubTree([module], uid_map)
    analyzer = _StubAnalyzer(tree)
    cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
    assert len(cands) >= 1


def test_analysis_utils_efficiency_and_fusion(tmp_path):

    row = pd.Series(
        {
            "FLOPS/Byte": 10.0,
            "TFLOPS/s_mean": 100.0,
            "TB/s_mean": 1.0,
            "Roofline Bound": "COMPUTE_BOUND",
            "Compute Spec": "matrix_bf16",
        }
    )
    result = au.calculate_efficiency(
        row, peak_maf_or_maf_dict={"matrix_bf16": 1000}, peak_hbm_bw=5300
    )
    assert result["bound_type"] == "compute"

    fusion_dir = tmp_path / "category_data"
    fusion_dir.mkdir()
    (fusion_dir / "kernel_fusion_metrics.json").write_text(
        json.dumps({"high_confidence_kernel_map": {"gemm_kernel": "fused_gemm"}})
    )
    assert au._load_fusion_map(str(tmp_path))["gemm_kernel"] == "fused_gemm"
    assert (
        au._match_fusion_op("{'name': 'gemm_kernel'}", {"gemm_kernel": "fused"})
        == "fused"
    )


def test_orchestrator_comparative_main(tmp_path, monkeypatch):

    out = str(tmp_path)
    _write_minimal_orchestrator_csvs(out, comparative=True)
    k1 = _kernel_event(10, "Cijk_a")
    k2 = _kernel_event(11, "ew_add")
    module = {
        "name": "nn.Module: MLP_0",
        "_category": "aten",
        "gpu_events": [10, 11],
    }
    tree = _StubTree([module], {10: k1, 11: k2})
    analyzer = _StubAnalyzer(tree)

    class _FakeTreePerfAnalyzer:
        @classmethod
        def from_file(cls, *args, **kwargs):
            return analyzer

    monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
    monkeypatch.setattr(
        op, "_extract_comparative_fusion_candidates", lambda *a, **k: []
    )
    monkeypatch.setattr(op, "_extract_standalone_fusion_candidates", lambda *a, **k: [])

    old_argv = sys.argv
    sys.argv = [
        "orchestrator_prepare",
        "--trace-path",
        "/fake/trace.json",
        "--platform",
        "MI300X",
        "--output-dir",
        out,
        "--comparison-scope",
        "comparative",
    ]
    try:
        op.main()
    finally:
        sys.argv = old_argv
    manifest = json.loads(
        open(os.path.join(out, "category_data", "category_manifest.json")).read()
    )
    assert manifest["comparison_scope"] == "comparative"


def test_kernel_fusion_and_tracediff_helpers(tmp_path):

    ops = [
        {
            "kernel_names": ["Cijk_a", "ew_add"],
            "base_name": "Block",
            "instance_count": 1,
        },
        {
            "kernel_names": ["Cijk_a", "ew_add"],
            "base_name": "Block",
            "instance_count": 1,
        },
    ]
    filtered = kfa._filter_and_dedup(ops)
    assert len(filtered) == 1

    df = pd.DataFrame({"name": ["aten::mm"], "Kernel Time (µs)_mean": [100.0]})
    diff = pd.DataFrame(
        {
            "name": ["aten::mm"],
            "kernel_time_delta_pct": [10.0],
            "lowest_common_ancestor_id": [1],
        }
    )
    report = {"unified_perf_summary": df}
    tde.enrich_perf_report_dict_inplace(report, diff)
    assert "unified_perf_summary" in report


class TestOrchestratorHelpersSweep:
    def test_helper_functions(self):

        assert op._strip_module_index("nn.Module: MLP_0") == "MLP"
        assert op._is_fusion_eligible("Cijk_gemm_kernel")
        assert not op._is_fusion_eligible("flash_attn_fused_kernel")
        tl = pd.DataFrame(
            {
                "type": ["total_time", "computation_time", "idle_time"],
                "time ms": [1000.0, 900.0, 100.0],
                "percent": [100.0, 90.0, 10.0],
            }
        )
        metrics = op._gpu_utilization_metrics_from_gpu_timeline_df(tl)
        assert metrics["total_time_ms"] == 1000.0


class TestOrchestratorPush95Coverage:
    def test_attention_core_narrowing(self):
        perf_lookup = {
            "Cijk_QK": {
                None: {"op_category": "GEMM", "data_in_mb": 1.0, "data_out_mb": 1.0},
            },
            "Cijk_PV": {
                None: {"op_category": "GEMM", "data_in_mb": 2.0, "data_out_mb": 2.0},
            },
            "softmax_kernel": {
                None: {
                    "op_category": "SDPA_fwd",
                    "data_in_mb": 0.5,
                    "data_out_mb": 0.5,
                },
            },
        }
        kernels = [
            {"name": "Cijk_QK", "type": "GEMM", "dur_us": 100},
            {"name": "softmax_kernel", "type": "SDPA", "dur_us": 50},
            {"name": "Cijk_PV", "type": "GEMM", "dur_us": 120},
            {
                "name": "vectorized_elementwise_kernel",
                "type": "Elementwise",
                "dur_us": 10,
            },
        ]
        core = _extract_attention_core(kernels, perf_lookup)
        assert core is not None
        assert len(core) == 3
        assert core[1]["name"] == "softmax_kernel"

    def test_gemm_norm_only_detection(self):
        entry = {
            "kernels": [
                {"name": "Cijk_gemm", "type": "GEMM"},
                {"name": "rmsnorm2d_kernel", "type": "NORM"},
            ]
        }
        assert _is_gemm_norm_only(entry) is True

    def test_standalone_attention_enrichment_and_duplicate_base(self, tmp_path):
        k_qk = _kernel_event(10, "Cijk_QK_gemm", dur=500)
        k_sm = _kernel_event(11, "softmax_warp_forward", dur=200)
        k_pv = _kernel_event(12, "Cijk_PV_gemm", dur=400)
        mod1 = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11, 12],
            "args": {"Input Dims": "[[2,8,128,64]]"},
        }
        mod2 = {
            "name": "nn.Module: Attn_1",
            "_category": "aten",
            "gpu_events": [10, 11, 12],
        }
        tree = _StubTree([mod1, mod2], {10: k_qk, 11: k_sm, 12: k_pv})
        analyzer = _StubAnalyzer(tree)

        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_QK_gemm'}]",
                    "[{'name': 'softmax_warp_forward'}]",
                    "[{'name': 'Cijk_PV_gemm'}]",
                ],
                "op category": ["GEMM", "SDPA_fwd", "GEMM"],
                "Data Moved (MB)": [10.0, 2.0, 8.0],
                "perf_params": ["{'M':2}", "{}", "{'M':2}"],
                "Input Dims": ["[[2,8,128,64]]", "[[2,8,128,64]]", "[[2,8,128,64]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)

    def test_standalone_keyerror_on_missing_uid(self, tmp_path):
        mod = {
            "name": "nn.Module: Broken_0",
            "_category": "aten",
            "gpu_events": [10, 99],
        }
        tree = _StubTree([mod], {10: _kernel_event(10, "Cijk_a")})
        analyzer = _StubAnalyzer(tree)
        csv_dir = tmp_path / "csv2"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": ["[{'name': 'Cijk_a'}]"],
                "op category": ["GEMM"],
                "Data Moved (MB)": [1.0],
                "perf_params": ["{}"],
                "Input Dims": ["[[1,1]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)
        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)

    def test_comparative_duplicate_base_accumulation(self, tmp_path):
        csv_dir = tmp_path / "trace1_csvs"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "name": ["Cijk_A", "Cijk_B"],
                "source": ["trace1", "trace1"],
                "lowest_common_ancestor_id": [100, 100],
                "kernel_time": [5000.0, 3000.0],
                "gpu_op_uid": [10, 11],
            }
        ).to_csv(csv_dir / "diff_stats.csv", index=False)
        uid_map = {
            10: {
                "name": "Cijk_A",
                "dur": 5000,
                "_category": "kernel",
                "gpu_events": [],
            },
            11: {
                "name": "Cijk_B",
                "dur": 3000,
                "_category": "kernel",
                "gpu_events": [],
            },
        }
        mod_a = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        mod_b = {
            "name": "nn.Module: Attn_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([mod_a, mod_b], uid_map)
        analyzer = _StubAnalyzer(tree)
        cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
        assert isinstance(cands, list)

    def test_orchestrator_main_alt_ops_summary_column(self, tmp_path, monkeypatch):

        out = str(tmp_path)
        csv_dir = os.path.join(out, "perf_report_csvs")
        os.makedirs(csv_dir)
        pd.DataFrame(
            {
                "type": ["total_time", "computation_time", "idle_time"],
                "time ms": [1000.0, 900.0, 100.0],
                "percent": [100.0, 90.0, 10.0],
            }
        ).to_csv(os.path.join(csv_dir, "gpu_timeline.csv"), index=False)
        pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Kernel Time (µs)_sum": [800000.0],
                "op category": ["GEMM"],
            }
        ).to_csv(os.path.join(csv_dir, "ops_summary.csv"), index=False)
        pd.DataFrame(
            {
                "name": ["aten::mm"],
                "op category": ["GEMM"],
                "Kernel Time (µs)": [800.0],
            }
        ).to_csv(os.path.join(csv_dir, "unified_perf_summary.csv"), index=False)
        pd.DataFrame({"name": ["aten::mm"], "op category": ["GEMM"]}).to_csv(
            os.path.join(csv_dir, "ops_summary_by_category.csv"), index=False
        )

        k1 = _kernel_event(10, "Cijk_a")
        k2 = _kernel_event(11, "ew_add")
        module = {
            "name": "nn.Module: MLP_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([module], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)

        class _FakeTreePerfAnalyzer:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
        monkeypatch.setattr(
            op, "_extract_standalone_fusion_candidates", lambda *a, **k: []
        )

        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            "/fake/trace.json",
            "--platform",
            "MI300X",
            "--output-dir",
            out,
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        assert manifest["comparison_scope"] == "standalone"


class TestOrchestratorPrepareFinal:
    def test_standalone_sibling_sequence_and_duplicate_base(self, tmp_path):
        k1 = _kernel_event(10, "Cijk_gemm_a", dur=500)
        k2 = _kernel_event(11, "vectorized_elementwise_kernel add", dur=300)
        k3 = _kernel_event(12, "Cijk_gemm_b", dur=400)
        parent = {
            "name": "aten::linear",
            "_category": "aten",
            "gpu_events": [12],
            "parent": None,
            "args": {"Input Dims": "[[2,3]]"},
        }
        mod1 = {
            "name": "nn.Module: MLP_0",
            "_category": "aten",
            "gpu_events": [10, 11],
            "args": {"Input Dims": "[[2,3]]"},
        }
        mod2 = {
            "name": "nn.Module: MLP_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        child_op = {
            "name": "aten::add",
            "_category": "aten",
            "gpu_events": [12],
            "parent": 99,
        }
        uid_map = {10: k1, 11: k2, 12: k3, 99: parent}
        tree = _StubTree(
            [mod1, mod2, parent, child_op], uid_map, parent_map={id(child_op): parent}
        )
        unified = [
            {"name": "aten::mm", "gpu_events": [10], "parent": 99},
            {"name": "aten::relu", "gpu_events": [11], "parent": 99},
        ]
        analyzer = _StubAnalyzer(tree, unified_events=unified)

        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "kernel_details_summary": [
                    "[{'name': 'Cijk_gemm_a'}]",
                    "[{'name': 'vectorized_elementwise_kernel add'}]",
                    "[{'name': 'Cijk_gemm_b'}]",
                ],
                "op category": ["GEMM", "elementwise", "GEMM"],
                "Data Moved (MB)": [10.0, 4.0, 8.0],
                "perf_params": ["{}", "{}", "{}"],
                "Input Dims": ["[[2,3]]", "[[4,4]]", "[[2,3]]"],
            }
        ).to_csv(csv_dir / "unified_perf_summary.csv", index=False)

        cands = _extract_standalone_fusion_candidates(analyzer, tree, str(csv_dir))
        assert isinstance(cands, list)

    def test_comparative_duplicate_base_accumulation(self, tmp_path, capsys):
        csv_dir = tmp_path / "trace1_csvs"
        csv_dir.mkdir()
        pd.DataFrame(
            {
                "name": ["Cijk_A", "Cijk_B"],
                "source": ["trace1", "trace1"],
                "lowest_common_ancestor_id": [100, 100],
                "kernel_time": [5000.0, 3000.0],
                "gpu_op_uid": [10, 11],
            }
        ).to_csv(csv_dir / "diff_stats.csv", index=False)

        uid_map = {
            10: {
                "name": "Cijk_A",
                "dur": 5000,
                "_category": "kernel",
                "gpu_events": [],
            },
            11: {
                "name": "Cijk_B",
                "dur": 3000,
                "_category": "kernel",
                "gpu_events": [],
            },
        }
        mod_a = {
            "name": "nn.Module: Attn_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        mod_b = {
            "name": "nn.Module: Attn_1",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([mod_a, mod_b], uid_map)
        analyzer = _StubAnalyzer(tree)
        cands = _extract_comparative_fusion_candidates(str(csv_dir), analyzer, tree)
        assert isinstance(cands, list)


class TestOrchestratorMainExtended:
    def test_orchestrator_disable_pseudo_ops(self, tmp_path, monkeypatch):

        out = str(tmp_path)
        _write_minimal_orchestrator_csvs(out, comparative=False)
        k1 = _kernel_event(10, "Cijk_a")
        k2 = _kernel_event(11, "ew_add")
        module = {
            "name": "nn.Module: MLP_0",
            "_category": "aten",
            "gpu_events": [10, 11],
        }
        tree = _StubTree([module], {10: k1, 11: k2})
        analyzer = _StubAnalyzer(tree)

        class _FakeTreePerfAnalyzer:
            @classmethod
            def from_file(cls, *args, **kwargs):
                return analyzer

        monkeypatch.setattr(op, "TreePerfAnalyzer", _FakeTreePerfAnalyzer)
        monkeypatch.setattr(
            op, "_extract_standalone_fusion_candidates", lambda *a, **k: []
        )

        old_argv = sys.argv
        sys.argv = [
            "orchestrator_prepare",
            "--trace-path",
            "/fake/trace.json",
            "--platform",
            "MI300X",
            "--output-dir",
            out,
            "--disable_pseudo_ops",
        ]
        try:
            op.main()
        finally:
            sys.argv = old_argv
        manifest = json.loads(
            open(os.path.join(out, "category_data", "category_manifest.json")).read()
        )
        assert manifest["comparison_scope"] == "standalone"
