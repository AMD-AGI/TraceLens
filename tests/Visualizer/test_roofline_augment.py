###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the orchestrator roofline-augmentation utility."""

from __future__ import annotations

import json

import pytest

from TraceLens.Visualizer.model_explorer_export.roofline_augment import (
    achieved_from_measurement,
    augment,
    augment_phases,
    classify_kernel,
    merge_into_priority_data,
    phase_argvs,
    report_notes,
    summarize_projection,
)
from TraceLens.Visualizer.model_explorer_export.roofline_cli import build_parser
from TraceLens.Visualizer.model_explorer_export.roofline_projection import (
    RooflineOp,
    RooflineProjection,
)


def test_classify_kernel_by_name_not_module():
    assert classify_kernel("_inkling_rel_attention_kernel.kd") == "Attention"
    assert classify_kernel("fused_moe_kernel.kd") == "MoE"
    # A bare Tensile GEMM is a GEMM even when launched from an attention module.
    assert classify_kernel("Cijk_Alik_Bljk_BBS_BH.kd", ["Model", "Attention"]) == "GEMM"
    assert classify_kernel("__amd_rocclr_copyBuffer.kd") is None
    assert classify_kernel("") is None


def _op(category, modeled, flops, bytes_, bound, *, instances=66):
    return RooflineOp(
        name=category.lower(),
        section="decoder",
        computation=category,
        operation="",
        category=category if modeled else None,
        modeled=modeled,
        reason=None if modeled else "unmodeled",
        output_elements=1,
        instances=instances,
        flops=flops,
        bytes=bytes_,
        roofline_time_us=1000.0 if modeled else None,
        roofline_bound=bound,
    )


def test_summarize_projection_derives_ai_bound_and_per_gpu():
    proj = RooflineProjection(batch_size=32, seq_len=1)
    # Attention: AI = 1e12 / 1.25e11 = 8.0, memory-bound.
    proj.ops.append(_op("Attention", True, 1.0e12, 1.25e11, "MEMORY_BOUND"))
    # MoE: AI = 2e12 / 5e9 = 400, compute-bound.
    proj.ops.append(_op("MoE", True, 2.0e12, 5.0e9, "COMPUTE_BOUND"))
    # Unmodeled op must be ignored.
    proj.ops.append(_op("other", False, None, None, None))

    summary = summarize_projection(proj, tp=8, ridge=200.7)
    assert set(summary) == {"Attention", "MoE"}
    assert summary["Attention"]["arithmetic_intensity"] == 8.0
    assert summary["Attention"]["bound"] == "MEMORY_BOUND"
    assert summary["Attention"]["vs_ridge"] == "below"
    # per-GPU floor = total roofline time (1000us * 66 instances) / tp.
    assert summary["Attention"]["roofline_us_per_gpu"] == (1000.0 * 66) / 8
    assert summary["MoE"]["arithmetic_intensity"] == 400.0
    assert summary["MoE"]["vs_ridge"] == "above"
    # Per-invocation (one layer) FLOPs/bytes of the dominant op are carried for
    # the measured-throughput comparison.
    assert summary["Attention"]["flops_per_invocation"] == 1.0e12
    assert summary["Attention"]["bytes_per_invocation"] == 1.25e11


def test_achieved_from_measurement_uses_ceiling_of_bound():
    arch = {"peak_tflops": 1000.0, "mem_bw_gbps": 1000.0}
    # Memory-bound: 1e11 bytes/invocation * 10 invocations / 1s = 1000 GB/s = 100% BW.
    mem = achieved_from_measurement(
        {
            "bound": "MEMORY_BOUND",
            "flops_per_invocation": 0.0,
            "bytes_per_invocation": 1.0e11,
        },
        time_ms=1000.0,
        count=10,
        arch=arch,
    )
    assert mem["ceiling"] == "memory bandwidth"
    assert round(mem["achieved_gbps"]) == 1000
    assert round(mem["pct_of_peak"]) == 100
    # Compute-bound: 1e14 FLOPs/invocation * 10 / 1s = 1000 TFLOP/s = 100% peak.
    comp = achieved_from_measurement(
        {
            "bound": "COMPUTE_BOUND",
            "flops_per_invocation": 1.0e14,
            "bytes_per_invocation": 0.0,
        },
        time_ms=1000.0,
        count=10,
        arch=arch,
    )
    assert comp["ceiling"] == "compute peak"
    assert round(comp["pct_of_peak"]) == 100
    # No measured time/count -> no achieved block.
    assert achieved_from_measurement({"bound": "MEMORY_BOUND"}, None, 10, arch) is None


def test_augment_picks_dominant_unmodeled_kernel_per_category(tmp_path):
    cat_dir = tmp_path / "category_data"
    cat_dir.mkdir()
    (cat_dir / "other_metrics.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "kernel_name": "_inkling_rel_attention_kernel.kd",
                        "time_ms": 8000.0,
                        "count": 528,
                        "impact_estimate_method": "heuristic",
                        "module_chain": ["Model", "Attention"],
                    },
                    {
                        "kernel_name": "fused_moe_kernel.kd",
                        "time_ms": 1600.0,
                        "count": 1024,
                        "impact_estimate_method": "heuristic",
                    },
                    # Smaller MoE aux kernel: same category, must NOT be picked
                    # (only the dominant per category is augmented).
                    {
                        "kernel_name": "moe_align_block_size_kernel.kd",
                        "time_ms": 90.0,
                        "count": 512,
                        "impact_estimate_method": "heuristic",
                    },
                    # Already trace-modeled: must be skipped.
                    {
                        "kernel_name": "sdpa_modeled.kd",
                        "time_ms": 50.0,
                        "count": 10,
                        "impact_estimate_method": "modeled",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "Attention": {
            "bound": "MEMORY_BOUND",
            "arithmetic_intensity": 8.0,
            "roofline_us_per_gpu": 1705.0,
            "roofline_time_us": 13640.0,
            "flops": 1.0,
            "bytes": 1.0,
            "flops_per_invocation": 0.0,
            "bytes_per_invocation": 1.0e9,
            "vs_ridge": "below",
        },
        "MoE": {
            "bound": "COMPUTE_BOUND",
            "arithmetic_intensity": 608.0,
            "roofline_us_per_gpu": 124.0,
            "roofline_time_us": 992.0,
            "flops": 1.0,
            "bytes": 1.0,
            "flops_per_invocation": 1.0e10,
            "bytes_per_invocation": 0.0,
            "vs_ridge": "above",
        },
    }
    regime = {"phase": "decode", "tp": 8}
    arch = {"peak_tflops": 1445.0, "mem_bw_gbps": 7200.0}
    payload = augment(tmp_path, summary, regime, ridge=200.7, arch=arch)

    kernels = {e["kernel"]: e for e in payload["entries"]}
    assert set(kernels) == {"_inkling_rel_attention_kernel.kd", "fused_moe_kernel.kd"}
    attn = kernels["_inkling_rel_attention_kernel.kd"]
    assert attn["roofline_category"] == "Attention"
    assert attn["bound"] == "MEMORY_BOUND"
    # Achieved throughput is computed from measured time/count against config FLOPs/bytes.
    assert attn["achieved"] is not None
    assert attn["achieved"]["ceiling"] == "memory bandwidth"
    assert kernels["fused_moe_kernel.kd"]["roofline_category"] == "MoE"
    # The JSON sidecar is written and report_notes reads it back keyed by kernel.
    assert (tmp_path / "metadata" / "roofline_augmentation.json").is_file()
    notes = report_notes(tmp_path)
    assert set(notes) == {"_inkling_rel_attention_kernel.kd", "fused_moe_kernel.kd"}
    assert "Analytical roofline" in notes["_inkling_rel_attention_kernel.kd"]


def test_report_notes_absent_returns_empty(tmp_path):
    assert report_notes(tmp_path) == {}


def test_merge_into_priority_data_is_additive(tmp_path):
    (tmp_path / "priority_data.json").write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "category": "other",
                        "global_rank": 1,
                        "estimate_method": "heuristic",
                        "bound_type": None,
                        "members": [
                            {
                                "operation": "hipModuleLaunchKernel->_inkling_rel_attention_kernel.kd (Synthetic Op)",
                                "efficiency_pct": None,
                            }
                        ],
                    },
                    {
                        "category": "elementwise",
                        "global_rank": 2,
                        "members": [{"operation": "aten::copy_"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "kernel": "_inkling_rel_attention_kernel.kd",
            "roofline_category": "Attention",
            "bound": "MEMORY_BOUND",
            "arithmetic_intensity": 8.0,
            "roofline_us_per_gpu": 1705.0,
            "achieved": {"pct_of_peak": 1.3, "ceiling": "memory bandwidth"},
            "note": "> roofline note",
        }
    ]
    merged = merge_into_priority_data(tmp_path, entries)
    assert merged == 1
    data = json.loads((tmp_path / "priority_data.json").read_text())
    f0, f1 = data["findings"]
    # The matching finding gains an additive roofline block...
    assert f0["roofline"]["bound"] == "MEMORY_BOUND"
    assert f0["roofline"]["source"] == "analytical-model-explorer"
    # ...without disturbing existing fields.
    assert f0["estimate_method"] == "heuristic"
    assert f0["bound_type"] is None
    assert f0["members"][0]["efficiency_pct"] is None
    # Non-matching finding is untouched.
    assert "roofline" not in f1


def _attention_summary(bound, ai):
    return {
        "bound": bound,
        "arithmetic_intensity": ai,
        "roofline_us_per_gpu": 1705.0 if bound == "MEMORY_BOUND" else 4000.0,
        "flops": 1.0,
        "bytes": 1.0,
        "flops_per_invocation": 0.0,
        "bytes_per_invocation": 1.0e9,
        "vs_ridge": "below",
    }


def test_augment_phases_omits_achieved_when_time_is_not_split(tmp_path):
    cat_dir = tmp_path / "category_data"
    cat_dir.mkdir()
    (cat_dir / "other_metrics.json").write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "kernel_name": "_inkling_rel_attention_kernel.kd",
                        "time_ms": 8000.0,
                        "count": 528,
                        "impact_estimate_method": "heuristic",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    arch = {"peak_tflops": 1445.0, "mem_bw_gbps": 7200.0}
    payload = augment_phases(
        tmp_path,
        [
            {
                "summary": {"Attention": _attention_summary("MEMORY_BOUND", 8.0)},
                "regime": {"phase": "decode", "batch_size": 32, "seq_len": 1},
                "ridge": 200.7,
                "arch": arch,
            },
            {
                "summary": {"Attention": _attention_summary("COMPUTE_BOUND", 200.0)},
                "regime": {"phase": "prefill", "batch_size": 1, "seq_len": 8192},
                "ridge": 200.7,
                "arch": arch,
            },
        ],
    )
    assert [entry["kernel"] for entry in payload["entries"]] == [
        "_inkling_rel_attention_kernel.kd"
    ]
    entry = payload["entries"][0]
    assert [phase["phase"] for phase in entry["phases"]] == ["decode", "prefill"]
    assert entry["achieved"] is None
    assert "decode" in entry["note"] and "prefill" in entry["note"]
    assert "no achieved-%" in entry["note"]


def test_merge_attaches_the_matching_phase(tmp_path):
    operation = "hipModuleLaunchKernel->_inkling_rel_attention_kernel.kd"
    (tmp_path / "priority_data.json").write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "phase": "decode",
                        "estimate_method": "heuristic",
                        "members": [{"operation": operation}],
                    },
                    {
                        "phase": "prefill",
                        "estimate_method": "heuristic",
                        "members": [{"operation": operation}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "kernel": "_inkling_rel_attention_kernel.kd",
            "roofline_category": "Attention",
            "bound": "MEMORY_BOUND",
            "arithmetic_intensity": 8.0,
            "roofline_us_per_gpu": 1705.0,
            "achieved": None,
            "note": "> combined",
            "phases": [
                {
                    "phase": "decode",
                    "bound": "MEMORY_BOUND",
                    "arithmetic_intensity": 8.0,
                    "roofline_us_per_gpu": 1705.0,
                    "achieved": {"pct_of_peak": 1.3, "ceiling": "memory bandwidth"},
                    "note": "> decode note",
                },
                {
                    "phase": "prefill",
                    "bound": "COMPUTE_BOUND",
                    "arithmetic_intensity": 200.0,
                    "roofline_us_per_gpu": 4000.0,
                    "achieved": {"pct_of_peak": 12.0, "ceiling": "compute peak"},
                    "note": "> prefill note",
                },
            ],
        }
    ]
    assert merge_into_priority_data(tmp_path, entries) == 2
    decode, prefill = json.loads((tmp_path / "priority_data.json").read_text())[
        "findings"
    ]
    assert decode["roofline"]["bound"] == "MEMORY_BOUND"
    assert decode["roofline"]["phase"] == "decode"
    assert decode["roofline"]["note"] == "> decode note"
    assert decode["estimate_method"] == "heuristic"
    assert prefill["roofline"]["bound"] == "COMPUTE_BOUND"
    assert prefill["roofline"]["note"] == "> prefill note"


def test_phase_argvs_skips_unresolved_and_builds_one_argv_per_phase():
    assert (
        phase_argvs(
            {
                "source": None,
                "phases": [{"phase": "decode", "batch_size": 1, "seq_len": 1}],
            }
        )
        == []
    )
    assert phase_argvs({"source": "org/model", "phases": []}) == []
    argvs = phase_argvs(
        {
            "source": "org/model",
            "engine": "vllm",
            "tp": 8,
            "pp": 1,
            "ep": 2,
            "gpu_arch_name": "MI300X",
            "phases": [
                {
                    "phase": "decode",
                    "batch_size": 32,
                    "seq_len": 1,
                    "dims": {"past_len": 65536},
                },
                {"phase": "prefill", "batch_size": 1, "seq_len": 8192},
            ],
        }
    )
    assert len(argvs) == 2
    parsed = build_parser().parse_args(argvs[0])
    assert parsed.source == "org/model"
    assert parsed.phase == "decode"
    assert parsed.batch_size == 32
    assert parsed.seq_len == 1
    assert parsed.tp == 8
    assert parsed.ep == 2
    assert parsed.engine == "vllm"
    assert parsed.gpu_arch_name == "MI300X"
    assert parsed.dim == ["past_len=65536"]
    assert build_parser().parse_args(argvs[1]).phase == "prefill"
    with pytest.raises(ValueError):
        phase_argvs(
            {
                "source": "org/model",
                "phases": [{"phase": "train", "batch_size": 1, "seq_len": 1}],
            }
        )
