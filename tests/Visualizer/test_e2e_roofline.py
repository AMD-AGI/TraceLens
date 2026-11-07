###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the projection-plus-profile end-to-end roofline plot."""

from __future__ import annotations

import json

import matplotlib
import pytest

matplotlib.use("Agg")

from TraceLens.Visualizer.model_explorer_export.e2e_roofline import (
    Regime,
    achieved_tflops,
    bandwidth_tb_s,
    blend_peak_tflops,
    build_points,
    load_profile_kernels,
    main,
    plot_e2e_roofline,
    regimes_from_augmentation,
    ridge_flops_per_byte,
)

_ARCH = {
    "name": "TestGPU",
    "mem_bw_gbps": 8000,
    "max_achievable_tflops": {"matrix_bf16": 2500, "matrix_fp4": 10100},
}


def _op(category, flops, bytes_, *, per_launch=None, spec=None):
    return {
        "category": category,
        "total_flops": flops,
        "total_bytes": bytes_,
        "compute_spec": spec,
        "flops_per_launch": per_launch,
    }


def _kernel(name, time_us, count):
    return {"name": name, "time_us": time_us, "count": count}


def test_ridge_is_peak_over_bandwidth():
    assert bandwidth_tb_s(_ARCH) == 8.0
    assert ridge_flops_per_byte(2500, 8.0) == 312.5
    assert ridge_flops_per_byte(10100, 8.0) == 1262.5


def test_blend_is_the_flop_weighted_harmonic_mean():
    # 25% of 10,100 and 75% of 2,500.
    peak, shares = blend_peak_tflops(
        {"matrix_fp4": 25.0, "matrix_bf16": 75.0},
        _ARCH["max_achievable_tflops"],
    )
    assert shares["matrix_fp4"] == 0.25
    assert peak == 1.0 / (0.25 / 10100 + 0.75 / 2500)


def test_point_uses_projection_intensity_and_profile_time():
    # One forward is 4e9 FLOPs. Each launch is 1e8, and the capture ran 40 of
    # them in 20 µs, so the achieved rate is 40 * 1e8 / 20e-6 / 1e12 = 200.
    regime = Regime(
        "decode",
        3,
        (
            _op(
                "Attention",
                4.0e9,
                2.0e8,
                per_launch=1.0e8,
                spec="matrix_bf16",
            ),
        ),
    )
    points, warnings = build_points(
        [regime],
        [_kernel("aiter::mla_qseqlen1", 20.0, 40)],
    )
    assert warnings == []
    point = next(item for item in points if item.series == "Verify attention")
    assert point.arithmetic_intensity == 4.0e9 / 2.0e8
    assert point.achieved_tflops == pytest.approx(
        achieved_tflops(4.0e9, 1.0e8, 40, 20.0)
    )
    assert point.achieved_tflops == pytest.approx(200.0)


def test_mixed_capture_keeps_phases_apart_and_skips_end_to_end():
    regimes = [
        Regime("decode", 3, (_op("Attention", 4.0e9, 2.0e8, per_launch=1.0e8),)),
        Regime("prefill", 1, (_op("Attention", 8.0e9, 1.0e9, per_launch=2.0e8),)),
    ]
    kernels = [
        _kernel("aiter::mla_qseqlen1", 1.0e4, 10),
        _kernel("opus_mla_v4_prefill", 5.0e4, 4),
        _kernel("mfma_moe1_t32", 9.0e4, 100),
        _kernel("aiter::cross_device_reduce_1stage", 6.5e9, 1),
    ]
    points, warnings = build_points(regimes, kernels)
    by_series = {point.series: point for point in points}
    assert set(by_series) == {"Verify attention", "Prefill attention"}
    assert by_series["Verify attention"].time_us == 1.0e4
    assert by_series["Prefill attention"].time_us == 5.0e4
    assert any("End to end" in warning for warning in warnings)


def test_include_assigns_an_unscoped_kernel_to_one_phase():
    regimes = [
        Regime(
            "decode",
            48,
            (_op("MoE", 2.0e9, 1.0e9, per_launch=1.0e6, spec="matrix_fp4"),),
        ),
        Regime(
            "prefill",
            1,
            (_op("MoE", 9.0e9, 3.0e7, per_launch=3.0e6, spec="matrix_fp4"),),
        ),
    ]
    kernels = [
        _kernel("mfma_moe1_t32x64x256", 1.0e5, 50),
        _kernel("mfma_moe1_t128x128x256", 2.0e5, 8),
    ]
    points, _warnings = build_points(
        regimes,
        kernels,
        includes={
            ("decode", "Routed MoE"): ("t32x64x256",),
            ("prefill", "Routed MoE"): ("t128x128x256",),
        },
    )
    moe = [point for point in points if point.series == "Routed MoE"]
    assert {(point.phase, point.time_us) for point in moe} == {
        ("decode", 1.0e5),
        ("prefill", 2.0e5),
    }


def test_same_launch_count_is_one_launch_and_different_counts_add():
    regime = Regime(
        "decode",
        3,
        (_op("MoE", 61.0e6, 30.0e6, per_launch=1.0e6),),
    )
    kernels = [
        _kernel("mfma_moe1_t32", 100.0, 10),
        _kernel("mfma_moe2_t32", 50.0, 10),
        _kernel("mfma_moe1_t64", 40.0, 4),
    ]
    points, _warnings = build_points(
        [regime],
        kernels,
        includes={("decode", "Routed MoE"): ("mfma_moe",)},
    )
    point = next(item for item in points if item.series == "Routed MoE")
    assert point.time_us == 190.0
    # Counts 10 and 4 are separate launches; the two kernels at 10 are one.
    assert point.achieved_tflops == pytest.approx(
        achieved_tflops(61.0e6, 1.0e6, 14, 190.0)
    )


def test_time_override_is_one_forward():
    regime = Regime(
        "decode",
        24,
        (
            _op("Attention", 4.0e9, 2.0e8, per_launch=1.0e8),
            _op("GEMM", 6.0e9, 1.0e9, spec="matrix_bf16"),
        ),
    )
    points, warnings = build_points(
        [regime],
        [_kernel("aiter::mla_qseqlen1", 20.0, 40)],
        time_overrides_us={("decode", "End to end"): 1.0e3},
    )
    assert warnings == []
    end_to_end = next(point for point in points if point.series == "End to end")
    # 10e9 FLOPs in 1000 µs. The per-launch cost is not used.
    assert end_to_end.achieved_tflops == pytest.approx(10.0e9 / 1.0e-3 / 1e12)
    assert end_to_end.arithmetic_intensity == 10.0e9 / 1.2e9


def test_profile_csv_and_plot(tmp_path):
    csv_path = tmp_path / "elementwise_ops.csv"
    csv_path.write_text(
        "name,operation_count,Kernel Time (µs)_sum\n"
        "hipGraphLaunch->aiter::mla_qseqlen1,40,20000\n",
        encoding="utf-8",
    )
    kernels = load_profile_kernels(tmp_path)
    assert kernels == [
        {"name": "hipGraphLaunch->aiter::mla_qseqlen1", "time_us": 20000.0, "count": 40}
    ]
    regime = Regime(
        "decode",
        3,
        (_op("Attention", 4.0e9, 2.0e8, per_launch=1.0e8, spec="matrix_bf16"),),
    )
    points, _warnings = build_points([regime], kernels)
    output = tmp_path / "roofline.png"
    plot_e2e_roofline(
        points,
        _ARCH,
        [regime],
        title="test roofline",
        output_path=output,
    )
    assert output.stat().st_size > 0


def test_cli_joins_augmentation_and_profile(tmp_path):
    augmentation = {
        "regimes": [
            {"phase": "decode", "batch_size": 3},
        ],
        "roofline_by_category": [
            {
                "Attention": {
                    "flops": 4.0e9,
                    "bytes": 2.0e8,
                    "flops_per_invocation": 1.0e8,
                }
            }
        ],
    }
    (tmp_path / "augmentation.json").write_text(
        json.dumps(augmentation), encoding="utf-8"
    )
    (tmp_path / "profile.json").write_text(
        json.dumps([{"name": "aiter::mla_qseqlen1", "time_us": 2.0e4, "count": 40}]),
        encoding="utf-8",
    )
    arch_path = tmp_path / "arch.json"
    arch_path.write_text(json.dumps(_ARCH), encoding="utf-8")
    output = tmp_path / "out.png"
    exit_code = main(
        [
            "--title",
            "cli roofline",
            "--output",
            str(output),
            "--gpu-arch-json-path",
            str(arch_path),
            "--augmentation",
            str(tmp_path / "augmentation.json"),
            "--profile-json",
            str(tmp_path / "profile.json"),
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    regimes = regimes_from_augmentation(augmentation)
    assert regimes[0].concurrency == 3
    assert regimes[0].ops[0]["flops_per_launch"] == 1.0e8
