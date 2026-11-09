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
    flag_points_above_roof,
    load_profile_kernels,
    points_from_report,
    main,
    plot_e2e_roofline,
    regimes_from_augmentation,
    roof_tflops,
    ridge_flops_per_byte,
)

_ARCH = {
    "name": "TestGPU",
    "mem_bw_gbps": 8000,
    "max_achievable_tflops": {"matrix_bf16": 2500, "matrix_fp4": 10100},
}


def _op(category, flops, bytes_, *, per_launch=None, spec=None, instances=0):
    return {
        "category": category,
        "total_flops": flops,
        "total_bytes": bytes_,
        "compute_spec": spec,
        "flops_per_launch": per_launch,
        "instances": instances,
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
    # One forward is 4e9 FLOPs over 40 launches, and the capture ran 40 launches
    # (one forward) in 20 µs, so the achieved rate is 4e9 / 20e-6 / 1e12 = 200.
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
                instances=40,
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
    assert point.achieved_tflops == pytest.approx(achieved_tflops(4.0e9, 40, 40, 20.0))
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
    by_series = {(point.series, point.phase): point for point in points}
    assert set(by_series) == {
        ("Verify attention", "decode"),
        ("Prefill attention", "prefill"),
        ("Whole model", "decode"),
        ("Whole model", "prefill"),
    }
    assert by_series["Verify attention", "decode"].time_us == 1.0e4
    assert by_series["Prefill attention", "prefill"].time_us == 5.0e4
    # Only the phase-tagged attention kernel is attributable. The unscoped MoE
    # kernel is not part of the whole-model point.
    assert by_series["Whole model", "decode"].time_us == 1.0e4
    assert by_series["Whole model", "prefill"].time_us == 5.0e4
    assert any("End to end" in warning for warning in warnings)
    assert any("unscoped" in warning for warning in warnings)


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
        (_op("MoE", 61.0e6, 30.0e6, per_launch=1.0e6, instances=61),),
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
        achieved_tflops(61.0e6, 61, 14, 190.0)
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
        {
            "name": "hipGraphLaunch->aiter::mla_qseqlen1",
            "time_us": 20000.0,
            "count": 40,
            "flops": None,
            "bytes": None,
        }
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


def test_profile_flops_replace_the_export_when_every_kernel_has_them():
    regime = Regime(
        "decode",
        3,
        (_op("Attention", 4.0e9, 2.0e8, per_launch=1.0e8, instances=40),),
    )
    kernel = _kernel("aiter::mla_qseqlen1", 20.0, 40)
    # The profile's own shapes, not the projection's 4e9 FLOPs / 2e8 bytes.
    kernel["flops"] = 8.0e9
    kernel["bytes"] = 1.0e8
    points, warnings = build_points([regime], [kernel])
    assert warnings == []
    point = next(item for item in points if item.series == "Verify attention")
    assert point.cost_source == "profile"
    assert point.arithmetic_intensity == 80.0
    assert point.achieved_tflops == pytest.approx(8.0e9 / 20.0e-6 / 1e12)


def test_one_uncosted_kernel_keeps_the_export_for_the_category():
    regime = Regime(
        "decode",
        3,
        (_op("MoE", 61.0e6, 30.0e6, per_launch=1.0e6, instances=61),),
    )
    costed = _kernel("mfma_moe1_t32", 100.0, 10)
    costed["flops"] = 1.0e15
    costed["bytes"] = 1.0e12
    bare = _kernel("mfma_moe_sorting", 50.0, 10)
    points, _warnings = build_points(
        [regime],
        [costed, bare],
        includes={("decode", "Routed MoE"): ("mfma_moe",)},
    )
    point = next(item for item in points if item.series == "Routed MoE")
    assert point.cost_source == "export"
    # The profile FLOPs are ignored: one kernel has no shape, so the export
    # describes the category. Same launch count, so the two kernels are one launch.
    assert point.achieved_tflops == pytest.approx(
        achieved_tflops(61.0e6, 61, 10, 150.0)
    )


def test_whole_model_uses_profile_per_category_and_export_for_the_rest():
    regime = Regime(
        "decode",
        3,
        (
            _op("Attention", 4.0e9, 2.0e8, instances=40),
            _op("GEMM", 6.0e9, 1.0e9, instances=20, spec="matrix_bf16"),
        ),
    )
    attention = _kernel("aiter::mla_qseqlen1", 20.0, 40)
    attention["flops"] = 8.0e9
    attention["bytes"] = 1.0e8
    gemm = _kernel("Cijk_gemm", 100.0, 20)
    points, _warnings = build_points([regime], [attention, gemm])
    whole = next(item for item in points if item.series == "Whole model")
    assert whole.cost_source == "mixed"
    assert whole.flops == pytest.approx(8.0e9 + 6.0e9)
    assert whole.bytes == pytest.approx(1.0e8 + 1.0e9)
    assert whole.time_us == 120.0
    assert whole.arithmetic_intensity == pytest.approx(whole.flops / whole.bytes)


def test_whole_model_include_adds_unscoped_kernels_without_dropping_tagged():
    regimes = [
        Regime(
            "decode",
            3,
            (
                _op("Attention", 4.0e9, 2.0e8, instances=10),
                _op("GEMM", 6.0e9, 1.0e9, instances=10),
            ),
        ),
        Regime("prefill", 1, (_op("Attention", 8.0e9, 1.0e9, instances=4),)),
    ]
    kernels = [
        _kernel("aiter::mla_qseqlen1", 100.0, 10),
        _kernel("Cijk_Alik_Bljk", 50.0, 10),
        _kernel("opus_mla_v4_prefill", 80.0, 4),
    ]
    points, _warnings = build_points(
        regimes,
        kernels,
        includes={("decode", "Whole model"): ("Cijk",)},
    )
    whole = {
        point.phase: point.time_us for point in points if point.series == "Whole model"
    }
    assert whole["decode"] == 150.0
    assert whole["prefill"] == 80.0


def test_points_from_report_use_the_stamped_graph_launch_costs(tmp_path):
    (tmp_path / "inferenceattention_ops.csv").write_text(
        "name,operation_count,Kernel Time (µs)_sum,GFLOPS,FLOPS/Byte,Input Dims\n"
        'aiter::mla_decode_v4_asm,10,100,2.0,4.0,"((4, 16, 512),)"\n'
        'aiter::mla_decode_v4_asm,1,1,0.001,4.0,"((1, 16, 512),)"\n',
        encoding="utf-8",
    )
    (tmp_path / "gemm_ops.csv").write_text(
        "name,operation_count,Kernel Time (µs)_sum,GFLOPS,FLOPS/Byte,Input Dims\n"
        'aiter::gemm_a8w8,10,200,8.0,16.0,"((4, 7168), (2048, 7168))"\n',
        encoding="utf-8",
    )
    # A launch the fill did not cover contributes nothing.
    (tmp_path / "other_ops.csv").write_text(
        "name,operation_count,Kernel Time (µs)_sum,GFLOPS,FLOPS/Byte,Input Dims\n"
        "hipGraphLaunch->mfma_moe1,61,1000,,,\n",
        encoding="utf-8",
    )
    points = {
        (item.series, item.concurrency): item for item in points_from_report(tmp_path)
    }
    assert ("Verify attention", 1) not in points
    attention = points[("Verify attention", 4)]
    assert attention.cost_source == "profile"
    assert attention.flops == pytest.approx(2.0e10)
    assert attention.arithmetic_intensity == pytest.approx(4.0)
    whole = points[("Whole model", 4)]
    assert whole.flops == pytest.approx(1.0e11)
    assert whole.time_us == pytest.approx(300.0)


def test_profile_csv_reads_flops_and_bytes(tmp_path):
    # 2 GFLOP/launch * 3 launches, at 4 FLOP/byte.
    csv_path = tmp_path / "gemm_ops.csv"
    csv_path.write_text(
        "name,operation_count,Kernel Time (µs)_sum,GFLOPS,FLOPS/Byte\n"
        "aten::mm,3,15,2.0,4.0\n"
        "hipGraphLaunch->mfma_moe1,61,100,,\n",
        encoding="utf-8",
    )
    kernels = {kernel["name"]: kernel for kernel in load_profile_kernels(tmp_path)}
    assert kernels["aten::mm"]["flops"] == pytest.approx(6.0e9)
    assert kernels["aten::mm"]["bytes"] == pytest.approx(1.5e9)
    assert kernels["hipGraphLaunch->mfma_moe1"]["flops"] is None


def test_launch_scaling_uses_the_mean_launch_not_the_costliest():
    # Two layer types: 30 launches of 309e9 and 31 of 69e9. A rollup keeps the
    # costliest launch as its per-launch cost. Scaling that by the launch count
    # would count the cheap layers at the expensive price.
    total = 30 * 309.0e9 + 31 * 69.0e9
    rollup = _op(
        "Attention",
        total,
        total / 8.0,
        per_launch=309.0e9,
        instances=61,
    )
    regime = Regime("prefill", 1, (rollup,))
    # Two forwards ran: 122 launches in 71.2 ms.
    points, _warnings = build_points(
        [regime],
        [_kernel("hipGraphLaunch->opus_mla_v4_prefill_a8w8", 71213.7, 122)],
    )
    point = next(item for item in points if item.series == "Prefill attention")
    assert point.achieved_tflops == pytest.approx(2 * total / 71213.7e-6 / 1e12)
    # The per-launch cost of the costliest layer overstates it by about 1.6x.
    assert point.achieved_tflops < 309.0e9 * 122 / 71213.7e-6 / 1e12 / 1.5


def test_a_rollup_with_no_launch_count_keeps_the_per_launch_fallback():
    # Files written before the launch count was recorded have only the
    # per-launch cost of one row.
    assert achieved_tflops(4.0e9, 0, 40, 20.0, flops_per_launch=1.0e8) == (
        pytest.approx(200.0)
    )
    assert achieved_tflops(4.0e9, 0, 0, 20.0, flops_per_launch=1.0e8) == (
        pytest.approx(4.0e9 / 20.0e-6 / 1e12)
    )


def test_roof_is_the_bandwidth_line_capped_by_the_highest_peak():
    assert roof_tflops(10.0, _ARCH) == 80.0
    assert roof_tflops(1.0e6, _ARCH) == 10100.0
    assert roof_tflops(10.0, {"mem_bw_gbps": 8000}) is None


def _point(series, intensity, achieved):
    from TraceLens.Visualizer.model_explorer_export.e2e_roofline import (
        RooflinePoint,
    )

    return RooflinePoint(
        series=series,
        phase="prefill",
        concurrency=1,
        arithmetic_intensity=intensity,
        achieved_tflops=achieved,
        flops=1.0,
        bytes=1.0,
        time_us=1.0,
    )


def test_a_point_above_the_roof_is_flagged_and_explained():
    below = _point("Routed MoE", 7.8, 60.0)
    within = _point("Verify attention", 7.8, 64.0)  # 2.6% over, in tolerance
    above = _point("Prefill attention", 7.8, 540.0)
    flagged, warnings = flag_points_above_roof([below, within, above], _ARCH)
    assert [point.above_roof for point in flagged] == [False, False, True]
    assert len(warnings) == 1
    assert "Prefill attention" in warnings[0]
    # 540 TFLOP/s at 7.8 FLOP/byte is 69 TB/s against an 8 TB/s roof.
    assert "69.2 TB/s" in warnings[0]
    assert "8.0 TB/s" in warnings[0]


def test_the_plot_outlines_a_point_above_the_roof(tmp_path):
    regime = Regime("prefill", 1, (_op("Attention", 4.0e9, 2.0e8, spec="matrix_bf16"),))
    flagged, _warnings = flag_points_above_roof(
        [_point("Prefill attention", 20.0, 5000.0)], _ARCH
    )
    assert flagged[0].above_roof
    output = tmp_path / "flagged.png"
    plot_e2e_roofline(
        flagged, _ARCH, [regime], title="above the roof", output_path=output
    )
    assert output.stat().st_size > 0


def test_cli_warns_when_a_point_is_above_the_roof(tmp_path, capsys):
    # 4e9 FLOPs at 20 FLOP/byte over 20 µs is 200 TFLOP/s. The arch roof there
    # is 8 TB/s * 20 = 160, so the profile time cannot belong to these bytes.
    augmentation = {
        "regimes": [{"phase": "decode", "batch_size": 3}],
        "roofline_by_category": [
            {
                "Attention": {
                    "flops": 4.0e9,
                    "bytes": 2.0e8,
                    "invocations": 40,
                }
            }
        ],
    }
    (tmp_path / "augmentation.json").write_text(
        json.dumps(augmentation), encoding="utf-8"
    )
    (tmp_path / "profile.json").write_text(
        json.dumps([{"name": "aiter::mla_qseqlen1", "time_us": 20.0, "count": 40}]),
        encoding="utf-8",
    )
    arch_path = tmp_path / "arch.json"
    arch_path.write_text(json.dumps(_ARCH), encoding="utf-8")
    exit_code = main(
        [
            "--title",
            "above",
            "--output",
            str(tmp_path / "above.png"),
            "--gpu-arch-json-path",
            str(arch_path),
            "--augmentation",
            str(tmp_path / "augmentation.json"),
            "--profile-json",
            str(tmp_path / "profile.json"),
        ]
    )
    assert exit_code == 0
    assert "above the" in capsys.readouterr().err
