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
    DEFAULT_SERIES,
    Regime,
    achieved_tflops,
    bandwidth_tb_s,
    blend_peak_tflops,
    build_points,
    flag_points_above_roof,
    load_profile_kernels,
    points_from_model,
    points_from_report,
    main,
    model_coverage,
    model_coverage_note,
    plot_e2e_roofline,
    regimes_from_augmentation,
    report_kernel_category,
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
    assert warnings == [
        "Whole model decode matches End to end, so only End to end is drawn"
    ]
    point = next(item for item in points if item.series == "Decode attention")
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
        ("Decode attention", "decode"),
        ("Prefill attention", "prefill"),
        ("Whole model", "decode"),
        ("Whole model", "prefill"),
    }
    assert by_series["Decode attention", "decode"].time_us == 1.0e4
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


def test_whole_model_is_omitted_when_it_is_the_same_forward_as_end_to_end():
    regime = Regime(
        "decode",
        1,
        (_op("GEMM", 6.0e9, 1.0e9, spec="matrix_bf16"),),
    )
    points, warnings = build_points(
        [regime],
        [_kernel("decode/gemm", 1.0e3, 1)],
        time_overrides_us={
            ("decode", "End to end"): 1.0e3,
            ("decode", "Whole model"): 1.0e3,
        },
    )
    assert "Whole model" not in {point.series for point in points}
    assert any(point.series == "End to end" for point in points)
    assert any("matches End to end" in warning for warning in warnings)


def test_a_kernel_is_counted_by_its_computation_not_its_launcher():
    conv = (
        "prefill/vllm::qwen_gdn_attention_core->_causal_conv1d_fwd_kernel.kd "
        "(Synthetic Op)"
    )
    chunk_state = (
        "prefill/vllm::qwen_gdn_attention_core->chunk_gdn_fwd_h_flydsl_opt.kd "
        "(Synthetic Op)"
    )
    chunk_output = (
        "prefill/vllm::qwen_gdn_attention_core->chunk_fwd_kernel_o_opt_vk.kd "
        "(Synthetic Op)"
    )
    prepare = (
        "prefill/vllm::qwen_gdn_attention_core->gdn_prepare_kernel.kd " "(Synthetic Op)"
    )
    post_conv = (
        "prefill/vllm::qwen_gdn_attention_core->_fused_post_conv_kernel.kd "
        "(Synthetic Op)"
    )
    recurrent = (
        "decode/hipGraphLaunch->fused_rearrange_sigmoid_gated_delta_rule_"
        "update_kernel.kd (Synthetic Op)"
    )
    launched_gemm = (
        "prefill/vllm::qwen_gdn_attention_core->aiter::gemm_a8w8_blockscale_ck "
        "(Synthetic Op)"
    )
    assert report_kernel_category(conv) == "CONV_fwd"
    assert report_kernel_category(recurrent) == "Attention"
    # The chunked kernels have no performance model. The launcher name must
    # not lend them the recurrent update's category.
    assert report_kernel_category(chunk_state) is None
    assert report_kernel_category(chunk_output) is None
    assert report_kernel_category(prepare) is None
    assert report_kernel_category(post_conv) is None
    assert report_kernel_category(launched_gemm) == "GEMM"


def test_a_fused_group_quant_kernel_joins_group_quant():
    # The act-mul quant is the SwiGLU epilogue. The registry leaves the device
    # name as other, which dropped its time and put GroupQuant above the roof.
    assert (
        report_kernel_category("prefill/vllm::rocm_aiter_act_mul_and_fp8_group_quant")
        == "GroupQuant"
    )
    assert (
        report_kernel_category(
            "decode/hipGraphLaunch->_ZN5aiter37dynamic_per_group_scaled_quant_kernel"
            "IDF16b (Synthetic Op)"
        )
        == "GroupQuant"
    )
    # A norm fused with the quant stays a norm, matching the registered op.
    assert (
        report_kernel_category(
            "prefill/vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant"
        )
        == "RMSNorm"
    )
    assert (
        report_kernel_category(
            "decode/hipGraphLaunch->_fused_rms_fp8_group_quant_kernel_BLOCK "
            "(Synthetic Op)"
        )
        == "RMSNorm"
    )


def test_series_markers_do_not_repeat():
    # Prefill and decode attention are one family, so they share the circle.
    # Every other series has a shape of its own.
    markers = [spec.marker for spec in DEFAULT_SERIES]
    assert markers.count("o") == 2
    rest = [marker for marker in markers if marker != "o"]
    assert len(rest) == len(set(rest))


def test_perf_report_categories_are_series_and_unknown_is_not():
    labels = {spec.label for spec in DEFAULT_SERIES}
    assert {"GEMM", "RMSNorm", "elementwise", "GroupQuant"} <= labels
    assert not any(label.lower() in {"unknown", "other", "others"} for label in labels)
    regime = Regime(
        "prefill",
        1,
        (
            _op("GEMM", 3.0e9, 1.0e9, instances=4),
            _op("RMSNorm", 1.0e6, 2.0e8, instances=4),
        ),
    )
    points, _warnings = build_points(
        [regime],
        [
            _kernel("prefill/aiter::gemm_a8w8_blockscale", 100.0, 4),
            _kernel(
                "prefill/vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant", 50.0, 4
            ),
            _kernel("prefill/not_a_known_kernel", 80.0, 4),
        ],
    )
    by_series = {point.series for point in points}
    assert {"GEMM", "RMSNorm"} <= by_series
    assert "other" not in by_series
    assert "unknown" not in by_series


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
    # The regime projects a GEMM and the profile has no GEMM kernel.
    assert warnings == ["GEMM decode: no profile time"]
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
    assert warnings == [
        "Whole model decode matches End to end, so only End to end is drawn"
    ]
    point = next(item for item in points if item.series == "Decode attention")
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
    points, warnings = build_points([regime], [attention, gemm])
    # The attributed sum is the whole forward, so Whole model is not drawn
    # beside End to end. End to end carries the same per-category mix.
    assert warnings == [
        "Whole model decode matches End to end, so only End to end is drawn"
    ]
    whole = next(item for item in points if item.series == "End to end")
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
    (tmp_path / "rmsnorm_ops.csv").write_text(
        "name,operation_count,Kernel Time (µs)_sum,GFLOPS,FLOPS/Byte,Input Dims\n"
        'vllm::rmsnorm,4,40,1.0,2.0,"((4, 5120),)"\n',
        encoding="utf-8",
    )
    # The unclassified bucket is not a series, even when the row has a cost.
    (tmp_path / "unknown_ops.csv").write_text(
        "name,operation_count,Kernel Time (µs)_sum,GFLOPS,FLOPS/Byte,Input Dims\n"
        'hipGraphLaunch->mystery,4,40,1.0,2.0,"((4, 5120),)"\n',
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
    assert ("Decode attention", 1) not in points
    assert ("unknown", 4) not in points
    assert ("Unknown", 4) not in points
    norm = points[("RMSNorm", 4)]
    assert norm.flops == pytest.approx(4.0e9)
    attention = points[("Decode attention", 4)]
    assert attention.cost_source == "profile"
    assert attention.flops == pytest.approx(2.0e10)
    assert attention.arithmetic_intensity == pytest.approx(4.0)
    whole = points[("Whole model", 4)]
    # Attention, GEMM, RMSNorm, and the unclassified row. Unknown is not its
    # own series, and its cost still belongs to the whole forward.
    assert whole.flops == pytest.approx(1.08e11)
    assert whole.time_us == pytest.approx(380.0)


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


def test_coverage_lists_unmodeled_kernels_over_one_percent_of_the_run():
    # 1000 µs. The GEMM is modeled. The two chunk launches are one kernel at
    # 7% once their phases are added. The prepare kernel is 2%. The filler is
    # 0.5%, under the line, so it counts but is not named.
    regime = Regime("prefill", 1, (_op("GEMM", 10.0, 2.0),))
    decode = Regime("decode", 1, (_op("GEMM", 10.0, 2.0),))
    chunk = (
        "vllm::qwen_gdn_attention_core->chunk_gdn_fwd_h_flydsl_opt.kd " "(Synthetic Op)"
    )
    prepare = "vllm::qwen_gdn_attention_core->gdn_prepare_kernel.kd (Synthetic Op)"
    kernels = [
        _kernel("prefill/aiter::gemm_a8w8_blockscale", 905.0, 1),
        _kernel(f"prefill/{chunk}", 40.0, 1),
        _kernel(f"decode/{chunk}", 30.0, 1),
        _kernel(f"prefill/{prepare}", 20.0, 1),
        _kernel("prefill/not_a_known_kernel", 5.0, 1),
    ]
    coverage = model_coverage([regime, decode], kernels)
    assert coverage.covered_share == pytest.approx(0.905)
    assert coverage.unmodeled_count == 3
    assert coverage.unmodeled_share == pytest.approx(0.095)
    assert [name for name, _share in coverage.listed] == [
        "chunk_gdn_fwd_h_flydsl_opt",
        "gdn_prepare_kernel",
    ]
    note = model_coverage_note(coverage)
    assert "90.5% of kernel time" in note
    assert "3 kernels are unmodeled (9.5% of kernel time)." in note
    assert "chunk_gdn_fwd_h_flydsl_opt (7.0%)" in note
    assert "gdn_prepare_kernel (2.0%)" in note
    assert "not_a_known_kernel" not in note


def test_unmodeled_kernels_under_one_percent_are_not_listed():
    regime = Regime("prefill", 1, (_op("GEMM", 10.0, 2.0),))
    kernels = [
        _kernel("prefill/aiter::gemm_a8w8_blockscale", 995.0, 1),
        _kernel("prefill/not_a_known_kernel", 5.0, 1),
    ]
    coverage = model_coverage([regime], kernels)
    assert coverage.unmodeled_count == 1
    assert coverage.listed == ()
    note = model_coverage_note(coverage)
    assert "99.5% of kernel time" in note
    assert "unmodeled" not in note


def test_communication_is_outside_the_coverage():
    regime = Regime("prefill", 1, (_op("GEMM", 10.0, 2.0),))
    kernels = [
        _kernel("prefill/aiter::gemm_a8w8_blockscale", 100.0, 1),
        _kernel("prefill/nccl_allreduce", 900.0, 1),
    ]
    coverage = model_coverage([regime], kernels)
    assert coverage.covered_share == pytest.approx(1.0)
    assert coverage.unmodeled_count == 0
    assert "100.0% of kernel time" in model_coverage_note(coverage)


def test_a_point_above_the_roof_is_flagged_and_explained():
    below = _point("Routed MoE", 7.8, 60.0)
    within = _point("Decode attention", 7.8, 64.0)  # 2.6% over, in tolerance
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


def test_model_only_points_sit_on_the_roof():
    # 2e8 bytes at 8 TB/s is 25 µs, and 4e9 FLOPs in that time is 160 TFLOP/s,
    # which is the bandwidth roof at 20 FLOP/byte.
    regime = Regime(
        "decode",
        48,
        (
            _op("Attention", 4.0e9, 2.0e8, spec="matrix_bf16"),
            _op("GEMM", 2.5e10, 2.5e7, spec="matrix_bf16"),
        ),
    )
    regime.ops[0]["roofline_time_us"] = 25.0
    regime.ops[1]["roofline_time_us"] = 10.0
    points, warnings = points_from_model([regime])
    by_series = {point.series: point for point in points}
    assert "Whole model" not in by_series
    assert any("Whole model" in warning for warning in warnings)
    attention = by_series["Decode attention"]
    assert attention.cost_source == "model"
    assert attention.arithmetic_intensity == pytest.approx(20.0)
    assert attention.achieved_tflops == pytest.approx(160.0)
    assert attention.achieved_tflops == pytest.approx(
        roof_tflops(attention.arithmetic_intensity, _ARCH)
    )
    gemm = by_series["GEMM"]
    assert gemm.achieved_tflops == pytest.approx(2500.0)
    assert gemm.time_us == 10.0
    end = by_series["End to end"]
    assert end.flops == pytest.approx(2.9e10)
    assert end.time_us == 35.0


def test_model_only_skips_a_series_with_no_roofline_time():
    regime = Regime("decode", 3, (_op("Attention", 4.0e9, 2.0e8),))
    points, warnings = points_from_model([regime])
    assert points == []
    assert any("no roofline time" in warning for warning in warnings)


def test_cli_model_only_reads_no_profile(tmp_path, capsys):
    projection = {
        "ops": [
            {
                "category": "Attention",
                "total_flops": 4.0e9,
                "total_bytes": 2.0e8,
                "compute_spec": "matrix_bf16",
                "instances": 40,
                "total_roofline_time_us": 25.0,
            }
        ]
    }
    path = tmp_path / "decode.json"
    path.write_text(json.dumps(projection), encoding="utf-8")
    arch_path = tmp_path / "arch.json"
    arch_path.write_text(json.dumps(_ARCH), encoding="utf-8")
    output = tmp_path / "model.png"
    exit_code = main(
        [
            "--model-only",
            "--title",
            "model",
            "--output",
            str(output),
            "--gpu-arch-json-path",
            str(arch_path),
            "--projection",
            f"decode:48:{path}",
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    stdout = capsys.readouterr().out
    assert "source=model" in stdout
    assert "Decode attention" in stdout
    refused = main(
        [
            "--model-only",
            "--title",
            "model",
            "--output",
            str(output),
            "--gpu-arch-json-path",
            str(arch_path),
            "--projection",
            f"decode:48:{path}",
            "--profile-json",
            str(path),
        ]
    )
    assert refused == 1
    assert "does not read a profile" in capsys.readouterr().err
