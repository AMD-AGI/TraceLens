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
    RooflinePoint,
    achieved_tflops,
    bandwidth_tb_s,
    blend_peak_tflops,
    build_points,
    flag_points_above_roof,
    load_profile_kernels,
    points_from_model,
    main,
    model_coverage,
    figure_kernel_note,
    figure_kernel_time_share,
    model_coverage_note,
    plot_e2e_roofline,
    regimes_from_augmentation,
    report_kernel_category,
    roof_tflops,
    ridge_flops_per_byte,
    top_kernel_series,
)

_ARCH = {
    "name": "TestGPU",
    "mem_bw_gbps": 8000,
    "max_achievable_tflops": {"matrix_bf16": 2500, "matrix_fp4": 10100},
}


def _op(
    category,
    flops,
    bytes_,
    *,
    per_launch=None,
    spec=None,
    instances=0,
    roofline_time_us=0,
):
    return {
        "category": category,
        "total_flops": flops,
        "total_bytes": bytes_,
        "compute_spec": spec,
        "flops_per_launch": per_launch,
        "instances": instances,
        "roofline_time_us": roofline_time_us,
    }


def _kernel(name, time_us, count):
    return {"name": name, "time_us": time_us, "count": count}


def _priced(name, time_us, count, flops=1.0e9, bytes_=1.0e6):
    kernel = _kernel(name, time_us, count)
    kernel["flops"] = flops
    kernel["bytes"] = bytes_
    return kernel


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


def test_projection_is_not_used_when_the_trace_has_no_flops():
    # The projection prices this attention op, but the traced kernel has no
    # perf-model shape. That graph is not drawn in its place.
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
        [_kernel("decode/flash_attn::_flash_attn_forward", 20.0, 40)],
    )
    assert warnings == []
    assert points == []


def test_mixed_capture_keeps_phases_apart_and_skips_end_to_end():
    regimes = [
        Regime("decode", 3, (_op("Attention", 4.0e9, 2.0e8, per_launch=1.0e8),)),
        Regime("prefill", 1, (_op("Attention", 8.0e9, 1.0e9, per_launch=2.0e8),)),
    ]
    kernels = [
        _priced("decode/flash_attn::_flash_attn_forward", 1.0e4, 10),
        _priced("prefill/flash_attn::_flash_attn_varlen_forward", 5.0e4, 4),
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
        _priced("aiter::moe_align_block_size", 1.0e5, 50),
        _priced("aiter::moe_sum", 2.0e5, 8),
    ]
    points, _warnings = build_points(
        regimes,
        kernels,
        includes={
            ("decode", "Routed MoE"): ("align_block",),
            ("prefill", "Routed MoE"): ("moe_sum",),
        },
    )
    moe = [point for point in points if point.series == "Routed MoE"]
    assert {(point.phase, point.time_us) for point in moe} == {
        ("decode", 1.0e5),
        ("prefill", 2.0e5),
    }


def test_kernels_without_profile_flops_are_not_priced_from_the_projection():
    regime = Regime(
        "decode",
        3,
        (_op("MoE", 61.0e6, 30.0e6, per_launch=1.0e6, instances=61),),
    )
    kernels = [
        _kernel("aiter::moe_align_block_size", 100.0, 10),
        _kernel("aiter::moe_sum", 50.0, 10),
        _kernel("aiter::moe_fused_gate", 40.0, 4),
    ]
    points, _warnings = build_points(
        [regime],
        kernels,
        includes={("decode", "Routed MoE"): ("aiter::moe",)},
    )
    assert not any(point.series == "Routed MoE" for point in points)


def test_whole_model_is_omitted_when_it_is_the_same_forward_as_end_to_end():
    regime = Regime(
        "decode",
        1,
        (_op("GEMM", 6.0e9, 1.0e9, spec="matrix_bf16"),),
    )
    points, warnings = build_points(
        [regime],
        [_priced("decode/gemm", 1.0e3, 1, flops=6.0e9, bytes_=1.0e9)],
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
    # The launcher name is not a category. The conv kernels are the priced
    # CONV_fwd op; the trace's ``conv_`` key is remapped to that name.
    assert report_kernel_category(conv) == "CONV_fwd"
    assert report_kernel_category(recurrent) is None
    assert report_kernel_category(post_conv) == "CONV_fwd"
    assert report_kernel_category(chunk_state) is None
    assert report_kernel_category(chunk_output) is None
    assert report_kernel_category(prepare) is None
    assert report_kernel_category(launched_gemm) == "GEMM"
    assert (
        report_kernel_category(
            "decode/hipGraphLaunch->Cijk_Alik_Bljk_MT64x16x128 (Synthetic Op)"
        )
        == "GEMM"
    )
    assert (
        report_kernel_category(
            "hipGraphLaunch->_sparse_attn_decode_gfx950_partial_kernel.kd "
            "(Synthetic Op)"
        )
        == "Attention"
    )


def test_registered_ops_keep_the_report_category():
    assert (
        report_kernel_category(
            "prefill/vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant"
        )
        == "RMSNorm"
    )
    assert (
        report_kernel_category("prefill/aiter::dynamic_per_group_scaled_quant")
        == "GroupQuant"
    )
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
    assert (
        report_kernel_category(
            "decode/hipGraphLaunch->_fused_rms_fp8_group_quant_kernel_BLOCK "
            "(Synthetic Op)"
        )
        == "GroupQuant"
    )
    assert (
        report_kernel_category(
            "prefill/triton_red_fused__to_copy_add_fused_add_rms_norm_2"
        )
        == "RMSNorm"
    )
    assert (
        report_kernel_category(
            "hipGraphLaunch->void at::native::elementwise_kernel (Synthetic Op)"
        )
        == "elementwise"
    )
    assert (
        report_kernel_category(
            "hipGraphLaunch->void wvSplitK_hf_sml_<bf16>(...) [clone .kd] "
            "(Synthetic Op)"
        )
        == "GEMM"
    )
    assert report_kernel_category("_fused_sconv_kernel") == "CONV_fwd"
    assert report_kernel_category("_add_rmsnorm_fwd_kernel") == "RMSNorm"
    assert report_kernel_category("mfma_moe1_silu_mul_afp4_wfp4_bf16") == "MoE"
    assert report_kernel_category("mfma_moe2_afp4_wfp4_bf16") == "MoE"
    assert (
        report_kernel_category("aiter::fused_mx_quant_moe_sort_kernel<bf16>")
        == "GroupQuant"
    )
    assert report_kernel_category("aiter::opus_moe_sorting_entry<K>") == "MoE"


def test_top_kernel_markers_do_not_repeat():
    kernels = []
    for index in range(10):
        kernel = _kernel(f"kernel_{index}", float(30 - index), 1)
        kernel["flops"] = 1.0e9
        kernel["bytes"] = 1.0e6
        kernels.append(kernel)
    markers = [
        spec.marker
        for spec in top_kernel_series(kernels, 10)
        if spec.kernel_name is not None
    ]
    assert len(markers) == len(set(markers))
    assert "D" not in markers


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
            _priced("prefill/aiter::gemm_a8w8_blockscale_ck", 100.0, 4),
            _priced(
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
        [_priced("decode/flash_attn::_flash_attn_forward", 20.0, 40, 8.0e9, 1.0e8)],
        time_overrides_us={("decode", "End to end"): 1.0e3},
    )
    # The regime projects a GEMM and the profile has no GEMM kernel.
    assert warnings == ["GEMM decode: no profile time"]
    end_to_end = next(point for point in points if point.series == "End to end")
    # The traced attention work, over the one-forward time. The GEMM
    # projection is not added.
    assert end_to_end.flops == pytest.approx(8.0e9)
    assert end_to_end.achieved_tflops == pytest.approx(8.0e9 / 1.0e-3 / 1e12)
    assert end_to_end.arithmetic_intensity == pytest.approx(80.0)


def test_profile_csv_and_plot(tmp_path):
    csv_path = tmp_path / "elementwise_ops.csv"
    csv_path.write_text(
        "name,operation_count,Kernel Time (µs)_sum\n"
        "hipGraphLaunch->decode/flash_attn::_flash_attn_forward,40,20000\n",
        encoding="utf-8",
    )
    kernels = load_profile_kernels(tmp_path)
    assert kernels == [
        {
            "name": "hipGraphLaunch->decode/flash_attn::_flash_attn_forward",
            "time_us": 20000.0,
            "count": 40,
            "flops": None,
            "bytes": None,
            "shape": "",
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
        json.dumps(
            [
                {
                    "name": "decode/flash_attn::_flash_attn_forward",
                    "time_us": 2.0e4,
                    "count": 40,
                    "flops": 8.0e9,
                    "bytes": 1.0e8,
                }
            ]
        ),
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
        (
            _op(
                "Attention",
                4.0e9,
                2.0e8,
                per_launch=1.0e8,
                instances=40,
                roofline_time_us=40.0,
            ),
        ),
    )
    kernel = _kernel("decode/flash_attn::_flash_attn_forward", 20.0, 40)
    kernel["flops"] = 8.0e9
    kernel["bytes"] = 1.0e8
    points, warnings = build_points([regime], [kernel])
    assert warnings == [
        "Whole model decode matches End to end, so only End to end is drawn"
    ]
    point = next(item for item in points if item.series == "Decode attention")
    assert point.cost_source == "profile"
    assert point.arithmetic_intensity == 80.0
    assert point.flops == pytest.approx(8.0e9)
    assert point.bytes == pytest.approx(1.0e8)
    assert point.achieved_tflops == pytest.approx(8.0e9 / 20.0e-6 / 1e12)


def test_uncosted_kernel_is_left_out_of_the_category():
    regime = Regime(
        "decode",
        3,
        (_op("MoE", 61.0e6, 30.0e6, per_launch=1.0e6, instances=61),),
    )
    costed = _kernel("aiter::moe_align_block_size", 100.0, 10)
    costed["flops"] = 1.0e15
    costed["bytes"] = 1.0e12
    bare = _kernel("aiter::moe_sum", 50.0, 10)
    points, _warnings = build_points(
        [regime],
        [costed, bare],
        includes={("decode", "Routed MoE"): ("aiter::moe",)},
    )
    point = next(item for item in points if item.series == "Routed MoE")
    # moe_sum has no traced shape. The static projection is not used for it,
    # and its time is not added to the kernels that do have one.
    assert point.cost_source == "profile"
    assert point.flops == pytest.approx(1.0e15)
    assert point.bytes == pytest.approx(1.0e12)
    assert point.time_us == 100.0


def test_whole_model_uses_only_kernels_the_perf_model_costed():
    regime = Regime(
        "decode",
        3,
        (
            _op("Attention", 4.0e9, 2.0e8, instances=40),
            _op("GEMM", 6.0e9, 1.0e9, instances=20, spec="matrix_bf16"),
        ),
    )
    attention = _kernel("decode/flash_attn::_flash_attn_forward", 20.0, 40)
    attention["flops"] = 8.0e9
    attention["bytes"] = 1.0e8
    gemm = _kernel("aten::mm", 100.0, 20)
    points, warnings = build_points([regime], [attention, gemm])
    # The attributed sum is the whole forward, so Whole model is not drawn
    # beside End to end. End to end carries the same per-category mix.
    assert warnings == [
        "Whole model decode matches End to end, so only End to end is drawn"
    ]
    whole = next(item for item in points if item.series == "End to end")
    # aten::mm has no traced shape, so the GEMM projection is not added.
    assert whole.cost_source == "profile"
    assert whole.flops == pytest.approx(8.0e9)
    assert whole.bytes == pytest.approx(1.0e8)
    assert whole.time_us == 20.0
    assert not any(point.series == "GEMM" for point in points)
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
        _priced("decode/flash_attn::_flash_attn_forward", 100.0, 10),
        _priced("aten::mm", 50.0, 10),
        _priced("prefill/flash_attn::_flash_attn_varlen_forward", 80.0, 4),
    ]
    points, _warnings = build_points(
        regimes,
        kernels,
        includes={("decode", "Whole model"): ("aten::mm",)},
    )
    whole = {
        point.phase: point.time_us for point in points if point.series == "Whole model"
    }
    assert whole["decode"] == 150.0
    assert whole["prefill"] == 80.0


def test_profile_csv_reads_flops_and_bytes(tmp_path):
    # 2 GFLOP/launch * 3 launches, at 4 FLOP/byte.
    csv_path = tmp_path / "gemm_ops.csv"
    csv_path.write_text(
        "name,operation_count,Kernel Time (µs)_sum,GFLOPS,FLOPS/Byte,"
        "Roofline Time (µs)_first,Origami Time (µs)_first\n"
        "aten::mm,3,15,2.0,4.0,10.0,30.0\n"
        "hipGraphLaunch->mfma_moe1,61,100,,,,\n",
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
    # Two forwards ran: 122 launches in 71.2 ms. The traced work is the real
    # total, not the rollup's costliest launch times the launch count.
    points, _warnings = build_points(
        [regime],
        [
            _priced(
                "prefill/flash_attn::_flash_attn_varlen_forward",
                71213.7,
                122,
                flops=2 * total,
                bytes_=2 * total / 8.0,
            )
        ],
    )
    point = next(item for item in points if item.series == "Prefill attention")
    assert point.achieved_tflops == pytest.approx(2 * total / 71213.7e-6 / 1e12)
    # The per-launch cost of the costliest layer overstates it by about 1.6x.
    assert point.achieved_tflops < 309.0e9 * 122 / 71213.7e-6 / 1e12 / 1.5


def test_a_rollup_with_no_launch_count_keeps_the_per_launch_fallback():
    # Files written before the launch count was recorded have only the
    # per-launch cost of one row.
    # 40 projected launches, 80 measured: two forwards, so the numerator doubles.
    assert achieved_tflops(4.0e9, 40, 80, 20.0) == pytest.approx(400.0)
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
        _kernel("prefill/aiter::gemm_a8w8_blockscale_ck", 905.0, 1),
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
        _kernel("prefill/aiter::gemm_a8w8_blockscale_ck", 995.0, 1),
        _kernel("prefill/not_a_known_kernel", 5.0, 1),
    ]
    coverage = model_coverage([regime], kernels)
    assert coverage.unmodeled_count == 1
    assert coverage.listed == ()
    note = model_coverage_note(coverage)
    assert "99.5% of kernel time" in note
    assert "unmodeled" not in note


def test_top_kernels_keep_one_marker_per_shape():
    wide = _kernel("aiter::gemm", 80.0, 2)
    wide["flops"] = 8.0e9
    wide["bytes"] = 1.0e6
    wide["shape"] = "((4096, 4096), ())"
    narrow = _kernel("aiter::gemm", 20.0, 1)
    narrow["flops"] = 1.0e9
    narrow["bytes"] = 1.0e6
    narrow["shape"] = "((128, 128), ())"
    same = _kernel("aiter::gemm", 10.0, 1)
    same["flops"] = 2.0e9
    same["bytes"] = 1.0e6
    same["shape"] = "((4096, 4096), ())"
    series = top_kernel_series([wide, narrow, same], 3)
    gemms = [spec for spec in series if spec.kernel_name == "aiter::gemm"]
    assert {spec.kernel_shape for spec in gemms} == {
        "((4096, 4096), ())",
        "((128, 128), ())",
    }
    assert {spec.label for spec in gemms} == {
        "aiter::gemm 4096×4096",
        "aiter::gemm 128×128",
    }
    regime = Regime("prefill", 1, (_op("GEMM", 1.0, 1.0),))
    points, _warnings = build_points([regime], [wide, narrow, same], series=series)
    by_label = {point.series: point for point in points}
    assert by_label["aiter::gemm 4096×4096"].flops == pytest.approx(10.0e9)
    assert by_label["aiter::gemm 4096×4096"].time_us == 90.0
    assert by_label["aiter::gemm 128×128"].flops == pytest.approx(1.0e9)
    assert by_label["aiter::gemm 128×128"].time_us == 20.0


def test_a_shared_shape_prefix_keeps_the_full_shape_in_the_label():
    # Both shapes start with the same numeric tensor, so the short prefix
    # does not tell them apart and the label keeps the whole Input Dims.
    left = _priced("gemm", 10.0, 1)
    left["shape"] = "((8, 8), foo)"
    right = _priced("gemm", 5.0, 1)
    right["shape"] = "((8, 8), bar)"
    series = top_kernel_series([left, right], 2)
    assert {spec.label for spec in series} == {
        "gemm ((8, 8), foo)",
        "gemm ((8, 8), bar)",
        "End to end",
    }


def test_a_top_kernel_without_flops_is_not_drawn():
    from TraceLens.Visualizer.model_explorer_export.e2e_roofline import Series

    bare = _kernel("plain_kernel", 20.0, 1)
    series = (
        Series(
            "plain_kernel", "o", None, None, kernel_name="plain_kernel", kernel_shape=""
        ),
        Series("End to end", "D", None, None),
    )
    points, warnings = build_points(
        [Regime("prefill", 1, (_op("GEMM", 1.0, 1.0),))],
        [bare],
        series=series,
    )
    assert any("no profile FLOPs" in warning for warning in warnings)
    assert not any(point.series == "plain_kernel" for point in points)


def test_cancelled_flops_leave_the_category_off_the_figure():
    negative = _priced("aiter::gemm_a8w8", 10.0, 1, flops=-5.0, bytes_=1.0)
    positive = _priced("aiter::gemm_a8w8", 10.0, 1, flops=5.0, bytes_=1.0)
    points, _warnings = build_points(
        [Regime("prefill", 1, (_op("GEMM", 1.0, 1.0),))],
        [negative, positive],
    )
    assert not any(point.series == "GEMM" for point in points)


def test_kernel_share_ignores_a_kernel_that_did_not_run():
    kernel = _priced("gemm", 10.0, 1)
    series = top_kernel_series([kernel], 1)
    points, _warnings = build_points(
        [Regime("prefill", 1, (_op("GEMM", 1.0, 1.0),))],
        [kernel],
        series=series,
    )
    idle = _kernel("idle", 0.0, 1)
    assert figure_kernel_time_share(points, series, [kernel, idle]) == pytest.approx(
        1.0
    )
    assert figure_kernel_time_share(points, series, [idle]) is None


def test_top_kernel_point_uses_the_traced_flops():
    kernel = _kernel("aiter::gemm_a8w8_blockscale_ck", 20.0, 40)
    kernel["flops"] = 8.0e9
    kernel["bytes"] = 1.0e8
    regime = Regime("prefill", 1, (_op("GEMM", 4.0e9, 2.0e8, instances=40),))
    series = top_kernel_series([kernel], 1)
    points, _warnings = build_points([regime], [kernel], series=series)
    point = next(item for item in points if item.series != "End to end")
    assert point.cost_source == "profile"
    assert point.flops == pytest.approx(8.0e9)
    assert point.bytes == pytest.approx(1.0e8)
    assert point.arithmetic_intensity == pytest.approx(80.0)
    assert point.achieved_tflops == pytest.approx(8.0e9 / 20.0e-6 / 1e12)


def test_top_kernel_note_is_the_share_drawn_on_the_figure():
    gemm = _kernel("gemm", 50.0, 1)
    gemm["flops"] = 1.0e9
    gemm["bytes"] = 1.0e6
    quant = _kernel("quant", 20.0, 1)
    quant["flops"] = 1.0e6
    quant["bytes"] = 1.0e6
    kernels = [gemm, _kernel("norm", 30.0, 1), quant]
    series = top_kernel_series(kernels, 3)
    # norm has no profile FLOPs, so the figure draws gemm and quant only.
    regime = Regime("prefill", 1, (_op("GEMM", 1.0e9, 1.0e6),))
    points, _warnings = build_points([regime], kernels, series=series)
    share = figure_kernel_time_share(points, series, kernels)
    assert share == pytest.approx(0.70)
    assert figure_kernel_note(share) == (
        "Figure shows kernels taking 70.0% of the run time."
    )
    assert figure_kernel_time_share(points, DEFAULT_SERIES, kernels) is None


def test_communication_is_outside_the_coverage():
    regime = Regime("prefill", 1, (_op("GEMM", 10.0, 2.0),))
    kernels = [
        _kernel("prefill/aiter::gemm_a8w8_blockscale_ck", 100.0, 1),
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
        json.dumps(
            [
                {
                    "name": "decode/flash_attn::_flash_attn_forward",
                    "time_us": 20.0,
                    "count": 40,
                    "flops": 4.0e9,
                    "bytes": 2.0e8,
                }
            ]
        ),
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
    refused_top = main(
        [
            "--model-only",
            "--top-kernels",
            "4",
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
    assert refused_top == 1
    assert "needs a profile" in capsys.readouterr().err
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


def test_profile_plot_keeps_the_slowest_kernels_and_end_to_end(tmp_path, capsys):
    kernels = []
    for index in range(12):
        kernels.append(
            {
                "name": f"hipGraphLaunch->kernel_{index:02d}",
                "time_us": (index + 1) * 10.0,
                "count": 1,
                "flops": 1.0e9 * (index + 1),
                "bytes": 1.0e8,
            }
        )
    kernels.append(
        {
            "name": "nccl_allreduce",
            "time_us": 1.0e9,
            "count": 1,
            "flops": 1.0,
            "bytes": 1.0,
        }
    )
    kernels.append(
        {"name": "hipGraphLaunch->bare_kernel", "time_us": 500.0, "count": 1}
    )
    series = top_kernel_series(kernels, 3)
    # bare_kernel is the longest run but has no FLOPs, so it does not take a slot.
    assert [spec.kernel_name for spec in series if spec.kernel_name] == [
        "hipGraphLaunch->kernel_11",
        "hipGraphLaunch->kernel_10",
        "hipGraphLaunch->kernel_09",
    ]
    assert [spec.label for spec in series] == [
        "kernel_11",
        "kernel_10",
        "kernel_09",
        "End to end",
    ]
    assert len(top_kernel_series(kernels)) == 11
    regime = Regime("decode", 3, (_op("GEMM", 4.0e9, 2.0e8, instances=1),))
    points, warnings = build_points([regime], kernels, series=series)
    assert not any("no profile FLOPs" in warning for warning in warnings)
    labels = {point.series for point in points}
    assert labels == {"kernel_11", "kernel_10", "kernel_09", "End to end"}
    slowest = next(point for point in points if point.series == "kernel_11")
    assert slowest.flops == pytest.approx(12.0e9)
    assert slowest.time_us == 120.0
    assert slowest.cost_source == "profile"
    end = next(point for point in points if point.series == "End to end")
    assert end.flops > slowest.flops

    launched = _kernel("hipGraphLaunch->gemm", 30.0, 1)
    launched["flops"] = 1.0e9
    launched["bytes"] = 1.0e6
    plain = _kernel("gemm", 20.0, 1)
    plain["flops"] = 1.0e9
    plain["bytes"] = 1.0e6
    peeled = top_kernel_series([launched, plain], 2)
    assert {spec.label for spec in peeled} == {
        "hipGraphLaunch->gemm",
        "gemm",
        "End to end",
    }

    projection = {
        "ops": [
            {
                "category": "GEMM",
                "total_flops": 4.0e9,
                "total_bytes": 2.0e8,
                "compute_spec": "matrix_bf16",
                "instances": 1,
                "total_roofline_time_us": 25.0,
            }
        ]
    }
    path = tmp_path / "decode.json"
    path.write_text(json.dumps(projection), encoding="utf-8")
    (tmp_path / "profile.json").write_text(json.dumps(kernels), encoding="utf-8")
    arch_path = tmp_path / "arch.json"
    arch_path.write_text(json.dumps(_ARCH), encoding="utf-8")
    output = tmp_path / "top.png"
    exit_code = main(
        [
            "--title",
            "top kernels",
            "--output",
            str(output),
            "--gpu-arch-json-path",
            str(arch_path),
            "--projection",
            f"decode:3:{path}",
            "--profile-json",
            str(tmp_path / "profile.json"),
            "--top-kernels",
            "3",
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    stdout = capsys.readouterr().out
    assert "kernel_11" in stdout
    assert "kernel_10" in stdout
    assert "End to end" in stdout
    assert "kernel_00" not in stdout
    assert "Decode attention" not in stdout
    assert "GEMM phase" not in stdout
    category = main(
        [
            "--title",
            "categories",
            "--output",
            str(output),
            "--gpu-arch-json-path",
            str(arch_path),
            "--projection",
            f"decode:3:{path}",
            "--profile-json",
            str(tmp_path / "profile.json"),
        ]
    )
    assert category == 0
    category_out = capsys.readouterr()
    assert "End to end" in category_out.out
    assert "kernel_11" not in category_out.out
    assert "GEMM" in category_out.err
    bare = main(
        [
            "--title",
            "top ten",
            "--output",
            str(output),
            "--gpu-arch-json-path",
            str(arch_path),
            "--projection",
            f"decode:3:{path}",
            "--profile-json",
            str(tmp_path / "profile.json"),
            "--top-kernels",
        ]
    )
    assert bare == 0
    bare_out = capsys.readouterr().out
    assert "kernel_03" in bare_out
    assert "kernel_01" not in bare_out


def test_uncovered_plot_and_cli_paths(tmp_path, monkeypatch, capsys):
    from TraceLens.Visualizer.model_explorer_export import e2e_roofline as e2e

    with pytest.raises(ValueError, match="at least 1"):
        e2e.top_kernel_series([_kernel("a", 1.0, 1)], 0)
    with pytest.raises(ValueError, match="positive"):
        achieved_tflops(1.0, 1, 1, 0.0)

    assert report_kernel_category("aten::add") == "elementwise"
    projection = {
        "ops": [
            {"total_flops": 1.0},
            {
                "category": "NormalizationRMS",
                "total_flops": 1.0,
                "total_bytes": 1.0,
                "roofline_time_us": 2.0,
                "instances": 3,
            },
            {"category": "Reduce", "total_flops": 1.0, "total_bytes": 1.0},
            {"category": "Conv1d", "total_flops": 1.0, "total_bytes": 1.0},
            {"category": "IndexedCopy", "total_flops": 1.0, "total_bytes": 1.0},
            {
                "category": "UnaryElementwise",
                "computation": "quant",
                "total_flops": 1.0,
                "total_bytes": 1.0,
            },
            {
                "category": "BinaryElementwise",
                "computation": "add",
                "total_flops": 1.0,
                "total_bytes": 1.0,
            },
            {"category": "Other", "total_flops": 1.0, "total_bytes": 1.0},
            {
                "category": "GEMM",
                "total_flops": 4.0,
                "total_bytes": 2.0,
                "flops": 1.0,
                "instances": 2,
                "total_roofline_time_us": 5.0,
            },
        ]
    }
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(projection), encoding="utf-8")
    regime = e2e.regime_from_projection(path, "decode", 1)
    categories = {op["category"] for op in regime.ops}
    assert categories == {
        "RMSNorm",
        "reduce",
        "CONV_fwd",
        "IndexedCopy",
        "GroupQuant",
        "elementwise",
        "Other",
        "GEMM",
    }
    assert regime.ops[0]["roofline_time_us"] == 6.0

    singular = e2e.regimes_from_augmentation(
        {
            "regime": {"phase": "decode", "batch_size": 1},
            "roofline_by_category": {"Attention": {"flops": 1.0, "bytes": 1.0}},
        }
    )
    assert singular[0].phase == "decode"
    with pytest.raises(ValueError, match="one category summary"):
        e2e.regimes_from_augmentation(
            {"regimes": [{}, {}], "roofline_by_category": [{}]}
        )

    (tmp_path / "not_a_list.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="list of kernel"):
        e2e.load_profile_json(tmp_path / "not_a_list.json")
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    (csv_dir / "bad_ops.csv").write_text(
        "name,operation_count\nfoo,1\n", encoding="utf-8"
    )
    (csv_dir / "ops_ops.csv").write_text(
        "name,Kernel Time (µs)_sum,operation_count,GFLOPS,FLOPS/Byte\n"
        "empty,,1,1,1\n"
        "zero,10,1,0,1\n"
        "ok,10,2,2,4\n",
        encoding="utf-8",
    )
    loaded = {kernel["name"]: kernel for kernel in load_profile_kernels(csv_dir)}
    assert loaded["zero"]["flops"] is None
    assert loaded["ok"]["flops"] == pytest.approx(4.0e9)

    attention = _kernel("decode/flash_attn::_flash_attn_forward", 20.0, 40)
    attention["flops"] = 8.0e9
    attention["bytes"] = 1.0e8
    scaled, _warnings = build_points(
        [
            Regime(
                "decode",
                3,
                (_op("Attention", 4.0e9, 2.0e8, instances=40),),
            )
        ],
        [attention],
        time_overrides_us={("decode", "Decode attention"): 10.0},
    )
    point = next(item for item in scaled if item.series == "Decode attention")
    assert point.time_us == 10.0
    assert point.cost_source == "profile"

    build_points(
        [Regime("decode", 1, (_op("Attention", 0.0, 0.0),))],
        [_kernel("decode/flash_attn::_flash_attn_forward", 20.0, 4)],
    )
    prefill = _kernel("prefill/flash_attn::_flash_attn_varlen_forward", 20.0, 1)
    prefill["flops"] = 1.0e9
    prefill["bytes"] = 1.0e8
    build_points(
        [Regime("decode", 1, (_op("GEMM", 1.0e9, 1.0e8, instances=1),))],
        [prefill],
        series=top_kernel_series([prefill], 1),
    )
    slow = _kernel("slow_kernel", 20.0, 1)
    slow["flops"] = 1.0e9
    slow["bytes"] = 1.0e8
    build_points(
        [Regime("decode", 1, (_op("GEMM", 1.0e9, 1.0e8, instances=1),))],
        [slow],
        series=top_kernel_series([slow], 1),
        time_overrides_us={("decode", "slow_kernel"): 0.0},
    )

    idle = Regime(
        "decode",
        1,
        (_op("Other", 0.0, 0.0), _op("Attention", 4.0e9, 2.0e8, spec="matrix_bf16")),
    )
    idle.ops[1]["roofline_time_us"] = 10.0
    assert points_from_model([idle])[0]
    empty_cat = Regime(
        "decode", 1, ({"category": "", "total_flops": 1.0, "total_bytes": 1.0},)
    )
    assert (
        model_coverage(
            [empty_cat],
            [_kernel("foo", 0.0, 1), _kernel("nccl_allreduce", 5.0, 1)],
        )
        is None
    )
    assert model_coverage([idle], []) is None

    flat = {
        "name": "Flat",
        "mem_bw_gbps": 8000,
        "max_achievable_tflops": {"matrix_bf16": 100.0, "matrix_fp8": 100.0},
    }
    mixed = (
        _op("Attention", 4.0e9, 2.0e8, spec="matrix_bf16"),
        _op("GEMM", 1.0e9, 1.0e8, spec="matrix_fp4"),
    )
    points = [
        RooflinePoint("Prefill attention", "prefill", 4, 20.0, 100.0, 1.0, 1.0, 1.0),
        RooflinePoint("Decode attention", "decode", 8, 20.0, 100.0, 1.0, 1.0, 1.0),
    ]
    output = tmp_path / "edges.png"
    plot_e2e_roofline(
        points,
        _ARCH,
        [Regime("decode", 4, mixed), Regime("prefill", 4, mixed)],
        title="edges",
        output_path=output,
    )
    plot_e2e_roofline([], flat, [], title="flat", output_path=output)
    plot_e2e_roofline([], _ARCH, [], title="empty", output_path=output)
    with pytest.raises(ValueError, match="no bfloat16"):
        plot_e2e_roofline(
            [], {"mem_bw_gbps": 8000}, [], title="no peaks", output_path=output
        )

    arch_path = tmp_path / "arch.json"
    arch_path.write_text(json.dumps(_ARCH), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            [
                {
                    "name": "decode/flash_attn::_flash_attn_forward",
                    "time_us": 20.0,
                    "count": 4,
                    "flops": 4.0e9,
                    "bytes": 2.0e8,
                }
            ]
        ),
        encoding="utf-8",
    )
    common = [
        "--title",
        "edges",
        "--output",
        str(output),
        "--gpu-arch-json-path",
        str(arch_path),
        "--projection",
        f"decode:1:{path}",
        "--profile-json",
        str(profile_path),
    ]
    assert (
        main(
            common
            + [
                "--kernels",
                "decode|Decode attention|mla",
                "--time-us",
                "decode|End to end|1000",
            ]
        )
        == 0
    )
    assert main(common + ["--time-us", "bad"]) == 1
    assert "expected phase" in capsys.readouterr().err
    assert main(common + ["--projection", "nocolon"]) == 1
    assert main(common + ["--projection", "decode:nope:file"]) == 1
    assert (
        main(
            [
                "--title",
                "t",
                "--output",
                str(output),
                "--projection",
                f"decode:1:{path}",
            ]
        )
        == 1
    )
    assert "gpu-arch" in capsys.readouterr().err
    assert (
        main(
            [
                "--title",
                "t",
                "--output",
                str(output),
                "--gpu-arch-json-path",
                str(arch_path),
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "--title",
                "t",
                "--output",
                str(output),
                "--gpu-arch-json-path",
                str(arch_path),
                "--projection",
                f"decode:1:{path}",
            ]
        )
        == 1
    )
    assert "profile" in capsys.readouterr().err
    assert (
        main(
            [
                "--title",
                "dir",
                "--output",
                str(output),
                "--gpu-arch-json-path",
                str(arch_path),
                "--projection",
                f"decode:1:{path}",
                "--profile-dir",
                str(csv_dir),
            ]
        )
        == 0
    )
    bare_projection = tmp_path / "bare.json"
    bare_projection.write_text(
        json.dumps(
            {"ops": [{"category": "Attention", "total_flops": 1.0, "total_bytes": 1.0}]}
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--model-only",
                "--title",
                "t",
                "--output",
                str(output),
                "--gpu-arch-name",
                "NotARealGPU",
                "--projection",
                f"decode:1:{bare_projection}",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "--model-only",
                "--title",
                "t",
                "--output",
                str(output),
                "--gpu-arch-json-path",
                str(arch_path),
                "--projection",
                f"decode:1:{bare_projection}",
            ]
        )
        == 1
    )
    assert "no points" in capsys.readouterr().err

    monkeypatch.setattr(e2e, "_CATEGORY_MARKERS", ())
    with pytest.raises(RuntimeError, match="distinct markers"):
        e2e._category_series(("GEMM",))
    monkeypatch.setattr(e2e, "_KERNEL_MARKERS", ("o",))
    costly = []
    for name, time_us in (("a", 2.0), ("b", 1.0)):
        kernel = _kernel(name, time_us, 1)
        kernel["flops"] = 1.0
        kernel["bytes"] = 1.0
        costly.append(kernel)
    with pytest.raises(ValueError, match="distinct markers"):
        e2e.top_kernel_series(costly, 2)
