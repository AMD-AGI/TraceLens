###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A replayed DeepSeek-V4 graph takes its shapes from the model launch at that position."""

from dataclasses import replace

from types import SimpleNamespace

from TraceLens.TreePerf.graph_schedule import (
    GraphModel,
    Invocation,
    align_kernels,
    graph_model_from_config,
    invocation_from_annotation,
    launches_from_architecture,
    mhc_fused_split,
    mhc_pre_split,
    num_cus_from_trace,
    replay_size,
    schedule_deepseek_v4,
)
from tests.fixtures.treeperf import _build_analyzer, _make_gpu_event, _mk_ac2g

# deepseek-ai/DeepSeek-V4-Pro-0813, the 61 layers the graph covers.
_RATIOS = (
    128,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
)


def _pro() -> GraphModel:
    return GraphModel(
        hidden_size=7168,
        q_lora_rank=1536,
        o_lora_rank=1024,
        num_attention_heads=128,
        head_dim=512,
        o_groups=16,
        moe_intermediate_size=3072,
        n_routed_experts=384,
        num_experts_per_tok=6,
        num_hidden_layers=61,
        compress_ratios=_RATIOS,
        num_hash_layers=3,
        vocab_size=129280,
        tp=8,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
        qk_rope_head_dim=64,
        hc_mult=4,
        n_shared_experts=1,
        expert_dtype="fp4",
        weight_block_size=(128, 128),
        max_position_embeddings=1048576,
    )


def test_decode_annotation_is_one_token_per_request_with_the_mean_past():
    invocation = invocation_from_annotation(
        "step[DECODE bs=3 g_sq=3 g_sqsq=9 g_sqsk=86718 g_sk=86718]"
    )
    assert invocation is not None
    assert invocation.batch_size == 3
    assert invocation.seq_len == 1
    assert invocation.tokens == 3
    assert invocation.past_len == 86718 // 3


def test_schedule_follows_compress_ratio_and_the_hash_layers():
    invocation = Invocation(batch_size=3, seq_len=1, past_len=28906, tokens=3)
    launches = schedule_deepseek_v4(_pro(), invocation)
    by_layer: dict[int, list[str]] = {}
    for launch in launches:
        if launch.layer is None:
            continue
        by_layer.setdefault(launch.layer, []).append(launch.key)
    assert set(by_layer) == set(range(61))
    assert by_layer[0][0] == "mhc_sqrsum"
    assert "c128" in by_layer[0] and "c4" not in by_layer[0]
    assert "hash_topk" in by_layer[0] and "topk" not in by_layer[0]
    assert "c128" in by_layer[1]
    assert "c4" in by_layer[2] and "hash_topk" in by_layer[2]
    assert "topk" in by_layer[3] and "hash_topk" not in by_layer[3]
    assert by_layer[60][-1] == "mhc_post"
    # The split-K reduce and the stage-2 reduction are kernels of the call
    # before them, so the collector keeps one row.
    for launch, previous in zip(launches[1:], launches):
        if launch.same_call:
            assert previous.op_name == launch.op_name
            assert launch.key in {"gemm_red", "moe_red", "cta_info"}

    wq_b = next(
        launch for launch in launches if launch.layer == 1 and launch.key == "ck_gemm"
    )
    # The CK block-scale weight is (N, K), then the activation and weight scales.
    assert wq_b.input_shapes == ((3, 1536), (8192, 1536), (3, 12), (64, 12))
    mla = next(
        launch for launch in launches if launch.layer == 1 and launch.key == "mla"
    )
    assert mla.input_shapes[1][0] == (28906 + 1) // 128
    c4_mla = next(
        launch for launch in launches if launch.layer == 2 and launch.key == "mla"
    )
    assert c4_mla.input_shapes[1][0] == (28906 + 1) // 4


def test_replay_size_uses_the_smallest_covering_bucket():
    assert replay_size(3, ()) == 3
    assert replay_size(3, (1, 2, 4, 8)) == 4
    assert replay_size(4, (1, 2, 4, 8)) == 4
    assert replay_size(16, (1, 2, 4, 8)) == 16


def test_fp4_experts_and_rope_take_the_checkpoint_layout():
    launches = schedule_deepseek_v4(
        _pro(), Invocation(batch_size=3, seq_len=1, past_len=0, tokens=3)
    )
    moe = next(
        launch for launch in launches if launch.layer == 3 and launch.key == "moe1"
    )
    assert moe.input_shapes[:4] == (
        (3, 7168),
        (385, 768, 3584),
        (385, 7168, 192),
        (3, 7),
    )
    rope = next(
        launch for launch in launches if launch.layer == 1 and launch.key == "rope"
    )
    assert rope.input_shapes == ((3, 16, 64), (1048576, 32), (3,))
    wo_a = next(
        launch for launch in launches if launch.layer == 1 and launch.key == "gemm_mx"
    )
    assert wo_a.input_shapes[:3] == ((3, 2, 4096), (2, 1024, 4096), (3, 2, 1024))
    covered = schedule_deepseek_v4(
        replace(_pro(), capture_buckets=(1, 4)),
        Invocation(batch_size=3, seq_len=1, past_len=0, tokens=3),
    )
    covered_moe = next(
        launch for launch in covered if launch.layer == 3 and launch.key == "moe1"
    )
    assert covered_moe.input_shapes[0] == (4, 7168)
    # w13's scale is already a multiple of 256 by 8. w2's 12 columns pad to 16.
    assert moe.input_shapes[-2:] == ((295680, 224), (2759680, 16))


def test_mhc_split_follows_the_device_policy():
    # gfx950 with 256 CUs, hidden 7168, four streams. Both token counts the
    # decode graph replays select the same split.
    assert mhc_pre_split(1, 4 * 7168, 256, "gfx950") == (112, 128)
    assert mhc_pre_split(4, 4 * 7168, 256, "gfx950")[0] == 112
    assert mhc_fused_split(1, 7168, 256, "gfx950") == 112
    assert mhc_fused_split(4, 7168, 256, "gfx950") == 112

    model = replace(_pro(), num_cus=256, gfx_arch="gfx950")
    launches = schedule_deepseek_v4(
        model, Invocation(batch_size=4, seq_len=1, past_len=0, tokens=4)
    )
    sqrsum = next(
        launch
        for launch in launches
        if launch.layer == 0 and launch.key == "mhc_sqrsum"
    )
    assert sqrsum.input_shapes == (
        (112, 4, 24),
        (112, 4),
        (4, 4, 7168),
        (24, 28672),
    )
    rms = next(
        launch
        for launch in launches
        if launch.layer == 1 and launch.key == "mhc_pre_rms"
    )
    assert rms.input_shapes == (
        (4, 4, 1),
        (4, 4, 4),
        (4, 7168),
        (112, 4, 24),
        (112, 4),
        (3,),
        (24,),
        (4, 4, 7168),
        (7168,),
    )
    fused = next(
        launch for launch in launches if launch.layer == 1 and launch.key == "mhc_fused"
    )
    assert fused.input_shapes[0] == (112, 4, 24)
    assert fused.input_shapes[5] == (4, 4)
    assert fused.input_shapes[7] == (24, 28672)


def test_compute_units_come_from_the_trace_device_block():
    devices = [{"name": "AMD Instinct MI355X", "numSms": 256}] * 8
    assert num_cus_from_trace({"deviceProperties": devices}) == 256
    assert num_cus_from_trace({}) is None
    assert (
        num_cus_from_trace({"deviceProperties": [{"name": "AMD Instinct MI355X"}]})
        is None
    )
    assert (
        num_cus_from_trace({"deviceProperties": [{"numSms": 256}, {"numSms": 104}]})
        is None
    )

    from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
    from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

    tree = TraceToTree(
        [],
        prune_nongpu_paths=False,
        trace_metadata={"deviceProperties": devices},
    )
    analyzer = TreePerfAnalyzer(
        tree,
        rebuild_tree=False,
        graph_model=replace(_pro(), num_cus=104, gfx_arch="gfx950"),
    )
    assert analyzer.graph_model.num_cus == 256
    assert analyzer.graph_model.gfx_arch == "gfx950"


def test_graph_model_reads_the_quant_and_expert_fields():
    model = graph_model_from_config(
        {
            "hidden_size": 32,
            "q_lora_rank": 16,
            "o_lora_rank": 8,
            "num_attention_heads": 8,
            "head_dim": 4,
            "o_groups": 8,
            "moe_intermediate_size": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 1,
            "compress_ratios": [128],
            "vocab_size": 16,
            "index_n_heads": 8,
            "index_head_dim": 4,
            "index_topk": 2,
            "qk_rope_head_dim": 2,
            "hc_mult": 4,
            "n_shared_experts": 1,
            "expert_dtype": "fp4",
            "max_position_embeddings": 32,
            "quantization_config": {"weight_block_size": [128, 128]},
            "capture_buckets": [1, 4],
        },
        tp=8,
    )
    assert model.weight_block_size == (128, 128)
    assert model.expert_dtype == "fp4"
    assert model.capture_buckets == (1, 4)
    assert model.qk_rope_head_dim == 2


def test_prefill_annotation_keeps_the_cached_past():
    invocation = invocation_from_annotation(
        "step[EXTEND bs=2 toks=2358 c_sq=2358 c_sqsq=2788532 c_sqsk=41422004 c_sk=35126]"
    )
    assert invocation is not None
    assert invocation.prefill
    assert invocation.tokens == 2358
    assert invocation.past_len == (35126 - 2358) // 2


def test_align_skips_kernels_that_are_not_the_next_launch():
    model = GraphModel(
        hidden_size=32,
        q_lora_rank=16,
        o_lora_rank=8,
        num_attention_heads=8,
        head_dim=4,
        o_groups=8,
        moe_intermediate_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        compress_ratios=(128,),
        num_hash_layers=0,
        vocab_size=16,
        tp=8,
        index_n_heads=8,
        index_head_dim=4,
        index_topk=2,
    )
    launches = schedule_deepseek_v4(
        model, Invocation(batch_size=3, seq_len=1, past_len=0, tokens=3)
    )
    names = [
        "void at::native::elementwise_kernel<8>()",
        "_vocab_parallel_embedding_kernel",
        "aiter::cross_device_reduce_1stage<std::bfloat16_t, 8>",
        "void aiter::mhc_pre_gemm_sqrsum_kernel<std::bfloat16_t>()",
    ]
    assigned = align_kernels(names, launches)
    assert assigned[0] is None
    assert assigned[1] is not None and assigned[1].key == "embed"
    assert assigned[1].input_shapes == ((3,), (2, 32))
    assert assigned[2] is not None and assigned[2].key == "allreduce"
    assert (
        assigned[3] is not None and assigned[3].op_name == "aiter::mhc_pre_gemm_sqrsum"
    )


def test_graph_launch_rows_take_the_invocation_shapes():
    model = GraphModel(
        hidden_size=32,
        q_lora_rank=16,
        o_lora_rank=8,
        num_attention_heads=8,
        head_dim=4,
        o_groups=8,
        moe_intermediate_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        compress_ratios=(128,),
        num_hash_layers=0,
        vocab_size=16,
        tp=8,
        index_n_heads=8,
        index_head_dim=4,
        index_topk=2,
    )
    corr = 7
    events = [
        _make_gpu_event(
            "step",
            900,
            800,
            "user_annotation",
            "step[DECODE bs=3 g_sq=3 g_sqsq=9 g_sqsk=9 g_sk=9]",
        ),
        _make_gpu_event(
            "rt",
            1000,
            50,
            "cuda_runtime",
            "hipGraphLaunch",
            args={"correlation": corr},
        ),
        _make_gpu_event(
            "k0",
            1010,
            5,
            "kernel",
            "void at::native::elementwise_kernel<8>()",
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _make_gpu_event(
            "k1",
            1020,
            5,
            "kernel",
            "_vocab_parallel_embedding_kernel",
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, 0, 7, 1010, "s"),
        _mk_ac2g(corr, 0, 7, 1030, "f"),
    ]
    analyzer = _build_analyzer(events, graph_model=model)
    collected = analyzer.collect_unified_perf_events()
    embed = next(
        event
        for event in collected
        if event["name"].endswith("vocab_parallel_embedding")
    )
    assert embed["args"]["Input Dims"] == [[3], [2, 32]]
    elementwise = [
        event for event in collected if "elementwise_kernel" in event["name"]
    ]
    assert elementwise
    assert "Input Dims" not in (elementwise[0].get("args") or {})


def test_a_stated_kernel_shape_from_the_engine_source_is_the_launch():
    node = SimpleNamespace(
        details=["kernel: aiter::mhc_pre", "shape: ((0, 7168), (7168, 7168))"],
        children=[],
    )
    spec = SimpleNamespace(export_block_trees=[("layer", node)])
    model = _pro()
    launches = launches_from_architecture(spec, model, Invocation(3, 1, 10, 3))
    assert launches[0].op_name == "aiter::mhc_pre"
    assert launches[0].input_shapes == ((3, 7168), (7168, 7168))

    analyzer = _build_analyzer([], graph_model=model)
    analyzer.architectures = {"decode": spec}
    analyzer._step_events = [
        {
            "name": "step[DECODE bs=3 g_sq=3 g_sqsq=9 g_sqsk=9 g_sk=9]",
            "ts": 0,
            "dur": 1000,
        }
    ]
    used = analyzer._launches_for({"name": "hipGraphLaunch", "ts": 10, "dur": 5})
    assert used[0].input_shapes == ((3, 7168), (7168, 7168))


def test_a_source_without_kernel_shapes_leaves_the_config_schedule():
    node = SimpleNamespace(details=["kernel: aiter::mhc_pre"], children=[])
    spec = SimpleNamespace(export_block_trees=[("layer", node)])
    assert launches_from_architecture(spec, _pro(), Invocation(1, 1, 0, 1)) is None


def _decode_launches(model, tokens=4, layer=None):
    launches = schedule_deepseek_v4(
        model, Invocation(batch_size=3, seq_len=1, past_len=0, tokens=tokens)
    )
    return [item for item in launches if layer is None or item.layer == layer]


def test_fused_wqkv_a_projects_q_lora_and_the_kv_head_together():
    fused = replace(_pro(), facts=(("fuse_wqa_wkv", True),))
    split = replace(_pro(), facts=(("fuse_wqa_wkv", False),))
    fused_gemms = [
        item for item in _decode_launches(fused, layer=2) if item.key == "gemm_a8w8"
    ]
    # One (q_lora + head_dim, hidden) projection and its scales, (1536 + 512).
    assert [item.input_shapes for item in fused_gemms] == [
        ((4, 7168), (2048, 7168), (4, 56), (16, 56))
    ]
    assert fused_gemms[0].input_types == (
        "c10::Float8_e4m3fn",
        "c10::Float8_e4m3fn",
        "float",
        "float",
    )
    split_gemms = [
        item for item in _decode_launches(split, layer=2) if item.key == "gemm_a8w8"
    ]
    assert [item.input_shapes[1] for item in split_gemms] == [
        (1536, 7168),
        (512, 7168),
    ]


def test_graph_launch_arguments_drive_the_roofline():
    """The captured call's dtype and stride record is what the roofline reads."""
    from TraceLens.TreePerf.graph_schedule import launch_arguments
    from TraceLens.PerfModel.extensions.moe_perf_model_extensions import (
        moe_flydsl_stage1,
    )
    from TraceLens.PerfModel.extensions.perf_model_extensions import (
        gemm_a8w8_blockscale_bpreshuffle,
    )

    gemm = next(
        item
        for item in _decode_launches(
            replace(_pro(), facts=(("fuse_wqa_wkv", True),)), layer=2
        )
        if item.key == "gemm_a8w8"
    )
    event = {"name": gemm.op_name, "args": launch_arguments(gemm), "kernel_details": []}
    model = gemm_a8w8_blockscale_bpreshuffle(event)
    assert model.flops() > 0
    assert model.bytes() > 0

    stage1 = next(
        item for item in _decode_launches(_moe_device(), layer=3) if item.key == "moe1"
    )
    event = {
        "name": stage1.op_name,
        "args": launch_arguments(stage1),
        "kernel_details": [],
    }
    moe = moe_flydsl_stage1(event)
    assert moe.flops() > 0
    assert moe.bytes() > 0

    from TraceLens.PerfModel.extensions.attention_perf_model_extensions import (
        mla_decode_v4_asm,
    )

    pooled = replace(_pooled(), num_cus=256)
    mla = next(item for item in _decode_launches(pooled, layer=3) if item.key == "mla")
    # The four scalars sit in front of the split buffers, as the captured call
    # records them, and the split count is the one the search returns.
    assert mla.input_shapes[9:13] == ((), (), (), ())
    assert mla.input_shapes[13][1] == 16
    assert mla.concrete[12] == "16"
    attended = mla_decode_v4_asm(
        {
            "name": mla.op_name,
            "args": launch_arguments(mla),
            "kernel_details": [],
        }
    )
    assert attended.param_details["num_kv_splits"] == 16
    assert attended.bytes() > 0


def test_compressor_width_follows_the_window_overlap():
    layers = {
        2: _decode_launches(_pro(), layer=2),
        3: _decode_launches(_pro(), layer=3),
    }
    # Ratio 4 overlaps its windows: 2 * 2 * head_dim for the attention
    # compressor and 2 * 2 * index_head_dim for the indexer's. Ratio 128 does
    # not overlap.
    c4_mm = [item.input_shapes for item in layers[2] if item.key == "tensile"]
    assert c4_mm == [((4, 7168), (7168, 512)), ((4, 7168), (7168, 2048))]
    c128_mm = [item.input_shapes for item in layers[3] if item.key == "tensile"]
    assert c128_mm == [((4, 7168), (7168, 1024))]


def test_c4_indexer_runs_weights_proj_as_bf16_and_wq_b_as_blockscale():
    layer = _decode_launches(_pro(), layer=2)
    hgemm = [item for item in layer if item.key == "hgemm"]
    assert hgemm[0].input_shapes == ((4, 7168), (64, 7168))
    quants = [item.input_shapes for item in layer if item.key == "quant"]
    # Output [T, K], the bf16 input viewed by the group, then the scales.
    assert quants[0] == ((4, 7168), (224, 128), (4, 56))
    assert quants[1] == ((4, 1536), (48, 128), (4, 12))
    assert quants[2] == ((4, 2048), (64, 128), (4, 16))
    ck = [item.input_shapes for item in layer if item.key == "ck_gemm"]
    assert ck[0] == ck[1] == ((4, 1536), (8192, 1536), (4, 12), (64, 12))


def test_indexer_query_scale_layout_comes_from_the_engine_facts():
    known = replace(_pro(), facts=(("indexer_q_scale_shape", (1, 4, 16, 4)),))
    rotary = next(
        item for item in _decode_launches(known, layer=2) if item.key == "rope_had"
    )
    assert rotary.input_shapes == (
        (4, 64, 64),
        (4, 1, 4, 16, 4),
        (4, 64, 128),
        (1048576, 32),
        (1048576, 32),
        (4,),
    )
    unknown = next(
        item for item in _decode_launches(_pro(), layer=2) if item.key == "rope_had"
    )
    assert unknown.input_shapes == ((4, 64, 128),)


def test_hc_head_and_hash_topk_follow_their_parameters():
    launches = _decode_launches(_pro())
    head = next(item for item in launches if item.key == "hc_head")
    assert head.input_shapes == ((4, 4, 7168), (4, 28672), (1,), (4,))
    hash_topk = next(item for item in launches if item.key == "hash_topk")
    assert hash_topk.input_shapes == ((4, 384), (4,), (129280, 6))
    post = next(item for item in launches if item.key == "mhc_post")
    assert post.input_shapes[3] == (4, 4)


def test_moe_sorting_pads_every_expert_to_the_tile_the_cu_search_picks():
    device = replace(_pro(), num_cus=256, gfx_arch="gfx950")
    decode = _decode_launches(device, tokens=4, layer=3)
    sort_shapes = [item.input_shapes for item in decode if item.key == "mx_sort"]
    # 4 * 7 + 385 * 32 - 7 slots, 32 rows of scale padding on top.
    assert sort_shapes[0] == (
        (4, 7168),
        (12352, 224),
        (4, 7168),
        (12341,),
        (2,),
        (12341,),
    )
    assert sort_shapes[1] == (
        (28, 384),
        (12352, 16),
        (28, 384),
        (12341,),
        (2,),
        (12341,),
    )
    moe_sort = next(item for item in decode if item.key == "moe_sort")
    assert moe_sort.input_shapes[-2:] == ((295680, 224), (2759680, 16))
    # A large batch makes the search prefer the 128-row tile.
    wide = _decode_launches(device, tokens=2048, layer=3)
    wide_sort = next(item for item in wide if item.key == "mx_sort")
    assert wide_sort.input_shapes[3] == (2048 * 7 + 385 * 128 - 7,)
    # Without a device there is no compute-unit count to search with.
    bare = _decode_launches(_pro(), tokens=4, layer=3)
    assert next(item for item in bare if item.key == "mx_sort").input_shapes == (
        (4, 7168),
    )


def _moe_device(**facts):
    base = {
        "moe_padded_tiers": (32768, 131072),
        "moe_tuned_rows": (
            (1, 32, "flydsl_moe1_t32", "opus_moe2_afp8_wfp4_atomic_t32"),
            (4, 32, "flydsl_moe1_t32", "flydsl_moe2_bf16_t32_reduce_persist"),
            (8, 32, "flydsl_moe1_t32", "flydsl_moe2_bf16_t32_atomic"),
            # A CK-tile first stage with an Opus second stage is dropped.
            (2048, 64, "cktile_moe1", "opus_moe2_atomic"),
        ),
        "moe_fallback_tiles": ((2048, 32), (4096, 64), (16384, 128), (None, 64)),
        "moe_fallback_stage2_mode": "atomic",
        "opus_stage2_contract": {
            "bk_logical": 256,
            "fp4_values_per_byte": 2,
            "mfma_m": 16,
            "scale_groups_per_row_pack": 8,
        },
        "opus_stage2_dummy_arg": True,
    }
    base.update(facts)
    return replace(_pro(), num_cus=256, gfx_arch="gfx950", facts=tuple(base.items()))


def test_aiter_pads_tokens_to_the_table_key():
    from TraceLens.TreePerf.graph_schedule import _padded_moe_tokens

    tiers = (32768, 131072)
    assert [_padded_moe_tokens(n, tiers) for n in (1, 2, 3, 4, 5, 1000)] == [
        1,
        2,
        4,
        4,
        8,
        1024,
    ]
    assert _padded_moe_tokens(32768, tiers) == 32768
    assert _padded_moe_tokens(100000, tiers) == 32768
    assert _padded_moe_tokens(200000, tiers) == 131072


def test_an_atomic_opus_stage2_is_one_decode_call_with_no_reduction():
    device = _moe_device()
    launches = _decode_launches(device, tokens=1, layer=3)
    stage2 = next(item for item in launches if item.key == "moe2")
    assert stage2.op_name == "aiter::_opus_moe_stage2_a8w4_decode_fwd_raw"
    # The dummy tensor of the custom op, then the activations, both weights
    # with their padded scales, the sort outputs, and the output.
    assert stage2.input_shapes == (
        (1,),
        (1, 7, 384),
        (385, 7168, 192),
        (12320, 16),
        (2759680, 16),
        (12320,),
        (12320,),
        (385,),
        (2,),
        (1, 7168),
    )
    assert not any(item.key == "moe_red" for item in launches)


def test_a_reduce_stage2_is_followed_by_the_reduction_with_the_same_tensors():
    device = _moe_device()
    launches = _decode_launches(device, tokens=4, layer=3)
    stage2 = next(item for item in launches if item.key == "moe2")
    reduction = next(item for item in launches if item.key == "moe_red")
    assert stage2.op_name == "pseudo_op::moe_flydsl_stage2"
    assert reduction.input_shapes == stage2.input_shapes
    assert reduction.input_shapes[-2:] == ((295680, 224), (2759680, 16))


def test_an_atomic_flydsl_stage2_accumulates_without_a_reduction():
    device = _moe_device()
    launches = _decode_launches(device, tokens=8, layer=3)
    assert next(item for item in launches if item.key == "moe2").op_name == (
        "pseudo_op::moe_flydsl_stage2"
    )
    assert not any(item.key == "moe_red" for item in launches)


def test_a_table_without_the_token_count_keeps_the_reduction_and_flydsl_shapes():
    device = _moe_device(moe_tuned_rows=(), moe_fallback_tiles=())
    launches = _decode_launches(device, tokens=1, layer=3)
    assert next(item for item in launches if item.key == "moe2").op_name == (
        "pseudo_op::moe_flydsl_stage2"
    )
    assert any(item.key == "moe_red" for item in launches)


def test_the_opus_call_without_the_postponed_annotation_has_no_dummy_tensor():
    device = _moe_device(opus_stage2_dummy_arg=False)
    stage2 = next(
        item
        for item in _decode_launches(device, tokens=1, layer=3)
        if item.key == "moe2"
    )
    assert stage2.input_shapes[0] == (1, 7, 384)


def _pooled(**facts):
    """The Pro model with the pool facts a trace and the engine source give."""
    base = {
        "kv_mapping_rows": 10700801,
        "req_to_token_shape": (9, 1048580),
        "page_size": 256,
        "kv_blocks_per_c128": {4: 32, 128: 1},
        "compress_state_ring_4": 8,
        "compress_state_ring_128": 128,
        "page_index_align": 64,
        "indexer_kv_block": 1,
        "indexer_page_bucket": 8,
        "indexer_cta_target": 1024,
        "indexer_ctas_per_query": 4,
        "indexer_cta_info_width": 4,
        "topk_plan_ints": 2,
    }
    base.update(facts)
    return replace(_pro(), sliding_window=128, facts=tuple(base.items()))


def _prep(model, tokens=4):
    launches = schedule_deepseek_v4(
        model, Invocation(batch_size=3, seq_len=1, past_len=0, tokens=tokens)
    )
    embed = next(i for i, item in enumerate(launches) if item.key == "embed")
    return launches[:embed]


def test_decode_replay_prep_follows_the_backend_metadata_builders():
    prep = _prep(_pooled())
    shapes = [item.input_shapes for item in prep]
    # ``get_swa_page_indices`` reads the request-to-token table, the allocator's
    # full-to-SWA map and the page table of the batch.
    assert ((9, 1048580), ()) in shapes
    assert ((10700801,), ()) in shapes
    assert ((9, 4097), ()) in shapes
    assert ((4, 4097), ()) in shapes
    # The ratio-128 indices pad to the index alignment, the top-k to its width.
    assert ((4, 8256), ()) in shapes
    assert ((4, 8194), (4, 8194), ()) in shapes
    assert ((4, 1024), (4, 1024), ()) in shapes
    paged = next(item for item in prep if item.key == "paged_idx")
    assert paged.input_shapes[-3:] == ((512,), (4608,), (33536,))
    plans = [item for item in prep if item.key == "plan"]
    assert [item.input_shapes for item in plans] == [((9, 1048580), (4,), (4,), ())] * 2
    # Three KV streams, each with an index-pointer pad of one extra entry.
    assert sum(item.input_shapes == ((5,), ()) for item in prep) == 3
    assert [item.key for item in prep][-3:] == ["pad_pt", "sched_prep", "cta_info"]
    assert prep[-3].input_shapes == ((4, 4097),)


def test_metadata_copies_record_the_engine_dtype_and_stride():
    """A cast, a memcpy and a strided column stay distinct records.

    ``seq_lens`` is int64 and the backend casts it to int32. The indptr pad
    copies int32 to int32. The plan column is the int32 view of the uint8
    DecodePlan, and a one-row column keeps that step through ``.to(int64)``.
    """
    from TraceLens.TreePerf.graph_schedule import launch_arguments

    records = [
        launch_arguments(item) for item in _prep(_pooled()) if item.key == "copy"
    ]
    assert records[0]["Input type"] == ["int", "long int", "Scalar"]
    assert records[0]["Input Strides"] == [[1], [1], []]
    assert records[0]["Concrete Inputs"] == ["", "", "False"]
    row_types = {
        tuple(record["Input type"])
        for record in records
        if record["Input Dims"][0] == [4]
    }
    assert ("int", "long int", "Scalar") in row_types
    assert ("int", "int", "Scalar") in row_types
    assert ("long int", "int", "Scalar") in row_types

    broadcast = next(
        record
        for record in records
        if record["Input Dims"][0] == [4, 128]
        and record["Input type"][:2] == ["long int", "long int"]
    )
    assert broadcast["Input Strides"][1] == [1, 0]
    padded = next(record for record in records if record["Input Dims"][0] == [4, 8194])
    assert padded["Input type"][:2] == ["int", "int"]
    assert padded["Input Strides"] == [[8256, 1], [8194, 1], []]
    column = next(record for record in records if record["Input Strides"][1] == [4])
    assert column["Input type"] == ["long int", "int", "Scalar"]
    assert column["Input Strides"][0] == [1]

    prep = _prep(_pooled())
    window = next(item for item in prep if item.input_shapes[:2] == ((4, 1), (1, 128)))
    assert launch_arguments(window)["Input type"] == ["int", "int", "Scalar"]
    assert launch_arguments(window)["Input Strides"] == [[1, 1], [128, 1], []]
    assert any(
        item.op_name == "aten::add" and item.input_types[:2] == ("long int", "long int")
        for item in prep
    )

    one = [
        launch_arguments(item)
        for item in _prep(_pooled(), tokens=1)
        if item.key == "copy"
    ]
    single = next(
        record
        for record in one
        if record["Input Dims"][0] == [1, 128]
        and record["Input type"][:2] == ["long int", "long int"]
    )
    assert single["Input Strides"][1] == [0, 0]
    kept = next(
        record
        for record in one
        if record["Input type"] == ["long int", "int", "Scalar"]
        and record["Input Strides"][1] == [4]
    )
    assert kept["Input Strides"][0] == [4]


def test_the_repeat_and_the_logits_gather_keep_their_view_strides():
    from TraceLens.TreePerf.graph_schedule import launch_arguments

    launches = _decode_launches(_pro(), tokens=4)
    expand = next(item for item in launches if item.key == "copy")
    args = launch_arguments(expand)
    assert args["Input type"] == ["c10::BFloat16", "c10::BFloat16", "Scalar"]
    assert args["Input Strides"] == [
        [114688, 7168, 7168, 28672, 7168, 1],
        [0, 0, 28672, 7168, 7168, 1],
        [],
    ]
    head = next(index for index, item in enumerate(launches) if item.key == "hc_head")
    gather, cast = [item for item in launches[head:] if item.key == "copy"]
    assert launch_arguments(gather)["Input Strides"] == [
        [129280, 16160, 1],
        [16160, 64640, 1],
        [],
    ]
    assert launch_arguments(cast)["Input type"] == ["float", "c10::BFloat16", "Scalar"]


def test_replay_prep_needs_the_pool_facts():
    assert _prep(_pro()) == []


def test_the_fp4_write_metadata_chain_exists_only_with_an_indexer_layer():
    model = replace(_pooled(), compress_ratios=(128,) * 61)
    keys = [item.key for item in _prep(model)]
    assert "pad_pt" not in keys and "cta_info" not in keys
    assert keys.count("plan") == 1


def test_copies_match_a_memcpy_and_an_elementwise_kernel():
    from TraceLens.TreePerf.graph_schedule import _matches

    assert _matches("__amd_rocclr_copyBuffer", "copy")
    assert _matches("void at::native::direct_copy_kernel_cuda(...)", "copy")
    assert not _matches("void at::native::FillFunctor<int>", "copy")


def test_hidden_states_expand_to_the_hyper_connection_streams():
    launches = _decode_launches(_pro(), tokens=2)
    expand = next(item for item in launches if item.key == "copy")
    assert expand.input_shapes == (
        (1, 4, 1, 2, 1, 7168),
        (1, 4, 1, 2, 1, 7168),
        (),
    )


def test_the_logits_gather_copy_needs_more_than_one_token():
    def tail_copies(tokens):
        launches = _decode_launches(_pro(), tokens=tokens)
        head = next(i for i, item in enumerate(launches) if item.key == "hc_head")
        return [item.input_shapes[0] for item in launches[head:] if item.key == "copy"]

    assert tail_copies(1) == [(1, 129280)]
    assert tail_copies(4) == [(4, 8, 16160), (4, 129280)]
    launches = _decode_launches(_pro(), tokens=4)
    gather = next(item for item in launches if item.key == "allgather")
    assert gather.input_shapes == ((32, 16160), (4, 16160))


def test_a_dropped_tuned_row_gives_way_to_the_atomic_flydsl_fallback():
    from TraceLens.TreePerf.graph_schedule import _moe_dispatch

    device = _moe_device()
    model = replace(device)
    # Tuned FlyDSL and Opus rows keep their own block and kernel.
    assert _moe_dispatch(model, 4) == (32, "flydsl_moe2_bf16_t32_reduce_persist")
    # The CK-tile + Opus row is dropped; the heuristic tile follows the tokens.
    assert _moe_dispatch(model, 2048) == (64, "flydsl_moe2_atomic")
    # 2358 tokens pad to 4096 and take the 64-row tier of the fallback.
    assert _moe_dispatch(model, 2358) == (64, "flydsl_moe2_atomic")
    assert _moe_dispatch(model, 5000) == (128, "flydsl_moe2_atomic")
    assert _moe_dispatch(model, 16384) == (64, "flydsl_moe2_atomic")


def test_kernels_of_one_stopped_op_share_its_row():
    """The collector keeps a reduce and index casts on the call that launched them."""
    from TraceLens.TreePerf.graph_schedule import Launch, group_stopped_ops

    gemm = Launch("aiter::gemm_a8w8_blockscale_bpreshuffle", "gemm_a8w8", ((4,),), 0)
    reduce = Launch(
        "aiter::gemm_a8w8_blockscale_bpreshuffle",
        "gemm_red",
        ((4,),),
        0,
        same_call=True,
    )
    cast = Launch(
        "aten::copy_",
        "copy",
        ((4, 128), (4, 128), ()),
        None,
        ("long int", "int", "Scalar"),
    )
    index = Launch("aten::index", "t_index", ((8, 16), ()), None)
    plain = Launch(
        "aten::copy_",
        "copy",
        ((4, 128), (4, 128), ()),
        None,
        ("int", "int", "Scalar"),
    )
    stage2 = Launch("pseudo_op::moe_flydsl_stage2", "moe2", ((4,),), 1)
    reduction = Launch(
        "pseudo_op::moe_flydsl_stage2", "moe_red", ((4,),), 1, same_call=True
    )
    assigned = [gemm, reduce, cast, index, plain, stage2, reduction]
    assert group_stopped_ops(assigned) == [
        (0, [0, 1]),
        (3, [2, 3]),
        (4, [4]),
        (5, [5, 6]),
    ]
    # The row is the call the schedule named. A second kernel of that call
    # does not invent a different name.
    assert assigned[0].op_name == assigned[1].op_name
    assert assigned[5].op_name == assigned[6].op_name


def test_a_scan_with_a_state_kernel_and_a_gemm_with_a_split_reduction_align():
    from TraceLens.TreePerf.graph_schedule import Launch, align_kernels

    scan = Launch("aten::cumsum", "t_scan", ((4,), (), ()), None)
    mm = Launch("aten::mm", "tensile", ((4, 4), (4, 4)), 0)
    names = [
        "void rocprim::detail::init_lookback_scan_state_kernel<x>",
        "void rocprim::detail::trampoline_kernel<x>",
        "Cijk_Alik_Bljk_BSS",
        "Cijk_Alik_Bljk_BSS_PostGSU",
    ]
    assert align_kernels(names, [scan, mm]) == [scan, scan, mm, mm]


def test_a_stage2_reduction_stays_on_the_call_when_the_dispatch_omits_it():
    """An atomic tuned row lists no reduction launch. The kernel still belongs
    to the stage-2 call when the graph runs it."""
    from TraceLens.TreePerf.graph_schedule import (
        Launch,
        align_kernels,
        group_stopped_ops,
    )

    stage2 = Launch("pseudo_op::moe_flydsl_stage2", "moe2", ((4,),), 1)
    later = Launch("aiter::cross_device_reduce", "allreduce", ((4, 8),), 1)
    names = [
        "mfma_moe2_afp8_wfp4_bf16_cshuffle_t32x128x128",
        "moe_reduction_kernel_0",
        "void aiter::cross_device_reduce_1stage<std::bfloat16_t>",
    ]
    assigned = align_kernels(names, [stage2, later])
    assert assigned == [stage2, stage2, later]
    assert group_stopped_ops(assigned) == [(0, [0, 1]), (2, [2])]


def test_the_prefill_graph_ends_with_the_norm_and_both_output_copies():
    from TraceLens.TreePerf.graph_schedule import Invocation, schedule_deepseek_v4

    model = _pro()
    launches = schedule_deepseek_v4(
        model, Invocation(batch_size=1, seq_len=64, past_len=0, tokens=64, prefill=True)
    )
    norm, hidden, streams = launches[-3:]
    assert norm.key == "rmsnorm"
    h = model.hidden_size
    assert hidden.input_shapes[0] == (64, h)
    assert streams.input_shapes[0] == (64, model.hc_mult * h)
