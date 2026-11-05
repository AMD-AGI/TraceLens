###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A replayed DeepSeek-V4 graph takes its shapes from the model launch at that position."""

from TraceLens.ModelUtils.graph_schedule import (
    GraphModel,
    Invocation,
    align_kernels,
    invocation_from_annotation,
    schedule_deepseek_v4,
)
from tests.fixtures.treeperf import _build_analyzer, _make_gpu_event, _mk_ac2g

# deepseek-ai/DeepSeek-V4-Pro-0813, the 61 layers the graph covers.
_RATIOS = (
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
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

    wq_b = next(
        launch
        for launch in launches
        if launch.layer == 1 and launch.key == "ck_gemm"
    )
    assert wq_b.input_shapes == ((3, 1536), (1536, 128 * 512 // 8))
    mla = next(launch for launch in launches if launch.layer == 1 and launch.key == "mla")
    assert mla.input_shapes[1][0] == (28906 + 1) // 128
    c4_mla = next(launch for launch in launches if launch.layer == 2 and launch.key == "mla")
    assert c4_mla.input_shapes[1][0] == (28906 + 1) // 4


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
    assert assigned[3] is not None and assigned[3].op_name == "aiter::mhc_pre_gemm_sqrsum"


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
