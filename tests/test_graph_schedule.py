###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A replayed graph's kernels take the shapes of the model launch they pair with."""

from tests.fixtures.treeperf import _build_analyzer, _make_gpu_event, _mk_ac2g
from TraceLens.ModelUtils import graph_launches as graph_launches_module
from TraceLens.ModelUtils.graph_launches import GraphModel, Launch
from TraceLens.ModelUtils.parallel_plan import parallel_degree_from_trace
from TraceLens.TreePerf.graph_schedule import (
    align_kernels,
    group_stopped_ops,
    invocation_from_annotation,
    kernel_is,
    launch_arguments,
    tp_from_trace,
)
from TraceLens.util import TraceEventUtils


def test_decode_annotation_is_one_token_per_request_with_the_mean_past():
    invocation = invocation_from_annotation(
        "step[DECODE bs=3 g_sq=3 g_sqsq=9 g_sqsk=86718 g_sk=86718]"
    )
    assert invocation is not None
    assert invocation.batch_size == 3
    assert invocation.seq_len == 1
    assert invocation.tokens == 3
    assert invocation.past_len == 86718 // 3


def test_prefill_annotation_keeps_the_cached_past():
    invocation = invocation_from_annotation(
        "step[EXTEND bs=2 toks=2358 c_sq=2358 c_sqsq=2788532 c_sqsk=41422004 c_sk=35126]"
    )
    assert invocation is not None
    assert invocation.prefill
    assert invocation.tokens == 2358
    assert invocation.past_len == (35126 - 2358) // 2


def test_tensor_parallel_degree_comes_from_the_trace_process_group():
    info = {
        "distributedInfo": {
            "world_size": 16,
            "pg_config": [
                {"pg_desc": "default_pg", "pg_size": 16},
                {"pg_desc": "tp:device", "pg_size": 8},
                {"pg_desc": "tp:cpu", "pg_size": 8},
                {"pg_desc": "moe_ep:device", "pg_size": 2},
            ],
        }
    }
    assert tp_from_trace(info) == 8
    assert tp_from_trace({"distributedInfo": {"world_size": 1}}) == 1
    assert tp_from_trace({}) is None
    assert (
        tp_from_trace(
            {
                "distributedInfo": {
                    "world_size": 8,
                    "pg_config": [
                        {"pg_desc": "tp:device", "pg_size": 8},
                        {"pg_desc": "tp:cpu", "pg_size": 4},
                    ],
                }
            }
        )
        is None
    )


def test_every_process_group_is_a_degree_of_the_served_model():
    info = {
        "distributedInfo": {
            "world_size": 8,
            "pg_config": [
                {"pg_desc": "default_pg", "pg_size": 8},
                {"pg_desc": "tp:device", "pg_size": 8},
                {"pg_desc": "tp:cpu", "pg_size": 8},
                {"pg_desc": "moe_ep:device", "pg_size": 2},
                {"pg_desc": "moe_dp:device", "pg_size": 4},
            ],
        }
    }
    model = GraphModel("org/model", parallel=parallel_degree_from_trace(info))
    assert model.tp == 8
    assert model.degree("attn_tp") == 8
    assert model.degree("moe_ep") == 2
    assert model.degree("moe_dp") == 4
    assert model.degree("pp") == 1


def test_the_device_block_gives_the_compute_units_and_the_gpu_target():
    devices = [
        {
            "name": "AMD Instinct MI355X",
            "numSms": 256,
            "computeMajor": 9,
            "computeMinor": 5,
        }
    ] * 8
    metadata = {"deviceProperties": devices}
    assert TraceEventUtils.num_cus_from_trace(metadata) == 256
    assert TraceEventUtils.gpu_target_prefix_from_trace(metadata) == "gfx95"
    nvidia = [{"name": "NVIDIA H100", "computeMajor": 9, "computeMinor": 0}]
    assert TraceEventUtils.gpu_target_prefix_from_trace(
        {"deviceProperties": nvidia}
    ) == ("sm_90")
    assert TraceEventUtils.gpu_target_prefix_from_trace({}) is None


def test_a_named_kernel_anchors_and_unnamed_launches_fill_between_in_order():
    norm = Launch("lib::norm", ((4, 64),), kernels=("rmsnorm_kernel",))
    cast = Launch("aten::copy_", ((4, 64), (4, 64)))
    gemm = Launch("lib::gemm", ((4, 64), (128, 64)), kernels=("gemm_kernel",))
    names = [
        "void lib::rmsnorm_kernel<bf16>(float*)",
        "void at::native::elementwise_kernel<8>()",
        "gemm_kernel_tile_128",  # a longer identifier is not the symbol
        "void gemm_kernel<1>()",
        "void gemm_kernel<2>()",
    ]
    assigned = align_kernels(names, [norm, cast, gemm])
    # The unclaimed kernel is no second kernel of the copy: another library's.
    assert assigned == [norm, cast, None, gemm, gemm]
    # A named launch whose kernels ran never takes one that is not its own.
    assert align_kernels(["void lib::rmsnorm_kernel<f>()", "other_kernel"], [norm]) == [
        norm,
        None,
    ]


def test_an_unnamed_launch_takes_a_second_kernel_only_where_no_launch_would():
    gemm = Launch("aten::mm", ((4, 8), (8, 16)))
    add = Launch("aten::add", ((4, 16),))
    # A split reduction after the GEMM: the GEMM's second kernel.
    names = ["Cijk_gemm", "Cijk_post_gsu_reduce", "elementwise_add"]
    assert align_kernels(names, [gemm, add]) == [gemm, gemm, add]
    # As many kernels as launches: one each.
    assert align_kernels(names[:1] + names[2:], [gemm, add]) == [gemm, add]
    # An extra kernel of another library is not the GEMM's.
    embed = Launch("aten::embedding", ((4,),))
    kernels = [
        "void at::native::gather_kernel<4>()",
        "Cijk_gemm",
        "Cijk_post_gsu_reduce",
        "elementwise_add",
    ]
    assert align_kernels(kernels, [embed, gemm, add]) == [embed, gemm, gemm, add]


def test_an_unnamed_launch_does_not_take_a_kernel_a_named_launch_claims():
    cast = Launch("aten::copy_", ((4,),))
    gemm = Launch("lib::gemm", ((4,),), kernels=("gemm_kernel",))
    assert align_kernels(["gemm_kernel"], [cast, gemm]) == [gemm]


def test_kernels_of_one_launch_and_a_same_call_launch_share_a_row():
    gemm = Launch("lib::gemm", ((4,),), kernels=("gemm",))
    reduce = Launch("lib::gemm", ((4,),), kernels=("reduce",), same_call=True)
    copy = Launch("aten::copy_", ((4,),))
    assigned = [gemm, gemm, reduce, None, copy, copy]
    assert group_stopped_ops(assigned) == [(0, [0, 1, 2]), (3, [3]), (4, [4, 5])]


def test_launch_arguments_are_the_record_a_captured_call_carries():
    launch = Launch(
        "lib::op",
        ((4, 8), ()),
        input_types=("c10::BFloat16", "Scalar"),
        concrete=("", "1"),
    )
    assert launch_arguments(launch) == {
        "Input Dims": [[4, 8], []],
        "Input type": ["c10::BFloat16", "Scalar"],
        "Input Strides": [[8, 1], []],
        "Concrete Inputs": ["", "1"],
    }


def test_graph_launch_rows_take_the_shapes_of_the_launch_they_pair_with(
    monkeypatch,
):
    seen = {}
    embed = Launch(
        "lib::embedding",
        ((3,), (2, 32)),
        input_types=("long int", "c10::BFloat16"),
        kernels=("_vocab_parallel_embedding_kernel",),
    )

    def launches(model, invocation, kernel_names, op_families=None):
        seen["invocation"] = invocation
        seen["kernels"] = kernel_names
        return [embed]

    monkeypatch.setattr(graph_launches_module, "graph_launches", launches)
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
    analyzer = _build_analyzer(events, graph_model=GraphModel("org/model"))
    collected = analyzer.collect_unified_perf_events()
    assert seen["invocation"].batch_size == 3
    assert seen["invocation"].scope == (
        "step[DECODE bs=3 g_sq=3 g_sqsq=9 g_sqsk=9 g_sk=9]",
    )
    assert seen["kernels"] == [
        "void at::native::elementwise_kernel<8>()",
        "_vocab_parallel_embedding_kernel",
    ]
    row = next(event for event in collected if event["name"] == "lib::embedding")
    assert row["args"]["Input Dims"] == [[3], [2, 32]]
    assert row["args"]["Input type"] == ["long int", "c10::BFloat16"]
    # The kernel no launch claims stays the synthetic launch row a capture
    # report emits for a launch it could not reparent.
    elementwise = [e for e in collected if "elementwise_kernel" in e["name"]]
    assert [e["name"] for e in elementwise] == [
        "hipGraphLaunch->void at::native::elementwise_kernel<8>() (Synthetic Op)"
    ]
    assert "Input Dims" not in (elementwise[0].get("args") or {})


def test_a_graph_without_a_batch_annotation_takes_the_latest_step_before_it(
    monkeypatch,
):
    seen = {}

    def launches(model, invocation, kernel_names, op_families=None):
        seen["invocation"] = invocation

    monkeypatch.setattr(graph_launches_module, "graph_launches", launches)
    corr = 9
    events = [
        _make_gpu_event(
            "step",
            100,
            50,
            "user_annotation",
            "step[VERIFY bs=32 g_sq=128 g_sqsq=512 g_sqsk=162440 g_sk=40610]",
        ),
        _make_gpu_event("draft", 900, 300, "user_annotation", "draft_extend"),
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
            "some_kernel",
            pid=0,
            tid=7,
            args={"correlation": corr, "stream": 7},
        ),
        _mk_ac2g(corr, 0, 7, 1010, "s"),
        _mk_ac2g(corr, 0, 7, 1030, "f"),
    ]
    analyzer = _build_analyzer(events, graph_model=GraphModel("org/model"))
    analyzer.collect_unified_perf_events()
    assert seen["invocation"].tokens == 128
    assert seen["invocation"].scope == ("draft_extend",)


def test_a_mangled_kernel_name_is_its_symbol():
    mangled = "_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li256EEEvPT0_"
    assert kernel_is(mangled, ("dynamic_per_group_scaled_quant_kernel",))
    # A longer name sharing the prefix is spelled with its own length.
    assert not kernel_is(mangled, ("dynamic_per_group_scaled_quant",))


def test_an_op_seen_outside_a_graph_takes_only_kernels_of_the_families_it_ran():
    embed = Launch("aten::embedding", ((4,),))
    norm = Launch("lib::norm", ((4,),))
    gemm = Launch("aten::mm", ((4, 8), (8, 16)))
    kernels = [
        "void at::native::gather_kernel<4>()",
        "Cijk_gemm",
        "Cijk_post_gsu",
        "void lib::norm_kernel<1>()",
    ]
    families = {"aten::embedding": {"at"}, "aten::mm": {"Cijk"}, "lib::norm": {"lib"}}
    assert align_kernels(kernels, [embed, gemm, norm], families) == [
        embed,
        gemm,
        gemm,
        norm,
    ]
    # Out of order, a GEMM kernel never goes to the norm or the embedding.
    assert align_kernels(kernels, [embed, norm, gemm], families) == [
        embed,
        gemm,
        gemm,
        None,
    ]
    # Without the families, the copy-like launch could take a GEMM kernel.
    assert align_kernels(kernels[:2], [embed, norm]) == [embed, norm]


def test_equal_pairings_prefer_a_kernel_named_like_the_op():
    embed = Launch("aten::embedding", ((4,),))
    norm = Launch("aiter::rmsnorm", ((4,),))
    cat = Launch("aten::cat", ((4,),))
    kernels = [
        "void at::native::gather_kernel<4>()",
        "void aiter::add_rmsnorm_quant_kernel<1>()",
        "void aiter::add_rmsnorm_quant_kernel<1>()",
        "void at::native::CatArrayBatchedCopy<4>()",
    ]
    # One kernel too many: the second norm kernel is the norm's, not the cat's.
    assert align_kernels(kernels, [embed, norm, cat]) == [embed, norm, norm, cat]


def test_a_launch_whose_symbols_the_replay_never_ran_pairs_as_unnamed():
    norm = Launch("lib::norm", ((4,),), kernels=("norm_kernel",))
    attention = Launch("lib::attention", ((4,),), kernels=("dropout_state_kernel",))
    kernels = ["void norm_kernel<1>()", "void ck::fmha_kernel<2>()"]
    assigned = align_kernels(kernels, [norm, attention])
    assert assigned[0] is norm
    assert assigned[1].op_name == "lib::attention" and assigned[1].kernels == ()


def test_a_pairing_repeated_is_the_same_and_its_copy_is_the_caller_s():
    from TraceLens.ModelUtils.graph_launches import Launch
    from TraceLens.TreePerf.graph_schedule import align_kernels

    launches = [
        Launch("lib::norm", ((4, 8),), kernels=("norm",)),
        Launch("aten::add", ((4, 8),)),
    ]
    first = align_kernels(["void norm<1>()", "add_kernel"], launches)
    first[0] = None
    assert align_kernels(["void norm<1>()", "add_kernel"], launches) == [
        launches[0],
        launches[1],
    ]
