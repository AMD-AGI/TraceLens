###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Device-partition plans read from a Hugging Face config."""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.parallel_plan import (
    GpuGroup,
    ParallelDegree,
    annotate_parallel_nodes,
    build_parallel_layout,
    gpu_group_report,
    layout_from_plans,
    parse_class_plans,
    parse_engine_class_plans,
    scope_graph_to_gpu_group,
    unique_gpu_groups,
)

_SOURCE = """
class TextConfig:
    model_type = "demo_text"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = None

class VisionConfig:
    model_type = "demo_vision"

class DemoConfig:
    model_type = "demo"
    sub_configs = {"text_config": TextConfig, "vision_config": VisionConfig}

class DemoForCausalLM:
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

class ExpertConfig:
    model_type = "expert"
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
"""


def _spec(**overrides) -> ArchitectureSpec:
    values = dict(
        name="demo",
        model_type="demo",
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=12288,
        vocab_size=128,
        raw_config={
            "model_type": "demo",
            "architectures": ["DemoForCausalLM"],
            "text_config": {"linear_num_value_heads": 32},
        },
    )
    values.update(overrides)
    return ArchitectureSpec(**values)


def _layout(degree: ParallelDegree, **spec_overrides):
    return layout_from_plans(
        parse_class_plans(_SOURCE),
        _spec(**spec_overrides),
        degree,
        _spec(**spec_overrides).raw_config,
    )


def test_tensor_parallel_plan_names_the_style_and_the_widths_that_divide():
    layout = _layout(ParallelDegree(tp=4))
    by_pattern = {rule.pattern: rule for rule in layout.rules}
    assert by_pattern["layers.*.self_attn.q_proj"].collective == "sharded output"
    assert by_pattern["layers.*.self_attn.o_proj"].collective == "all-reduce"
    assert by_pattern["layers.*.linear_attn.in_proj_qkv"].collective == "gather"
    assert by_pattern["lm_head"].style == "colwise_gather_output"
    assert by_pattern["lm_head"].partitions == 4
    text = "\n".join(layout.fact_lines())
    assert "num_attention_heads 16 divides by 4" in text
    assert "num_key_value_heads 4 divides by 4" in text
    assert "intermediate_size 12288 divides by 4" in text
    assert "linear_num_value_heads 32 divides by 4" in text
    assert "vocab_size 128 divides by 4" in text


def test_a_width_that_does_not_divide_is_reported():
    layout = _layout(
        ParallelDegree(tp=8),
        num_key_value_heads=4,
        raw_config={
            "model_type": "demo",
            "architectures": ["DemoForCausalLM"],
        },
    )
    text = "\n".join(layout.fact_lines())
    assert "num_key_value_heads 4 does not divide by 8" in text
    assert "num_attention_heads 16 divides by 8" in text


def test_pipeline_stages_follow_the_declared_cut_points():
    even = _layout(ParallelDegree(pp=4))
    assert [(stage.layer_start, stage.layer_end) for stage in even.stages] == [
        (0, 8),
        (8, 16),
        (16, 24),
        (24, 32),
    ]
    assert even.stages[0].modules == ("embed_tokens",)
    assert even.stages[-1].modules == ("norm", "lm_head")
    assert "num_hidden_layers 32 divides by 4" in "\n".join(even.fact_lines())

    uneven = _layout(ParallelDegree(pp=3))
    assert [stage.layer_end - stage.layer_start for stage in uneven.stages] == [
        11,
        11,
        10,
    ]
    assert "does not divide by 3" in "\n".join(uneven.fact_lines())


def test_expert_plan_shards_experts_and_keeps_the_tensor_parallel_collective():
    classes = parse_class_plans(_SOURCE)
    spec = ArchitectureSpec(
        name="expert",
        model_type="expert",
        num_experts=64,
        raw_config={"model_type": "expert", "architectures": []},
    )
    layout = layout_from_plans(
        classes, spec, ParallelDegree(tp=2, ep=8), spec.raw_config
    )
    by_pattern = {rule.pattern: rule for rule in layout.rules}
    assert by_pattern["layers.*.mlp.experts.gate_up_proj"].partitions == 8
    assert by_pattern["layers.*.mlp.experts.gate_up_proj"].mesh == "ep"
    assert by_pattern["layers.*.mlp.experts"].mesh == "tp"
    assert by_pattern["layers.*.mlp.experts"].partitions == 2
    assert "num_experts 64 divides by 8" in "\n".join(layout.fact_lines())

    indivisible = layout_from_plans(
        classes, spec, ParallelDegree(ep=3), spec.raw_config
    )
    assert "num_experts 64 does not divide by 3" in "\n".join(indivisible.fact_lines())


def test_a_requested_axis_with_no_plan_is_reported():
    classes = parse_class_plans(_SOURCE)
    spec = ArchitectureSpec(
        name="expert",
        model_type="expert",
        num_experts=8,
        raw_config={"model_type": "expert"},
    )
    layout = layout_from_plans(classes, spec, ParallelDegree(tp=4), spec.raw_config)
    assert layout.rules == []
    assert "The model declares no tensor-parallel plan." in layout.notes


def test_a_tied_embedding_is_sharded_rowwise():
    spec = _spec(
        raw_config={
            "model_type": "demo",
            "architectures": ["DemoForCausalLM"],
            "tie_word_embeddings": True,
        }
    )
    layout = layout_from_plans(
        parse_class_plans(_SOURCE), spec, ParallelDegree(tp=4), spec.raw_config
    )
    tied = next(rule for rule in layout.rules if rule.pattern == "embed_tokens")
    assert tied.style == "embedding_rowwise"


def test_matching_modules_are_stamped_with_their_style():
    layout = _layout(ParallelDegree(tp=4))
    graph = {
        "nodes": [
            {
                "id": "model/layers/0/self_attn/q_proj",
                "label": "q_proj",
                "namespace": "model/layers/0/self_attn",
                "attrs": [{"key": "attr_name", "value": "q_proj"}],
            },
            {
                "id": "norm",
                "label": "norm",
                "namespace": "",
                "attrs": [{"key": "attr_name", "value": "norm"}],
            },
        ]
    }
    annotate_parallel_nodes(graph, layout)
    stamped = graph["nodes"][0]["attrs"][-1]
    assert stamped["key"] == "parallel"
    assert "colwise" in stamped["value"]
    assert all(item.get("key") != "parallel" for item in graph["nodes"][1]["attrs"])


def test_plans_are_read_from_the_modeling_file_and_its_configuration_sibling(tmp_path):
    modeling = tmp_path / "modeling_demo.py"
    modeling.write_text("class DemoForCausalLM:\n    pass\n", encoding="utf-8")
    config = tmp_path / "configuration_demo.py"
    config.write_text(
        "class DemoConfig:\n"
        "    model_type = 'demo'\n"
        "    base_model_tp_plan = {'layers.*.self_attn.q_proj': 'colwise'}\n",
        encoding="utf-8",
    )
    spec = ArchitectureSpec(
        name="demo",
        model_type="demo",
        num_attention_heads=8,
        code_paths=[str(modeling)],
        raw_config={"model_type": "demo", "architectures": ["DemoForCausalLM"]},
    )
    layout = build_parallel_layout(spec, ParallelDegree(tp=4))
    assert any(rule.pattern.endswith("q_proj") for rule in layout.rules)
    assert "num_attention_heads 8 divides by 4" in "\n".join(layout.fact_lines())


def test_fact_sheet_lists_the_partition_counts():
    from TraceLens.Visualizer.model_explorer_export.fact_sheet import (
        build_fact_sheet_viewer,
    )

    spec = _spec()
    spec.parallel_layout = _layout(ParallelDegree(tp=4, pp=2))
    body = build_fact_sheet_viewer(spec)["body"]
    assert "Tensor parallel: 4 partitions" in body
    assert "Pipeline parallel: 2 partitions" in body
    model_type = next(
        index
        for index, line in enumerate(body.splitlines())
        if line.startswith("- Model type:")
    )
    assert body.splitlines()[model_type + 1] == "- Parallelism: TP 4, PP 2"


def _shape(node: dict) -> str | None:
    return next(
        (item["value"] for item in node["attrs"] if item["key"] == "output_shape"),
        None,
    )


def _full_shape(node: dict) -> str | None:
    return next(
        (item["value"] for item in node["attrs"] if item["key"] == "full_shape"),
        None,
    )


def _edge(source: str) -> dict:
    return {
        "sourceNodeId": source,
        "sourceNodeOutputId": "0",
        "targetNodeInputId": "0",
    }


def _wired(producer: dict, consumer_id: str = "consumer") -> dict:
    producer["incomingEdges"] = [_edge("@input")]
    return {
        "nodes": [
            {
                "id": "@input",
                "label": "hidden_states",
                "namespace": "",
                "attrs": [{"key": "synthetic", "value": "@input"}],
            },
            producer,
            {
                "id": consumer_id,
                "label": "consumer",
                "namespace": "",
                "attrs": [{"key": "synthetic", "value": "@output"}],
                "incomingEdges": [_edge(producer["id"])],
            },
        ]
    }


def test_rowwise_output_is_all_reduced_on_its_way_to_consumers():
    from TraceLens.Visualizer.model_explorer_export.type_check import (
        integrity_check_graph_nodes,
    )

    layout = _layout(ParallelDegree(tp=4))
    producer = {
        "id": "48x_Block/mlp/down_proj",
        "label": "down_proj",
        "namespace": "48x_Block/mlp",
        "attrs": [
            {"key": "attr_name", "value": "down_proj"},
            {"key": "output_shape", "value": "[B, S, 4096] bfloat16"},
        ],
    }
    graph = _wired(producer)
    annotate_parallel_nodes(graph, layout)
    comm = next(
        node for node in graph["nodes"] if node["label"].startswith("all-reduce")
    )
    assert comm["label"] == "all-reduce x48"
    assert "8192*B*S bytes" in next(
        item["value"] for item in comm["attrs"] if item["key"] == "parallel"
    )
    consumer = next(node for node in graph["nodes"] if node["id"] == "consumer")
    assert consumer["incomingEdges"][0]["sourceNodeId"] == comm["id"]
    assert _shape(producer) == "[B, S, 4096] bfloat16"
    assert _full_shape(producer) == "[B, S, 4096] bfloat16"
    assert integrity_check_graph_nodes(graph["nodes"]) == []


def test_a_column_parallel_output_stays_sharded_without_a_collective():
    layout = _layout(ParallelDegree(tp=4))
    producer = {
        "id": "block/self_attn/q_proj",
        "label": "q_proj",
        "namespace": "block/self_attn",
        "attrs": [
            {"key": "attr_name", "value": "q_proj"},
            {"key": "output_shape", "value": "[2, 4, 8] float32"},
        ],
    }
    graph = _wired(producer)
    annotate_parallel_nodes(graph, layout)
    assert [node["label"] for node in graph["nodes"] if node["id"] != "@input"] == [
        "q_proj",
        "consumer",
    ]


def test_a_gather_moves_each_ranks_shard():
    layout = _layout(ParallelDegree(tp=4))
    producer = {
        "id": "block/linear_attn/in_proj_qkv",
        "label": "in_proj_qkv",
        "namespace": "block/linear_attn",
        "attrs": [
            {"key": "attr_name", "value": "in_proj_qkv"},
            {"key": "output_shape", "value": "[2, 4, 8] float32"},
        ],
    }
    graph = _wired(producer)
    annotate_parallel_nodes(graph, layout)
    comm = next(node for node in graph["nodes"] if node["label"] == "gather")
    # Full output is 2*4*8*4 = 256 bytes; the shard that moves is that divided by 4.
    assert "64 bytes" in next(
        item["value"] for item in comm["attrs"] if item["key"] == "parallel"
    )
    assert comm["attrs"][2]["value"] == "[2, 4, 8] float32"
    assert _shape(producer) == "[2, 4, 2] float32"
    assert _full_shape(producer) == "[2, 4, 8] float32"
    assert _full_shape(comm) == "[2, 4, 8] float32"


def test_a_column_parallel_shape_is_per_gpu_and_keeps_the_full_tensor():
    layout = _layout(ParallelDegree(tp=4))
    producer = {
        "id": "block/self_attn/q_proj",
        "label": "q_proj",
        "namespace": "block/self_attn",
        "attrs": [
            {"key": "attr_name", "value": "q_proj"},
            {"key": "output_shape", "value": "[2, 4, 8] float32"},
        ],
        "outputsMetadata": [
            {"id": "0", "attrs": [{"key": "shape", "value": "[2, 4, 8] float32"}]}
        ],
    }
    consumer = {
        "id": "block/self_attn/view",
        "label": "view",
        "namespace": "block/self_attn",
        "attrs": [{"key": "output_shape", "value": "[2, 4, 8] float32"}],
        "incomingEdges": [_edge(producer["id"])],
    }
    graph = _wired(producer)
    graph["nodes"].append(consumer)
    annotate_parallel_nodes(graph, layout)
    assert _shape(producer) == "[2, 4, 2] float32"
    assert _full_shape(producer) == "[2, 4, 8] float32"
    assert producer["outputsMetadata"][0]["attrs"][0]["value"] == "[2, 4, 2] float32"
    assert producer["outputsMetadata"][0]["attrs"][1]["value"] == "[2, 4, 8] float32"
    assert _shape(consumer) == "[2, 4, 2] float32"
    assert _full_shape(consumer) == "[2, 4, 8] float32"

    uneven = {
        "id": "block/self_attn/q_proj",
        "label": "q_proj",
        "namespace": "block/self_attn",
        "attrs": [
            {"key": "attr_name", "value": "q_proj"},
            {"key": "output_shape", "value": "[5] float32"},
        ],
    }
    uneven_graph = _wired(uneven)
    annotate_parallel_nodes(uneven_graph, _layout(ParallelDegree(tp=2)))
    assert _shape(uneven) == "[3] float32"
    assert _full_shape(uneven) == "[5] float32"
    from TraceLens.ModelUtils.parallel_plan import _localize_gpu_shapes

    uneven_layout = _layout(ParallelDegree(tp=2))
    uneven_layout.gpu_group = GpuGroup(index=1, gpus=(1,))
    # Rank 0 was drawn first. Rank 1 reads the saved full shape, not that shard.
    _localize_gpu_shapes({"nodes": uneven_graph["nodes"]}, uneven_layout)
    assert _shape(uneven) == "[2] float32"
    assert _full_shape(uneven) == "[5] float32"


def test_expert_count_on_one_gpu_is_the_local_slice():
    classes = parse_class_plans(_SOURCE)
    spec = ArchitectureSpec(
        name="expert",
        model_type="expert",
        num_experts=64,
        raw_config={"model_type": "expert", "architectures": []},
    )
    layout = layout_from_plans(classes, spec, ParallelDegree(ep=8), spec.raw_config)
    one_hot = {
        "id": "block/mlp/sidefeed:2:experts:@op_one_hot:1",
        "label": "One hot",
        "namespace": "block/mlp",
        "attrs": [
            {"key": "attr_name", "value": "@op_one_hot"},
            {"key": "output_shape", "value": "[B*S, 4, 64] int64"},
        ],
    }
    graph = {"nodes": [one_hot]}
    annotate_parallel_nodes(graph, layout)
    assert _shape(one_hot) == "[B*S, 4, 8] int64"
    assert _full_shape(one_hot) == "[B*S, 4, 64] int64"


def test_gradient_all_reduce_is_drawn_only_for_a_training_run():
    producer = {
        "id": "block/self_attn/q_norm",
        "label": "q_norm",
        "namespace": "block/self_attn",
        "attrs": [{"key": "attr_name", "value": "q_norm"}],
    }
    inference = _layout(ParallelDegree(tp=4))
    graph = _wired(producer)
    annotate_parallel_nodes(graph, inference)
    assert all(node["label"] != "gradient all-reduce" for node in graph["nodes"])

    training = _layout(ParallelDegree(tp=4))
    training.mode = "training"
    graph = _wired(producer)
    annotate_parallel_nodes(graph, training)
    assert any(node["label"] == "gradient all-reduce" for node in graph["nodes"])


def test_pipeline_boundaries_send_the_hidden_state():
    from TraceLens.Visualizer.model_explorer_export.type_check import (
        integrity_check_graph_nodes,
    )

    layout = _layout(ParallelDegree(pp=3))
    hidden = {
        "id": "32x_Block/@output:hidden_states",
        "label": "hidden_states",
        "namespace": "32x_Block",
        "attrs": [{"key": "output_shape", "value": "[B, S, 128] bfloat16"}],
        "incomingEdges": [_edge("@input")],
    }
    graph = {
        "nodes": [
            {
                "id": "@input",
                "label": "hidden_states",
                "namespace": "",
                "attrs": [{"key": "synthetic", "value": "@input"}],
            },
            hidden,
            {
                "id": "next",
                "label": "next",
                "namespace": "",
                "attrs": [{"key": "synthetic", "value": "@output"}],
                "incomingEdges": [_edge(hidden["id"])],
            },
        ]
    }
    annotate_parallel_nodes(graph, layout)
    labels = [node["label"] for node in graph["nodes"]]
    assert labels.count("send stage 0->1 hidden_states") == 1
    assert labels.count("send stage 1->2 hidden_states") == 1
    assert sum(label.startswith("recv") for label in labels) == 2
    recv = next(node for node in graph["nodes"] if node["label"].startswith("recv"))
    assert next(
        item["value"] for item in recv["attrs"] if item["key"] == "synthetic"
    ) == ("communication_sink")
    # The send watches the activation. The stack's own consumer stays on it.
    assert graph["nodes"][2]["incomingEdges"][0]["sourceNodeId"] == hidden["id"]
    assert integrity_check_graph_nodes(graph["nodes"]) == []
    assert "send/recv hidden_states across 2 pipeline boundaries" in "\n".join(
        layout.fact_lines()
    )


def test_expert_dispatch_and_combine_move_tokens_times_hidden():
    classes = parse_class_plans(_SOURCE)
    spec = ArchitectureSpec(
        name="expert",
        model_type="expert",
        num_experts=64,
        num_experts_per_tok=2,
        hidden_size=32,
        raw_config={"model_type": "expert", "architectures": []},
    )
    layout = layout_from_plans(
        classes, spec, ParallelDegree(tp=2, ep=8), spec.raw_config
    )
    router = {
        "id": "block/mlp/gate",
        "label": "gate",
        "namespace": "block/mlp",
        "attrs": [
            {"key": "attr_name", "value": "gate"},
            {"key": "output_shape", "value": "[B, S, 32] bfloat16"},
        ],
    }
    experts = {
        "id": "block/mlp/experts",
        "label": "experts",
        "namespace": "block/mlp",
        "attrs": [
            {"key": "attr_name", "value": "experts"},
            {"key": "output_shape", "value": "[B, S, 32] bfloat16"},
        ],
    }
    graph = {
        "nodes": [
            router,
            experts,
            {
                "id": "after-router",
                "label": "scores",
                "namespace": "block/mlp",
                "incomingEdges": [_edge(router["id"])],
            },
            {
                "id": "after-experts",
                "label": "hidden",
                "namespace": "block/mlp",
                "incomingEdges": [_edge(experts["id"])],
            },
        ]
    }
    annotate_parallel_nodes(graph, layout)
    dispatch = next(node for node in graph["nodes"] if node["label"] == "dispatch")
    combine = next(node for node in graph["nodes"] if node["label"] == "combine")
    # B*S*32*2 experts/token * 2 bytes
    assert "128*B*S bytes" in next(
        item["value"] for item in dispatch["attrs"] if item["key"] == "parallel"
    )
    reduced = next(node for node in graph["nodes"] if node["label"] == "all-reduce")
    assert combine["incomingEdges"][0]["sourceNodeId"] == experts["id"]
    assert reduced["incomingEdges"][0]["sourceNodeId"] == combine["id"]
    assert graph["nodes"][2]["incomingEdges"][0]["sourceNodeId"] == dispatch["id"]
    assert graph["nodes"][3]["incomingEdges"][0]["sourceNodeId"] == reduced["id"]
    text = "\n".join(layout.fact_lines())
    assert "dispatch ep x8" in text
    assert "combine ep x8" in text


def test_activation_bytes_accept_a_packed_batch_and_skip_an_index_dtype():
    layout = _layout(ParallelDegree(tp=4))
    producer = {
        "id": "48x_Block/mlp/down_proj",
        "label": "down_proj",
        "namespace": "48x_Block/mlp",
        "attrs": [
            {"key": "attr_name", "value": "down_proj"},
            {"key": "output_shape", "value": "[B*S, 4096] bfloat16"},
        ],
    }
    graph = _wired(producer)
    annotate_parallel_nodes(graph, layout)
    reduced = next(node for node in graph["nodes"] if "all-reduce" in node["label"])
    assert "8192*B*S bytes" in next(
        item["value"] for item in reduced["attrs"] if item["key"] == "parallel"
    )

    classes = parse_class_plans(_SOURCE)
    spec = ArchitectureSpec(
        name="expert",
        model_type="expert",
        num_experts=64,
        num_experts_per_tok=2,
        hidden_size=32,
        raw_config={"model_type": "expert", "architectures": []},
    )
    expert = layout_from_plans(
        classes, spec, ParallelDegree(tp=2, ep=8), spec.raw_config
    )
    router = {
        "id": "block/mlp/gate",
        "label": "gate",
        "namespace": "block/mlp",
        "attrs": [
            {"key": "attr_name", "value": "gate"},
            {"key": "output_shape", "value": "[B*S, 4] int64"},
        ],
    }
    experts = {
        "id": "block/mlp/experts",
        "label": "experts",
        "namespace": "block/mlp",
        "attrs": [
            {"key": "attr_name", "value": "experts"},
            {"key": "output_shape", "value": "[B*S, 32] bfloat16"},
        ],
    }
    graph = {
        "nodes": [
            router,
            experts,
            {
                "id": "after-router",
                "label": "scores",
                "namespace": "block/mlp",
                "incomingEdges": [_edge(router["id"])],
            },
            {
                "id": "after-experts",
                "label": "hidden",
                "namespace": "block/mlp",
                "incomingEdges": [_edge(experts["id"])],
            },
        ]
    }
    annotate_parallel_nodes(graph, expert)
    dispatch = next(node for node in graph["nodes"] if node["label"] == "dispatch")
    combine = next(node for node in graph["nodes"] if node["label"] == "combine")
    reduced = next(node for node in graph["nodes"] if node["label"] == "all-reduce")
    dispatch_text = next(
        item["value"] for item in dispatch["attrs"] if item["key"] == "parallel"
    )
    assert "bytes" not in dispatch_text
    assert "128*B*S bytes" in next(
        item["value"] for item in combine["attrs"] if item["key"] == "parallel"
    )
    assert "64*B*S bytes" in next(
        item["value"] for item in reduced["attrs"] if item["key"] == "parallel"
    )


def test_a_collective_matches_the_attribute_path_carried_in_the_node_id():
    layout = _layout(ParallelDegree(tp=4))
    producer = {
        "id": "decoder/48x_Block/linear_attn/seq:0:in_proj_qkv:in_proj_qkv:0",
        "label": "Linear",
        "namespace": "48x_Block/SomeAttention",
        "attrs": [
            {"key": "attr_name", "value": "in_proj_qkv"},
            {"key": "output_shape", "value": "[2, 4, 8] float32"},
        ],
    }
    nested = {
        "id": (
            "decoder/48x_Block/self_attn/seq:1:indexer@l12:indexer@l12:0/"
            "seq:0:q_proj:q_proj:0"
        ),
        "label": "Linear",
        "namespace": "48x_Block/Indexer",
        "attrs": [{"key": "attr_name", "value": "q_proj"}],
        "incomingEdges": [_edge("@input")],
    }
    graph = _wired(producer)
    graph["nodes"].append(nested)
    annotate_parallel_nodes(graph, layout)
    comm = next(node for node in graph["nodes"] if node["label"].startswith("gather"))
    assert "64 bytes" in next(
        item["value"] for item in comm["attrs"] if item["key"] == "parallel"
    )
    consumer = next(node for node in graph["nodes"] if node["id"] == "consumer")
    assert consumer["incomingEdges"][0]["sourceNodeId"] == comm["id"]
    assert all(item.get("key") != "parallel" for item in nested["attrs"])


def test_an_inlined_module_gets_one_collective_on_the_edge_that_leaves():
    from TraceLens.Visualizer.model_explorer_export.type_check import (
        integrity_check_graph_nodes,
    )

    classes = parse_class_plans(_SOURCE)
    spec = ArchitectureSpec(
        name="expert",
        model_type="expert",
        num_experts=64,
        num_experts_per_tok=2,
        hidden_size=32,
        raw_config={"model_type": "expert", "architectures": []},
    )
    layout = layout_from_plans(
        classes, spec, ParallelDegree(tp=2, ep=8), spec.raw_config
    )
    zeros = "decoder/48x_Block/mlp/sidefeed:2:experts:@op_zeros:0"
    inner = "decoder/48x_Block/mlp/sidefeed:2:experts:@op_inner:1"
    index_add = "decoder/48x_Block/mlp/sidefeed:2:experts:@op_index_add:2"
    gate = "decoder/48x_Block/mlp/sideproducer:2:gate:@op_l2/@output:scores"
    scores = "decoder/48x_Block/mlp/sidefeed:2:experts:@op_one_hot:3"
    boundary = "decoder/48x_Block/mlp/@input:initial"
    body = "decoder/48x_Block/mlp/@body_out:final"

    def op(node_id: str, sources: list[str], **extra: object) -> dict:
        node = {
            "id": node_id,
            "label": node_id.rsplit("/", 1)[-1],
            "namespace": "48x_Block",
            "incomingEdges": [_edge(source) for source in sources],
        }
        if extra:
            node.update(extra)
        return node

    graph = {
        "nodes": [
            {
                "id": "@input",
                "label": "hidden_states",
                "namespace": "",
                "attrs": [{"key": "synthetic", "value": "@input"}],
            },
            op(zeros, ["@input"]),
            op(inner, [zeros]),
            op(
                gate,
                ["@input"],
                attrs=[{"key": "output_shape", "value": "[B, S, 32] bfloat16"}],
            ),
            op(scores, [gate]),
            op(
                index_add,
                [inner, scores],
                attrs=[{"key": "output_shape", "value": "[B, S, 32] bfloat16"}],
            ),
            op(boundary, [zeros]),
            op(body, [index_add]),
            {
                "id": "consumer",
                "label": "consumer",
                "namespace": "",
                "attrs": [{"key": "synthetic", "value": "@output"}],
                "incomingEdges": [_edge(body)],
            },
        ]
    }
    annotate_parallel_nodes(graph, layout)
    labels = [node["label"] for node in graph["nodes"]]
    assert sum(label.startswith("dispatch") for label in labels) == 1
    assert sum(label.startswith("combine") for label in labels) == 1
    assert sum(label.startswith("all-reduce") for label in labels) == 1
    by_id = {node["id"]: node for node in graph["nodes"]}
    assert by_id[inner]["incomingEdges"][0]["sourceNodeId"] == zeros
    assert by_id[boundary]["incomingEdges"][0]["sourceNodeId"] == zeros
    dispatch = next(
        node for node in graph["nodes"] if node["label"].startswith("dispatch")
    )
    combine = next(
        node for node in graph["nodes"] if node["label"].startswith("combine")
    )
    reduced = next(
        node for node in graph["nodes"] if node["label"].startswith("all-reduce")
    )
    assert by_id[scores]["incomingEdges"][0]["sourceNodeId"] == dispatch["id"]
    assert combine["incomingEdges"][0]["sourceNodeId"] == index_add
    assert reduced["incomingEdges"][0]["sourceNodeId"] == combine["id"]
    assert by_id[body]["incomingEdges"][0]["sourceNodeId"] == reduced["id"]
    assert all(
        item.get("key") != "parallel" for item in by_id[zeros].get("attrs") or []
    )
    assert integrity_check_graph_nodes(graph["nodes"]) == []


def test_gpus_with_the_same_stage_and_shapes_share_a_subgraph():
    pipeline = _layout(ParallelDegree(pp=4))
    groups = unique_gpu_groups(pipeline, _spec(num_experts=4))
    # Stages of 8 layers: stage 0 carries the embedding, the last carries the
    # head, and the two middle stages are the same drawing.
    assert [group.gpus for group in groups] == [(0,), (1, 2), (3,)]
    assert groups[1].experts == (0, 1, 2, 3)
    assert "subgraph 1: GPUs 1-2" in "\n".join(
        gpu_group_report(groups, pipeline.degree)
    )

    uneven_pipeline = _layout(ParallelDegree(pp=3))
    assert [group.gpus for group in unique_gpu_groups(uneven_pipeline, _spec())] == [
        (0,),
        (1,),
        (2,),
    ]

    even_tensor = _layout(ParallelDegree(tp=4))
    assert [group.gpus for group in unique_gpu_groups(even_tensor, _spec())] == [
        (0, 1, 2, 3)
    ]


def test_expert_slices_and_uneven_shards_split_groups_when_shapes_differ():
    experts = _layout(ParallelDegree(pp=2, ep=2))
    groups = unique_gpu_groups(experts, _spec(num_experts=4))
    assert [group.gpus for group in groups] == [(0, 1), (2, 3)]
    assert groups[0].expert_sets == (((0,), (0, 1)), ((1,), (2, 3)))
    text = "\n".join(_fact_lines_for(experts, groups[0]))
    assert "Subgraph 0: GPUs 0-1" in text
    assert "experts 0-3" in text
    assert "GPU 0: experts 0-1" in text
    assert "GPU 1: experts 2-3" in text

    uneven = _layout(ParallelDegree(ep=2), num_experts=3)
    split = unique_gpu_groups(uneven, _spec(num_experts=3))
    assert [group.experts for group in split] == [(0, 1), (2,)]

    widths = _layout(ParallelDegree(tp=2), num_attention_heads=3)
    rank_groups = unique_gpu_groups(widths, _spec(num_attention_heads=3))
    assert [group.gpus for group in rank_groups] == [(0,), (1,)]
    assert rank_groups[0].shards == (("num_attention_heads", 2),)
    assert rank_groups[1].shards == (("num_attention_heads", 1),)


def _fact_lines_for(layout, group: GpuGroup) -> list[str]:
    layout.gpu_group = group
    return layout.fact_lines()


def test_a_pipeline_group_draws_only_that_stage():
    layout = _layout(ParallelDegree(pp=4))
    groups = unique_gpu_groups(layout, _spec())
    layout.gpu_group = groups[0]
    graph = {
        "nodes": [
            _module("embed_tokens"),
            _module("32x_Block/mlp/down_proj", namespace="32x_Block/mlp"),
            _module("32x_Block/input_layernorm", namespace="32x_Block"),
            _module("norm"),
            _module("lm_head"),
            _send("32x_Block", "0→1"),
            _send("32x_Block", "1→2"),
        ],
        "groupNodeAttributes": {
            "": {"architecture_fact_sheet": "old"},
            "32x_Block": {"repeat": "32x_Block"},
            "norm": {"title": "norm"},
        },
    }
    scope_graph_to_gpu_group(graph, layout)
    ids = [node["id"] for node in graph["nodes"]]
    assert "embed_tokens" in ids
    assert "8x_Block/mlp/down_proj" in ids
    assert "8x_Block/input_layernorm" in ids
    assert "norm" not in ids and "lm_head" not in ids
    assert any("stage 0→1" in node_id for node_id in ids)
    assert not any("stage 1→2" in node_id for node_id in ids)
    assert "8x_Block" in graph["groupNodeAttributes"]
    assert graph["groupNodeAttributes"]["8x_Block"]["repeat"] == "8x_Block"
    assert "norm" not in graph["groupNodeAttributes"]
    text = "\n".join(layout.fact_lines())
    assert "Subgraph 0: GPU 0" in text
    assert "pipeline stage 0" in text
    assert "across 1 pipeline boundary" in text

    layout.gpu_group = groups[1]
    middle = {
        "nodes": [
            _module("embed_tokens"),
            _module("32x_Block/mlp/down_proj", namespace="32x_Block/mlp"),
            _module("norm"),
            _send("32x_Block", "0→1"),
            _send("32x_Block", "1→2"),
            _send("32x_Block", "2→3"),
        ]
    }
    scope_graph_to_gpu_group(middle, layout)
    middle_ids = [node["id"] for node in middle["nodes"]]
    assert "embed_tokens" not in middle_ids and "norm" not in middle_ids
    assert "8x_Block/mlp/down_proj" in middle_ids
    assert any("stage 0→1" in node_id for node_id in middle_ids)
    assert any("stage 1→2" in node_id for node_id in middle_ids)
    assert not any("stage 2→3" in node_id for node_id in middle_ids)


def _module(node_id: str, namespace: str | None = None) -> dict:
    leaf = node_id.rsplit("/", 1)[-1]
    return {
        "id": node_id,
        "label": leaf,
        "namespace": namespace if namespace is not None else "",
        "attrs": [{"key": "attr_name", "value": leaf}],
    }


def _send(namespace: str, hop: str) -> dict:
    return {
        "id": f"{namespace}/@output:hidden_states/@comm:send:stage {hop} hidden_states",
        "label": f"send stage {hop} hidden_states",
        "namespace": namespace,
    }


# Trimmed to the lines that matter, but the idioms (class/kwarg names, the
# make_layers/make_pp_layers tuple order, the is_first_rank/is_last_rank
# conditionals) are read straight off vllm==0.32.0 and sglang==0.5.22 --
# the engines this plan recognizes.
_VLLM_ENGINE_SOURCE = """
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, prefix=""):
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, prefix=f"{prefix}.gate_up_proj"
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, prefix=f"{prefix}.down_proj"
        )

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, prefix=""):
        self.qkv_proj = QKVParallelLinear(hidden_size, head_dim, total_num_heads)
        self.o_proj = RowParallelLinear(hidden_size, hidden_size)

class LlamaModel(nn.Module):
    def __init__(self, config, prefix=""):
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, lambda prefix: LlamaDecoderLayer(config, prefix), prefix
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size)
        else:
            self.norm = PPMissingLayer()

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
"""

_SGLANG_ENGINE_SOURCE = """
class MiniMaxM3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, prefix=""):
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)

class MiniMaxM3MoE(nn.Module):
    def __init__(self, config, layer_id, prefix=""):
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            prefix=add_prefix("experts", prefix),
        )

class MiniMaxM3SparseAttention(nn.Module):
    def __init__(self, hidden_size, prefix=""):
        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(hidden_size, head_dim, total_num_heads)
        self.o_proj = RowParallelLinear(hidden_size, hidden_size, reduce_results=False)

class MiniMaxM3Model(nn.Module):
    def __init__(self, config, prefix=""):
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = PPMissingLayer()
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers, layer_fn, prefix=add_prefix("layers", prefix)
        )
        if self.pp_group.is_last_rank:
            if self.use_gemma_norm:
                self.norm = GemmaRMSNorm(config.hidden_size)
            else:
                self.norm = RMSNorm(config.hidden_size)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

class MiniMaxM3SparseForCausalLM(nn.Module):
    def __init__(self, config):
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        else:
            self.lm_head = PPMissingLayer()
"""


def test_vllm_linear_classes_read_as_the_matching_hf_style():
    classes = {c.name: c for c in parse_engine_class_plans(_VLLM_ENGINE_SOURCE)}
    assert classes["LlamaMLP"].tp == {
        "gate_up_proj": "packed_colwise",
        "down_proj": "rowwise",
    }
    assert classes["LlamaAttention"].tp == {
        "qkv_proj": "packed_colwise",
        "o_proj": "rowwise",
    }
    assert classes["LlamaModel"].tp == {"embed_tokens": "embedding_rowwise"}
    assert classes["LlamaForCausalLM"].tp == {"lm_head": "embedding_rowwise"}


def test_vllm_pipeline_boundaries_follow_pp_group_rank_checks():
    classes = {c.name: c for c in parse_engine_class_plans(_VLLM_ENGINE_SOURCE)}
    assert classes["LlamaModel"].pp == [
        ("embed_tokens", ["input_ids"], ["inputs_embeds"]),
        ("layers", ["hidden_states"], ["hidden_states"]),
        ("norm", ["hidden_states"], []),
    ]
    # lm_head's real-module branch runs tie_weights() after the assignment;
    # the boundary still reads off the FIRST self.lm_head assignment.
    assert classes["LlamaForCausalLM"].pp == [("lm_head", ["hidden_states"], [])]


def test_sglang_reduce_results_false_reads_as_rowwise_split_input():
    classes = {c.name: c for c in parse_engine_class_plans(_SGLANG_ENGINE_SOURCE)}
    assert classes["MiniMaxM3SparseAttention"].tp == {
        "qkv_proj": "packed_colwise",  # matched via its indexer-subclass suffix
        "o_proj": "rowwise_split_input",
    }


def test_sglang_expert_factory_is_read_by_its_num_experts_keyword():
    classes = {c.name: c for c in parse_engine_class_plans(_SGLANG_ENGINE_SOURCE)}
    assert classes["MiniMaxM3MoE"].ep == {"experts": "grouped_gemm"}
    assert classes["MiniMaxM3MoE"].tp is None


def test_sglang_make_pp_layers_puts_modules_first_not_last():
    classes = {c.name: c for c in parse_engine_class_plans(_SGLANG_ENGINE_SOURCE)}
    # A nested if/else picking GemmaRMSNorm vs. RMSNorm still reads as one
    # norm boundary entry, on the pp_group.is_last_rank side.
    assert classes["MiniMaxM3Model"].pp == [
        ("embed_tokens", ["input_ids"], ["inputs_embeds"]),
        ("layers", ["hidden_states"], ["hidden_states"]),
        ("norm", ["hidden_states"], []),
    ]


def _engine_spec(
    engine: str, path, architectures: list[str], **overrides
) -> ArchitectureSpec:
    values = dict(
        name=architectures[0],
        model_type=architectures[0],
        engine=engine,
        code_paths=[str(path)],
        raw_config={"model_type": architectures[0], "architectures": architectures},
    )
    values.update(overrides)
    return ArchitectureSpec(**values)


def test_vllm_engine_layout_shards_a_dense_model_across_tp_and_pp(tmp_path):
    path = tmp_path / "llama.py"
    path.write_text(_VLLM_ENGINE_SOURCE, encoding="utf-8")
    spec = _engine_spec(
        "vllm",
        path,
        ["LlamaForCausalLM"],
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
    )
    layout = build_parallel_layout(spec, ParallelDegree(tp=8, pp=2))
    by_pattern = {rule.pattern: rule for rule in layout.rules}
    assert by_pattern["down_proj"].collective == "all-reduce"
    assert by_pattern["qkv_proj"].collective == "sharded output"
    assert [(stage.layer_start, stage.layer_end) for stage in layout.stages] == [
        (0, 40),
        (40, 80),
    ]
    assert layout.stages[0].modules == ("embed_tokens",)
    assert layout.stages[-1].modules == ("norm", "lm_head")
    text = "\n".join(layout.fact_lines())
    assert "vocab_size 128256 divides by 8" in text


def test_sglang_engine_layout_shards_a_moe_model_across_tp_pp_and_ep(tmp_path):
    path = tmp_path / "minimax_m3.py"
    path.write_text(_SGLANG_ENGINE_SOURCE, encoding="utf-8")
    spec = _engine_spec(
        "sglang",
        path,
        ["MiniMaxM3SparseForCausalLM"],
        num_hidden_layers=60,
        num_attention_heads=64,
        num_key_value_heads=4,
        intermediate_size=3072,
        vocab_size=200064,
        num_experts=128,
        num_experts_per_tok=4,
    )
    layout = build_parallel_layout(spec, ParallelDegree(tp=8, pp=2, ep=4))
    by_pattern = {rule.pattern: rule for rule in layout.rules}
    assert by_pattern["experts"].mesh == "ep"
    assert by_pattern["experts"].partitions == 4
    assert by_pattern["o_proj"].collective == "local output"
    groups = unique_gpu_groups(layout, spec)
    assert len(groups) == 4  # one per expert-parallel rank; tp/pp ranks match shapes
    text = "\n".join(layout.fact_lines())
    assert "num_key_value_heads 4 does not divide by 8" in text
    assert "num_experts 128 divides by 4" in text


def test_default_engine_is_unaffected_by_the_new_parser():
    classes = parse_class_plans(_SOURCE)
    assert classes  # the existing Hugging Face fixture still parses as before
    assert parse_engine_class_plans(_SOURCE) == []  # no vLLM/SGLang idiom in it


def test_partition_flags_default_to_one_and_reject_zero(capsys):
    from TraceLens.Visualizer.model_explorer_export import cli

    args = cli.build_parser().parse_args(["org/model"])
    assert (args.tp, args.pp, args.ep) == (1, 1, 1)
    with pytest.raises(SystemExit):
        cli.main(["org/model", "--ep", "0"])
    assert "--ep" in capsys.readouterr().err
