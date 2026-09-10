###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Coverage-focused unit tests for shape_inference internals.

These exercise the individual per-op shape branches, module ctor parsing, the
torch/meta op builders, and the many small helper functions directly. They do
not modify any source file and are independent of the existing
``test_shape_inference.py`` suite.
"""

from __future__ import annotations

import ast as _ast
import json
import logging
from pathlib import Path

import pytest

from TraceLens.ModelUtils.ast_analyze import analyze_source
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import BlockNode, build_block_node
from TraceLens.ModelUtils.extract import ArchitectureSpec, load_architecture
from TraceLens.ModelUtils.model_graph import (
    ModelGraphNode,
    NodeKind,
    OperationKind,
    build_model_graph,
)
from TraceLens.ModelUtils.shape_inference import (
    build_operator_export,
    save_operator_export,
    serialize_dim,
    subgraph_boundary_signature,
    ModuleConvSpec,
    ModuleDimRegistry,
    ModuleEmbeddingSpec,
    ModuleLinearSpec,
    ModuleParameterSpec,
    ShapeContext,
    ShapeInferencer,
    Symbol,
    TensorSpec,
    _call_class_name,
    _config_dtype,
    _default_hidden_shape,
    _merge_flatten_dim,
    _module_path_matches,
    _module_path_segments,
    _normalize_module_patterns,
    _quant_storage_dtype,
    _resolve_cast_dtype,
    _eval_config_condition,
    _eval_dim_expr,
    _export_operation_kind,
    _first_tensor_shape,
    _heuristic_linear_out_features,
    _infer_einsum_shape,
    _is_embedding,
    _looks_valid,
    _low_level_computation,
    _operator_name,
    _output_tensor_name,
    _parse_module_ctor,
    _parse_split_sizes,
    _parse_tensor_ctor_shape,
    _resolve_dim_expr,
    _resolve_dim_name,
    _resolve_op_int,
    _resolve_view_shape,
    _symbolic_binop,
    _torch_dtype,
    _torch_op_chunk,
    _torch_op_split,
    _torch_op_unbind,
    _torch_op_unflatten,
)


# ---------------------------------------------------------------------------
# Local helpers (mirrors the patterns in test_shape_inference.py)
# ---------------------------------------------------------------------------


def _make_inferencer(**dims: int) -> ShapeInferencer:
    spec = ArchitectureSpec(
        name="Test",
        model_type="test",
        hidden_size=dims.get("hidden_size", 4096),
        raw_config={"hidden_size": dims.get("hidden_size", 4096)},
    )
    ctx = ShapeContext.from_spec(spec)
    for key, val in dims.items():
        ctx.dims[key] = val
    return ShapeInferencer(spec, context=ctx)


def _node(
    label: str,
    *,
    details: list[str] | None = None,
    external_inputs: list[str] | None = None,
    operation: OperationKind = OperationKind.TORCH_FUNCTIONAL,
    kind: NodeKind = NodeKind.LEAF,
    node_id: str = "n1",
    class_name: str | None = None,
    meta: dict | None = None,
) -> ModelGraphNode:
    m: dict = {"class_name": class_name if class_name is not None else label}
    if details:
        m["details"] = details
    if external_inputs:
        m["external_inputs"] = external_inputs
    if meta:
        m.update(meta)
    return ModelGraphNode(
        id=node_id, kind=kind, label=label, operation=operation, metadata=m
    )


# ---------------------------------------------------------------------------
# ShapeContext.from_spec
# ---------------------------------------------------------------------------


def test_shape_context_uses_explicit_head_dim():
    spec = ArchitectureSpec(
        name="t",
        model_type="t",
        hidden_size=4096,
        num_attention_heads=32,
        head_dim=64,
        raw_config={"hidden_size": 4096},
    )
    ctx = ShapeContext.from_spec(spec)
    # Explicit head_dim wins over hidden//heads (which would be 128).
    assert ctx.dims[Symbol.HEAD_DIM.value] == 64


# ---------------------------------------------------------------------------
# ModuleDimRegistry._walk_init_body branch handling
# ---------------------------------------------------------------------------


INIT_BRANCH_SRC = """
import torch.nn as nn


class M(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.flag_true:
            self.a = nn.Linear(config.hidden_size, 10)
        if config.flag_false:
            self.b = nn.Linear(config.hidden_size, 20)
        else:
            self.c = nn.Linear(config.hidden_size, 30)
        for _ in range(2):
            self.d = nn.Linear(config.hidden_size, 40)
        try:
            self.e = nn.Linear(config.hidden_size, 50)
        except Exception:
            pass
"""


def test_walk_init_body_evaluates_conditionals_loops_and_try():
    config = {"hidden_size": 8, "flag_true": 1, "flag_false": 0}
    analysis = analyze_source(INIT_BRANCH_SRC, config=config)
    spec = ArchitectureSpec(
        name="m", model_type="t", hidden_size=8, raw_config=config,
        class_registry=analysis.class_registry,
    )
    ctx = ShapeContext.from_spec(spec)
    reg = ModuleDimRegistry.from_registry(
        analysis.class_registry, config=config, context=ctx
    )
    assert reg.linear_by_attr["a"].out_features == 10  # if True branch
    assert reg.linear_by_attr["c"].out_features == 30  # if False -> else branch
    assert "b" not in reg.linear_by_attr
    assert reg.linear_by_attr["d"].out_features == 40  # for body
    assert reg.linear_by_attr["e"].out_features == 50  # try body


INIT_COND_SRC = """
import torch.nn as nn


class N(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        if self.dim:
            self.p = nn.Linear(config.hidden_size, 11)
        if not config.zero_flag:
            self.q = nn.Linear(config.hidden_size, 12)
        if config.maybe is not None:
            self.r = nn.Linear(config.hidden_size, 13)
"""


def test_walk_init_body_self_attr_not_and_isnotnone_conditions():
    config = {"hidden_size": 8, "zero_flag": 0, "maybe": 5}
    analysis = analyze_source(INIT_COND_SRC, config=config)
    spec = ArchitectureSpec(
        name="n", model_type="t", hidden_size=8, raw_config=config,
        class_registry=analysis.class_registry,
    )
    ctx = ShapeContext.from_spec(spec)
    reg = ModuleDimRegistry.from_registry(
        analysis.class_registry, config=config, context=ctx
    )
    assert reg.linear_by_attr["p"].out_features == 11  # self.dim truthy
    assert reg.linear_by_attr["q"].out_features == 12  # not zero_flag
    assert reg.linear_by_attr["r"].out_features == 13  # maybe is not None


def test_eval_config_condition_returns_none_when_unresolvable():
    test = _ast.parse("config.unknown_attr", mode="eval").body
    assert _eval_config_condition(test, config={}, local_vars={}) is None


# ---------------------------------------------------------------------------
# Torch dtype / tensor shape helpers
# ---------------------------------------------------------------------------


def test_torch_dtype_resolves_and_falls_back():
    import torch

    assert _torch_dtype(torch, "float32") is torch.float32
    assert _torch_dtype(torch, "torch.float16") is torch.float16
    assert _torch_dtype(torch, "not_a_dtype") is torch.float32


def test_first_tensor_shape_variants():
    import torch

    assert _first_tensor_shape(torch.zeros(2, 3)) == (2, 3)
    assert _first_tensor_shape([torch.zeros(4, 5), 7]) == (4, 5)
    assert _first_tensor_shape((1, 2, 3)) is None
    assert _first_tensor_shape(None) is None


def test_resolve_op_int():
    assert _resolve_op_int(None, {}) is None
    assert _resolve_op_int("5", {}) == 5
    assert _resolve_op_int(3, {}) == 3
    assert _resolve_op_int("qkv", {"qkv": 42}) == 42
    assert _resolve_op_int("missing", {}) is None


# ---------------------------------------------------------------------------
# Torch op builders (via meta tensors)
# ---------------------------------------------------------------------------


def _meta(shape, dtype="float16"):
    import torch

    return torch.zeros(shape, dtype=torch.float16, device="meta")


def test_torch_op_split_builder():
    import torch

    t = _meta((2, 137, 8))
    out = _torch_op_split(torch, [t], ["split_size: 4", "dim: 2"], {})
    assert tuple(out.shape) == (2, 137, 4)
    # Missing / non-positive size returns None.
    assert _torch_op_split(torch, [t], ["dim: 2"], {}) is None


def test_torch_op_chunk_builder():
    import torch

    t = _meta((2, 137, 8))
    out = _torch_op_chunk(torch, [t], ["chunks: 2", "dim: 2"], {})
    assert tuple(out.shape) == (2, 137, 4)
    assert _torch_op_chunk(torch, [t], ["dim: 2"], {}) is None


def test_torch_op_unbind_builder():
    import torch

    t = _meta((2, 137))
    out = _torch_op_unbind(torch, [t], ["dim: 0"], {})
    assert tuple(out.shape) == (137,)


def test_torch_op_unflatten_builder():
    import torch

    t = _meta((2, 137, 8))
    out = _torch_op_unflatten(torch, [t], ["dim: 2", "sizes: (2, 4)"], {})
    assert tuple(out.shape) == (2, 137, 2, 4)
    # No sizes -> None.
    assert _torch_op_unflatten(torch, [t], ["dim: 2"], {}) is None
    # More than one -1 -> None.
    assert (
        _torch_op_unflatten(torch, [t], ["dim: 2", "sizes: (-1, -1)"], {}) is None
    )


def test_torch_op_shape_end_to_end_split():
    inf = _make_inferencer()
    node = _node(
        "Split",
        details=["split_size: 4", "dim: 2"],
        operation=OperationKind.NN_MODULE,
    )
    inp = TensorSpec(shape=("B", "S", 8), dtype="float16")
    out = inf._torch_op_shape(node, [inp])
    assert out is not None
    assert out.shape == ("B", "S", 4)


def test_torch_op_shape_returns_none_when_dim_unresolved():
    inf = _make_inferencer()
    node = _node("Split", details=["split_size: 4", "dim: -1"])
    inp = TensorSpec(shape=("UNKNOWN", 8), dtype="float16")
    assert inf._torch_op_shape(node, [inp]) is None


def test_torch_op_shape_no_inputs_or_unknown_builder():
    inf = _make_inferencer()
    assert inf._torch_op_shape(_node("Split"), []) is None
    assert (
        inf._torch_op_shape(_node("NotAnOp"), [TensorSpec(("B", "S", 8))]) is None
    )


def test_concrete_dim_and_shape_helpers():
    inf = _make_inferencer(head_dim=64)
    assert inf._concrete_dim("B") == 2
    assert inf._concrete_dim("S") == 137
    assert inf._concrete_dim(9) == 9
    assert inf._concrete_dim("head_dim") == 64
    assert inf._concrete_dim("no_such_dim") is None
    assert inf._concrete_shape(TensorSpec(("B", "S", 64))) == (2, 137, 64)
    assert inf._concrete_shape(TensorSpec(("B", "nope"))) is None


# ---------------------------------------------------------------------------
# _infer_node_output op branches
# ---------------------------------------------------------------------------


def test_meta_shape_takes_priority():
    inf = _make_inferencer()
    inf._meta_shapes["q_proj"] = TensorSpec(("B", "S", 512), "float16")
    node = _node("Linear", operation=OperationKind.NN_MODULE, meta={"attr_name": "q_proj"})
    out = inf._infer_node_output(node, [TensorSpec(("B", "S", 4096))], root=None)
    assert out.shape == ("B", "S", 512)


def test_synthetic_output_passthrough_with_inputs():
    inf = _make_inferencer()
    node = _node("out", meta={"synthetic": "@output"})
    out = inf._infer_node_output(node, [TensorSpec(("B", "S", 4096))], root=None)
    assert out.shape == ("B", "S", 4096)


def test_synthetic_catchall_no_inputs():
    inf = _make_inferencer(hidden_size=128)
    node = _node("hidden", meta={"synthetic": "@hidden_states"})
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == ("B", "S", 128)


def test_chunk_non_integer_split_size_falls_through():
    inf = _make_inferencer()
    inp = TensorSpec(("B", "S", 4096), "float16")
    node = _node("Chunk", details=["split_size: not_a_number", "dim: -1"])
    out = inf._infer_node_output(node, [inp], root=None)
    assert out.shape == ("B", "S", 4096)


def test_concat_no_inputs_and_stack():
    inf = _make_inferencer(hidden_size=256)
    concat_empty = inf._infer_node_output(_node("Concat"), [], root=None)
    assert concat_empty.shape == ("B", "S", 256)

    a = TensorSpec(("B", "S", 8), "float16")
    b = TensorSpec(("B", "S", 8), "float16")
    stacked = inf._infer_node_output(_node("Stack"), [a, b], root=None)
    assert stacked.shape == (2, "B", "S", 8)


def test_matmul_single_and_no_inputs():
    inf = _make_inferencer(hidden_size=64)
    single = inf._infer_node_output(_node("MatMul"), [TensorSpec(("B", "S", 8))], root=None)
    assert single.shape == ("B", "S", 8)
    empty = inf._infer_node_output(_node("MatMul"), [], root=None)
    assert empty.shape == ("B", "S", 64)


def test_einsum_fallbacks():
    inf = _make_inferencer(hidden_size=32)
    # Equation without '->' resolution: fall back to inputs[0].
    node = _node("Einsum", details=["equation: bij"])
    out = inf._infer_node_output(node, [TensorSpec(("B", "S", 8))], root=None)
    assert out.shape == ("B", "S", 8)
    # No inputs -> default hidden.
    empty = inf._infer_node_output(_node("Einsum", details=["equation: ij,jk->ik"]), [], root=None)
    assert empty.shape == ("B", "S", 32)


def test_causal_conv1d_no_inputs():
    inf = _make_inferencer(hidden_size=48)
    out = inf._infer_node_output(_node("Causal Conv1d"), [], root=None)
    assert out.shape == ("B", "S", 48)


def test_gather_no_inputs_uses_default():
    inf = _make_inferencer(hidden_size=16)
    inf.context.dims[Symbol.EXPERTS_PER_TOK.value] = 4
    out = inf._infer_node_output(_node("Gather"), [], root=None)
    # replace_last_dim of default hidden shape with top-k.
    assert out.shape == ("B", "S", 4)


def test_reduction_argmax_with_dim_keeps_shape_int64():
    inf = _make_inferencer()
    inp = TensorSpec(("B", "S", 8), "float16")
    node = _node("argmax", details=["dim: 1"])
    out = inf._infer_node_output(node, [inp], root=None)
    assert out.shape == ("B", "S", 8)
    assert out.dtype == "int64"


def test_pointwise_no_inputs():
    inf = _make_inferencer(hidden_size=64)
    out = inf._infer_node_output(_node("relu"), [], root=None)
    assert out.shape == ("B", "S", 64)


def test_linear_out_features_defaults_to_hidden():
    inf = _make_inferencer(hidden_size=100)
    node = _node("Linear", operation=OperationKind.TORCH_FUNCTIONAL)
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == ("B", "S", 100)


def test_conv_empty_source_shape_returns_channel_only():
    inf = _make_inferencer(hidden_size=768)
    node = _node("Conv1d", operation=OperationKind.NN_MODULE)
    out = inf._infer_node_output(node, [TensorSpec((), "float16")], root=None)
    assert out.shape == (768,)


def test_norm_without_inputs():
    inf = _make_inferencer(hidden_size=128)
    node = _node("RMSNorm", operation=OperationKind.NN_MODULE)
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == ("B", "S", 128)


def test_gpu_kernel_falls_back_to_hidden_when_no_introspection():
    inf = _make_inferencer(hidden_size=64)
    node = _node("flash_attn", operation=OperationKind.GPU_KERNEL, class_name="KernelOp")
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == ("B", "S", 64)


def test_torch_functional_unknown_warns_and_passes_through(caplog):
    inf = _make_inferencer(hidden_size=64)
    node = _node("frobnicate", operation=OperationKind.TORCH_FUNCTIONAL)
    with caplog.at_level(logging.WARNING, logger="TraceLens.ModelUtils.shape_inference"):
        out = inf._infer_node_output(node, [TensorSpec(("B", "S", 8))], root=None)
    assert out.shape == ("B", "S", 8)
    assert "No shape inference rule" in caplog.text


def test_torch_functional_unknown_no_inputs():
    inf = _make_inferencer(hidden_size=32)
    node = _node("frobnicate", operation=OperationKind.TORCH_FUNCTIONAL)
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == ("B", "S", 32)


def test_block_composite_without_inputs():
    inf = _make_inferencer(hidden_size=77)
    node = _node("DecoderLayer", operation=OperationKind.NN_MODULE, kind=NodeKind.BLOCK)
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == ("B", "S", 77)


def test_attention_name_fallback():
    inf = _make_inferencer(hidden_size=99)
    node = _node(
        "MysteryAttention",
        operation=OperationKind.NN_MODULE,
        kind=NodeKind.LEAF,
        meta={"attr_name": "self_attention"},
    )
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == ("B", "S", 99)


def test_unhandled_op_uses_torch_op_builder():
    inf = _make_inferencer()
    node = _node(
        "Unflatten",
        operation=OperationKind.NN_MODULE,
        kind=NodeKind.LEAF,
        details=["dim: 2", "sizes: (3, 4)"],
        meta={"attr_name": "reshaper"},
    )
    out = inf._infer_node_output(node, [TensorSpec(("B", "S", 12))], root=None)
    assert out.shape == ("B", "S", 3, 4)


def test_unknown_op_with_inputs_passes_through(caplog):
    inf = _make_inferencer()
    node = _node(
        "totallyunknown",
        operation=OperationKind.NN_MODULE,
        kind=NodeKind.LEAF,
        meta={"attr_name": "mystery"},
    )
    with caplog.at_level(logging.WARNING, logger="TraceLens.ModelUtils.shape_inference"):
        out = inf._infer_node_output(node, [TensorSpec(("B", "S", 5))], root=None)
    assert out.shape == ("B", "S", 5)


def test_unknown_op_no_inputs_defaults_hidden():
    inf = _make_inferencer(hidden_size=64)
    node = _node(
        "totallyunknown",
        operation=OperationKind.NN_MODULE,
        kind=NodeKind.LEAF,
        meta={"attr_name": "mystery"},
    )
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == ("B", "S", 64)


# ---------------------------------------------------------------------------
# FX / meta lookup fallbacks
# ---------------------------------------------------------------------------


def test_fx_op_shape_without_checkpoint_returns_none():
    inf = _make_inferencer()
    node = _node("View", node_id="blk:0:mod:@op_l10_c5_view:0")
    assert inf._fx_op_shape(node, [TensorSpec(("B", "S", 8))]) is None


def test_fx_op_shape_with_checkpoint_but_no_capture():
    inf = _make_inferencer()
    inf._meta_checkpoint = "/no/such/checkpoint"
    node = _node("View", node_id="blk:0:mod:@op_l10_c5_view:0")
    # Tracing fails -> None, and the empty cache is memoised.
    assert inf._fx_op_shape(node, [TensorSpec(("B", "S", 8))]) is None
    assert inf._op_fx_shapes == {}


def test_fx_op_shape_no_line_key():
    inf = _make_inferencer()
    inf._meta_checkpoint = "/no/such"
    node = _node("View", node_id="plain_id")
    assert inf._fx_op_shape(node, []) is None


def test_lookup_meta_shape_via_id_segment():
    inf = _make_inferencer()
    inf._meta_shapes["mod"] = TensorSpec(("B", "S", 321), "float16")
    node = _node("Linear", node_id="blk:mod:leaf", operation=OperationKind.NN_MODULE)
    node.metadata.pop("class_name", None)
    out = inf._lookup_meta_shape(node)
    assert out is not None and out.shape == ("B", "S", 321)


def test_lookup_meta_shape_empty_returns_none():
    inf = _make_inferencer()
    assert inf._lookup_meta_shape(_node("Linear")) is None


# ---------------------------------------------------------------------------
# Forward introspection
# ---------------------------------------------------------------------------


def test_introspect_forward_shape_no_class_name_returns_none():
    inf = _make_inferencer()
    node = ModelGraphNode(id="n1", kind=NodeKind.LEAF, label="", operation=None, metadata={})
    assert inf._introspect_forward_shape(node, [TensorSpec(("B", "S", 8))], root=None) is None


FUZZY_SRC = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x


class SwiGlu(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        parts = torch.chunk(x, 2, dim=-1)
        return F.silu(parts[0]) * parts[1]
"""


def _fuzzy_inferencer() -> ShapeInferencer:
    analysis = analyze_source(FUZZY_SRC, config={"hidden_size": 8})
    spec = ArchitectureSpec(
        name="f", model_type="t", hidden_size=8, raw_config={"hidden_size": 8},
        class_registry=analysis.class_registry,
    )
    return ShapeInferencer(spec)


def test_fuzzy_attention_lookup_returns_bsh():
    inf = _fuzzy_inferencer()
    node = _node("core_attention", operation=OperationKind.NN_MODULE, class_name="core_attention")
    out = inf._introspect_forward_shape(node, [TensorSpec(("B", "S", 4))], root=None)
    assert out is not None
    assert out.shape == ("B", "S", 8)


def test_fuzzy_non_attention_simulates_forward():
    inf = _fuzzy_inferencer()
    node = _node("swi_glu", operation=OperationKind.NN_MODULE, class_name="swi_glu")
    out = inf._introspect_forward_shape(node, [TensorSpec(("B", "S", 8))], root=None)
    assert out is not None
    # chunk halves the last dim 8 -> 4.
    assert out.shape[-1] == 4


INLINE_SRC = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        def swiglu(x):
            parts = torch.chunk(x, 2, dim=-1)
            return F.silu(parts[0]) * parts[1]

        self.activation_func = swiglu

    def forward(self, hidden_states):
        return self.activation_func(hidden_states)
"""


def test_introspect_inline_function():
    analysis = analyze_source(INLINE_SRC, config={"hidden_size": 8})
    spec = ArchitectureSpec(
        name="i", model_type="t", hidden_size=8, raw_config={"hidden_size": 8},
        class_registry=analysis.class_registry,
    )
    inf = ShapeInferencer(spec)
    out = inf._introspect_inline_function(
        "activation_func", [TensorSpec(("B", "S", 8))], root=None
    )
    assert out is not None
    assert out.shape[-1] == 4


METHOD_SRC = """
import torch
import torch.nn as nn


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden = config.hidden_size

    def get_pooled(self, hidden_states):
        pooled = hidden_states.sum(dim=1)
        return pooled
"""


def test_introspect_method_shape():
    analysis = analyze_source(METHOD_SRC, config={"hidden_size": 8})
    spec = ArchitectureSpec(
        name="p", model_type="t", hidden_size=8, raw_config={"hidden_size": 8},
        class_registry=analysis.class_registry,
    )
    inf = ShapeInferencer(spec)
    out = inf._introspect_method_shape("get_pooled", [TensorSpec(("B", "S", 8))], root=None)
    assert out is not None
    assert out.shape[-1] == 8


def test_simulate_forward_ops_no_operations_returns_none():
    analysis = analyze_source(FUZZY_SRC, config={"hidden_size": 8})
    spec = ArchitectureSpec(
        name="f", model_type="t", hidden_size=8, raw_config={"hidden_size": 8},
        class_registry=analysis.class_registry,
    )
    inf = ShapeInferencer(spec)
    core = analysis.class_registry["CoreAttention"]
    # CoreAttention.forward just returns x -> no shape-bearing operations.
    assert inf._simulate_forward_ops(core, [TensorSpec(("B", "S", 8))], root=None, guard_name="g") is None


# ---------------------------------------------------------------------------
# Spec lookups keyed by (class_name, attr)
# ---------------------------------------------------------------------------


def test_lookup_linear_conv_embedding_by_class_and_attr():
    inf = _make_inferencer()
    inf.module_dims.linear[("C", "proj")] = ModuleLinearSpec(8, 16)
    inf.module_dims.conv[("C", "cv")] = ModuleConvSpec(3, 32)
    inf.module_dims.embedding[("C", "emb")] = ModuleEmbeddingSpec(100, 64)

    lin_node = _node("Linear", class_name="C", meta={"attr_name": "proj"})
    assert inf._lookup_linear_spec(lin_node, root=None).out_features == 16

    cv_node = _node("Conv2d", class_name="C", meta={"attr_name": "cv"})
    assert inf._lookup_conv_spec(cv_node, root=None).out_channels == 32

    emb_node = _node("Embedding", class_name="C", meta={"attr_name": "emb"})
    assert inf._lookup_embedding_spec(emb_node, root=None).embedding_dim == 64


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_resolve_dim_name_suffix_heuristic():
    assert _resolve_dim_name("hc", {"hc_mult": 4}) == 4
    assert _resolve_dim_name("self.hc", {"hc": 7}) == 7
    assert _resolve_dim_name("nope", {}) is None


def test_resolve_view_shape_edge_cases():
    src = TensorSpec(("B", "S", 8), "float16")
    assert _resolve_view_shape("", src, {}) is None
    assert _resolve_view_shape(",", src, {}) is None
    # Starred prefix without an explicit slice keeps the full source shape.
    out = _resolve_view_shape("*x.shape, 4", src, {})
    assert out == ("B", "S", 8, 4)
    # Unresolvable trailing name gives up.
    assert _resolve_view_shape("nope_name", src, {}) is None


def test_looks_valid():
    assert _looks_valid(TensorSpec(()), []) is False
    assert _looks_valid(TensorSpec(("B", "S", 8)), []) is True
    # Input had batch, result lost it -> invalid.
    assert _looks_valid(TensorSpec((8, 16)), [TensorSpec(("B", "S", 8))]) is False
    assert _looks_valid(TensorSpec(("B", "S", 8)), [TensorSpec(("B", "S", 8))]) is True


def test_eval_dim_expr():
    assert _eval_dim_expr("1 +", {}) is None  # SyntaxError
    assert _eval_dim_expr("3 + 4", {}) == 7
    assert _eval_dim_expr("10 - 3", {}) == 7
    assert _eval_dim_expr("12 // 4", {}) == 3
    assert _eval_dim_expr("2 * 5", {}) == 10
    assert _eval_dim_expr("-5", {}) == -5
    assert _eval_dim_expr("hc", {"hc": 4}) == 4
    assert _eval_dim_expr("self.hc", {"hc": 9}) == 9
    assert _eval_dim_expr("missing + 1", {}) is None


def test_parse_split_sizes():
    assert _parse_split_sizes("[hc] * 3", {"hc": 5}) == [5, 5, 5]
    assert _parse_split_sizes("[missing] * 2", {}) is None
    assert _parse_split_sizes("[a, b, a]", {"a": 2, "b": 3}) == [2, 3, 2]
    assert _parse_split_sizes("2048", {}) == [2048]
    assert _parse_split_sizes("a, b", {"a": 1, "b": 2}) == [1, 2]
    assert _parse_split_sizes("[garbage_name]", {}) is None


def test_infer_einsum_shape_failure_paths():
    a = TensorSpec(("B", "S", 8))
    b = TensorSpec((8, 16))
    assert _infer_einsum_shape("bij", [a]) is None  # no '->'
    assert _infer_einsum_shape("ij,jk->ik", [a]) is None  # arity mismatch
    assert _infer_einsum_shape("ij->i", [a]) is None  # sub len != shape rank
    assert _infer_einsum_shape("ij->ik", [TensorSpec((4, 5))]) is None  # k unknown
    assert _infer_einsum_shape("ij,jk->ik", [TensorSpec((4, 5)), TensorSpec((5, 6))]) == (4, 6)


def test_config_dtype():
    assert _config_dtype({"torch_dtype": "torch.bfloat16"}) == "bfloat16"
    assert _config_dtype({}) == "float16"


def test_call_class_name():
    call_attr = _ast.parse("nn.Linear(4, 8)").body[0].value
    assert _call_class_name(call_attr) == "Linear"
    call_name = _ast.parse("Parameter(3)").body[0].value
    assert _call_class_name(call_name) == "Parameter"
    non_call = _ast.parse("(lambda: 1)()").body[0].value
    assert _call_class_name(non_call) is None


def _ctx() -> ShapeContext:
    return ShapeContext.from_spec(ArchitectureSpec(name="t", model_type="t", raw_config={}))


def test_resolve_dim_expr_binops_and_symbolic():
    ctx = _ctx()
    sub = _ast.parse("config.hidden_size - 2", mode="eval").body
    assert _resolve_dim_expr(sub, config={"hidden_size": 8}, local_vars={}, context=ctx) == 6
    unresolved = _ast.parse("unknown_name - 2", mode="eval").body
    assert _resolve_dim_expr(unresolved, config={}, local_vars={}, context=ctx) is None
    # Symbolic: B is a string dim, so the binop renders as algebra.
    sym = _ast.parse("B + 1", mode="eval").body
    out = _resolve_dim_expr(sym, config={}, local_vars={}, context=ctx)
    assert out == "B+1"


def test_symbolic_binop_unsupported_op_returns_none():
    assert _symbolic_binop("B", "S", _ast.Mod()) is None


def test_parse_module_ctor_embedding_keywords():
    call = _ast.parse("nn.Embedding(num_embeddings=100, embedding_dim=8)").body[0].value
    spec = _parse_module_ctor(call, config={}, local_vars={}, context=_ctx())
    assert isinstance(spec, ModuleEmbeddingSpec)
    assert spec.num_embeddings == 100
    assert spec.embedding_dim == 8


def test_parse_module_ctor_parameter():
    call = _ast.parse("nn.Parameter(torch.empty(3, 4))").body[0].value
    spec = _parse_module_ctor(call, config={}, local_vars={}, context=_ctx())
    assert isinstance(spec, ModuleParameterSpec)
    assert spec.shape == (3, 4)


def test_parse_tensor_ctor_shape_unresolved_returns_none():
    call = _ast.parse("torch.empty(unknown_name)").body[0].value
    assert _parse_tensor_ctor_shape(call, config={}, local_vars={}, context=_ctx()) is None


def test_output_tensor_name_synthetic_input():
    node = _node("input_ids", meta={"synthetic": "@input"})
    assert _output_tensor_name(node) == "input_ids"


def test_operator_name_synthetic_and_elementwise():
    inp = _node("input_ids", node_id="@in1", meta={"synthetic": "@input"})
    assert _operator_name(inp) == "input_ids"
    mul = _node("Multiply", node_id="@m1")
    assert _operator_name(mul) == "×"
    add = _node("Add", node_id="@a1")
    assert _operator_name(add) == "+"
    ew = _node("Elementwise ×", node_id="@e1")
    assert _operator_name(ew) == "Elementwise ×"


def test_export_operation_kind_none():
    node = ModelGraphNode(id="n", kind=NodeKind.LEAF, label="x", operation=None, metadata={})
    assert _export_operation_kind(node) == "unknown"


def test_low_level_computation_variants():
    gpu = ModelGraphNode(
        id="n", kind=NodeKind.LEAF, label="flash", operation=OperationKind.GPU_KERNEL, metadata={}
    )
    assert _low_level_computation(gpu) == "flash"
    inp = ModelGraphNode(
        id="n", kind=NodeKind.LEAF, label="x", operation=OperationKind.SYNTHETIC,
        metadata={"synthetic": "@input"},
    )
    assert _low_level_computation(inp) == "input"
    add = ModelGraphNode(id="n", kind=NodeKind.LEAF, label="+", operation=None, metadata={})
    assert _low_level_computation(add) == "elementwise_add"
    mul = ModelGraphNode(id="n", kind=NodeKind.LEAF, label="×", operation=None, metadata={})
    assert _low_level_computation(mul) == "elementwise_mul"


def test_is_embedding_by_role():
    node = ModelGraphNode(
        id="n", kind=NodeKind.LEAF, label="", operation=OperationKind.NN_MODULE,
        metadata={"role": "embedding"},
    )
    assert _is_embedding("", node) is True


def test_heuristic_linear_out_features():
    ctx = _make_inferencer(
        hidden_size=4096, intermediate_size=11008, vocab_size=32000
    ).context
    ctx.dims[Symbol.INTERMEDIATE.value] = 11008
    ctx.dims[Symbol.VOCAB.value] = 32000
    ctx.dims[Symbol.EXPERTS.value] = 64
    assert _heuristic_linear_out_features("lm_head", ctx) == 32000
    assert _heuristic_linear_out_features("gate_proj", ctx) == 11008
    assert _heuristic_linear_out_features("o_proj", ctx) == 4096
    assert _heuristic_linear_out_features("router", ctx) == 64
    assert _heuristic_linear_out_features("expert_proj", ctx) == 11008
    assert _heuristic_linear_out_features("something_proj", ctx) == 4096
    assert _heuristic_linear_out_features(None, ctx) is None
    assert _heuristic_linear_out_features("nomatch", ctx) is None


def test_default_hidden_shape():
    ctx = _make_inferencer(hidden_size=256).context
    assert _default_hidden_shape(ctx) == ("B", "S", 256)


# ---------------------------------------------------------------------------
# Integration tests: full export pipeline (sweeps infer_model_graph,
# export_operators, export_architecture, topological ordering, dedup, etc.)
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_build_operator_export_full_pipeline(tmp_path):
    spec = load_architecture(
        FIXTURES / "custom_model",
        name="Custom MLA MoE",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    payload = build_operator_export(spec)
    assert payload["name"] == "Custom MLA MoE"
    assert payload["sections"]
    all_ops = [op for s in payload["sections"] for op in s["operators"]]
    assert any(op["name"] == "input" and op["operation"] == "input" for op in all_ops)
    assert any(op["name"] == "output" for op in all_ops)
    # JSON serialisable and round-trips through save_operator_export.
    json.dumps(payload)
    out = save_operator_export(payload, tmp_path / "export.json")
    assert out.exists()
    reloaded = json.loads(out.read_text())
    assert reloaded["name"] == "Custom MLA MoE"


def test_build_operator_export_without_model_output():
    spec = load_architecture(
        FIXTURES / "custom_model",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    payload = build_operator_export(spec, include_model_output=False)
    all_ops = [op for s in payload["sections"] for op in s["operators"]]
    assert not any(op["name"] == "output" for op in all_ops)


def test_llama_like_fixture_export():
    spec = load_architecture(
        FIXTURES / "llama_like",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
    )
    payload = build_operator_export(spec)
    assert payload["sections"]
    for section in payload["sections"]:
        for op in section["operators"]:
            assert "shape" in op["output"]


# ---------------------------------------------------------------------------
# HyperConnection special-cased inference (stream slot specs)
# ---------------------------------------------------------------------------

HYPERCONNECTION_SOURCE = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Norm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        return hidden_states * self.weight


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = Norm(config)
        self.fn = nn.Parameter(torch.empty(24, config.hidden_size))

    def forward(self, hidden_streams):
        flat = self.input_norm(hidden_streams)
        mix = F.linear(flat, self.fn)
        pre = mix.sigmoid()
        collapsed = (pre * hidden_streams).sum(dim=2)
        return collapsed
"""


def test_hyperconnection_root_special_casing():
    config = {"hidden_size": 4096, "hc_mult": 4}
    analysis = analyze_source(HYPERCONNECTION_SOURCE, config=config)
    basic = BasicOpFilter.for_detailed()
    block = build_block_node(
        attr_name="hc",
        class_name="HyperConnection",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    spec = ArchitectureSpec(
        name="hc",
        model_type="test",
        hidden_size=4096,
        raw_config=config,
        class_registry=analysis.class_registry,
        basic_ops=basic,
        export_block_trees=[("HyperConnection", block)],
    )
    graph = build_model_graph(block, title="HyperConnection", basic_ops=basic)
    from TraceLens.ModelUtils.shape_inference import ShapeInferencer as _SI

    specs = _SI(spec).infer_model_graph(graph, root=block)
    assert specs  # every node id received a spec


# ---------------------------------------------------------------------------
# Kimi router export (real torch-functional ops + TopK/Gather/Sum shapes)
# ---------------------------------------------------------------------------


def test_kimi_gate_export_integration():
    config = {
        "hidden_size": 7168,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_expert_group": 1,
        "topk_group": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
    }
    analysis = analyze_source((FIXTURES / "kimi_moe_gate.py").read_text(), config=config)
    basic = BasicOpFilter.for_detailed()
    gate = build_block_node(
        attr_name="gate",
        class_name="KimiMoEGate",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    spec = ArchitectureSpec(
        name="Kimi gate",
        model_type="kimi",
        hidden_size=7168,
        num_experts=896,
        num_experts_per_tok=16,
        raw_config=config,
        class_registry=analysis.class_registry,
        basic_ops=basic,
        export_block_trees=[("KimiMoEGate", gate)],
    )
    payload = build_operator_export(spec)
    ops = payload["sections"][0]["operators"]
    by_comp = {op["computation"]: op for op in ops}
    assert by_comp["TopK"]["output"] == {"shape": ["B", "S", 16], "dtype": "int64"}


# ---------------------------------------------------------------------------
# subgraph_boundary_signature edge cases
# ---------------------------------------------------------------------------


def test_subgraph_boundary_signature_variants():
    from TraceLens.ModelUtils.shape_inference import OperatorRecord

    # No compute ops -> None.
    inp = OperatorRecord("input", "input", "input", [], TensorSpec(("B", "S", 8)))
    assert subgraph_boundary_signature([inp]) is None

    # With an input op present.
    comp = OperatorRecord("proj", "Linear", "nn_module", ["input"], TensorSpec(("B", "S", 16)))
    sig = subgraph_boundary_signature([inp, comp], class_name="Block")
    assert sig is not None and sig[0] == "Block"

    # Without an input op present ("no_input" marker).
    sig2 = subgraph_boundary_signature([comp])
    assert sig2 is not None and sig2[-1] == "no_input"


# ---------------------------------------------------------------------------
# model_output_operator when vocab is unknown
# ---------------------------------------------------------------------------


def test_model_output_operator_none_without_vocab():
    spec = ArchitectureSpec(name="t", model_type="t", hidden_size=8, raw_config={"hidden_size": 8})
    inf = ShapeInferencer(spec)
    assert inf.model_output_operator() is None


def test_serialize_dim_public_wrapper():
    assert serialize_dim(128) == 128
    assert serialize_dim("H") == "H"


# ---------------------------------------------------------------------------
# load_meta_shapes: unavailable checkpoint returns False
# ---------------------------------------------------------------------------


def test_load_meta_shapes_returns_false_when_unavailable():
    inf = _make_inferencer()
    assert inf.load_meta_shapes("/definitely/not/a/real/checkpoint") is False


# ---------------------------------------------------------------------------
# Conv constructor parsing (positional + keyword channels)
# ---------------------------------------------------------------------------


def test_parse_module_ctor_conv_keyword_channels():
    call = _ast.parse("nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)").body[0].value
    spec = _parse_module_ctor(call, config={}, local_vars={}, context=_ctx())
    assert isinstance(spec, ModuleConvSpec)
    assert spec.in_channels == 3
    assert spec.out_channels == 64


def test_parse_module_ctor_linear_keyword_features():
    call = _ast.parse("nn.Linear(in_features=8, out_features=16)").body[0].value
    spec = _parse_module_ctor(call, config={}, local_vars={}, context=_ctx())
    assert isinstance(spec, ModuleLinearSpec)
    assert spec.in_features == 8
    assert spec.out_features == 16


def test_parse_module_ctor_non_call_returns_none():
    node = _ast.parse("42", mode="eval").body
    assert _parse_module_ctor(node, config={}, local_vars={}, context=_ctx()) is None


# ---------------------------------------------------------------------------
# Additional per-op branch coverage in _infer_node_output
# ---------------------------------------------------------------------------


def _synth(label, synthetic, *, port_label=None, class_name=None, node_id="n1"):
    meta = {"synthetic": synthetic}
    if port_label is not None:
        meta["port_label"] = port_label
    if class_name is not None:
        meta["class_name"] = class_name
    return ModelGraphNode(
        id=node_id, kind=NodeKind.LEAF, label=label,
        operation=OperationKind.SYNTHETIC, metadata=meta,
    )


def test_tensor_synthetic_weight_bias_scalar_and_parameter():
    inf = _make_inferencer(hidden_size=32)
    inf.context.dims[Symbol.EXPERTS.value] = 8
    w = inf._infer_node_output(_synth("w", "@tensor", port_label="expert_weight"), [], root=None)
    # Unquantized fixture: weight storage dtype resolves to the model dtype.
    assert w.shape == (8, 32) and w.dtype == "float16"
    b = inf._infer_node_output(_synth("b", "@tensor", port_label="gate_bias"), [], root=None)
    assert b.shape == (8,)
    s = inf._infer_node_output(_synth("s", "@tensor", port_label="scalar"), [], root=None)
    assert s.shape == ()
    inf.module_dims.parameter_by_attr["myp"] = ModuleParameterSpec((3, 4))
    p = inf._infer_node_output(_synth("myp", "@tensor", port_label="myp"), [], root=None)
    assert p.shape == (3, 4)


def test_elementwise_and_catchall_synthetic():
    inf = _make_inferencer(hidden_size=64)
    add = inf._infer_node_output(_node("+", operation=OperationKind.SYNTHETIC), [], root=None)
    assert add.shape == ("B", "S", 64)
    # catch-all synthetic with inputs passes through inputs[0].
    node = _synth("x", "@kernel_port_in")
    out = inf._infer_node_output(node, [TensorSpec(("B", "S", 5))], root=None)
    assert out.shape == ("B", "S", 5)


def test_external_spec_weight_and_bias_for_view_without_inputs():
    inf = _make_inferencer(hidden_size=16)
    inf.context.dims[Symbol.EXPERTS.value] = 4
    w = inf._infer_node_output(
        _node("view", external_inputs=["router_weight"]), [], root=None
    )
    # Unquantized fixture: weight storage dtype resolves to the model dtype.
    assert w.shape == (4, 16) and w.dtype == "float16"
    b = inf._infer_node_output(
        _node("view", external_inputs=["gate_bias"]), [], root=None
    )
    assert b.shape == (4,)


def test_view_resolved_flatten_and_fallback():
    inf = _make_inferencer(hidden_size=16)
    src = TensorSpec(("B", "S", 16), "float16")
    # Resolvable structured shape.
    r = inf._infer_node_output(_node("view", details=["shape: 4, 8"]), [src], root=None)
    assert r.shape == (4, 8)
    # -1 marker -> collapse leading dims.
    m = inf._infer_node_output(_node("reshape", details=["shape: -1, unknownname"]), [src], root=None)
    assert m.shape == ("B*S", 16)
    # Unresolvable, no -1 -> pass through source.
    f = inf._infer_node_output(_node("flatten", details=["shape: unknownname"]), [src], root=None)
    assert f.shape == ("B", "S", 16)


def test_unsqueeze_adds_leading_axis():
    inf = _make_inferencer()
    out = inf._infer_node_output(_node("unsqueeze"), [TensorSpec(("B", "S"))], root=None)
    assert out.shape == (1, "B", "S")


def test_split_with_explicit_size_and_external_fallback():
    inf = _make_inferencer(qkv_dim=2730)
    src = TensorSpec(("B", "S", 8192), "float16")
    sized = inf._infer_node_output(
        _node("split", details=["split_size: 2048", "dim: -1"]), [src], root=None
    )
    assert sized.shape == ("B", "S", 2048)
    ext = inf._infer_node_output(
        _node("split", details=["dim: -1"], external_inputs=["qkv_dim"]), [src], root=None
    )
    assert ext.shape == ("B", "S", 2730)


def test_concat_with_inputs_and_symbolic_fallback():
    inf = _make_inferencer()
    a = TensorSpec(("B", "S", 1024), "float16")
    b = TensorSpec(("B", "S", 2048), "float16")
    out = inf._infer_node_output(_node("concat", details=["dim: -1"]), [a, b], root=None)
    assert out.shape == ("B", "S", 3072)
    # Symbolic last dims -> not all int -> return widest base.
    c = TensorSpec(("B", "S", "H"), "float16")
    d = TensorSpec(("B", "S", "H"), "float16")
    out2 = inf._infer_node_output(_node("concat", details=["dim: -1"]), [c, d], root=None)
    assert out2.shape == ("B", "S", "H")


def test_transpose_and_permute():
    inf = _make_inferencer()
    src = TensorSpec(("B", "S", 32, 128), "float16")
    t = inf._infer_node_output(_node("transpose", details=["dim0: 1", "dim1: 2"]), [src], root=None)
    assert t.shape == ("B", 32, "S", 128)
    # permute is a no-op in symbolic inference -> passes source through.
    p = inf._infer_node_output(_node("permute", details=["dims: (0, 2, 1, 3)"]), [src], root=None)
    assert p.shape == ("B", "S", 32, 128)


def test_matmul_two_inputs():
    inf = _make_inferencer()
    a = TensorSpec(("B", "S", 4096), "float16")
    b = TensorSpec((4096, 1024), "float16")
    out = inf._infer_node_output(_node("matmul"), [a, b], root=None)
    assert out.shape == ("B", "S", 1024)


def test_einsum_success_via_node():
    inf = _make_inferencer()
    a = TensorSpec((4, 5), "float16")
    b = TensorSpec((5, 6), "float16")
    out = inf._infer_node_output(_node("einsum", details=["equation: ij,jk->ik"]), [a, b], root=None)
    assert out.shape == (4, 6)


def test_nonzero_and_one_hot():
    inf = _make_inferencer()
    inf.context.dims[Symbol.EXPERTS.value] = 8
    nz = inf._infer_node_output(_node("nonzero"), [TensorSpec(("B", "S", 4))], root=None)
    assert nz.shape == ("nnz", 3) and nz.dtype == "int64"
    nz_empty = inf._infer_node_output(_node("nonzero"), [], root=None)
    assert nz_empty.shape[0] == "nnz"
    oh = inf._infer_node_output(_node("one hot"), [TensorSpec(("B", "S"))], root=None)
    assert oh.shape == ("B", "S", 8) and oh.dtype == "int64"


def test_causal_conv1d_passthrough():
    inf = _make_inferencer()
    out = inf._infer_node_output(_node("causal conv1d"), [TensorSpec(("B", "S", 16))], root=None)
    assert out.shape == ("B", "S", 16)


def test_cast_contiguous_squeeze_expand():
    inf = _make_inferencer()
    src = TensorSpec(("B", "S", 16), "float16")
    casted = inf._infer_node_output(_node("cast", details=["dtype: float32"]), [src], root=None)
    assert casted.dtype == "float32"
    squeezed = inf._infer_node_output(_node("squeeze"), [src], root=None)
    assert squeezed.shape == ("B", "S", 16)
    # expand without inputs -> default hidden.
    expanded = inf._infer_node_output(_node("expand"), [], root=None)
    assert expanded.shape[-1] == 4096


def test_topk_with_empty_source_shape():
    inf = _make_inferencer()
    inf.context.dims[Symbol.EXPERTS_PER_TOK.value] = 4
    out = inf._infer_node_output(_node("topk"), [TensorSpec((), "float16")], root=None)
    assert out.shape == ("B", "S", 4)


def test_linear_out_features_from_input_last_dim():
    inf = _make_inferencer(hidden_size=4096)
    node = _node("Linear", operation=OperationKind.TORCH_FUNCTIONAL)
    out = inf._infer_node_output(node, [TensorSpec(("B", "S", 64))], root=None)
    assert out.shape == ("B", "S", 64)


def test_conv_matches_in_channel_axis():
    inf = _make_inferencer()
    inf.module_dims.conv_by_attr["cv"] = ModuleConvSpec(in_channels=3, out_channels=64)
    node = _node("Conv2d", operation=OperationKind.NN_MODULE, meta={"attr_name": "cv"})
    out = inf._infer_node_output(node, [TensorSpec(("B", 3, 8, 8))], root=None)
    assert out.shape == ("B", 64, 8, 8)


def test_router_module_output():
    inf = _make_inferencer()
    inf.context.dims[Symbol.EXPERTS.value] = 8
    node = _node("TopKRouter", operation=OperationKind.NN_MODULE, class_name="TopKRouter")
    out = inf._infer_node_output(node, [TensorSpec(("B", "S", 16))], root=None)
    assert out.shape == ("B", "S", 8)


def test_block_with_inputs_passes_through():
    inf = _make_inferencer()
    node = _node("DecoderLayer", operation=OperationKind.NN_MODULE, kind=NodeKind.BLOCK)
    out = inf._infer_node_output(node, [TensorSpec(("B", "S", 12))], root=None)
    assert out.shape == ("B", "S", 12)


def test_lookup_meta_shape_no_match_returns_none():
    inf = _make_inferencer()
    inf._meta_shapes["other.module"] = TensorSpec(("B", "S", 8))
    node = _node("Linear", node_id="blk:0:mystery", operation=OperationKind.NN_MODULE)
    node.metadata.pop("class_name", None)
    assert inf._lookup_meta_shape(node) is None


# ---------------------------------------------------------------------------
# infer_block_tree convenience wrapper
# ---------------------------------------------------------------------------


def test_infer_block_tree():
    config = {"hidden_size": 4096, "hc_mult": 4}
    analysis = analyze_source(HYPERCONNECTION_SOURCE, config=config)
    basic = BasicOpFilter.for_detailed()
    block = build_block_node(
        attr_name="hc", class_name="HyperConnection",
        registry=analysis.class_registry, basic_ops=basic,
    )
    spec = ArchitectureSpec(
        name="hc", model_type="test", hidden_size=4096, raw_config=config,
        class_registry=analysis.class_registry, basic_ops=basic,
        export_block_trees=[("HyperConnection", block)],
    )
    specs = ShapeInferencer(spec).infer_block_tree(block, title="HyperConnection")
    assert specs


# ---------------------------------------------------------------------------
# ShapeContext.from_spec: nested aliases + forward-local scalars
# ---------------------------------------------------------------------------


FORWARD_LOCAL_SRC = """
import torch.nn as nn


class Attn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        half = hidden_size // 2
        return x
"""


def test_shape_context_folds_forward_local_scalars_and_nested_aliases():
    config = {
        "hidden_size": 4096,
        "linear_attn_config": {"head_dim": 128, "num_heads": 64},
    }
    analysis = analyze_source(FORWARD_LOCAL_SRC, config=config)
    spec = ArchitectureSpec(
        name="a", model_type="t", hidden_size=4096, raw_config=config,
        class_registry=analysis.class_registry,
    )
    ctx = ShapeContext.from_spec(spec)
    assert ctx.dims["linear_head_dim"] == 128  # nested alias
    assert ctx.dims["half"] == 2048  # forward-local scalar


# ---------------------------------------------------------------------------
# ModuleDimRegistry: conv recording, plain locals, ambiguous parameters
# ---------------------------------------------------------------------------


REGISTRY_SRC = """
import torch
import torch.nn as nn


class A(nn.Module):
    def __init__(self, config):
        super().__init__()
        scale = config.hidden_size
        self.cv = nn.Conv2d(3, 16, 3)
        self.weight = nn.Parameter(torch.empty(config.hidden_size))


class B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
"""


def test_registry_records_conv_locals_and_ambiguous_parameters():
    config = {"hidden_size": 8}
    analysis = analyze_source(REGISTRY_SRC, config=config)
    spec = ArchitectureSpec(
        name="r", model_type="t", hidden_size=8, raw_config=config,
        class_registry=analysis.class_registry,
    )
    ctx = ShapeContext.from_spec(spec)
    reg = ModuleDimRegistry.from_registry(analysis.class_registry, config=config, context=ctx)
    assert reg.conv_by_attr["cv"].out_channels == 16
    # 'weight' declared with differing shapes across classes -> ambiguous.
    assert "weight" in reg.ambiguous_parameters
    # Class-qualified lookup still resolves.
    assert reg.lookup_parameter("weight", "A").shape == (8,)
    # Ambiguous bare lookup returns None.
    assert reg.lookup_parameter("weight", None) is None
    # Missing attr returns None.
    assert reg.lookup_parameter("", "A") is None


# ---------------------------------------------------------------------------
# Extra helper coverage
# ---------------------------------------------------------------------------


def test_torch_op_unflatten_with_unresolved_dim_uses_negative_one():
    import torch

    t = _meta((2, 137, 8))
    out = _torch_op_unflatten(torch, [t], ["dim: 2", "sizes: (unknownname, 4)"], {})
    assert tuple(out.shape) == (2, 137, 2, 4)


def test_resolve_view_shape_starred_with_slice_and_unresolved():
    src = TensorSpec(("B", "S", 8, 16), "float16")
    out = _resolve_view_shape("*x.shape[:-1], 4", src, {})
    assert out == ("B", "S", 8, 4)
    # Starred slice but a trailing name that cannot resolve.
    assert _resolve_view_shape("*x.shape[:-1], nope", src, {}) is None


def test_parse_split_sizes_nested_and_failure():
    assert _parse_split_sizes("[a, (b)]", {"a": 1, "b": 2}) == [1, 2]
    assert _parse_split_sizes("a, unknown", {"a": 1}) is None


def test_resolve_dim_expr_subscript_int_and_symbolic_paren():
    ctx = _ctx()
    # config.sub["head_dim"] nested subscript.
    sub = _ast.parse('config.linear_attn_config["head_dim"]', mode="eval").body
    val = _resolve_dim_expr(
        sub, config={"linear_attn_config": {"head_dim": 128}}, local_vars={}, context=ctx
    )
    assert val == 128
    # int(config.hidden_size) call wrapper.
    intcall = _ast.parse("int(config.hidden_size)", mode="eval").body
    assert _resolve_dim_expr(intcall, config={"hidden_size": 8}, local_vars={}, context=ctx) == 8
    # Symbolic algebra requiring parenthesisation.
    paren = _ast.parse("(B + 1) * 2", mode="eval").body
    assert _resolve_dim_expr(paren, config={}, local_vars={}, context=ctx) == "(B+1)*2"


def test_parse_module_ctor_tensor_factory_and_full():
    zeros = _ast.parse("torch.zeros((3, 4))").body[0].value
    spec = _parse_module_ctor(zeros, config={}, local_vars={}, context=_ctx())
    assert isinstance(spec, ModuleParameterSpec) and spec.shape == (3, 4)
    full = _ast.parse("torch.full((3, 4), 0.0)").body[0].value
    spec2 = _parse_module_ctor(full, config={}, local_vars={}, context=_ctx())
    assert isinstance(spec2, ModuleParameterSpec) and spec2.shape == (3, 4)


# ---------------------------------------------------------------------------
# Remaining targeted coverage
# ---------------------------------------------------------------------------


def test_unflatten_skips_empty_size_entries():
    import torch

    t = _meta((2, 137, 8))
    out = _torch_op_unflatten(torch, [t], ["dim: 2", "sizes: (2, 4,)"], {})
    assert tuple(out.shape) == (2, 137, 2, 4)


def test_external_spec_uses_registered_parameter():
    inf = _make_inferencer()
    inf.module_dims.parameter_by_attr["myp"] = ModuleParameterSpec((5, 6))
    node = _node("view", external_inputs=["myp"])
    out = inf._infer_node_output(node, [], root=None)
    assert out.shape == (5, 6)


def test_ensure_op_fx_shapes_without_checkpoint_and_memoisation():
    inf = _make_inferencer()
    # No checkpoint -> builds an empty cache.
    inf._ensure_op_fx_shapes()
    assert inf._op_fx_shapes == {}
    # Second call short-circuits on the already-built cache.
    inf._ensure_op_fx_shapes()
    assert inf._op_fx_shapes == {}


def test_fx_op_shape_returns_prebuilt_shape():
    inf = _make_inferencer()
    inf._meta_checkpoint = "cached"
    inf._op_fx_shapes = {(10, "view", 0): (2, 137, 3)}
    node = _node("View", node_id="blk:0:m:@op_l10_c5_view:0")
    out = inf._fx_op_shape(node, [TensorSpec(("B", "S", 8))])
    assert out is not None
    assert out.shape == (2, 137, 3)


def test_introspect_direct_registry_class_and_id_parts():
    inf = _fuzzy_inferencer()
    # Exact class name in registry -> simulate its forward.
    node = _node("SwiGlu", operation=OperationKind.NN_MODULE, class_name="SwiGlu",
                 node_id="a:b:SwiGlu")
    out = inf._introspect_forward_shape(node, [TensorSpec(("B", "S", 8))], root=None)
    assert out is not None
    assert out.shape[-1] == 4


def test_fuzzy_suffix_match_lookup():
    inf = _fuzzy_inferencer()
    # 'swiglu' matches 'SwiGlu' via the case/underscore-insensitive suffix rule.
    assert inf._fuzzy_registry_lookup("swiglu") is not None


METHOD_NOOP_SRC = """
import torch.nn as nn


class Passthrough(nn.Module):
    def get_x(self, hidden_states):
        return hidden_states
"""


def test_introspect_method_without_operations_returns_none():
    analysis = analyze_source(METHOD_NOOP_SRC, config={"hidden_size": 8})
    spec = ArchitectureSpec(
        name="p", model_type="t", hidden_size=8, raw_config={"hidden_size": 8},
        class_registry=analysis.class_registry,
    )
    inf = ShapeInferencer(spec)
    assert inf._introspect_method_shape("get_x", [TensorSpec(("B", "S", 8))], root=None) is None


def test_owner_class_name_from_node_id_path():
    child = BlockNode(attr_name="gate", class_name="Router", role="other", label="Router")
    root = BlockNode(attr_name="blk", class_name="Block", role="other", label="Block", children=[child])
    inf = _make_inferencer()
    node = _node("linear", node_id="blk:0:gate:@op_l1_c1_linear:0")
    assert inf._owner_class_name(node, root) == "Router"
    assert inf._owner_class_name(node, None) is None


def test_resolve_view_shape_starred_slice_with_symbolic_name():
    src = TensorSpec(("B", "S", 8, 16), "float16")
    out = _resolve_view_shape("*x.shape[:-1], myname", src, {"myname": 7})
    assert out == ("B", "S", 8, 7)


def test_eval_config_condition_self_attr_absent():
    test = _ast.parse("self.missing", mode="eval").body
    assert _eval_config_condition(test, config={}, local_vars={}) is None


def test_resolve_dim_expr_name_in_locals_and_add():
    ctx = _ctx()
    name = _ast.parse("local_dim", mode="eval").body
    assert _resolve_dim_expr(name, config={}, local_vars={"local_dim": 12}, context=ctx) == 12
    add = _ast.parse("config.hidden_size + 2", mode="eval").body
    assert _resolve_dim_expr(add, config={"hidden_size": 8}, local_vars={}, context=ctx) == 10


def test_operator_name_forward_op_display_label():
    node = ModelGraphNode(
        id="n", kind=NodeKind.LEAF, label="View",
        operation=OperationKind.TORCH_FUNCTIONAL,
        metadata={"attr_name": "@op_l1_c1_view", "class_name": "View"},
    )
    # is_forward_operation attr -> routed through operation_display_label.
    assert isinstance(_operator_name(node), str)


def test_low_level_computation_fallback_label():
    node = ModelGraphNode(id="n", kind=NodeKind.LEAF, label="Custom", operation=None, metadata={})
    assert _low_level_computation(node) == "Custom"


def test_is_embedding_by_name_and_label():
    node = ModelGraphNode(id="n", kind=NodeKind.LEAF, label="Embedding", operation=OperationKind.NN_MODULE, metadata={})
    assert _is_embedding("Embedding", node) is True


def test_torch_op_shape_builder_returns_none_when_size_missing():
    inf = _make_inferencer()
    node = _node("Split", details=["dim: 2"], operation=OperationKind.NN_MODULE)
    inp = TensorSpec(("B", "S", 8), "float16")
    assert inf._torch_op_shape(node, [inp]) is None


def test_torch_op_shape_returns_none_on_builder_exception():
    inf = _make_inferencer()
    # sizes (3, 3) = 9 don't divide the size-8 axis -> torch.unflatten raises.
    node = _node("Unflatten", details=["dim: 2", "sizes: (3, 3)"], operation=OperationKind.NN_MODULE)
    inp = TensorSpec(("B", "S", 8), "float16")
    assert inf._torch_op_shape(node, [inp]) is None


def test_topological_order_handles_cycles():
    from TraceLens.ModelUtils.model_graph import GraphEdge, ModelGraph

    a = _synth("a", "@input", node_id="a")
    b = _node("relu", node_id="b")
    c = _node("relu", node_id="c")
    # b <-> c form a cycle that has no zero-indegree entry.
    graph = ModelGraph(
        title="cyclic",
        nodes=[a, b, c],
        edges=[
            GraphEdge(source="a", target="b"),
            GraphEdge(source="b", target="c"),
            GraphEdge(source="c", target="b"),
        ],
    )
    inf = _make_inferencer(hidden_size=32)
    specs = inf.infer_model_graph(graph)
    # Every node still receives a spec even though b/c cycle.
    assert set(specs) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Real compute dtype + per-module quantization dtype (Phase 3)
# ---------------------------------------------------------------------------


def test_config_dtype_reads_nested_and_top_level_and_quant_compute():
    # Newer HF configs put the real dtype under text_config as ``dtype``.
    assert _config_dtype({"text_config": {"dtype": "bfloat16"}}) == "bfloat16"
    # Legacy top-level ``torch_dtype`` with a ``torch.`` prefix is stripped.
    assert _config_dtype({"torch_dtype": "torch.float16"}) == "float16"
    # A quant config's explicit compute dtype wins.
    assert (
        _config_dtype({"quantization_config": {"bnb_4bit_compute_dtype": "bfloat16"}})
        == "bfloat16"
    )
    # Nothing found -> conservative default.
    assert _config_dtype({}) == "float16"


def test_quant_storage_dtype_across_methods():
    assert _quant_storage_dtype({"quant_method": "fp8", "fmt": "e4m3"}) == "fp8_e4m3"
    assert _quant_storage_dtype({"quant_method": "fp8", "fmt": "e5m2"}) == "fp8_e5m2"
    assert _quant_storage_dtype({"fmt": "e4m3"}) == "fp8_e4m3"
    assert (
        _quant_storage_dtype(
            {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            }
        )
        == "nf4"
    )
    assert (
        _quant_storage_dtype({"quant_method": "bitsandbytes", "load_in_8bit": True})
        == "int8"
    )
    assert _quant_storage_dtype({"quant_method": "gptq", "bits": 4}) == "int4"
    assert _quant_storage_dtype({"quant_method": "awq", "bits": 8}) == "int8"
    assert _quant_storage_dtype({"quant_method": "unknown"}) is None


def test_normalize_module_patterns_strips_prefix_and_indices():
    patterns = _normalize_module_patterns(
        [
            "model.layers.0.self_attn.q_proj",
            "model.layers.12.mlp.gate",
            "lm_head",
            "visual",
        ]
    )
    assert ("self_attn", "q_proj") in patterns
    assert ("mlp", "gate") in patterns
    assert ("lm_head",) in patterns
    assert ("visual",) in patterns
    # Non-list input is ignored.
    assert _normalize_module_patterns(None) == ()


def test_module_path_segments_and_matches():
    segs = _module_path_segments(
        "decoder/45x_Layer/self_attn/seq:0:q_proj:q_proj:0"
    )
    assert segs == ["decoder", "self_attn", "q_proj"]
    assert _module_path_matches(segs, ("self_attn", "q_proj"))
    assert not _module_path_matches(segs, ("mlp", "gate"))
    # Single-segment pattern matches anywhere.
    assert _module_path_matches(["encoder", "visual", "blocks"], ("visual",))
    # A multi-segment pattern must be contiguous, so shared_experts.down_proj
    # never collides with visual.merger.down_proj.
    shared = ["decoder", "mlp", "shared_experts", "down_proj"]
    assert _module_path_matches(shared, ("shared_experts", "down_proj"))
    assert not _module_path_matches(shared, ("merger", "down_proj"))
    assert not _module_path_matches(segs, ())


def test_weight_dtype_fp8_experts_vs_bf16_not_convert():
    not_convert = _normalize_module_patterns(
        [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.gate",
            "lm_head",
            "visual",
        ]
    )
    ctx = ShapeContext(
        dtype="bfloat16", quant_dtype="fp8_e4m3", not_convert=not_convert
    )
    # MoE routed + shared experts stay quantized.
    assert ctx.weight_dtype("d/L/mlp/sidefeed:1:experts:@op_bmm:2") == "fp8_e4m3"
    assert (
        ctx.weight_dtype("d/L/mlp/sidefeed:3:shared_experts:down_proj:6")
        == "fp8_e4m3"
    )
    # Everything in modules_to_not_convert keeps the compute dtype.
    assert ctx.weight_dtype("d/L/self_attn/seq:0:q_proj:q_proj:0") == "bfloat16"
    assert ctx.weight_dtype("d/L/mlp/seq:0:gate:0") == "bfloat16"
    assert ctx.weight_dtype("e/visual/blocks/attn/seq:0:qkv:0") == "bfloat16"
    assert ctx.weight_dtype("d/lm_head:0") == "bfloat16"
    # A non-quantized context returns the compute dtype for every weight.
    plain = ShapeContext(dtype="bfloat16")
    assert plain.weight_dtype("d/L/mlp/sidefeed:1:experts:@op_bmm:2") == "bfloat16"


def test_resolve_cast_dtype_low_precision_tokens():
    assert _resolve_cast_dtype("torch.float8_e4m3fn", "float32", "bf16") == "fp8_e4m3"
    assert _resolve_cast_dtype("float8_e5m2", "float32", "bf16") == "fp8_e5m2"
    assert _resolve_cast_dtype("uint4", "float32", "bf16") == "uint4"
    assert _resolve_cast_dtype("int4", "float32", "bf16") == "int4"
    # Generic float still resolves to float32, not fp8.
    assert _resolve_cast_dtype("torch.float", "float16", "bf16") == "float32"


def test_merge_flatten_dim_conservation_and_fallback():
    assert _merge_flatten_dim(("B", "S", 4096), ["4096"]) == "B*S"
    assert _merge_flatten_dim(("B", "S", 8, 288), ["8", "288"]) == "B*S"
    assert _merge_flatten_dim(("B", "S", "H"), ["B", "S"]) == "H"
    # Placeholder ``-1`` in the target is ignored, not treated as a factor.
    assert _merge_flatten_dim(("B", "S", 4096), ["-1", "4096"]) == "B*S"
    # Non-divisible numerics -> None (caller falls back).
    assert _merge_flatten_dim(("B", 7), ["3"]) is None
    # A target symbol with no source match -> None.
    assert _merge_flatten_dim(("B", "S"), ["Z"]) is None
