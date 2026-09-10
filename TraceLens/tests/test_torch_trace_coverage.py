###############################################################################
# Coverage tests for TraceLens/ModelUtils/torch_trace.py
#
# These exercise the config-patching, Auto-class resolution, meta-device
# instantiation, rotary-embedding patching, per-module FX tracing (including
# the two source-rewriting fallbacks) and FX ShapeProp propagation helpers
# against small, real torch nn.Modules and tiny HF checkpoints instantiated
# on the meta device.
###############################################################################

from __future__ import annotations

import os


import sys
import types

import pytest
import torch
import torch.nn as nn
import transformers
from transformers.activations import ACT2FN

from TraceLens.ModelUtils import torch_trace as tt


# ---------------------------------------------------------------------------
# Module-level test modules.  They MUST live at module scope in a real .py
# file so that ``inspect.getsource`` (used by the FX fallbacks) can read them.
# ---------------------------------------------------------------------------


class ViewUnpackMod(nn.Module):
    """Uses the ``x.view(*x.shape[:-1], a, b)`` idiom that breaks plain FX."""

    def forward(self, x):
        return x.view(*x.shape[:-1], 2, 2)


class ActFromDictMod(nn.Module):
    """RMSNorm-like module that looks up an unregistered activation module."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self.variance_epsilon = 1e-6
        self.activation = "gelu"

    def forward(self, x):
        act = ACT2FN[self.activation]
        return act(x * self.weight)


class ActNoStrMod(nn.Module):
    """Has ACT2FN in source but ``activation`` is not a string -> None."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self.activation = 123  # not a str

    def forward(self, x):
        act = ACT2FN[self.activation]
        return act(x) + x.view(*x.shape[:-1], 2, 4)  # also breaks plain FX


class ActUnknownMod(nn.Module):
    """ACT2FN lookup with an unknown key -> plain trace fails and the
    mirror fallback finds no activation module (``ACT2FN.get`` -> None)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self.activation = "no_such_activation_xyz"

    def forward(self, x):
        act = ACT2FN[self.activation]
        return act(x * self.weight)


# ---------------------------------------------------------------------------
# Tiny checkpoint fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llama_ckpt(tmp_path_factory):
    from transformers import LlamaConfig

    d = tmp_path_factory.mktemp("llama_ckpt")
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
        max_position_embeddings=128,
    )
    cfg.architectures = ["LlamaForCausalLM"]
    cfg.save_pretrained(str(d))
    return str(d)


@pytest.fixture(scope="module")
def failing_ckpt(tmp_path_factory):
    """A ViT config with a bogus ``ForCausalLM`` architecture suffix so the
    resolved Auto class (AutoModelForCausalLM) rejects it."""
    from transformers import ViTConfig

    d = tmp_path_factory.mktemp("vit_ckpt")
    cfg = ViTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=16,
        patch_size=8,
    )
    cfg.architectures = ["ViTForCausalLM"]
    cfg.save_pretrained(str(d))
    return str(d)


# ---------------------------------------------------------------------------
# _patch_config
# ---------------------------------------------------------------------------


def test_patch_config_adds_missing_and_defaults():
    class Cfg:
        def __init__(self):
            self.existing = 1  # already present -> hasattr True branch

        def to_dict(self):
            return {"existing": 1, "newattr": 42}

    c = Cfg()
    tt._patch_config(c)
    assert c.existing == 1
    assert c.newattr == 42  # line 44 setattr
    assert c.max_length == 131072
    assert c.use_cache is True


def test_patch_config_keeps_existing_defaults():
    class Cfg:
        def __init__(self):
            self.max_length = 7
            self.use_cache = False

        def to_dict(self):
            return {}

    c = Cfg()
    tt._patch_config(c)
    assert c.max_length == 7
    assert c.use_cache is False


# ---------------------------------------------------------------------------
# _resolve_auto_classes
# ---------------------------------------------------------------------------


def test_resolve_auto_map_preferred():
    c = types.SimpleNamespace(auto_map={"AutoModelForCausalLM": "x.Foo"})
    assert tt._resolve_auto_classes(c) == [transformers.AutoModelForCausalLM]


def test_resolve_arch_causal_suffix():
    c = types.SimpleNamespace(auto_map=None, architectures=["FooForCausalLM"])
    assert tt._resolve_auto_classes(c) == [transformers.AutoModelForCausalLM]


def test_resolve_arch_conditional_suffix():
    c = types.SimpleNamespace(
        auto_map={}, architectures=["FooForConditionalGeneration"]
    )
    res = tt._resolve_auto_classes(c)
    assert transformers.AutoModelForCausalLM in res
    assert transformers.AutoModel in res
    assert len(res) == 3


def test_resolve_arch_seq2seq_suffix():
    c = types.SimpleNamespace(auto_map={}, architectures=["FooForSeq2SeqLM"])
    assert tt._resolve_auto_classes(c) == [transformers.AutoModelForSeq2SeqLM]


def test_resolve_arch_unknown_suffix_falls_through():
    c = types.SimpleNamespace(auto_map={}, architectures=["FooBar"])
    assert tt._resolve_auto_classes(c) == [transformers.AutoModel]


def test_resolve_last_resort_no_metadata():
    c = types.SimpleNamespace(auto_map={}, architectures=[])
    assert tt._resolve_auto_classes(c) == [transformers.AutoModel]


# ---------------------------------------------------------------------------
# _instantiate_meta
# ---------------------------------------------------------------------------


def test_instantiate_meta_success(llama_ckpt):
    model, config = tt._instantiate_meta(llama_ckpt)
    assert type(model).__name__ == "LlamaForCausalLM"
    # instantiated on meta device -> params carry no real storage
    assert all(p.device.type == "meta" for p in model.parameters())
    assert config is not None


def test_instantiate_meta_all_classes_fail(failing_ckpt):
    with pytest.raises(ValueError, match="Could not instantiate"):
        tt._instantiate_meta(failing_ckpt)


# ---------------------------------------------------------------------------
# _patch_rotary_embeddings
# ---------------------------------------------------------------------------


def test_patch_rotary_embeddings_children_fallback():
    """Force ``sys.modules.get(type(model).__module__)`` to be None so the
    children-scan fallback runs and finds ``apply_rotary_pos_emb``."""
    fake_mod = types.ModuleType("fake_rotary_child_mod")
    fake_mod.apply_rotary_pos_emb = lambda x, *a, **k: x
    sys.modules["fake_rotary_child_mod"] = fake_mod

    class RotaryEmbeddingChild(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = 8

        def forward(self, x):
            return x

    RotaryEmbeddingChild.__module__ = "fake_rotary_child_mod"

    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = RotaryEmbeddingChild()

        def forward(self, x):
            return self.child(x)

    Parent.__module__ = "totally_missing_module_zzz"
    sys.modules.pop("totally_missing_module_zzz", None)

    try:
        parent = Parent()
        tt._patch_rotary_embeddings(parent)
        # apply_rotary_pos_emb replaced with the shape-preserving lambda
        t = torch.zeros(2, 3)
        assert fake_mod.apply_rotary_pos_emb(t) is t
        # the rotary submodule's forward was replaced with a meta producer
        out = parent.child.forward(4096)
        assert out.device.type == "meta"
        assert out.shape == (4096, 4, 2)
        # non-int arg -> default max_seq path
        out2 = parent.child.forward(torch.zeros(1))
        assert out2.shape == (4096, 4, 2)
    finally:
        sys.modules.pop("fake_rotary_child_mod", None)


def test_patch_rotary_embeddings_module_found(llama_ckpt):
    model, _ = tt._instantiate_meta(llama_ckpt)
    # module is resolvable via sys.modules -> the non-fallback branch
    tt._patch_rotary_embeddings(model)


# ---------------------------------------------------------------------------
# _fx_trace_module
# ---------------------------------------------------------------------------


def test_fx_trace_plain_module():
    graph = tt._fx_trace_module(nn.Linear(4, 4))
    assert graph is not None
    ops = {n.op for n in graph.nodes}
    assert "call_module" in ops or "call_function" in ops


def test_fx_trace_view_unpack_fallback():
    graph = tt._fx_trace_module(ViewUnpackMod())
    assert graph is not None
    # the rewritten forward uses unflatten
    targets = {str(n.target) for n in graph.nodes}
    assert any("unflatten" in t for t in targets)


def test_fx_trace_act2fn_fallback():
    graph = tt._fx_trace_module(ActFromDictMod())
    assert graph is not None
    # the mirror's forward (with self.act_fn substituted) traced to real ops
    assert any(
        n.op in ("call_module", "call_function", "call_method")
        for n in graph.nodes
    )


def test_fx_trace_act2fn_non_string_returns_none():
    # ACT2FN present but activation not a str -> ACT2FN branch returns None
    assert tt._fx_trace_module(ActNoStrMod()) is None


def test_fx_trace_act2fn_unknown_key_returns_none():
    # ACT2FN.get(unknown) -> None -> ACT2FN branch returns None
    assert tt._fx_trace_module(ActUnknownMod()) is None


def test_fx_trace_untraceable_no_fallback_returns_none():
    class Bad(nn.Module):
        def forward(self, x):
            if x.sum() > 0:  # data-dependent control flow, no ACT2FN/view
                return x
            return -x

    assert tt._fx_trace_module(Bad()) is None


# ---------------------------------------------------------------------------
# _propagate_fx_node_shapes
# ---------------------------------------------------------------------------


def test_propagate_none_input_returns_empty():
    lin = nn.Linear(8, 8).to("meta")
    graph = tt._fx_trace_module(lin)
    assert tt._propagate_fx_node_shapes(lin, graph, None) == {}


def test_propagate_success_with_model_dtype():
    lin = nn.Linear(8, 8).to("meta")
    graph = tt._fx_trace_module(lin)
    res = tt._propagate_fx_node_shapes(
        lin, graph, (2, 8), model_dtype=torch.float32
    )
    assert res  # non-empty
    # the linear output node keeps the (2, 8) shape
    shapes = {shape for shape, _dtype in res.values()}
    assert (2, 8) in shapes
    assert any(dt == torch.float32 for _s, dt in res.values())


def test_propagate_success_dtype_from_params():
    lin = nn.Linear(8, 8).to("meta")
    graph = tt._fx_trace_module(lin)
    res = tt._propagate_fx_node_shapes(lin, graph, (2, 8), model_dtype=None)
    assert res
    shapes = {shape for shape, _dtype in res.values()}
    assert (2, 8) in shapes


def test_propagate_failure_returns_empty():
    lin = nn.Linear(8, 8).to("meta")
    graph = tt._fx_trace_module(lin)
    # wrong reduction dim -> ShapeProp raises -> {}
    assert tt._propagate_fx_node_shapes(lin, graph, (2, 3)) == {}


def test_propagate_no_float_params_uses_float32_default():
    # A parameter-less module: dtype falls back to float32.
    class Id(nn.Module):
        def forward(self, x):
            return x + x

    m = Id().to("meta")
    graph = tt._fx_trace_module(m)
    res = tt._propagate_fx_node_shapes(m, graph, (2, 8), model_dtype=None)
    assert res
    shapes = {shape for shape, _dtype in res.values()}
    assert (2, 8) in shapes
