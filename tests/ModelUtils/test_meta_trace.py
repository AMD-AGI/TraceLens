###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

###############################################################################
# Coverage tests for TraceLens/ModelUtils/meta_trace.py
#
# Exercises the meta-device shape tracing entrypoints (``trace_meta_shapes``
# and ``trace_meta_op_shapes``) end-to-end against a tiny real Llama checkpoint
# instantiated on the meta device, plus the pure helper functions and the
# torch-unavailable / instantiation-failure / no-shapes fallback branches.
###############################################################################

from __future__ import annotations

import types
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from TraceLens.ModelUtils import meta_trace as mt
from TraceLens.ModelUtils import torch_trace as tt

# ---------------------------------------------------------------------------
# Tiny Llama checkpoint fixture
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Synthetic model exercising trace_meta_op_shapes control-flow branches.
# ``Block`` lives at module scope so inspect.getsourcefile resolves it; the
# dynamically-created ``DynMod`` has ``__module__ = "builtins"`` so
# getsourcefile raises and the "no source file" branch runs.
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(x) + x


_DynMod = type("DynMod", (nn.Module,), {"forward": lambda self, x: x})
_DynMod.__module__ = "builtins"  # getsourcefile -> TypeError


class SyntheticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 8)
        self.a = Block()
        self.dyn = _DynMod()
        self.b = Block()  # same class as ``a`` -> "seen class" skip

    def forward(self, ids):
        h = self.emb(ids)
        h = self.a(h)
        h = self.dyn(h)
        h = self.b(h)
        return h


@pytest.fixture(scope="module")
def llama_ckpt(tmp_path_factory):
    from transformers import LlamaConfig

    d = tmp_path_factory.mktemp("llama_ckpt_meta")
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


# ---------------------------------------------------------------------------
# trace_meta_shapes
# ---------------------------------------------------------------------------


def test_trace_meta_shapes_success(llama_ckpt):
    shapes = mt.trace_meta_shapes(llama_ckpt, seq_len=16, batch_size=1)
    assert shapes is not None
    assert len(shapes) > 0
    # every recorded value is a shape tuple
    assert all(isinstance(v, tuple) for v in shapes.values())
    # some layer submodule captured a 3-D (B, S, H) activation shape
    assert any(len(v) >= 2 for v in shapes.values())


def test_trace_meta_shapes_torch_unavailable(monkeypatch):
    # Make ``import torch`` fail inside the function.
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert mt.trace_meta_shapes("whatever") is None


def test_trace_meta_shapes_instantiation_failure(monkeypatch):
    def _boom(_ckpt):
        raise RuntimeError("cannot load")

    monkeypatch.setattr(tt, "_instantiate_meta", _boom)
    assert mt.trace_meta_shapes("bad/checkpoint") is None


def test_trace_meta_shapes_no_shapes_captured(monkeypatch):
    """Model whose forward raises before any post-hook fires -> no shapes."""

    class NoShapeModel(nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("forward exploded immediately")

    def _fake_instantiate(_ckpt):
        return NoShapeModel(), None

    monkeypatch.setattr(tt, "_instantiate_meta", _fake_instantiate)
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda m: None)
    assert mt.trace_meta_shapes("bad/checkpoint") is None


def test_trace_meta_shapes_tuple_output(monkeypatch):
    """Cover the tuple/list output branch of the forward hook."""

    class TupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            # x arrives as long tokens; ignore and emit a tuple of tensors
            t = torch.zeros(1, 4, device="meta")
            return ("not a tensor", t)

    def _fake_instantiate(_ckpt):
        return TupleModel(), None

    monkeypatch.setattr(tt, "_instantiate_meta", _fake_instantiate)
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda m: None)
    shapes = mt.trace_meta_shapes("x", seq_len=4)
    assert shapes is not None
    # the outer module's tuple output was recorded
    assert shapes.get("") == (1, 4)


# ---------------------------------------------------------------------------
# _norm_op / _fx_op_base / _fx_source_line
# ---------------------------------------------------------------------------


def test_norm_op():
    assert mt._norm_op("Add_1") == "add1"
    assert mt._norm_op("torch.nn.functional.relu") == "torchnnfunctionalrelu"


def test_fx_op_base_call_module_and_function():
    call_module = types.SimpleNamespace(op="call_module", target="a.b.q_proj")
    assert mt._fx_op_base(call_module) == "qproj"

    call_fn = types.SimpleNamespace(op="call_function", name="add_12")
    assert mt._fx_op_base(call_fn) == "add"


def test_fx_source_line_variants():
    make = lambda meta: types.SimpleNamespace(meta=meta)
    # no source file
    assert mt._fx_source_line(make({}), None) is None
    # no stack trace
    assert mt._fx_source_line(make({}), "/mod.py") is None
    # stack trace without a File frame
    assert mt._fx_source_line(make({"stack_trace": "nope"}), "/mod.py") is None
    # frame in a different file
    assert (
        mt._fx_source_line(make({"stack_trace": 'File "/other.py", line 5'}), "/mod.py")
        is None
    )
    # matching frame -> returns the line number
    assert (
        mt._fx_source_line(
            make({"stack_trace": 'File "/mod.py", line 42, in forward'}),
            "/mod.py",
        )
        == 42
    )


# ---------------------------------------------------------------------------
# trace_meta_op_shapes
# ---------------------------------------------------------------------------


def test_trace_meta_op_shapes_success(llama_ckpt):
    result = mt.trace_meta_op_shapes(llama_ckpt)
    assert result is not None
    assert len(result) > 0
    # keys are (line, op_name, occurrence)
    for key, val in result.items():
        assert isinstance(key, tuple) and len(key) == 3
        line, op, idx = key
        assert isinstance(line, int)
        assert isinstance(op, str)
        assert isinstance(idx, int)
        assert isinstance(val, tuple)
    # RMSNorm's mean(-1, keepdim=True) reduces the last dim to 1 somewhere
    assert any(v and v[-1] == 1 for v in result.values())


def test_trace_meta_op_shapes_synthetic_branches(monkeypatch):
    """Drive the per-class loop: same-class skip, missing-source-file skip,
    and successful trace/propagate of a custom module."""

    def _fake_instantiate(_ckpt):
        return SyntheticNet().to("meta"), None

    monkeypatch.setattr(tt, "_instantiate_meta", _fake_instantiate)
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda m: None)
    result = mt.trace_meta_op_shapes("x", seq_len=137, batch_size=2)
    assert result is not None
    # Block.forward's ops (lin + add) were captured with symbolic B/S dims.
    assert any(v == ("B", "S", 8) for v in result.values())


def test_trace_meta_op_shapes_fx_trace_raises(monkeypatch):
    """When _fx_trace_module raises for every module, nothing is captured."""

    def _fake_instantiate(_ckpt):
        return SyntheticNet().to("meta"), None

    def _boom(_mod):
        raise RuntimeError("trace exploded")

    monkeypatch.setattr(tt, "_instantiate_meta", _fake_instantiate)
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda m: None)
    monkeypatch.setattr(tt, "_fx_trace_module", _boom)
    assert mt.trace_meta_op_shapes("x", seq_len=137, batch_size=2) is None


def test_trace_meta_op_shapes_torch_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert mt.trace_meta_op_shapes("whatever") is None


def test_trace_meta_op_shapes_instantiation_failure(monkeypatch):
    def _boom(_ckpt):
        raise RuntimeError("nope")

    monkeypatch.setattr(tt, "_instantiate_meta", _boom)
    assert mt.trace_meta_op_shapes("bad/checkpoint") is None


# ---------------------------------------------------------------------------
# symbolise_meta_shape
# ---------------------------------------------------------------------------


def test_symbolise_meta_shape():
    out = mt.symbolise_meta_shape((1, 128, 32), batch_size=1, seq_len=128)
    assert out == ("B", "S", 32)
    # no substitution when dims don't match
    assert mt.symbolise_meta_shape((7, 9), batch_size=1, seq_len=128) == (7, 9)


def test_symbolise_meta_shape_batch2_keeps_size_one():
    """A genuine size-1 dim (an ``unsqueeze(1)`` head axis) must stay literal 1,
    not alias onto ``B`` -- the reason ``load_meta_shapes`` traces with
    ``batch_size=2``. With ``batch_size=1`` a compressor output ``(1, 1, 32, 512)``
    would print as a nonsensical ``[B, B, 32, 512]``; with ``batch_size=2`` only
    the true batch dim maps to ``B`` and the singleton axis survives."""
    out = mt.symbolise_meta_shape((2, 1, 32, 512), batch_size=2, seq_len=128)
    assert out == ("B", 1, 32, 512)


# ---------------------------------------------------------------------------
# walk_meta_module_tree (forward-free structural ModuleList walk)
# ---------------------------------------------------------------------------


def test_walk_meta_module_tree_llama(llama_ckpt):
    groups = mt.walk_meta_module_tree(llama_ckpt)
    assert groups is not None
    layer_groups = [g for g in groups if g.path.endswith("layers")]
    assert layer_groups, [g.path for g in groups]
    g = layer_groups[0]
    assert g.length == 2  # num_hidden_layers in the fixture
    assert g.element_class.endswith("DecoderLayer")
    assert len(g.signatures) == g.length


def test_walk_meta_module_tree_torch_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert mt.walk_meta_module_tree("whatever") is None


def test_walk_meta_module_tree_instantiation_failure(monkeypatch):
    def _boom(_ckpt):
        raise RuntimeError("cannot load")

    monkeypatch.setattr(tt, "_instantiate_meta", _boom)
    assert mt.walk_meta_module_tree("bad/checkpoint") is None


def test_walk_meta_module_tree_multiple_lists(monkeypatch):
    """Two independent ModuleLists -> two MetaModuleGroups (vision-tower case)."""

    class TwoLists(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(3)])
            self.vision = nn.ModuleList([nn.Linear(4, 4) for _ in range(5)])

    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (TwoLists(), None))
    groups = mt.walk_meta_module_tree("x")
    assert groups is not None
    by_path = {g.path: g for g in groups}
    assert by_path["layers"].length == 3
    assert by_path["layers"].element_class == "Block"
    assert by_path["vision"].length == 5
    assert by_path["vision"].element_class == "Linear"


def test_walk_meta_module_tree_mixed_signatures(monkeypatch):
    """Elements with different child structure bucket into distinct signatures."""
    from collections import Counter

    class DenseLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)

    class MoELayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(4, 4)
            self.moe = nn.ModuleList([nn.Linear(4, 4)])

    class Hybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [DenseLayer() for _ in range(5)] + [MoELayer() for _ in range(3)]
            )

    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (Hybrid(), None))
    groups = mt.walk_meta_module_tree("x")
    assert groups is not None
    layer_group = next(g for g in groups if g.path == "layers")
    assert layer_group.length == 8
    buckets = Counter(layer_group.signatures)
    assert sorted(buckets.values()) == [3, 5]


def test_walk_meta_module_tree_splits_on_nested_divergence(monkeypatch):
    """Layers with identical immediate children but a differing GRANDCHILD split apart.

    A shallow immediate-children signature would collapse these into one uniform group;
    the full nested descriptor separates them and records the diverging path so a caller
    can render each variant's own subtree.
    """
    from collections import Counter

    class Attn(nn.Module):
        def __init__(self, compressor):
            super().__init__()
            self.q = nn.Linear(4, 4)
            if compressor is not None:
                self.compressor = compressor

    class Layer(nn.Module):
        def __init__(self, compressor):
            super().__init__()
            self.self_attn = Attn(compressor)  # same immediate class either way
            self.mlp = nn.Linear(4, 4)

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [Layer(None), Layer(None)]  # no nested compressor
                + [Layer(nn.Linear(4, 4)) for _ in range(3)]  # nested compressor
            )

    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (Stack(), None))
    groups = mt.walk_meta_module_tree("x")
    layer_group = next(g for g in groups if g.path == "layers")
    # Immediate children are identical (self_attn=Attn, mlp=Linear) for all 5 layers.
    immediate = {
        tuple(cls for path, cls in desc if "." not in path)
        for desc in layer_group.descriptors
    }
    assert len(immediate) == 1
    # But the full signature splits 2 (no compressor) vs 3 (compressor present).
    assert sorted(Counter(layer_group.signatures).values()) == [2, 3]
    # The divergent path is discoverable from the descriptors.
    paths = {path for desc in layer_group.descriptors for path, _ in desc}
    assert "self_attn.compressor" in paths


def test_repair_missing_config_attr_copies_from_subconfig():
    """A value already defined on a sub-config is reused, not overwritten with 1."""

    class Cfg:
        def to_dict(self):
            return {}

    top = Cfg()
    sub = Cfg()
    sub.temporal_patch_size = 2  # real value lives on the sub-config
    top.vision_config = sub

    assert mt._repair_missing_config_attr(top, "temporal_patch_size") is True
    assert top.temporal_patch_size == 2  # copied from the sub-config


def test_repair_missing_config_attr_neutral_default_on_top_and_subs():
    """When no config defines the attr, a neutral 1 is set on every config."""

    class Cfg:
        def to_dict(self):
            return {}

    top = Cfg()
    sub = Cfg()
    top.vision_config = sub

    assert mt._repair_missing_config_attr(top, "mystery_attr") is True
    assert top.mystery_attr == 1
    assert sub.mystery_attr == 1


def test_repair_missing_config_attr_noop_when_already_present():
    """Nothing to patch (and no subconfigs) -> returns False."""

    class Cfg:
        def to_dict(self):
            return {}

    top = Cfg()
    top.foo = 5
    assert mt._repair_missing_config_attr(top, "foo") is False


def test_instantiate_meta_robust_happy_path_delegates(monkeypatch):
    """When the canonical instantiator succeeds, its result is returned as-is."""
    sentinel = (object(), object())
    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: sentinel)
    assert mt._instantiate_meta_robust("x") is sentinel


class _FakeConfig:
    def to_dict(self):
        return {}


def test_instantiate_meta_robust_repairs_missing_attr(monkeypatch):
    """A missing config attr is filled in generically, then instantiation succeeds."""
    import transformers

    fake_cfg = _FakeConfig()

    class FakeModel(nn.Module):
        def eval(self):
            return self

    class FakeAuto:
        @staticmethod
        def from_config(config, trust_remote_code=False):
            if not hasattr(config, "needed_attr"):
                raise AttributeError(
                    "'PreTrainedConfig' object has no attribute 'needed_attr'"
                )
            return FakeModel()

    def _boom(_ckpt):
        raise AttributeError("'PreTrainedConfig' object has no attribute 'needed_attr'")

    monkeypatch.setattr(tt, "_instantiate_meta", _boom)
    monkeypatch.setattr(tt, "_patch_config", lambda c: None)
    monkeypatch.setattr(tt, "_resolve_auto_classes", lambda c: [FakeAuto])
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: fake_cfg
    )

    result = mt._instantiate_meta_robust("some/model")
    assert result is not None
    _model, cfg = result
    assert cfg is fake_cfg
    assert cfg.needed_attr == 1  # neutral default was applied


def test_instantiate_meta_robust_degrades_on_missing_dependency(monkeypatch):
    """A missing optional dependency (e.g. einops) degrades to None, no raise."""
    import transformers

    fake_cfg = _FakeConfig()

    class FakeAuto:
        @staticmethod
        def from_config(config, trust_remote_code=False):
            raise ImportError("This modeling file requires einops")

    def _boom(_ckpt):
        raise ImportError("This modeling file requires einops")

    monkeypatch.setattr(tt, "_instantiate_meta", _boom)
    monkeypatch.setattr(tt, "_patch_config", lambda c: None)
    monkeypatch.setattr(tt, "_resolve_auto_classes", lambda c: [FakeAuto])
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: fake_cfg
    )

    assert mt._instantiate_meta_robust("some/model") is None


def test_instantiate_meta_robust_gives_up_on_unrepairable_attr(monkeypatch):
    """An attr that keeps raising even after a default is set -> bounded, None."""
    import transformers

    fake_cfg = _FakeConfig()

    class FakeAuto:
        @staticmethod
        def from_config(config, trust_remote_code=False):
            # Raises for the same attr regardless of whether it was set, so the
            # repair can never satisfy it — the loop must give up, not spin.
            raise AttributeError("'X' object has no attribute 'stubborn'")

    def _boom(_ckpt):
        raise AttributeError("'X' object has no attribute 'stubborn'")

    monkeypatch.setattr(tt, "_instantiate_meta", _boom)
    monkeypatch.setattr(tt, "_patch_config", lambda c: None)
    monkeypatch.setattr(tt, "_resolve_auto_classes", lambda c: [FakeAuto])
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: fake_cfg
    )

    assert mt._instantiate_meta_robust("some/model") is None


def test_instantiate_meta_robust_text_submodel_fallback(monkeypatch):
    """A composite/multimodal config whose *full* build fails inside a non-text
    sub-tower falls back to building only the recognised TEXT sub-model.

    Guards the MiniMax vision-tower regression: the full build dies in the vision
    tower, so the fallback must (a) take the declared text sub-config via
    ``get_text_config``, (b) re-type it through the text ``*ForCausalLM`` class'
    ``config_class`` so derived/structured fields (which a scalar neutral default
    could not synthesise) are recomputed, and (c) build the text model. It must
    stay opt-in: without the flag the shared instantiator still returns ``None``.
    """
    import sys

    import transformers

    class _RawTextSubConfig:
        """The degraded sub-config as loaded: lacks the structured field."""

        def __init__(self, **kw):
            self.__dict__.update(kw)

        def to_dict(self):
            return dict(self.__dict__)

    class _TypedTextConfig:
        """The properly-typed text config: its __init__ recomputes a structured
        (list) field the modeling code indexes — mirrors ``layer_types`` /
        ``rope_parameters``."""

        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.derived_structured = [0, 0]

        def to_dict(self):
            return dict(self.__dict__)

    class _TopConfig:
        def __init__(self):
            self._text = _RawTextSubConfig(hidden=8)

        def get_text_config(self):
            return self._text

        def to_dict(self):
            return {}

    top_cfg = _TopConfig()

    class FakeTextForCausalLM(nn.Module):
        config_class = _TypedTextConfig

        def __init__(self, config):
            super().__init__()
            # Indexing a scalar neutral default would raise TypeError; only the
            # typed config supplies a real list here.
            _ = config.derived_structured[1]
            self.config = config

        @classmethod
        def _from_config(cls, config):
            return cls(config)

        def eval(self):
            return self

    class FakeConditionalGeneration(nn.Module):
        """The full (composite) model class, in the same module as the text one."""

    # The text class is defined in this function; expose it at module scope so the
    # fallback's ``*ForCausalLM`` module scan finds it (monkeypatch auto-reverts).
    monkeypatch.setattr(
        sys.modules[__name__], "FakeTextForCausalLM", FakeTextForCausalLM, raising=False
    )

    class _Mapping:
        def __contains__(self, key):
            return key is _TopConfig

        def __getitem__(self, key):
            return FakeConditionalGeneration

    class FakeAuto:
        _model_mapping = _Mapping()

        @staticmethod
        def from_config(config, trust_remote_code=False):
            raise ValueError("vision sub-tower cannot build on meta")

    def _boom(_ckpt):
        raise AttributeError(
            "'PreTrainedConfig' object has no attribute 'temporal_patch_size'"
        )

    monkeypatch.setattr(tt, "_instantiate_meta", _boom)
    monkeypatch.setattr(tt, "_patch_config", lambda c: None)
    monkeypatch.setattr(tt, "_resolve_auto_classes", lambda c: [FakeAuto])
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: top_cfg
    )

    # Opt-out (the shape tracers' default): no text fallback, stays None.
    assert mt._instantiate_meta_robust("some/vlm") is None

    # Opt-in (attention-group harvest): the text sub-model builds.
    result = mt._instantiate_meta_robust("some/vlm", text_submodel_fallback=True)
    assert result is not None
    model, cfg = result
    assert isinstance(model, FakeTextForCausalLM)
    assert isinstance(cfg, _TypedTextConfig)


def test_structured_module_output_reports_its_result_tensor():
    hidden = torch.empty(2, 5, 8)
    cache = torch.empty(2, 4, 5, 2)
    output = {
        "past_key_values": cache,
        "last_hidden_state": hidden,
    }
    assert mt._tensor_shape_from_output(hidden) == (2, 5, 8)
    assert mt._tensor_shape_from_output((cache, hidden)) == (2, 4, 5, 2)
    # A plain mapping makes no promise about which tensor is the result.
    # Its declared insertion order is the only structural ordering available.
    assert mt._tensor_shape_from_output(output) == (2, 4, 5, 2)

    class Result:
        def __init__(self):
            self.cache = torch.empty(2, 4, 5, 2)
            self.hidden = hidden

        def to_tuple(self):
            # This is the output contract ModelOutput exposes. No field name
            # has to be known to select its result tensor.
            return (self.hidden, self.cache)

    assert mt._tensor_shape_from_output(Result()) == (2, 5, 8)

    @dataclass
    class DataclassResult:
        result: torch.Tensor
        cache: torch.Tensor

    assert mt._tensor_shape_from_output(DataclassResult(hidden, cache)) == (2, 5, 8)

    class Ordered(dict):
        def to_tuple(self):
            return (self["hidden"], self["cache"])

    assert mt._tensor_shape_from_output(Ordered(cache=cache, hidden=hidden)) == (
        2,
        5,
        8,
    )

    class Broken:
        def __init__(self):
            self.hidden = hidden

        def to_tuple(self):
            raise RuntimeError("the positional contract is unavailable")

    assert mt._tensor_shape_from_output(Broken()) == (2, 5, 8)

    cycle = Broken()
    cycle.self = cycle
    assert mt._tensor_shape_from_output(cycle) == (2, 5, 8)
    assert mt._tensor_shape_from_output((None, "name", [hidden])) == (2, 5, 8)
    assert mt._tensor_shape_from_output("name") is None
    assert mt._tensor_shape_from_output(b"name") is None
    assert mt._tensor_shape_from_output(object()) is None


# ---------------------------------------------------------------------------
# Config-attribute repair edges
# ---------------------------------------------------------------------------


def test_meta_build_rejects_an_attribute_error_it_cannot_parse():
    """An AttributeError that does not name a missing attribute is not repaired."""

    def builder(_config):
        raise AttributeError("the modeling code failed for another reason")

    assert mt._meta_build_with_attr_repair(_FakeConfig(), [builder], "ckpt") is None


def test_meta_build_stops_when_the_repair_budget_is_spent(monkeypatch):
    """A repair that would need another attempt past the budget returns None."""
    monkeypatch.setattr(mt, "_MAX_CONFIG_ATTR_REPAIRS", 0)
    config = _FakeConfig()

    def builder(_config):
        raise AttributeError("'X' object has no attribute 'fresh_attr'")

    assert mt._meta_build_with_attr_repair(config, [builder], "ckpt") is None
    assert config.fresh_attr == 1


# ---------------------------------------------------------------------------
# Text sub-model fallback early exits
# ---------------------------------------------------------------------------


def _reload_config(monkeypatch, config):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: config
    )


def test_text_submodel_returns_none_when_the_config_cannot_be_reloaded(monkeypatch):
    import transformers

    def boom(*_a, **_k):
        raise OSError("config is unavailable")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", boom)
    assert mt._instantiate_text_submodel_meta(object(), [], "ckpt") is None


def test_text_submodel_returns_none_when_text_config_lookup_fails(monkeypatch):
    class Cfg:
        def get_text_config(self):
            raise RuntimeError("no text config")

    _reload_config(monkeypatch, Cfg())
    assert mt._instantiate_text_submodel_meta(object(), [], "ckpt") is None


def test_text_submodel_returns_none_when_there_is_no_distinct_text_config(monkeypatch):
    class Same:
        def get_text_config(self):
            return self

    class Missing:
        def get_text_config(self):
            return None

    _reload_config(monkeypatch, Same())
    assert mt._instantiate_text_submodel_meta(object(), [], "ckpt") is None
    _reload_config(monkeypatch, Missing())
    assert mt._instantiate_text_submodel_meta(object(), [], "ckpt") is None


def test_text_submodel_returns_none_when_no_model_class_resolves(monkeypatch):
    class Cfg:
        def get_text_config(self):
            return object()

    class Unmapped:
        pass

    _reload_config(monkeypatch, Cfg())
    assert mt._instantiate_text_submodel_meta(object(), [Unmapped], "ckpt") is None


def test_text_submodel_returns_none_when_the_model_mapping_lookup_fails(monkeypatch):
    class Cfg:
        def get_text_config(self):
            return object()

    class BadMap:
        def __contains__(self, _key):
            raise RuntimeError("mapping is unusable")

    class Auto:
        _model_mapping = BadMap()

    _reload_config(monkeypatch, Cfg())
    assert mt._instantiate_text_submodel_meta(object(), [Auto], "ckpt") is None


def test_text_submodel_returns_none_when_the_modeling_module_is_not_loaded(monkeypatch):
    class Full:
        pass

    Full.__module__ = "tracelens_modeling_module_not_loaded"

    class Mapping:
        def __contains__(self, _key):
            return True

        def __getitem__(self, _key):
            return Full

    class Auto:
        _model_mapping = Mapping()

    class Cfg:
        def get_text_config(self):
            return object()

    _reload_config(monkeypatch, Cfg())
    assert mt._instantiate_text_submodel_meta(object(), [Auto], "ckpt") is None


def test_text_submodel_returns_none_when_the_module_has_no_causal_lm(monkeypatch):
    import sys
    import types

    module = types.ModuleType("tracelens_modeling_without_causal_lm")
    module.Helper = type("Helper", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Full:
        pass

    Full.__module__ = module.__name__

    class Mapping:
        def __contains__(self, _key):
            return True

        def __getitem__(self, _key):
            return Full

    class Auto:
        _model_mapping = Mapping()

    class Cfg:
        def get_text_config(self):
            return object()

    _reload_config(monkeypatch, Cfg())
    assert mt._instantiate_text_submodel_meta(object(), [Auto], "ckpt") is None


def test_text_submodel_uses_the_raw_config_when_retyping_fails(monkeypatch):
    """A config class that cannot be constructed is discarded, and a build that
    still fails on that raw config returns None."""
    import sys
    import types

    module = types.ModuleType("tracelens_modeling_retype_fails")

    class BadConfig:
        def __init__(self, **_kw):
            raise TypeError("typed config cannot be built")

    class TextForCausalLM:
        config_class = BadConfig

        @classmethod
        def _from_config(cls, _config):
            raise RuntimeError("text model cannot be built")

    module.TextForCausalLM = TextForCausalLM
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(tt, "_patch_config", lambda _c: None)

    class Full:
        pass

    Full.__module__ = module.__name__

    class Mapping:
        def __contains__(self, _key):
            return True

        def __getitem__(self, _key):
            return Full

    class Auto:
        _model_mapping = Mapping()

    class Sub:
        def to_dict(self):
            return {"hidden": 8}

    class Cfg:
        def get_text_config(self):
            return Sub()

    _reload_config(monkeypatch, Cfg())
    assert mt._instantiate_text_submodel_meta(object(), [Auto], "ckpt") is None


def test_instantiate_meta_robust_returns_none_when_imports_fail(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert mt._instantiate_meta_robust("ckpt") is None


# ---------------------------------------------------------------------------
# harvest_meta_tensors
# ---------------------------------------------------------------------------


def test_harvest_meta_tensors_indexes_tied_weights_and_buffers(monkeypatch):
    class TiedHead(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

    class HarvestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(5, 4)
            self.head = TiedHead(self.embed.weight)
            self.register_buffer("inv_freq", torch.empty(3))

    model = HarvestModel()
    monkeypatch.setattr(mt, "_instantiate_meta_robust", lambda _c: (model, None))
    index = mt.harvest_meta_tensors("ckpt")
    assert index is not None
    assert index.by_qualified["embed.weight"].shape == (5, 4)
    assert index.by_qualified["head.weight"].shape == (5, 4)
    assert index.by_qualified["inv_freq"].shape == (3,)
    assert index.by_qualified["embed.weight"].dtype == "float32"
    assert index.by_class_attr[("Embedding", "weight")].shape == (5, 4)
    assert index.by_class_attr[("TiedHead", "weight")].shape == (5, 4)
    assert index.by_class_attr[("HarvestModel", "inv_freq")].shape == (3,)


def test_harvest_meta_tensors_torch_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert mt.harvest_meta_tensors("ckpt") is None


def test_harvest_meta_tensors_returns_none_when_instantiation_fails(monkeypatch):
    monkeypatch.setattr(mt, "_instantiate_meta_robust", lambda _c: None)
    assert mt.harvest_meta_tensors("ckpt") is None


def test_harvest_meta_tensors_returns_none_when_the_walk_fails(monkeypatch):
    class Unwalkable(nn.Module):
        def named_modules(self, *args, **kwargs):
            raise RuntimeError("cannot walk")

    monkeypatch.setattr(mt, "_instantiate_meta_robust", lambda _c: (Unwalkable(), None))
    assert mt.harvest_meta_tensors("ckpt") is None


# ---------------------------------------------------------------------------
# Tracer failure branches
# ---------------------------------------------------------------------------


def test_trace_meta_shapes_returns_none_when_rotary_patch_fails(monkeypatch):
    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (nn.Linear(2, 2), None))

    def boom(_model):
        raise RuntimeError("rotary patch failed")

    monkeypatch.setattr(tt, "_patch_rotary_embeddings", boom)
    assert mt.trace_meta_shapes("x") is None


def test_trace_meta_shapes_ignores_an_output_whose_shape_cannot_be_read(monkeypatch):
    class Unreadable:
        @property
        def shape(self):
            raise RuntimeError("shape is unavailable")

    class BoomOut(nn.Module):
        def forward(self, _x):
            return Unreadable()

    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (BoomOut(), None))
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda _m: None)
    assert mt.trace_meta_shapes("x", seq_len=4, batch_size=1) is None


def test_trace_meta_input_specs_torch_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert mt.trace_meta_input_specs("x") is None


def test_trace_meta_input_specs_returns_none_when_rotary_patch_fails(monkeypatch):
    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (nn.Linear(2, 2), None))

    def boom(_model):
        raise RuntimeError("rotary patch failed")

    monkeypatch.setattr(tt, "_patch_rotary_embeddings", boom)
    assert mt.trace_meta_input_specs("x") is None


def test_trace_meta_input_specs_names_uninspectable_forward_arguments(monkeypatch):
    class Odd(nn.Module):
        def forward(self, x):
            return torch.zeros(1, device=x.device)

    Odd.forward.__signature__ = "not-a-signature"
    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (Odd().to("meta"), None))
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda _m: None)
    result = mt.trace_meta_input_specs("x", seq_len=5, batch_size=2)
    assert result is not None
    resolved, by_class = result
    assert resolved["arg0"] == (("B", "S"), "int64")
    assert by_class[("Odd", "arg0")] == (("B", "S"), "int64")


def test_trace_meta_input_specs_returns_none_when_nothing_is_captured(monkeypatch):
    class Silent:
        def forward(self, x):
            raise AssertionError("the probe must fail before forward")

        def named_modules(self):
            return [("", self)]

        def register_forward_pre_hook(self, hook, with_kwargs=False):
            return types.SimpleNamespace(remove=lambda: None)

        def __call__(self, *args, **kwargs):
            raise RuntimeError("probe failed before any tensor was recorded")

    monkeypatch.setattr(mt, "_instantiate_meta_robust", lambda _c: (Silent(), None))
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda _m: None)
    assert mt.trace_meta_input_specs("x") is None


def test_trace_meta_op_shapes_returns_none_when_rotary_patch_fails(monkeypatch):
    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (nn.Linear(4, 4), None))

    def boom(_model):
        raise RuntimeError("rotary patch failed")

    monkeypatch.setattr(tt, "_patch_rotary_embeddings", boom)
    assert mt.trace_meta_op_shapes("x") is None


def test_trace_meta_op_shapes_keeps_going_when_the_probe_forward_fails(monkeypatch):
    class Exploding(nn.Module):
        def forward(self, _x):
            raise RuntimeError("probe forward failed")

    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (Exploding(), None))
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda _m: None)
    assert mt.trace_meta_op_shapes("x", seq_len=137, batch_size=2) is None


def test_trace_meta_op_shapes_skips_nodes_without_a_source_line(monkeypatch):
    monkeypatch.setattr(
        tt, "_instantiate_meta", lambda _c: (SyntheticNet().to("meta"), None)
    )
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda _m: None)
    monkeypatch.setattr(mt, "_fx_source_line", lambda _node, _source: None)
    assert mt.trace_meta_op_shapes("x", seq_len=137, batch_size=2) is None


class _AddWide(nn.Module):
    def forward(self, x):
        return x + 1


class _AddNarrow(nn.Module):
    def forward(self, x):
        return x + 1


class _ConflictNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = _AddWide()
        self.narrow = _AddNarrow()

    def forward(self, x):
        return self.wide(x)[:, :1] + self.narrow(x[:, :1])


def test_trace_meta_op_shapes_drops_conflicting_shapes(monkeypatch):
    """Two classes whose same op lands on one source line with different shapes
    are dropped instead of keeping either guess."""
    monkeypatch.setattr(
        tt, "_instantiate_meta", lambda _c: (_ConflictNet().to("meta"), None)
    )
    monkeypatch.setattr(tt, "_patch_rotary_embeddings", lambda _m: None)
    monkeypatch.setattr(mt, "_fx_source_line", lambda _node, _source: 1)
    result = mt.trace_meta_op_shapes("x", seq_len=137, batch_size=2)
    assert result is not None
    # Occurrence 0 is the add both classes emit, at two different shapes, so it
    # is dropped. A later add that only one class produces is kept.
    assert (1, "add", 0) not in result
    assert result[(1, "add", 1)] == ("B", 1)


def test_conv_geometry_detail_reads_live_module():
    # A standard conv: weight is [C_out, C_in/groups, kH, kW], bias present.
    conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
    assert (
        mt.conv_geometry_detail(conv)
        == "conv_weight=[64, 3, 7, 7]; conv_bias=1; conv_transposed=0"
    )
    # Grouped, bias-free conv: C_in/groups is reflected in the weight shape.
    grouped = nn.Conv2d(32, 32, kernel_size=3, groups=32, bias=False)
    assert (
        mt.conv_geometry_detail(grouped)
        == "conv_weight=[32, 1, 3, 3]; conv_bias=0; conv_transposed=0"
    )
    # Transposed conv sets the flag.
    deconv = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, bias=False)
    assert "conv_transposed=1" in mt.conv_geometry_detail(deconv)
    # Not a convolution -> None.
    assert mt.conv_geometry_detail(nn.Linear(8, 8)) is None
    assert mt.conv_geometry_detail(object()) is None


def test_meta_main_input_feeds_pixel_values_for_a_vision_model():
    # A vision model's main_input_name is pixel_values; it must get a
    # [B, C, H, W] image, not the [B, S] token dummy its forward would reject.
    vision_model = types.SimpleNamespace(main_input_name="pixel_values")
    vision_cfg = types.SimpleNamespace(num_channels=3, image_size=224)
    dummy = mt._meta_main_input(vision_model, vision_cfg, batch_size=2)
    assert isinstance(dummy, dict)
    assert tuple(dummy["pixel_values"].shape) == (2, 3, 224, 224)

    # A text model stays on the [B, S] token dummy.
    text_model = types.SimpleNamespace(main_input_name="input_ids")
    text_dummy = mt._meta_main_input(text_model, types.SimpleNamespace(), batch_size=1)
    assert isinstance(text_dummy, torch.Tensor)
    assert text_dummy.dtype == torch.long and text_dummy.shape[0] == 1


def test_trace_meta_conv_ops_captures_live_conv_instances(monkeypatch):
    # A two-conv vision stub: instantiation is stubbed out so the test stays
    # offline, but the hook/forward/capture path is the real one.
    class TinyVision(nn.Module):
        main_input_name = "pixel_values"

        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=True)
            self.dw = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)

        def forward(self, pixel_values):
            return self.dw(self.stem(pixel_values))

    cfg = types.SimpleNamespace(num_channels=3, image_size=16)
    monkeypatch.setattr(
        mt,
        "_instantiate_meta_robust",
        lambda checkpoint: (TinyVision().to("meta"), cfg),
    )
    convs = mt.trace_meta_conv_ops("ignored", batch_size=1)
    assert convs is not None
    by_path = {c["path"]: c for c in convs}
    assert by_path["stem"]["output_shape"] == [1, 8, 4, 4]
    assert by_path["stem"]["weight_shape"] == [8, 3, 4, 4]
    assert by_path["stem"]["bias"] is True
    # Depthwise block: groups == channels -> weight [C, 1, k, k], no bias.
    assert by_path["dw"]["weight_shape"] == [8, 1, 3, 3]
    assert by_path["dw"]["bias"] is False
    assert by_path["dw"]["input_shape"] == [1, 8, 4, 4]
