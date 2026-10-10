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


# ---------------------------------------------------------------------------
# harvest_meta_attention_groups (forward-free grouped-query repeat-factor walk)
# ---------------------------------------------------------------------------


def test_attention_group_factor_prefers_precomputed_groups():
    assert (
        mt._attention_group_factor(types.SimpleNamespace(num_key_value_groups=16)) == 16
    )


def test_attention_group_factor_derives_from_live_head_counts():
    # No precomputed groups -> derive from the module's own head counts (not config).
    mod = types.SimpleNamespace(num_attention_heads=64, num_key_value_heads=4)
    assert mt._attention_group_factor(mod) == 16


def test_attention_group_factor_rejects_bool_zero_and_absent():
    assert (
        mt._attention_group_factor(types.SimpleNamespace(num_key_value_groups=True))
        is None
    )
    assert (
        mt._attention_group_factor(
            types.SimpleNamespace(num_attention_heads=64, num_key_value_heads=0)
        )
        is None
    )
    assert mt._attention_group_factor(types.SimpleNamespace()) is None


def test_harvest_meta_attention_groups_by_class(monkeypatch):
    class Attn(nn.Module):
        def __init__(self, groups):
            super().__init__()
            self.num_key_value_groups = groups

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Attn(16) for _ in range(3)])
            self.other = nn.Linear(4, 4)  # no group attr -> skipped

    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (Model(), None))
    assert mt.harvest_meta_attention_groups("x") == {"Attn": 16}


def test_harvest_meta_attention_groups_conflict_keeps_first(monkeypatch):
    class Attn(nn.Module):
        def __init__(self, groups):
            super().__init__()
            self.num_key_value_groups = groups

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Attn(16)
            self.b = Attn(8)  # same class, different factor -> keep first

    monkeypatch.setattr(tt, "_instantiate_meta", lambda _c: (Model(), None))
    assert mt.harvest_meta_attention_groups("x") == {"Attn": 16}


def test_harvest_meta_attention_groups_torch_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert mt.harvest_meta_attention_groups("whatever") is None


def test_harvest_meta_attention_groups_instantiation_failure(monkeypatch):
    def _boom(_ckpt):
        raise RuntimeError("cannot load")

    monkeypatch.setattr(tt, "_instantiate_meta", _boom)
    assert mt.harvest_meta_attention_groups("bad/checkpoint") is None


# ---------------------------------------------------------------------------
# Robust meta-instantiation: generic missing-config-attr repair + graceful
# degradation for a genuinely uninstantiable model (missing optional dep).
# ---------------------------------------------------------------------------


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
