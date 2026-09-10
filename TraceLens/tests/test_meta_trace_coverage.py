###############################################################################
# Coverage tests for TraceLens/ModelUtils/meta_trace.py
#
# Exercises the meta-device shape tracing entrypoints (``trace_meta_shapes``
# and ``trace_meta_op_shapes``) end-to-end against a tiny real Llama checkpoint
# instantiated on the meta device, plus the pure helper functions and the
# torch-unavailable / instantiation-failure / no-shapes fallback branches.
###############################################################################

from __future__ import annotations

import os


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
        mt._fx_source_line(
            make({"stack_trace": 'File "/other.py", line 5'}), "/mod.py"
        )
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
