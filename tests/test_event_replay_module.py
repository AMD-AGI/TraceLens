###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens/EventReplay.

Schema parsing, IR construction, custom-init name matching, and
``extract_batch_context`` run without torch. Torch-dependent paths are skipped
when torch is missing. GPU replay uses ``@pytest.mark.gpu`` (excluded from
CPU-only CI). For a longer GPU smoke (profile + replay), run
``python examples/event_replay_gpu_smoke.py``.
"""

from __future__ import annotations


import importlib.util
import sys
from unittest.mock import patch

import pytest

from TraceLens.EventReplay.custom_inits import (
    CustomInit,
    PagedAttentionInit,
    extract_batch_context,
)
from TraceLens.EventReplay.event_replay import EventReplayer
from TraceLens.EventReplay.utils import TensorCfg, list_profile_tensor_types

HAS_TORCH = importlib.util.find_spec("torch") is not None


class _FakeSchema:
    def __init__(self, schema_str: str):
        self._schema_str = schema_str

    def __str__(self) -> str:
        return self._schema_str


def _require_torch():
    return pytest.importorskip("torch")


def _require_cuda_gpu():
    torch = _require_torch()
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA/HIP with at least one visible GPU")
    return torch


ADD_EVENT = {
    "name": "aten::add",
    "args": {
        "Input Dims": [[2, 4], [2, 4], ()],
        "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"],
        "Input Strides": [[4, 1], [4, 1], ()],
        "Concrete Inputs": ["", "", "1.0"],
    },
}

ADD_SCHEMA = _FakeSchema(
    "aten::add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
)

MM_EVENT = {
    "name": "aten::mm",
    "args": {
        "Input Dims": [[4, 8], [8, 16]],
        "Input type": ["c10::BFloat16", "c10::BFloat16"],
        "Input Strides": [[8, 1], [16, 1]],
        "Concrete Inputs": ["", ""],
    },
}


class TestTensorCfg:
    def test_tensor_cfg_fields(self):
        cfg = TensorCfg(
            shape=[2, 4],
            dtype="c10::BFloat16",
            strides=[4, 1],
            init="normal",
        )
        assert cfg.shape == [2, 4]
        assert cfg.dtype == "c10::BFloat16"
        assert cfg.init == "normal"

    def test_list_profile_tensor_types_contains_common_dtypes(self):
        assert "c10::BFloat16" in list_profile_tensor_types
        assert "float" in list_profile_tensor_types


class TestParseSchemaString:
    def test_positional_and_keyword_args(self):
        op_name, pos, kw, ret = EventReplayer.parse_schema_string(ADD_SCHEMA)
        assert op_name == "aten::add"
        assert [a["arg_name"] for a in pos] == ["self", "other"]
        assert [a["arg_name"] for a in kw] == ["alpha"]
        assert ret == "Tensor"

    def test_positional_only_schema(self):
        schema = _FakeSchema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
        op_name, pos, kw, ret = EventReplayer.parse_schema_string(schema)
        assert op_name == "aten::mm"
        assert len(pos) == 2
        assert kw == []
        assert ret == "Tensor"

    def test_invalid_schema_raises(self):
        with pytest.raises(ValueError, match="Cannot parse schema"):
            EventReplayer.parse_schema_string(_FakeSchema("not a schema"))


class TestSchemaMatching:
    def test_is_schema_match_accepts_matching_add_event(self):
        assert EventReplayer._is_schema_match(ADD_EVENT, ADD_SCHEMA)

    def test_is_schema_match_rejects_wrong_arg_count(self):
        short_event = {
            "name": "aten::add",
            "args": {
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": [""],
                "Input Dims": [[2, 4]],
                "Input Strides": [[4, 1]],
            },
        }
        assert not EventReplayer._is_schema_match(short_event, ADD_SCHEMA)

    def test_is_schema_match_rejects_bad_scalar(self):
        bad_event = {
            "name": "aten::add",
            "args": {
                "Input Dims": [[2, 4], [2, 4], ()],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "Scalar"],
                "Input Strides": [[4, 1], [4, 1], ()],
                "Concrete Inputs": ["", "", "not-a-number"],
            },
        }
        assert not EventReplayer._is_schema_match(bad_event, ADD_SCHEMA)


class TestEventReplayIR:
    def test_get_event_replay_ir_builds_tensor_and_scalar_args(self):
        ir = EventReplayer._get_event_replay_IR(ADD_EVENT, ADD_SCHEMA)
        pos_values = [entry["value"] for entry in ir["list_pos_args"]]
        assert all(isinstance(v, TensorCfg) for v in pos_values)
        assert pos_values[0].shape == [2, 4]
        kw_values = {entry["arg_name"]: entry["value"] for entry in ir["list_kwargs"]}
        assert kw_values["alpha"] == 1.0


class _FakeReplayer:
    def __init__(self, name):
        self.event = {"name": name}


class _NoOpInit(CustomInit):
    def initialize(self, replayer, **kwargs):
        return None


class TestCustomInitAppliesTo:
    def test_exact_event_name_matches(self):
        init = _NoOpInit()
        init.op_patterns = ["aten::mm"]
        assert init.applies_to(_FakeReplayer("aten::mm"))

    def test_substring_does_not_match(self):
        init = _NoOpInit()
        init.op_patterns = ["mm"]
        assert not init.applies_to(_FakeReplayer("aten::mm"))


class TestCustomInitRegistry:
    @pytest.fixture(autouse=True)
    def _isolate_custom_init_registry(self):
        saved = EventReplayer._custom_init_registry[:]
        yield
        EventReplayer._custom_init_registry = saved

    def test_register_prepends_so_last_wins(self):
        builtin = _NoOpInit()
        builtin.op_patterns = ["_rocm_C::paged_attention"]
        user = _NoOpInit()
        user.op_patterns = ["_rocm_C::paged_attention"]
        EventReplayer._custom_init_registry = [builtin]
        EventReplayer.register_custom_init(user)
        assert EventReplayer._custom_init_registry[0] is user
        first = next(
            i
            for i in EventReplayer._custom_init_registry
            if i.applies_to(_FakeReplayer("_rocm_C::paged_attention"))
        )
        assert first is user


class _FakeAnalyzer:
    def __init__(self, events):
        self.tree = type("Tree", (), {"events": events})()


def _vllm_annotation(ts=0, dur=100):
    return {
        "cat": "user_annotation",
        "name": "execute_context_2(18)_generation_5(5)",
        "ts": ts,
        "dur": dur,
    }


def _cpu_op(name, ts=10):
    return {
        "name": name,
        "ts": ts,
        "args": {"Input Dims": [[1, 1]]},
    }


class TestExtractBatchContext:
    def test_exact_paged_attention_is_annotated(self):
        op = _cpu_op("_rocm_C::paged_attention")
        n = extract_batch_context(_FakeAnalyzer([_vllm_annotation(), op]))
        assert n == 1
        assert op["batch_context"]["n_prefill"] == 2
        assert "_rocm_C::paged_attention" in PagedAttentionInit.op_patterns

    def test_substring_name_is_not_annotated(self):
        op = _cpu_op("aiter::paged_attention_v1")
        n = extract_batch_context(_FakeAnalyzer([_vllm_annotation(), op]))
        assert n == 0
        assert "batch_context" not in op


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestEventReplayIRWithTorch:
    def test_get_args_kwargs_cpu(self):
        ir = EventReplayer._get_event_replay_IR(ADD_EVENT, ADD_SCHEMA)
        pos_args, kwargs = EventReplayer._get_args_kwargs(ir, device="cpu")
        assert len(pos_args) == 2
        assert pos_args[0].shape == (2, 4)
        assert kwargs["alpha"] == 1.0
        assert all(t.device.type == "cpu" for t in pos_args)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestEventReplayUtils:
    def test_get_torch_or_raise_returns_torch_module(self):
        from TraceLens.EventReplay.utils import _get_torch_or_raise

        torch = _get_torch_or_raise()
        assert torch.__name__ == "torch"

    def test_get_torch_or_raise_import_error(self):
        import TraceLens.EventReplay.utils as replay_utils

        replay_utils._torch_module = None
        with patch.dict(sys.modules, {"torch": None}), pytest.raises(
            ImportError, match="PyTorch is required"
        ):
            replay_utils._get_torch_or_raise()
        replay_utils._torch_module = None

    def test_build_tensor_cpu_and_summarize(self):
        from TraceLens.EventReplay.utils import build_tensor, summarize_tensor

        cfg = TensorCfg(shape=[3, 5], dtype="float", strides=[5, 1])
        tensor = build_tensor(cfg, device="cpu")
        summary = summarize_tensor(tensor)
        assert "shape=(3, 5)" in summary
        assert "device=cpu" in summary

    def test_build_tensor_rejects_normal_init_for_int(self):
        from TraceLens.EventReplay.utils import build_tensor

        cfg = TensorCfg(shape=[2], dtype="int", strides=[1])
        with pytest.raises(ValueError, match="Cannot initialize tensor"):
            build_tensor(cfg, device="cpu")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestEventReplayerCpu:
    @pytest.fixture(autouse=True)
    def _isolate_custom_init_registry(self):
        saved = EventReplayer._custom_init_registry[:]
        yield
        EventReplayer._custom_init_registry = saved

    def test_event_replayer_lazy_cpu_replay(self):
        replayer = EventReplayer(MM_EVENT, device="cpu", lazy=True)
        result = replayer.replay()
        torch = _require_torch()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 16)
        assert hasattr(replayer, "args")

    def test_get_repro_info_serializes_tensor_cfg(self):
        replayer = EventReplayer(MM_EVENT, device="cpu", lazy=True)
        info = replayer.get_repro_info()
        assert info["op_name"] == "aten::mm"
        pos0 = info["replay_ir"]["list_pos_args"][0]["value"]
        assert pos0["shape"] == [4, 8]
        assert pos0["dtype"] == "c10::BFloat16"

    def test_get_repro_info_idempotent_and_does_not_mutate_ir(self):
        replayer = EventReplayer(MM_EVENT, device="cpu", lazy=True)
        assert replayer.get_repro_info() == replayer.get_repro_info()
        for arg in replayer.event_replay_IR["list_pos_args"]:
            if arg["arg_type"].startswith("Tensor"):
                assert isinstance(arg["value"], TensorCfg)
        torch = _require_torch()
        result = replayer.replay()
        assert isinstance(result, torch.Tensor)

    def test_lazy_custom_init_sees_args(self):
        accessed = {}

        class ProbeInit(CustomInit):
            op_patterns = ["aten::mm"]

            def initialize(self, replayer, **kwargs):
                accessed["args"] = replayer.args
                accessed["kwargs"] = replayer.kwargs
                return None

        EventReplayer.register_custom_init(ProbeInit())
        EventReplayer(MM_EVENT, device="cpu", lazy=True, auto_init=True).replay()
        assert "args" in accessed
        assert len(accessed["args"]) == 2

    def test_lazy_custom_init_reruns_after_rebuild(self):
        seen_ids = []

        class ProbeInit(CustomInit):
            op_patterns = ["aten::mm"]

            def initialize(self, replayer, **kwargs):
                seen_ids.append(id(replayer.args[0]))

        EventReplayer.register_custom_init(ProbeInit())
        replayer = EventReplayer(MM_EVENT, device="cpu", lazy=True, auto_init=True)
        replayer.replay()
        replayer.replay()
        assert len(seen_ids) == 2
        assert seen_ids[0] != seen_ids[1]

    def test_eager_custom_init_runs_once(self):
        log = []

        class ProbeInit(CustomInit):
            op_patterns = ["aten::mm"]

            def initialize(self, replayer, **kwargs):
                log.append("init")

        EventReplayer.register_custom_init(ProbeInit())
        replayer = EventReplayer(MM_EVENT, device="cpu", lazy=False, auto_init=True)
        replayer.replay()
        replayer.replay()
        assert log == ["init"]

    def test_first_matching_custom_init_wins(self):
        log = []

        class InitA(CustomInit):
            op_patterns = ["aten::mm"]

            def initialize(self, replayer, **kwargs):
                log.append("A")

        class InitB(CustomInit):
            op_patterns = ["aten::mm"]

            def initialize(self, replayer, **kwargs):
                log.append("B")

        EventReplayer._custom_init_registry = [InitA(), InitB()]
        EventReplayer(MM_EVENT, device="cpu", auto_init=True).replay()
        assert log == ["A"]

    def test_register_custom_init_overrides_builtin(self):
        log = []

        class BuiltinInit(CustomInit):
            op_patterns = ["aten::mm"]

            def initialize(self, replayer, **kwargs):
                log.append("builtin")

        class UserInit(CustomInit):
            op_patterns = ["aten::mm"]

            def initialize(self, replayer, **kwargs):
                log.append("user")

        EventReplayer._custom_init_registry = [BuiltinInit()]
        EventReplayer.register_custom_init(UserInit())
        EventReplayer(MM_EVENT, device="cpu", auto_init=True).replay()
        assert log == ["user"]

    def test_auto_init_false_skips_custom_inits(self):
        log = []

        class AlwaysInit(CustomInit):
            op_patterns = ["aten::mm"]

            def initialize(self, replayer, **kwargs):
                log.append("ran")

        EventReplayer._custom_init_registry = [AlwaysInit()]
        EventReplayer(MM_EVENT, device="cpu", auto_init=False).replay()
        assert log == []


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestBatchedReplayHelpers:
    def test_get_args_kwargs_from_ir_cpu(self):
        _require_torch()
        from TraceLens.EventReplay import batched_replay as br

        replay_ir = {
            "list_pos_args": [
                {
                    "arg_name": "self",
                    "arg_type": "Tensor",
                    "value": {
                        "shape": [2, 3],
                        "dtype": "float",
                        "strides": [3, 1],
                        "init": "normal",
                    },
                }
            ],
            "list_kwargs": [],
        }
        pos_args, kwargs = br._get_args_kwargs_from_ir(replay_ir, device="cpu")
        assert len(pos_args) == 1
        assert pos_args[0].shape == (2, 3)
        assert kwargs == {}

    def test_resolve_replay_func_aten_mm(self):
        from TraceLens.EventReplay.batched_replay import resolve_replay_func

        func = resolve_replay_func("aten::mm")
        assert callable(func)

    def test_resolve_replay_func_missing_op_raises(self):
        from TraceLens.EventReplay.batched_replay import resolve_replay_func

        with pytest.raises(RuntimeError, match="Cannot resolve"):
            resolve_replay_func("pr607missing::no_such_op")

    def test_resolve_replay_func_python_module_op(self):
        import types
        from TraceLens.EventReplay.batched_replay import resolve_replay_func

        mod = types.ModuleType("pr607climod")

        def add_one(x):
            return x + 1

        mod.add_one = add_one
        sys.modules["pr607climod"] = mod
        try:
            func = resolve_replay_func("pr607climod::add_one")
            assert func is add_one
            assert func(3) == 4
        finally:
            sys.modules.pop("pr607climod", None)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestEventReplayGpu:
    def test_build_tensor_on_cuda(self):
        _require_cuda_gpu()
        from TraceLens.EventReplay.utils import build_tensor

        cfg = TensorCfg(shape=[4, 4], dtype="c10::BFloat16", strides=[4, 1])
        tensor = build_tensor(cfg, device="cuda")
        assert tensor.device.type == "cuda"

    def test_event_replayer_cuda_replay(self):
        _require_cuda_gpu()
        replayer = EventReplayer(MM_EVENT, device="cuda", lazy=True)
        replayer.replay()

    def test_benchmark_func_cuda(self):
        torch = _require_cuda_gpu()
        from TraceLens.EventReplay.utils import benchmark_func

        a = torch.randn(8, 8, device="cuda", dtype=torch.float16)
        b = torch.randn(8, 8, device="cuda", dtype=torch.float16)

        def matmul():
            torch.matmul(a, b)

        metrics = benchmark_func(
            matmul, device=torch.device("cuda"), warmup=1, avg_steps=2
        )
        assert metrics["mean_us"] > 0
