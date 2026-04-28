###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for EventReplay core functionality.

All tests use CPU-only ops (aten::mm) so they run without a GPU.
Run from the repo root:
    python -m pytest TraceLens/EventReplay/test_event_replay.py -v
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from TraceLens.EventReplay.event_replay import EventReplayer  # noqa: E402
from TraceLens.EventReplay.custom_inits import CustomInit  # noqa: E402
from TraceLens.EventReplay.utils import TensorCfg  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mm_event(M=4, K=8, N=16):
    """Minimal profiler event dict for aten::mm  (M x K) @ (K x N)."""
    return {
        "name": "aten::mm",
        "args": {
            "Input Dims": [[M, K], [K, N]],
            "Input type": ["float", "float"],
            "Input Strides": [[K, 1], [N, 1]],
            "Concrete Inputs": ["", ""],
        },
    }


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save and restore the global custom-init registry around every test."""
    saved = EventReplayer._custom_init_registry[:]
    yield
    EventReplayer._custom_init_registry = saved


# ---------------------------------------------------------------------------
# BUG-1: lazy=True + auto_init=True must not crash
# ---------------------------------------------------------------------------

class TestLazyAutoInit:
    def test_lazy_replay_sets_self_args(self):
        """replay() in lazy mode must populate self.args."""
        replayer = EventReplayer(_make_mm_event(), device="cpu", lazy=True, auto_init=False)
        assert not hasattr(replayer, "args")
        replayer.replay()
        assert hasattr(replayer, "args")
        assert isinstance(replayer.args, list)

    def test_lazy_with_custom_init_no_crash(self):
        """A custom init that reads replayer.args must work in lazy mode."""
        accessed = {}

        class ProbeInit(CustomInit):
            op_patterns = ["aten::mm"]
            def initialize(self, replayer, **kwargs):
                accessed["args"] = replayer.args
                accessed["kwargs"] = replayer.kwargs
                return "[probe] ok"

        EventReplayer.register_custom_init(ProbeInit())
        replayer = EventReplayer(_make_mm_event(), device="cpu", lazy=True, auto_init=True)
        replayer.replay()
        assert "args" in accessed
        assert len(accessed["args"]) == 2  # self, mat2


# ---------------------------------------------------------------------------
# BUG-2: get_repro_info() must not corrupt event_replay_IR
# ---------------------------------------------------------------------------

class TestGetReproInfo:
    def test_idempotent(self):
        """Calling get_repro_info() twice must produce identical output."""
        replayer = EventReplayer(_make_mm_event(), device="cpu", lazy=True)
        assert replayer.get_repro_info() == replayer.get_repro_info()

    def test_does_not_mutate_ir(self):
        """TensorCfg objects in the IR must survive get_repro_info()."""
        replayer = EventReplayer(_make_mm_event(), device="cpu", lazy=True)
        replayer.get_repro_info()
        for arg in replayer.event_replay_IR["list_pos_args"]:
            if arg["arg_type"].startswith("Tensor"):
                assert isinstance(arg["value"], TensorCfg), (
                    f"arg '{arg['arg_name']}' is {type(arg['value'])}, expected TensorCfg"
                )

    def test_replay_works_after_get_repro_info(self):
        """replay() must succeed after get_repro_info() (IR still intact)."""
        replayer = EventReplayer(_make_mm_event(), device="cpu", lazy=True)
        replayer.get_repro_info()
        result = replayer.replay()
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# CLAIM-1: first-match-wins (only one init runs)
# ---------------------------------------------------------------------------

class TestFirstMatchWins:
    def test_only_first_matching_init_runs(self):
        """When two inits match, only the first registered one executes."""
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
        EventReplayer(_make_mm_event(), device="cpu", auto_init=True).replay()
        assert log == ["A"]


# ---------------------------------------------------------------------------
# CLAIM-4: replay() returns the op result
# ---------------------------------------------------------------------------

class TestReplayReturn:
    def test_returns_tensor(self):
        """replay() of aten::mm must return a correctly-shaped Tensor."""
        replayer = EventReplayer(_make_mm_event(M=4, K=8, N=16), device="cpu")
        result = replayer.replay()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 16)

    def test_returns_tensor_lazy(self):
        """Lazy replay must also return the result."""
        result = EventReplayer(_make_mm_event(), device="cpu", lazy=True).replay()
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# Exact name matching for op_patterns
# ---------------------------------------------------------------------------

class TestExactNameMatching:
    def test_exact_match_hits(self):
        """op_patterns=["aten::mm"] matches event name "aten::mm"."""
        matched = []

        class ExactInit(CustomInit):
            op_patterns = ["aten::mm"]
            def initialize(self, replayer, **kwargs):
                matched.append(True)

        EventReplayer._custom_init_registry = [ExactInit()]
        EventReplayer(_make_mm_event(), device="cpu", auto_init=True).replay()
        assert matched == [True]

    def test_substring_does_not_match(self):
        """op_patterns=["mm"] must NOT match "aten::mm" (exact only)."""
        matched = []

        class SubstringInit(CustomInit):
            op_patterns = ["mm"]
            def initialize(self, replayer, **kwargs):
                matched.append(True)

        EventReplayer._custom_init_registry = [SubstringInit()]
        EventReplayer(_make_mm_event(), device="cpu", auto_init=True).replay()
        assert matched == []


# ---------------------------------------------------------------------------
# auto_init=False skips all inits
# ---------------------------------------------------------------------------

class TestAutoInitDisabled:
    def test_no_init_runs_when_disabled(self):
        log = []

        class AlwaysInit(CustomInit):
            op_patterns = ["aten::mm"]
            def initialize(self, replayer, **kwargs):
                log.append("ran")

        EventReplayer._custom_init_registry = [AlwaysInit()]
        EventReplayer(_make_mm_event(), device="cpu", auto_init=False).replay()
        assert log == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
