###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A branch ``__init__`` already decided is not a choice.

DeepSeek's attention appends compressed entries only where the layer has a
compressor::

    kv = self.kv_proj(hidden_states)
    if self.compressor is not None:
        compressed_kv, block_bias = self.compressor(...)
        kv = torch.cat([kv, compressed_kv], dim=2)

Whether `self.compressor` exists is settled in `__init__`, and the variant split
already knows it: `20x_CSACompressor_TopKRouter` always has one,
`2x_no_compressor_HashRouter` never does. Yet the phi over `kv` carried BOTH
arms in every variant, so it published `[B, 1, S, 512]` against the compressor
path's `[B, 1, S + n_windows, 512]` -- and a phi whose arms disagree reports a
shape that is wrong whenever the other arm runs. Here the other arm never runs
at all.

The ABSENCE arm cannot be removed the way the presence arm already is (by
dropping an omitted submodule's ops): it is the PRE-BRANCH value of `kv`, which
the `cat` on the live arm also reads, so killing its producer would take the
live arm with it. The phi's EDGE to it goes instead, leaving one arm to fold.

The surviving arm is identified by CONNECTION -- its producer carries the same
``condition:`` the phi merges on -- not by position in the predecessor list.
"""

from __future__ import annotations

from types import SimpleNamespace

from TraceLens.ModelUtils.ast_analyze import ForwardOperation
from TraceLens.ModelUtils.block_tree import _statically_decided_phi_arms


def _op(attr_name, label, details=(), predecessors=()):
    return ForwardOperation(
        attr_name=attr_name,
        class_name=label,
        label=label,
        details=tuple(details),
        predecessors=tuple(predecessors),
    )


def _cls(operations):
    # The helper reads one attribute; a real ClassStructure needs five unrelated
    # constructor arguments that say nothing about this behaviour.
    return SimpleNamespace(forward_operations={op.attr_name: op for op in operations})


_GUARD = "self.compressor is not None"


def _attention_like():
    return _cls(
        [
            _op("@op_kv", "Linear"),
            _op("@op_cat", "Concat", [f"condition: {_GUARD}"], ["@op_kv"]),
            _op("@op_merge", "Merge", [f"merge: {_GUARD}"], ["@op_cat", "@op_kv"]),
        ]
    )


class TestAVariantThatBuildsTheSubmodule:
    def test_the_absence_arm_is_dropped(self) -> None:
        live = _statically_decided_phi_arms(
            _attention_like(), frozenset({"compressor"})
        )
        assert live == {"@op_merge": ("@op_cat",)}

    def test_the_arm_is_chosen_by_its_condition_not_its_position(self) -> None:
        """Reversing the predecessor order keeps the same surviving arm."""
        structure = _cls(
            [
                _op("@op_kv", "Linear"),
                _op("@op_cat", "Concat", [f"condition: {_GUARD}"], ["@op_kv"]),
                _op(
                    "@op_merge",
                    "Merge",
                    [f"merge: {_GUARD}"],
                    ["@op_kv", "@op_cat"],
                ),
            ]
        )
        live = _statically_decided_phi_arms(structure, frozenset({"compressor"}))
        assert live == {"@op_merge": ("@op_cat",)}


class TestWhenItLeavesThePhiAlone:
    def test_a_submodule_this_variant_does_not_build(self) -> None:
        """Absent is the OTHER pass's job -- it prunes the presence arm's ops."""
        assert _statically_decided_phi_arms(_attention_like(), frozenset()) == {}

    def test_a_phi_on_a_condition_init_does_not_decide(self) -> None:
        """``attention_mask is not None`` is a runtime fact, not a built one."""
        guard = "attention_mask is not None"
        structure = _cls(
            [
                _op("@op_a", "Linear"),
                _op("@op_b", "Concat", [f"condition: {guard}"], ["@op_a"]),
                _op("@op_merge", "Merge", [f"merge: {guard}"], ["@op_b", "@op_a"]),
            ]
        )
        assert _statically_decided_phi_arms(structure, frozenset({"compressor"})) == {}

    def test_a_phi_whose_arms_are_all_guarded(self) -> None:
        """Both arms under the same condition is not this situation; leave it."""
        structure = _cls(
            [
                _op("@op_a", "Linear", [f"condition: {_GUARD}"]),
                _op("@op_b", "Concat", [f"condition: {_GUARD}"]),
                _op("@op_merge", "Merge", [f"merge: {_GUARD}"], ["@op_a", "@op_b"]),
            ]
        )
        assert _statically_decided_phi_arms(structure, frozenset({"compressor"})) == {}

    def test_an_op_that_is_not_a_phi(self) -> None:
        structure = _cls([_op("@op_add", "Add", ["condition: x"], ["@op_a"])])
        assert _statically_decided_phi_arms(structure, frozenset({"compressor"})) == {}
