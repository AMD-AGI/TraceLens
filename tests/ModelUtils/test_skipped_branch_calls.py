###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A branch that is not taken does not call anything either.

An ``@op_l`` or ``@fn_l`` id in this file is a call key this test builds so
``_call_source_line`` can be checked. It is not a lookup into an upstream
modeling file.


The list of a forward's calls is read off the function's AST, which knows
nothing about which arms ran. So deciding a branch removed the ops inside it
but LEFT the helpers it calls: Kimi kept a ``get_unpad_data`` frame whose
``Nonzero`` and ``Max`` fed nothing at all, because the statements that read
them were never walked. Four dead nodes, in a graph that is supposed to have
none.

A call goes when its key names a line inside a skipped arm. A submodule has no
line in its key, so it goes only when the live walk never recorded it -- one
called in both arms is called whichever arm runs.
"""

from __future__ import annotations

from types import SimpleNamespace

from TraceLens.ModelUtils.ast_analyze import _call_source_line, _without_skipped_calls


def _analysis(spans=(), attrs=frozenset()):
    return SimpleNamespace(skipped_spans=tuple(spans), skipped_attrs=frozenset(attrs))


CALLS = [
    "@fn_l568_get_unpad_data",
    "@fn_l570_rearrange",
    "@op_l569_c28_unsqueeze",
    "q_proj",
    "k_proj",
]


class TestTheLineACallKeyNames:
    def test_a_free_function_frame(self) -> None:
        assert _call_source_line("@fn_l568_get_unpad_data") == 568

    def test_an_operation(self) -> None:
        assert _call_source_line("@op_l569_c28_unsqueeze") == 569

    def test_a_repeated_submodule_call_site(self) -> None:
        assert _call_source_line("rotary_emb@l1773") == 1773

    def test_a_plain_submodule_names_no_line(self) -> None:
        assert _call_source_line("q_proj") is None


class TestWhatASkippedArmTakesWithIt:
    def test_everything_on_its_lines(self) -> None:
        kept = _without_skipped_calls(CALLS, _analysis(spans=[(566, 571)]), set())
        assert kept == ["q_proj", "k_proj"]

    def test_and_nothing_outside_them(self) -> None:
        kept = _without_skipped_calls(CALLS, _analysis(spans=[(600, 700)]), set())
        assert kept == CALLS

    def test_a_submodule_called_only_there(self) -> None:
        kept = _without_skipped_calls(CALLS, _analysis(attrs={"k_proj"}), set())
        assert "k_proj" not in kept and "q_proj" in kept

    def test_but_not_one_the_live_walk_also_recorded(self) -> None:
        # Called in both arms: whichever arm runs, it runs.
        kept = _without_skipped_calls(CALLS, _analysis(attrs={"k_proj"}), {"k_proj"})
        assert "k_proj" in kept

    def test_nothing_skipped_changes_nothing(self) -> None:
        assert _without_skipped_calls(CALLS, _analysis(), set()) is CALLS
