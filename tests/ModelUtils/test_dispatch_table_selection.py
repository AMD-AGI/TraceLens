###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A runtime table subscripted by a config value names what it selects.

``attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]``
picks its callee at run time, so the subscript on its own names nothing -- which
is why such a call is recorded as a dispatch rather than a kernel. But the key is
a config value, and once the config is resolved (including whatever the model's
own constructor writes back onto it) the key is a plain string. The table is then
being asked for an entry the source does name, and the node can say which.

Nothing here knows what any table holds, or which table it is. A key the config
does not answer stays unresolved rather than being guessed: naming the wrong
entry would draw a kernel the model does not run.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.ast_analyze import _config_selected_entry_name


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


class TestWhatItResolves:
    def test_a_config_attribute_key(self) -> None:
        assert (
            _config_selected_entry_name(
                _expr("ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]"),
                {"_attn_implementation": "flash_attention_2"},
            )
            == "flash_attention_2"
        )

    def test_any_table_at_all(self) -> None:
        """The rule is about the shape of the lookup, not the table's name."""
        assert (
            _config_selected_entry_name(
                _expr("SOME_OTHER_REGISTRY[config.hidden_act]"),
                {"hidden_act": "silu"},
            )
            == "silu"
        )

    def test_a_literal_key(self) -> None:
        assert (
            _config_selected_entry_name(
                _expr("TABLE['sdpa']"), {"_attn_implementation": "eager"}
            )
            == "sdpa"
        )

    def test_a_bare_name_key(self) -> None:
        assert (
            _config_selected_entry_name(_expr("TABLE[impl]"), {"impl": "eager"})
            == "eager"
        )

    def test_the_written_down_fallback_argument(self) -> None:
        """``get_interface(key, eager_attention_forward)`` names its fallback."""
        assert (
            _config_selected_entry_name(
                _expr("TABLE.get_interface(self.config.impl, eager_attention_forward)"),
                {},
            )
            == "eager_attention_forward"
        )

    def test_bound_to_an_implementation_outright(self) -> None:
        assert (
            _config_selected_entry_name(_expr("eager_attention_forward"), {})
            == "eager_attention_forward"
        )

    def test_surrounding_space_is_dropped(self) -> None:
        assert (
            _config_selected_entry_name(_expr("TABLE[config.impl]"), {"impl": "  x  "})
            == "x"
        )


class TestWhatStaysUnresolved:
    def test_a_key_the_config_does_not_answer(self) -> None:
        assert (
            _config_selected_entry_name(
                _expr("TABLE[self.config._attn_implementation]"), {}
            )
            is None
        )

    def test_an_empty_config_value(self) -> None:
        assert (
            _config_selected_entry_name(_expr("TABLE[config.impl]"), {"impl": "   "})
            is None
        )

    def test_a_non_string_config_value(self) -> None:
        """A table keyed by a number is not naming an implementation."""
        assert (
            _config_selected_entry_name(_expr("TABLE[config.layers]"), {"layers": 32})
            is None
        )

    def test_a_computed_key(self) -> None:
        assert (
            _config_selected_entry_name(_expr("TABLE[a + b]"), {"a": "x", "b": "y"})
            is None
        )

    def test_a_binding_that_names_nothing(self) -> None:
        """An attribute read names no implementation this can resolve."""
        for source in ("self.attn", "a.b.c"):
            assert _config_selected_entry_name(_expr(source), {"impl": "x"}) is None

    def test_a_call_handing_over_several_functions(self) -> None:
        """Ambiguous -- two candidates, so neither is the one it runs."""
        assert _config_selected_entry_name(_expr("pick(eager, flash)"), {}) is None

    def test_no_binding_at_all(self) -> None:
        assert _config_selected_entry_name(None, {"impl": "x"}) is None

    def test_no_config_at_all(self) -> None:
        assert _config_selected_entry_name(_expr("TABLE[config.impl]"), None) is None
