###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A rendered shape must not print a symbol that should have resolved.

``B``/``S``/``Img``/``Pv`` vary at run time and are legitimately symbolic.
``N``/``H``/``D`` and the rest are checkpoint CONSTANTS: the model states them
in its config, and the shape machinery records which keys each resolves from.
When one reaches a rendered shape as a bare letter it did not resolve -- and in
a shape a bare letter is indistinguishable from a real runtime axis, so the
reader cannot tell a resolved graph from a broken one. Nothing else catches it:
the existing shape check looks for ``?``, ``[]``, ``-1`` and ``.shape[...]``,
all of which a bare ``N`` passes.

Which symbols are constants is not a list kept in the check. It is every symbol
the shape machinery does not call a runtime axis, read from there, so a symbol
added is covered automatically and a runtime one is never flagged.
"""

from __future__ import annotations

from TraceLens.Visualizer.model_explorer_export.type_check import (
    unresolved_config_dims,
)


def _node(shape: str, node_id: str = "m/@op") -> dict:
    return {"id": node_id, "attrs": [{"key": "output_shape", "value": shape}]}


class TestAConfigConstantMustResolve:
    def test_a_bare_head_count_is_reported(self) -> None:
        messages = unresolved_config_dims([_node("[B, N, S, D] bfloat16")])
        assert messages, "a bare 'N' is an unresolved head count"

    def test_the_message_says_what_it_is_and_where_to_look(self) -> None:
        """So the reader knows where to fix it, not just that it is broken.

        It no longer quotes a list of key spellings: there is no such list to
        quote, because the names a model reads its constants under come from the
        model's own config class.
        """
        (message,) = [
            m for m in unresolved_config_dims([_node("[B, N, S] f32")]) if "'N'" in m
        ]
        assert "heads" in message, "which dimension failed"
        assert "attribute_map" in message, "where a rename would be declared"
        assert "m/@op" in message, "and which node"

    def test_every_offending_dim_is_found_not_every_other_one(self) -> None:
        """``[B, N, S, D]`` shares the comma between neighbours.

        A pattern that CONSUMES the delimiters matches every other dim, so half
        the offenders go unreported -- which is how a first draft of this check
        passed a shape carrying two unresolved symbols.
        """
        symbols = {
            m.split("bare symbol ")[1].split(" ")[0].strip("'")
            for m in unresolved_config_dims([_node("[B, N, S, D] bfloat16")])
        }
        assert symbols == {"N", "D"}, symbols


class TestRuntimeAxesAreNeverFlagged:
    def test_batch_sequence_images_and_patches_are_legitimate(self) -> None:
        assert unresolved_config_dims([_node("[B, S, Img, Pv] bfloat16")]) == []

    def test_a_resolved_shape_is_silent(self) -> None:
        assert unresolved_config_dims([_node("[B, 32, S, 128] bfloat16")]) == []

    def test_a_node_without_a_shape_is_skipped(self) -> None:
        assert unresolved_config_dims([{"id": "m/@x", "attrs": []}]) == []


class TestThePartitionComesFromTheShapeMachinery:
    def test_constants_are_every_symbol_that_is_not_a_runtime_axis(self) -> None:
        """Not a letter list kept in the check, so it follows the enum."""
        from TraceLens.ModelUtils.shape_inference import _RUNTIME_SYMBOLS, Symbol

        config = {s.value for s in Symbol if s not in _RUNTIME_SYMBOLS}
        runtime = {s.value for s in Symbol if s in _RUNTIME_SYMBOLS}
        assert runtime == {"B", "S", "Pv", "Img"}, runtime
        for value in config:
            assert unresolved_config_dims([_node(f"[{value}, 4]")]), value
        for value in runtime:
            assert unresolved_config_dims([_node(f"[{value}, 4]")]) == [], value
