###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A spec field is read through the model's declaration, not a list of spellings.

Building an ``ArchitectureSpec`` read each field through a hardcoded fallback
chain -- ``_get(config, "hidden_size", "n_embd", "d_model")`` and its kin. Those
lists are guesses, and a guess is both too wide and too narrow at once:

* too wide, because it claims a name globally (the same flaw that let ``n_heads``
  shadow GLM's sparse indexer head count);
* too narrow, because nobody thought of ``n_positions``. GPT-2 stores
  ``max_position_embeddings`` under exactly that name, so GPT-2's position count
  did not resolve AT ALL while the list carried two spellings no model uses.

``GPT2Config`` states the whole mapping itself::

    attribute_map = {"hidden_size": "n_embd", "max_position_embeddings":
                     "n_positions", "num_attention_heads": "n_head",
                     "num_hidden_layers": "n_layer"}

Measured across the four pinned models: every one of these fields is present
natively, so none of them ever reached the guess lists.
"""

from __future__ import annotations

from TraceLens.ModelUtils.extract import _declared_get

GPT2 = {
    "hidden_size": "n_embd",
    "max_position_embeddings": "n_positions",
    "num_attention_heads": "n_head",
    "num_hidden_layers": "n_layer",
}


class TestTheDeclarationIsConsulted:
    def test_a_key_stored_under_its_declared_name_resolves(self) -> None:
        config = {"n_embd": 768}
        assert _declared_get(config, "hidden_size", GPT2) == 768

    def test_the_name_no_guess_list_had(self) -> None:
        """``n_positions`` -- the case the hardcoded chain missed entirely."""
        config = {"n_positions": 1024}
        assert _declared_get(config, "max_position_embeddings", GPT2) == 1024

    def test_every_declared_field_resolves(self) -> None:
        config = {"n_embd": 768, "n_positions": 1024, "n_head": 12, "n_layer": 12}
        resolved = {
            canonical: _declared_get(config, canonical, GPT2) for canonical in GPT2
        }
        assert resolved == {
            "hidden_size": 768,
            "max_position_embeddings": 1024,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
        }


class TestWhatTheConfigStatesItselfWins:
    def test_the_canonical_name_is_preferred(self) -> None:
        config = {"hidden_size": 4096, "n_embd": 768}
        assert _declared_get(config, "hidden_size", GPT2) == 4096

    def test_a_declaration_is_not_required(self) -> None:
        assert _declared_get({"hidden_size": 4096}, "hidden_size", {}) == 4096

    def test_an_undeclared_missing_field_stays_missing(self) -> None:
        """GPT-2 has no grouped-query attention, so this is CORRECTLY absent."""
        assert _declared_get({"n_embd": 768}, "num_key_value_heads", GPT2) is None

    def test_a_declaration_pointing_nowhere_resolves_nothing(self) -> None:
        assert _declared_get({}, "hidden_size", GPT2) is None
