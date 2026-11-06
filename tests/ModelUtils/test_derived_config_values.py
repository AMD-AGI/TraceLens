###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A config class derives what the checkpoint never writes down.

DeepSeek ships neither ``layer_types`` nor ``mlp_layer_types``; its config
class expands them in ``__post_init__`` into one entry PER LAYER -- 43 of each,
``['sliding_attention', 'sliding_attention', 'compressed_sparse_attention',
...]`` and ``3x 'hash_moe'`` then ``'moe'``. Those lists are the only statement
of what an individual layer IS, so without them nothing read from source can
tell layer 2 from layer 3, and a per-layer read like
``config.layer_types[layer_idx]`` stays unresolved however good the parser is.

Instantiating a CONFIG is far cheaper, and far more likely to succeed, than
instantiating a model on a meta device -- so taking the lists from one narrows
that dependency rather than adding another.

Only PER-LAYER lists are taken. The first draft merged everything the live
config carried, and that is how ``id2label`` -- keyed by ``int`` on a config
OBJECT, by string in JSON -- reached a parser that assumed string keys and
crashed the entire GPT-2 build.
"""

from __future__ import annotations

from TraceLens.ModelUtils.extract import (
    _config_has_per_layer_typing,
    _with_derived_config_values,
)


class _Live:
    """Stands in for a config object: attributes, like the real one."""

    def __init__(self, **values):
        self.__dict__.update(values)


class TestOnlyThePerLayerLists:
    def test_a_list_with_one_entry_per_layer_is_taken(self, monkeypatch) -> None:
        live = _Live(
            num_hidden_layers=3,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        )
        out = _patched(monkeypatch, {"num_hidden_layers": 3}, live)
        assert out["layer_types"] == [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]

    def test_a_list_of_another_length_is_not(self, monkeypatch) -> None:
        # Says nothing about any one layer.
        live = _Live(num_hidden_layers=3, architectures=["SomeForCausalLM"])
        out = _patched(monkeypatch, {"num_hidden_layers": 3}, live)
        assert "architectures" not in out

    def test_a_scalar_is_not(self, monkeypatch) -> None:
        live = _Live(num_hidden_layers=3, hidden_size=4096)
        out = _patched(monkeypatch, {"num_hidden_layers": 3}, live)
        assert "hidden_size" not in out

    def test_a_dict_keyed_by_int_is_not(self, monkeypatch) -> None:
        # `id2label` is keyed by int on an object and by string in JSON; taking
        # it crashed a parser that assumed `key.lower()` works.
        live = _Live(num_hidden_layers=2, id2label={0: "LABEL_0", 1: "LABEL_1"})
        out = _patched(monkeypatch, {"num_hidden_layers": 2}, live)
        assert "id2label" not in out

    def test_what_the_checkpoint_states_is_never_overwritten(self, monkeypatch) -> None:
        live = _Live(num_hidden_layers=2, layer_types=["a", "b"])
        out = _patched(
            monkeypatch, {"num_hidden_layers": 2, "layer_types": ["x", "y"]}, live
        )
        assert out["layer_types"] == ["x", "y"]

    def test_no_checkpoint_leaves_the_config_alone(self) -> None:
        config = {"num_hidden_layers": 2}
        assert _with_derived_config_values(config, None) is config


class TestAConfigKeyIsNotAlwaysAString:
    """The crash that taught the lesson, pinned on the function that crashed."""

    def test_an_int_keyed_entry_does_not_raise(self) -> None:
        config = {"id2label": {0: "LABEL_0"}, "num_hidden_layers": 2}
        assert _config_has_per_layer_typing(config) is False

    def test_and_a_real_per_layer_list_is_still_found(self) -> None:
        config = {"id2label": {0: "LABEL_0"}, "moe_layer_freq": [0, 1, 1]}
        assert _config_has_per_layer_typing(config) is True


def _patched(monkeypatch, config, live):
    """Run the filler against *live* without touching the hub."""
    import transformers

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return live

    monkeypatch.setattr(transformers, "AutoConfig", _AutoConfig, raising=False)
    return _with_derived_config_values(config, "some/checkpoint")
