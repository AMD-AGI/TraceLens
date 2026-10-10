###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A key a checkpoint omits still has the value its config class declares.

``config.json`` records only what differs from the defaults, and the defaults
live in the configuration class::

    class GPT2Config(PreTrainedConfig):
        reorder_and_upcast_attn: bool = False
        add_cross_attention: bool = False

Read without them, those keys are ``None`` -- which is not ``False``. A branch
the model CANNOT take then looks merely unresolved, so it gets drawn: GPT-2
rendered an upcast-attention path its own config switches off, and that path was
the source of every type-check warning and every dead node in the model.

The defaults have to be SCOPED or they repeat the mistake they fix. A flat merge
claims a name globally, and a name is not global:

* ``Glm5NextVisionConfig`` defaults ``image_size`` to 336 while GLM's checkpoint
  states 448 in its ``vision_config``; a top-level write shadows the real value.
* A vision tower's ``num_heads`` of 16 is not the text tower's 64.

Each config class states its ``model_type`` and so does each section of the
checkpoint, so the two are matched rather than guessed -- and a key the section
already states at any depth is left alone.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from TraceLens.ModelUtils.config_resolve import declared_config_defaults
from TraceLens.ModelUtils.extract import _with_declared_defaults

SOURCE = """
class DemoConfig:
    model_type = "demo"
    attribute_map = {"hidden_size": "n_embd"}
    keys_to_ignore_at_inference = ["past_key_values"]
    n_embd: int = 768
    add_cross_attention: bool = False
    reorder_and_upcast_attn: bool = False


class DemoVisionConfig:
    model_type = "demo_vision"
    num_heads: int = 16
    image_size: int = 336


class Unplaceable:
    some_default: int = 7
"""


def _sources(tmp_path: Path, body: str = SOURCE) -> list[Path]:
    directory = tmp_path / "demo"
    directory.mkdir()
    (directory / "configuration_demo.py").write_text(
        textwrap.dedent(body), encoding="utf-8"
    )
    modeling = directory / "modeling_demo.py"
    modeling.write_text("", encoding="utf-8")
    return [modeling]


class TestTheDefaultsAreReadPerConfigClass:
    def test_each_class_keeps_its_own(self, tmp_path: Path) -> None:
        declared = declared_config_defaults(_sources(tmp_path))
        assert declared["demo"]["add_cross_attention"] is False
        assert declared["demo_vision"]["num_heads"] == 16

    def test_a_class_that_names_no_config_is_not_placed(self, tmp_path: Path) -> None:
        """Nothing says which section it would belong to."""
        declared = declared_config_defaults(_sources(tmp_path))
        assert all("some_default" not in values for values in declared.values())

    def test_machinery_is_not_mistaken_for_a_value(self, tmp_path: Path) -> None:
        declared = declared_config_defaults(_sources(tmp_path))
        assert "attribute_map" not in declared["demo"]
        assert "keys_to_ignore_at_inference" not in declared["demo"]


class TestApplyingThemToACheckpoint:
    def test_an_omitted_switch_becomes_false_not_none(self, tmp_path: Path) -> None:
        """``None`` is not ``False``: it leaves a dead branch merely unresolved."""
        config = {"model_type": "demo", "n_embd": 768}
        filled = _with_declared_defaults(config, _sources(tmp_path))
        assert filled["add_cross_attention"] is False
        assert filled["reorder_and_upcast_attn"] is False

    def test_what_the_checkpoint_states_is_untouched(self, tmp_path: Path) -> None:
        config = {"model_type": "demo", "add_cross_attention": True}
        filled = _with_declared_defaults(config, _sources(tmp_path))
        assert filled["add_cross_attention"] is True

    def test_a_section_is_filled_from_its_own_class(self, tmp_path: Path) -> None:
        config = {
            "model_type": "demo",
            "vision_config": {"model_type": "demo_vision"},
        }
        filled = _with_declared_defaults(config, _sources(tmp_path))
        assert filled["vision_config"]["num_heads"] == 16
        assert "num_heads" not in filled, "a vision default is not a model-wide one"


class TestWhatMustNotBeShadowed:
    def test_a_nested_value_wins_over_its_class_default(self, tmp_path: Path) -> None:
        """GLM states image_size 448 in vision_config; the class says 336."""
        config = {
            "model_type": "demo",
            "vision_config": {"model_type": "demo_vision", "image_size": 448},
        }
        filled = _with_declared_defaults(config, _sources(tmp_path))
        assert filled["vision_config"]["image_size"] == 448

    def test_a_key_stated_only_deeper_is_not_written_shallower(
        self, tmp_path: Path
    ) -> None:
        """Writing it at the top would outrank the nested registration."""
        config = {
            "model_type": "demo_vision",
            "inner": {"image_size": 448},
        }
        filled = _with_declared_defaults(config, _sources(tmp_path))
        assert "image_size" not in filled or filled["image_size"] == 448

    def test_a_section_naming_no_known_class_is_left_alone(
        self, tmp_path: Path
    ) -> None:
        config = {"model_type": "something_else"}
        assert _with_declared_defaults(config, _sources(tmp_path)) == config
