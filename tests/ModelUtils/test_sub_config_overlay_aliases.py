###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A sub-config answers the canonical names its config OBJECT would.

``Glm5NextVisionConfig`` stores only ``num_heads``, yet the object answers
``config.num_attention_heads`` because the class declares::

    attribute_map = {"num_attention_heads": "num_heads"}

A plain dict does not. So overlaying that sub-config onto a parent which DOES
define ``num_attention_heads`` (the text tower's 64) lets the parent leak into
the vision tower, which has 16 heads.

This used to be bridged by a list of names a config might store a value under.
Such a list claims a name globally, and a name is not global: ``n_heads`` was on
it as another word for the attention head count, while in GLM it is the sparse
indexer's own (``self.n_heads = config.index_n_heads``, 32). Reading the model's
declaration states exactly the renames that model makes, per class.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from TraceLens.ModelUtils.config_resolve import (
    apply_config_attribute_aliases,
    declared_config_aliases,
)

CONFIG_SOURCE = """
class DemoVisionConfig:
    attribute_map = {"num_attention_heads": "num_heads"}
"""


def _source_dir(tmp_path: Path, body: str = CONFIG_SOURCE) -> list[Path]:
    directory = tmp_path / "demo"
    directory.mkdir()
    (directory / "configuration_demo.py").write_text(
        textwrap.dedent(body), encoding="utf-8"
    )
    modeling = directory / "modeling_demo.py"
    modeling.write_text("", encoding="utf-8")
    return [modeling]


class TestTheOverlayCarriesTheCanonicalName:
    def test_the_sub_config_gains_the_name_it_answers_to(self, tmp_path: Path) -> None:
        declared = declared_config_aliases(_source_dir(tmp_path))
        overlaid = apply_config_attribute_aliases({"num_heads": 16}, declared)
        assert overlaid["num_attention_heads"] == 16

    def test_the_parent_cannot_leak_through_it(self, tmp_path: Path) -> None:
        """The whole point: text 64 must not reach a 16-head vision tower."""
        declared = declared_config_aliases(_source_dir(tmp_path))
        parent = {"num_attention_heads": 64, "hidden_size": 4096}
        vision = apply_config_attribute_aliases({"num_heads": 16}, declared)
        assert {**parent, **vision}["num_attention_heads"] == 16

    def test_a_value_the_sub_config_states_itself_wins(self, tmp_path: Path) -> None:
        declared = declared_config_aliases(_source_dir(tmp_path))
        overlaid = apply_config_attribute_aliases(
            {"num_heads": 16, "num_attention_heads": 12}, declared
        )
        assert overlaid["num_attention_heads"] == 12

    def test_the_input_is_left_alone(self, tmp_path: Path) -> None:
        declared = declared_config_aliases(_source_dir(tmp_path))
        overlay = {"num_heads": 16}
        apply_config_attribute_aliases(overlay, declared)
        assert overlay == {"num_heads": 16}


class TestAModelThatRenamesNothing:
    def test_nothing_declared_is_a_plain_copy(self, tmp_path: Path) -> None:
        declared = declared_config_aliases(
            _source_dir(tmp_path, "class Plain:\n    x = 1\n")
        )
        assert declared == {}
        assert apply_config_attribute_aliases({"num_heads": 16}, declared) == {
            "num_heads": 16
        }

    def test_no_declaration_at_all_is_accepted(self) -> None:
        assert apply_config_attribute_aliases({"num_heads": 16}) == {"num_heads": 16}

    def test_a_name_this_model_does_not_rename_is_not_invented(
        self, tmp_path: Path
    ) -> None:
        """``n_heads`` is not a synonym the overlay may assume."""
        declared = declared_config_aliases(_source_dir(tmp_path))
        overlaid = apply_config_attribute_aliases({"n_heads": 32}, declared)
        assert "num_attention_heads" not in overlaid
