###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Modeling code reads the config OBJECT's names; the dict is given them.

``config.num_local_experts`` is answered by a config object that stores
``n_routed_experts``, because the class says so in ``attribute_map``. A plain
dict does not answer it, so the analyser used to consult a hardcoded list of
synonyms a config *might* use -- the same guess-the-spelling approach that let
``n_heads`` shadow GLM's sparse indexer head count, kept in a second place.

Giving the dict the names the object answers to removes the need for the list:
every model in the suite declares these renames itself, and a model that does
not is reading names its own config already has.

A second table went with it: ``linear_lower_bound`` was mapped by hand to
``("linear_attn_config", "gate_lower_bound")`` -- one model's nested layout
written into the analyser. The config class declares that default, so it
resolves without the bespoke path.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from TraceLens.ModelUtils.ast_analyze import analyze_source
from TraceLens.ModelUtils.config_resolve import (
    apply_config_attribute_aliases,
    declared_config_aliases,
)

CONFIG_SOURCE = """
class DemoConfig:
    model_type = "demo"
    attribute_map = {"num_local_experts": "n_routed_experts"}
"""

MODEL_SOURCE = """
import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.num_local_experts, 8))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)
"""


def _sources(tmp_path: Path) -> list[Path]:
    directory = tmp_path / "demo"
    directory.mkdir()
    (directory / "configuration_demo.py").write_text(
        textwrap.dedent(CONFIG_SOURCE), encoding="utf-8"
    )
    modeling = directory / "modeling_demo.py"
    modeling.write_text("", encoding="utf-8")
    return [modeling]


class TestTheDictAnswersWhatTheObjectWould:
    def test_the_read_name_resolves_from_the_stored_key(self, tmp_path: Path) -> None:
        declared = declared_config_aliases(_sources(tmp_path))
        config = apply_config_attribute_aliases({"n_routed_experts": 288}, declared)
        assert config["num_local_experts"] == 288

    def test_the_analyser_then_reads_it(self, tmp_path: Path) -> None:
        """``config.num_local_experts`` with no synonym list anywhere."""
        import ast

        from TraceLens.ModelUtils.ast_analyze import _config_value

        declared = declared_config_aliases(_sources(tmp_path))
        config = apply_config_attribute_aliases(
            {"n_routed_experts": 288, "hidden_size": 8}, declared
        )
        read = ast.parse("config.num_local_experts", mode="eval").body
        assert _config_value(read, config, {}) == 288

    def test_the_model_still_analyses(self, tmp_path: Path) -> None:
        declared = declared_config_aliases(_sources(tmp_path))
        config = apply_config_attribute_aliases(
            {"n_routed_experts": 288, "hidden_size": 8}, declared
        )
        analysis = analyze_source(MODEL_SOURCE, config=config)
        assert "Router" in analysis.class_registry


class TestNoSynonymTableRemains:
    def test_the_guessed_expert_spellings_are_gone(self) -> None:
        from TraceLens.ModelUtils import ast_analyze

        assert not hasattr(ast_analyze, "_CONFIG_ATTR_ALIASES")

    def test_the_hand_written_nested_path_is_gone(self) -> None:
        """One model's nested config layout is not the analyser's business."""
        from TraceLens.ModelUtils import ast_analyze

        assert not hasattr(ast_analyze, "_CONFIG_NESTED_ALIASES")

    def test_an_undeclared_name_resolves_nothing(self, tmp_path: Path) -> None:
        """Without a declaration there is no guess to fall back on."""
        declared = declared_config_aliases(_sources(tmp_path))
        config = apply_config_attribute_aliases({"moe_num_experts": 288}, declared)
        assert "num_local_experts" not in config
