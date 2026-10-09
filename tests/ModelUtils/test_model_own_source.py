###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A model's own source arrives whole, and the spec says where it is.

Fetching a transformers-native model from GitHub took one file -- the modeling
module -- and left behind the neighbour it opens with::

    from .configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig

That module is part of the model. It states the head counts, the expert counts
and the names the modeling code reads them under; without it those have to come
from whatever transformers version happens to be installed, which is a DIFFERENT
revision from the pinned modeling file we analyse.

Two bounds matter as much as the fetch:

* Only the model's OWN directory. ``from ...cache_utils import Cache`` reaches
  into the library around the model, and following those would fetch
  transformers a file at a time.
* The cache mirrors the repo layout. Flattening ``src/transformers/models/x/``
  into one directory of ``src__transformers__models__x__*.py`` names puts the
  neighbours somewhere no relative import resolves.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from TraceLens.ModelUtils.github import cached_file_path, parse_github_url
from TraceLens.ModelUtils.source import _own_package_imports


class TestWhichNeighboursCount:
    def test_a_sibling_module_is_named(self) -> None:
        source = "from .configuration_glm5_next import Glm5NextConfig\n"
        assert _own_package_imports(source) == ["configuration_glm5_next"]

    def test_the_library_around_the_model_is_not(self) -> None:
        """``...cache_utils`` is transformers, not the model."""
        source = textwrap.dedent("""
            from ...cache_utils import Cache
            from ...activations import ACT2FN
            from ..shared import Helper
            """)
        assert _own_package_imports(source) == []

    def test_the_bare_spelling_names_modules_in_its_names(self) -> None:
        """``from . import x`` puts the module in ``names``, not ``module``."""
        assert _own_package_imports("from . import configuration_x\n") == [
            "configuration_x"
        ]

    def test_each_module_is_named_once(self) -> None:
        source = textwrap.dedent("""
            from .configuration_x import A
            from .configuration_x import B
            """)
        assert _own_package_imports(source) == ["configuration_x"]

    def test_source_that_does_not_parse_yields_nothing(self) -> None:
        assert _own_package_imports("def f(:\n") == []


class TestTheCacheMirrorsTheRepo:
    def test_a_file_keeps_its_directories(self) -> None:
        """So ``from .configuration_x`` resolves beside the modeling module."""
        ref = parse_github_url("github:huggingface/transformers@abc123")
        modeling = cached_file_path(
            ref, "src/transformers/models/glm5_next/modeling_glm5_next.py"
        )
        configuration = cached_file_path(
            ref, "src/transformers/models/glm5_next/configuration_glm5_next.py"
        )
        assert modeling.parent == configuration.parent
        assert modeling.name == "modeling_glm5_next.py", "not a flattened name"


class TestTheSpecSaysWhereTheSourceIs:
    """``code_sources`` holds provenance labels, which name no file."""

    @pytest.mark.parametrize(
        "model",
        ["zai-org/GLM-5.3-Flash", "deepseek-ai/DeepSeek-V4-Flash"],
    )
    def test_the_model_s_configuration_module_is_readable(self, model: str) -> None:
        pytest.importorskip("huggingface_hub")
        from TraceLens.ModelUtils.loader import load_model_spec

        from tests.model_pins import pin_for

        pin = pin_for(model)
        spec = load_model_spec(
            model, detailed=True, revision=pin.revision if pin else None
        )
        assert spec.code_paths, "the spec must say where its source is"
        found = [
            path
            for directory in {Path(p).parent for p in spec.code_paths}
            for path in directory.glob("configuration_*.py")
        ]
        assert found, (model, spec.code_paths)
