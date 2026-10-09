###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A model's helpers are read at the commit its modeling file is read at.

The modeling file for a transformers-native model is fetched at a pinned commit,
but everything it imports -- ``vision_utils``, ``masking_utils``,
``activations``, even its own ``configuration_*`` -- resolved through
``sys.path`` to whatever version of transformers happened to be INSTALLED. That
is two revisions of one library describing one model, and helpers are where
several of a graph's frames come from: GLM's ``get_vision_position_ids`` frame
is defined in ``transformers/vision_utils.py``.

Measured before the fix: 27 modules per model came from installed transformers.

Where the repo keeps its top-level packages is not configured anywhere. It is
derived: the same file is named twice, as a repo path and as a dotted module
(``src/transformers/models/x/modeling_x.py`` against
``transformers.models.x.modeling_x``), so stripping as many trailing path
segments as the module has dotted parts leaves the prefix.
"""

from __future__ import annotations

import pytest

from TraceLens.ModelUtils import github as G
from TraceLens.ModelUtils.github import (
    cached_file_path,
    parse_github_url,
    pinned_module_origin,
    register_pinned_module_source,
)

MODELING = "src/transformers/models/glm5_next/modeling_glm5_next.py"
MODULE = "transformers.models.glm5_next.modeling_glm5_next"


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """A registered repo whose files live under a temporary cache."""
    monkeypatch.setattr(G, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(G, "_PINNED_MODULE_SOURCES", [])
    monkeypatch.setattr(G, "_PINNED_MODULE_MISSES", set())
    ref = parse_github_url("github:huggingface/transformers@abc123")
    register_pinned_module_source(ref, MODELING, MODULE)
    return ref


def _place(ref, subpath: str) -> None:
    path = cached_file_path(ref, subpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n", encoding="utf-8")


class TestThePrefixIsDerivedNotConfigured:
    def test_a_helper_resolves_inside_the_pinned_repo(self, pinned) -> None:
        """``transformers.vision_utils`` is ``src/transformers/vision_utils.py``."""
        _place(pinned, "src/transformers/vision_utils.py")
        origin = pinned_module_origin("transformers.vision_utils")
        assert origin is not None
        assert origin.endswith("src/transformers/vision_utils.py")

    def test_a_package_resolves_through_its_init(self, pinned) -> None:
        _place(pinned, "src/transformers/utils/__init__.py")
        origin = pinned_module_origin("transformers.utils")
        assert origin is not None
        assert origin.endswith("src/transformers/utils/__init__.py")

    def test_the_model_s_own_configuration_is_pinned_too(self, pinned) -> None:
        _place(pinned, "src/transformers/models/glm5_next/configuration_glm5_next.py")
        origin = pinned_module_origin(
            "transformers.models.glm5_next.configuration_glm5_next"
        )
        assert origin is not None
        assert origin.endswith("configuration_glm5_next.py")


class TestWhatItDoesNotClaim:
    def test_nothing_registered_resolves_nothing(self, tmp_path, monkeypatch) -> None:
        """So the caller falls back to the installed library."""
        monkeypatch.setattr(G, "_PINNED_MODULE_SOURCES", [])
        monkeypatch.setattr(G, "_PINNED_MODULE_MISSES", set())
        assert pinned_module_origin("transformers.vision_utils") is None

    def test_a_module_the_repo_lacks_falls_back(self, pinned, monkeypatch) -> None:
        """Best effort per module -- it must not invent a path."""
        monkeypatch.setattr(G, "fetch_github_file", lambda *a, **k: None)
        assert pinned_module_origin("transformers.not_published") is None

    def test_an_empty_name_is_not_a_module(self, pinned) -> None:
        assert pinned_module_origin("") is None

    def test_a_path_shorter_than_its_module_registers_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """A mismatched pair cannot say where the packages start."""
        monkeypatch.setattr(G, "CACHE_ROOT", tmp_path)
        monkeypatch.setattr(G, "_PINNED_MODULE_SOURCES", [])
        ref = parse_github_url("github:x/y@z")
        register_pinned_module_source(ref, "modeling_x.py", "a.b.c.modeling_x")
        assert G._PINNED_MODULE_SOURCES == []


class TestTheNetworkIsNotRePriced:
    def test_a_cached_file_is_found_without_fetching(self, pinned, monkeypatch) -> None:
        """A warm cache must cost no requests at all."""
        _place(pinned, "src/transformers/vision_utils.py")

        def _forbidden(*args, **kwargs):
            raise AssertionError("fetched a file already on disk")

        monkeypatch.setattr(G, "fetch_github_file", _forbidden)
        assert pinned_module_origin("transformers.vision_utils") is not None

    def test_a_package_does_not_pay_for_the_module_spelling(
        self, pinned, monkeypatch
    ) -> None:
        """``utils`` misses ``utils.py`` every run; the 404 is not cached."""
        _place(pinned, "src/transformers/utils/__init__.py")

        def _forbidden(*args, **kwargs):
            raise AssertionError("spent a request on a known miss")

        monkeypatch.setattr(G, "fetch_github_file", _forbidden)
        assert pinned_module_origin("transformers.utils") is not None

    def test_a_missing_module_is_asked_for_once(self, pinned, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            G, "fetch_github_file", lambda ref, sub, **k: calls.append(sub)
        )
        assert pinned_module_origin("transformers.absent") is None
        before = len(calls)
        assert pinned_module_origin("transformers.absent") is None
        assert len(calls) == before, calls
