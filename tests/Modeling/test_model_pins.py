###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A test that asserts on a model's graph names the revision it describes.

A Hugging Face repo is mutable. ``deepseek-ai/DeepSeek-V4-Flash`` published an
``inference/`` reference implementation and an ``encoding/`` chat-template helper
mid-session; the export began reading them, 2267 nodes became 1472, and 20 tests
went red for a reason that had nothing to do with this repo. A floating
checkpoint makes "upstream published something" and "we broke it" look identical
in a test report.

These tests guard the pin itself. The matching runtime guard lives in the
``model_graph_nodes`` fixture, which every graph test funnels through.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from TraceLens.ModelUtils.model_pins import (
    MODEL_REVISIONS,
    is_pinned,
    pinned_revision,
)

_TESTS = Path(__file__).resolve().parent.parent
# Entry points that reach the hub with a model id.
_HUB_ENTRY_POINTS = {"load_model_spec", "model_graph_nodes"}
_FULL_SHA = re.compile(r"\A[0-9a-f]{40}\Z")


class TestThePins:
    def test_every_pin_is_a_full_commit_sha(self) -> None:
        """A branch or tag moves, which is the thing being guarded against."""
        for model_id, revision in MODEL_REVISIONS.items():
            assert _FULL_SHA.match(revision), (model_id, revision)

    def test_lookup_is_exact(self) -> None:
        assert pinned_revision("deepseek-ai/DeepSeek-V4-Flash") == (
            MODEL_REVISIONS["deepseek-ai/DeepSeek-V4-Flash"]
        )
        assert pinned_revision("deepseek-ai/DeepSeek-V4-Flash ") is not None

    def test_an_unlisted_model_is_not_pinned(self) -> None:
        """Someone exporting their own checkpoint gets their checkpoint."""
        assert pinned_revision("some-org/some-model") is None
        assert not is_pinned("some-org/some-model")

    def test_nothing_is_not_pinned(self) -> None:
        assert pinned_revision(None) is None
        assert pinned_revision("") is None


class TestThePinReachesTheHub:
    def test_both_hub_calls_carry_the_pinned_revision(self, monkeypatch) -> None:
        """A pin nothing passes to the hub is decoration."""
        import sys
        import types

        from TraceLens.ModelUtils import source

        model_id = "deepseek-ai/DeepSeek-V4-Flash"
        seen: list[str | None] = []
        module = types.ModuleType("huggingface_hub")

        def list_repo_files(name: str, revision: str | None = None):
            seen.append(revision)
            return ["modeling_x.py"]

        def hf_hub_download(name: str, filename: str, revision: str | None = None):
            seen.append(revision)
            return "/nonexistent/modeling_x.py"

        module.list_repo_files = list_repo_files
        module.hf_hub_download = hf_hub_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)

        source._list_repo_python_files(model_id)
        source._download_repo_files(model_id, ["modeling_x.py"])
        assert seen == [MODEL_REVISIONS[model_id]] * 2, seen

    def test_a_pinned_model_never_falls_back_to_the_moving_ref(
        self, tmp_path, monkeypatch
    ) -> None:
        """An uncached pin resolves to nothing, so the download fetches it.

        Reading ``refs/main`` instead would quietly undo the pin and hand back a
        different revision than the one the assertions were written against.
        """
        from TraceLens.ModelUtils import model_pins, source

        model_id = "pinned-org/pinned-model"
        base = tmp_path / ("models--" + model_id.replace("/", "--"))
        (base / "snapshots" / "cafe").mkdir(parents=True)
        (base / "refs").mkdir(parents=True)
        (base / "refs" / "main").write_text("cafe\n", encoding="utf-8")
        monkeypatch.setattr(source, "_hub_cache_root", lambda: tmp_path)
        monkeypatch.setitem(model_pins.MODEL_REVISIONS, model_id, "0" * 40)

        assert source._hub_snapshot_root(model_id) is None
        # The same cache resolves fine for the revision that IS pinned.
        monkeypatch.setitem(model_pins.MODEL_REVISIONS, model_id, "cafe")
        assert source._hub_snapshot_root(model_id) == base / "snapshots" / "cafe"


def _literal_model_ids(path: Path) -> set[str]:
    """Model ids passed as literals to a hub entry point in one test file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute) else None
        )
        if name not in _HUB_ENTRY_POINTS or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
    return found


class TestTheSuiteOnlyAssertsOnPinnedModels:
    def test_every_literal_hub_model_in_the_tests_is_pinned(self) -> None:
        """A new model test must bring its revision with it.

        Only ``owner/name`` literals handed straight to a hub entry point count;
        a local directory or a stub id in a unit test is not a hub checkout.
        Indirect ids (a module-level ``_MODEL``, a loop variable) go through the
        ``model_graph_nodes`` fixture, which asserts the same thing at run time.
        """
        unpinned: set[str] = set()
        for path in sorted(_TESTS.rglob("test_*.py")):
            for model_id in _literal_model_ids(path):
                if "/" not in model_id or model_id.startswith("."):
                    continue
                if Path(model_id).exists():
                    continue
                if not is_pinned(model_id):
                    unpinned.add(f"{model_id}  ({path.relative_to(_TESTS)})")
        assert not unpinned, (
            "these tests read a hub model at no fixed revision; add each to "
            "TraceLens/ModelUtils/model_pins.py:\n  " + "\n  ".join(sorted(unpinned))
        )
