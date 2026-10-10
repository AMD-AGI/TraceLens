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

Generation deliberately does NOT pin: exporting a model pulls its current code
and the libraries it currently needs. Only these tests pin, by passing the
revision and library versions explicitly. These tests guard the pins; the
matching runtime guard lives in the ``model_graph_nodes`` fixture, which every
graph test funnels through.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from model_pins import MODEL_PINS, is_pinned, pin_for

_TESTS = Path(__file__).resolve().parent.parent
# Entry points that reach the hub with a model id.
_HUB_ENTRY_POINTS = {"load_model_spec", "model_graph_nodes"}
_FULL_SHA = re.compile(r"\A[0-9a-f]{40}\Z")


class TestThePins:
    def test_every_revision_is_a_full_commit_sha(self) -> None:
        """A branch or tag moves, which is the thing being guarded against."""
        for model_id, pin in MODEL_PINS.items():
            assert _FULL_SHA.match(pin.revision), (model_id, pin.revision)

    def test_every_pin_names_the_transformers_it_was_written_against(self) -> None:
        """The library decides what the modeling code resolves to."""
        for model_id, pin in MODEL_PINS.items():
            assert pin.transformers, model_id

    def test_pinned_libraries_carry_exact_versions(self) -> None:
        """``fla-core`` without a version is not a pin."""
        for model_id, pin in MODEL_PINS.items():
            for requirement in pin.packages:
                assert "==" in requirement, (model_id, requirement)

    def test_only_a_model_with_its_own_environment_pins_libraries(self) -> None:
        """Libraries are installed into a provisioned env; without one there is
        nowhere to put them, and naming them would be decoration."""
        for model_id, pin in MODEL_PINS.items():
            if pin.packages:
                assert pin.own_environment, model_id

    def test_an_unlisted_model_is_not_pinned(self) -> None:
        """Someone exporting their own checkpoint gets their checkpoint."""
        assert pin_for("some-org/some-model") is None
        assert not is_pinned("some-org/some-model")

    def test_nothing_is_not_pinned(self) -> None:
        assert pin_for(None) is None
        assert pin_for("") is None


class TestGenerationIsNotPinned:
    def test_the_library_reads_the_head_by_default(self) -> None:
        """An export draws the model as it is today.

        ``revision`` is a parameter the caller passes, not a table the library
        consults -- so nothing outside these tests is pinned to anything.
        """
        import inspect

        from TraceLens.ModelUtils.loader import load_model_spec
        from TraceLens.ModelUtils.source import resolve_source_files

        for func in (load_model_spec, resolve_source_files):
            assert inspect.signature(func).parameters["revision"].default is None

    def test_no_library_module_imports_the_pins(self) -> None:
        """The pins live with the tests; the product must not reach for them."""
        root = Path(__file__).resolve().parents[2] / "TraceLens"
        offenders = [
            str(path.relative_to(root))
            for path in root.rglob("*.py")
            if "model_pins" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, offenders


class TestAnExplicitRevisionReachesTheHub:
    def test_both_hub_calls_carry_the_revision_they_were_given(
        self, monkeypatch
    ) -> None:
        """A revision nothing passes to the hub is decoration."""
        import sys
        import types

        from TraceLens.ModelUtils import source

        revision = MODEL_PINS["deepseek-ai/DeepSeek-V4-Flash"].revision
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

        source._list_repo_python_files("org/model", revision)
        source._download_repo_files("org/model", ["modeling_x.py"], revision)
        assert seen == [revision] * 2, seen

    def test_asking_for_nothing_asks_the_hub_for_nothing(self, monkeypatch) -> None:
        """What an export does: no revision, so the checkpoint's head answers."""
        import sys
        import types

        from TraceLens.ModelUtils import source

        seen: list[str | None] = []
        module = types.ModuleType("huggingface_hub")
        module.list_repo_files = lambda name, revision=None: (
            seen.append(revision) or ["modeling_x.py"]
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)

        source._list_repo_python_files("org/model")
        assert seen == [None], seen

    def test_a_requested_revision_never_falls_back_to_the_moving_ref(
        self, tmp_path, monkeypatch
    ) -> None:
        """An uncached revision resolves to nothing, so the download fetches it.

        Reading ``refs/main`` instead would hand back different source than the
        caller asked for -- the failure the revision exists to prevent.
        """
        from TraceLens.ModelUtils import source

        model_id = "org/model"
        base = tmp_path / ("models--" + model_id.replace("/", "--"))
        (base / "snapshots" / "cafe").mkdir(parents=True)
        (base / "refs").mkdir(parents=True)
        (base / "refs" / "main").write_text("cafe\n", encoding="utf-8")
        monkeypatch.setattr(source, "_hub_cache_root", lambda: tmp_path)

        assert source._hub_snapshot_root(model_id, "0" * 40) is None
        assert (
            source._hub_snapshot_root(model_id, "cafe") == base / "snapshots" / "cafe"
        )
        # No revision: the head, as an export reads it.
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
