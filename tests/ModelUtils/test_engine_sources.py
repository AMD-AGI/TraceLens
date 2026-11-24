###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An engine's releases and dependencies are found from its own sources."""

import subprocess
import types

from TraceLens.ModelUtils import engine_sources
from TraceLens.ModelUtils.engine_sources import linked_repository, release_tags


def test_release_tags_are_newest_first_without_prereleases(monkeypatch):
    listing = "\n".join(
        f"abc\trefs/tags/{tag}"
        for tag in (
            "v0.5.9",
            "v0.5.10",
            "v0.5.10^{}",
            "v0.6.0rc1",
            "nightly",
            "v0.5.21",
        )
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(stdout=listing)
    )
    assert release_tags("o", "r") == ["v0.5.21", "v0.5.10", "v0.5.9"]


def test_a_dependency_s_repository_is_the_one_the_sources_link_under_its_name(tmp_path):
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "README.md").write_text(
        "Kernels come from https://github.com/ROCm/FlyDSL and torch from github.com/pytorch/pytorch.\n"
    )
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "setup.py").write_text('FLYDSL = "flydsl==0.3.2"\n')
    assert linked_repository("flydsl", [tmp_path / "engine", tmp_path / "lib"]) == (
        "ROCm",
        "FlyDSL",
    )
    assert linked_repository("absent", [tmp_path / "engine"]) is None


def test_a_dependency_linked_to_two_repositories_is_not_guessed(tmp_path):
    (tmp_path / "a.md").write_text(
        "flydsl: github.com/ROCm/FlyDSL or github.com/fork/flydsl\n"
    )
    assert linked_repository("flydsl", [tmp_path]) is None


def test_a_pinned_dependency_is_fetched_at_its_release_tag(monkeypatch, tmp_path):
    library = tmp_path / "lib"
    library.mkdir()
    (library / "requirements.txt").write_text("flydsl==0.3.2\n")
    (library / "README.md").write_text("See https://github.com/ROCm/FlyDSL\n")
    fetched = []

    def checkout(owner, repo, ref, declared_by_engine=False):
        fetched.append((owner, repo, ref))
        if ref != "v0.3.2":
            raise OSError("no such ref")
        tree = tmp_path / "flydsl_tree"
        (tree / "python" / "flydsl").mkdir(parents=True, exist_ok=True)
        (tree / "python" / "flydsl" / "__init__.py").write_text("")
        return tree

    monkeypatch.setattr(engine_sources, "checkout", checkout)
    sources = engine_sources.EngineSources("sglang", "v1", tmp_path / "engine")
    sources.checkouts.append(library)
    assert sources._library_requirement("flydsl") == tmp_path / "flydsl_tree"
    assert fetched == [("ROCm", "FlyDSL", "v0.3.2")]


def test_an_image_built_on_the_repository_s_base_image_has_its_libraries(tmp_path):
    from TraceLens.ModelUtils.engine_sources import image_stages, library_pins

    docker = tmp_path / "docker"
    docker.mkdir()
    (docker / "Dockerfile.rocm_base").write_text(
        'ARG AITER_BRANCH="v0.1.24"\nARG AITER_REPO="https://github.com/ROCm/aiter.git"\n'
        "FROM rocm/dev-ubuntu:7.2 AS base\nARG AITER_BRANCH\nARG AITER_REPO\nENV BASE_ONLY=1\n"
    )
    (docker / "Dockerfile.rocm").write_text(
        "ARG BASE_IMAGE=rocm/engine-dev:base\nFROM ${BASE_IMAGE} AS final\nENV ENGINE_FLAG=1\n"
    )
    stages = image_stages(docker / "Dockerfile.rocm", "gfx950")
    pins = {
        name: ref for stage in stages for name, (_, ref) in library_pins(stage).items()
    }
    assert pins == {"aiter": "v0.1.24"}
    environment = {key: value for stage in stages for key, value in stage.env.items()}
    assert environment["BASE_ONLY"] == "1" and environment["ENGINE_FLAG"] == "1"
