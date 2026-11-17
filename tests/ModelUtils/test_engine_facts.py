###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The engine release and its library commits are read from source, not installed."""

from TraceLens.ModelUtils import engine_facts
from TraceLens.ModelUtils.engine_facts import (
    dockerfile_pins,
    kernel_library_commit,
    traced_engine_source,
)


def _frame(name, ts, dur, tid=1):
    return {"cat": "python_function", "name": name, "ts": ts, "dur": dur, "tid": tid}


def test_dockerfile_pins_follow_the_stage_that_sets_them():
    text = (
        "FROM rocm AS base\nENV AITER_COMMIT_DEFAULT=skip\n"
        'FROM base AS gfx950\nENV AITER_COMMIT_DEFAULT="abc123"\n'
        "FROM base AS gfx942\nENV AITER_COMMIT_DEFAULT=abc123\n"
    )
    assert dockerfile_pins(text, "AITER_COMMIT_DEFAULT") == {
        "base": "skip",
        "gfx950": "abc123",
        "gfx942": "abc123",
    }


def test_the_kernel_library_commit_is_the_one_the_device_stages_agree_on(
    monkeypatch, tmp_path
):
    dockerfile = tmp_path / "rocm.Dockerfile"
    monkeypatch.setattr(
        engine_facts, "engine_source_of", lambda engine: "github:o/r@v1.0.0"
    )
    monkeypatch.setattr(engine_facts, "fetch_github_file", lambda ref, path: dockerfile)
    dockerfile.write_text(
        "FROM x AS gfx950\nENV K=aa\nFROM x AS gfx942\nENV K=aa\n", encoding="utf-8"
    )
    assert kernel_library_commit("sglang", "K") == "aa"
    dockerfile.write_text(
        "FROM x AS gfx950\nENV K=aa\nFROM x AS gfx942\nENV K=bb\n", encoding="utf-8"
    )
    assert kernel_library_commit("sglang", "K") is None


def test_the_trace_places_the_run_at_the_release_whose_defs_match_its_frames(
    monkeypatch, tmp_path
):
    sources = {
        "github:o/r@v1.0.2": "\n\n\ndef run():\n    pass\n",
        "github:o/r@v1.0.1": "\n\ndef run():\n    pass\n",
    }

    def fetch(ref, path):
        text = sources.get(f"github:{ref.owner}/{ref.repo}@{ref.ref}")
        if text is None:
            return None
        found = tmp_path / ref.ref / path
        found.parent.mkdir(parents=True, exist_ok=True)
        found.write_text(text, encoding="utf-8")
        return found

    monkeypatch.setattr(engine_facts, "fetch_github_file", fetch)
    monkeypatch.setattr(
        "TraceLens.ModelUtils.source.ENGINE_GITHUB_SOURCES",
        {"sglang": "github:o/r@v1.0.2"},
    )
    frame = "/x/python/sglang/srt/a.py(3): run"
    events = [_frame(frame, 0, 1)]
    assert traced_engine_source("sglang", events) == "github:o/r@v1.0.1"
    assert traced_engine_source("sglang", []) == "github:o/r@v1.0.2"
    assert traced_engine_source("vllm", events) is None
