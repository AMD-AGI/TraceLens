###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Engine facts are read from the pinned source text, not from an installed package."""

import pytest

from TraceLens.ModelUtils import engine_facts
from TraceLens.ModelUtils.engine_facts import (
    dockerfile_pins,
    environment_default,
    kernel_library_commit,
    load_engine_facts,
    module_constant,
    trace_runtime_facts,
    traced_engine_source,
)


def test_module_constants_evaluate_literals_and_earlier_names():
    source = "_A = 2\n_B = (1, 4, _A * 8, 4)\n_C: int = -3\n_D = open('x')\n"
    assert module_constant(source, "_B") == (1, 4, 16, 4)
    assert module_constant(source, "_C") == -3
    with pytest.raises(KeyError):
        module_constant(source, "_D")


def test_environment_default_reads_the_literal_of_the_envs_attribute():
    source = (
        "class Other:\n    FLAG = EnvBool(False)\n"
        "class Envs:\n    FLAG = EnvBool(True)\n    COUNT = EnvInt(7)\n"
    )
    assert environment_default(source, "FLAG") is True
    assert environment_default(source, "COUNT") == 7
    with pytest.raises(KeyError):
        environment_default(source, "MISSING")


def test_load_engine_facts_skips_what_the_source_does_not_give(monkeypatch):
    texts = {
        "sglang.srt.environ": "class Envs:\n    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)\n",
        "sglang.kernels.ops.attention.dsv4.fp4_indexer_hip": "_OTHER = 1\n",
    }
    monkeypatch.setattr(
        engine_facts, "_module_text", lambda source, module: texts.get(module)
    )
    assert load_engine_facts("sglang") == (("fuse_wqa_wkv", True),)
    assert load_engine_facts("vllm") == ()
    assert load_engine_facts(None) == ()


def _frame(name, ts, dur, tid=1):
    return {"cat": "python_function", "name": name, "ts": ts, "dur": dur, "tid": tid}


def _op(name, ts, dims=None, concrete=None, tid=1):
    args = {}
    if dims is not None:
        args["Input Dims"] = dims
    if concrete is not None:
        args["Concrete Inputs"] = concrete
    return {"cat": "cpu_op", "name": name, "ts": ts, "dur": 1, "tid": tid, "args": args}


def test_runtime_facts_read_the_pool_sizes_and_the_replay_widths_from_eager_ops():
    events = [
        _frame("a.py(1): translate_loc_from_full_to_swa", 0, 10),
        _op("aten::index", 2, dims=[[10700801], [5]]),
        _frame("a.py(9): alloc_for_decode", 20, 10),
        _op("aten::index", 22, dims=[[9, 1048580], [3]]),
        _frame("r.py(152): build_replay_fb_view", 40, 10),
        _op("aten::slice", 42, concrete=["", "0", "0", "4", "1"]),
        _op("aten::slice", 43, concrete=["", "0", "0", "4", "1"]),
        _frame("r.py(152): build_replay_fb_view", 60, 10),
        _op("aten::slice", 62, concrete=["", "0", "0", "1", "1"]),
        # An op on another thread is not the frame's own.
        _op("aten::slice", 63, concrete=["", "0", "0", "8", "1"], tid=2),
    ]
    assert dict(trace_runtime_facts(events)) == {
        "kv_mapping_rows": 10700801,
        "req_to_token_shape": (9, 1048580),
        "replay_tokens": (1, 4),
    }
    assert trace_runtime_facts([]) == ()


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
