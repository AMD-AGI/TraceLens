###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The engine release a run executed is the one its python frames agree with."""

from TraceLens.ModelUtils import engine_facts
from TraceLens.ModelUtils.engine_facts import (
    agreeing_frames,
    traced_engine_release,
    traced_frames,
)


def _frame(name):
    return {"cat": "python_function", "name": name, "ts": 0, "dur": 1}


_SOURCE = """import functools


@functools.lru_cache
def cached(x):
    return [y for y in x]


def plain():
    return 1
"""


def test_frames_are_read_from_the_last_mention_of_the_package():
    events = [
        _frame("/sgl-workspace/sglang/python/sglang/srt/a.py(5): cached"),
        _frame("/usr/lib/python3/threading.py(1): run"),
        {"cat": "cpu_op", "name": "aten::mm"},
    ]
    assert traced_frames(events, "sglang") == {"sglang/srt/a.py": {(5, "cached")}}


def test_a_frame_agrees_anywhere_in_its_function_from_the_first_decorator():
    frames = {
        (4, "cached"),
        (6, "cached"),
        (6, "<listcomp>"),
        (9, "plain"),
        (4, "plain"),
    }
    # The decorator line and body lines of `cached`, its comprehension, and
    # `plain`'s def agree; line 4 is not inside `plain`.
    assert agreeing_frames(_SOURCE, frames) == 4


def test_the_release_agreeing_with_the_most_frames_is_the_one_that_ran(monkeypatch):
    shifted = "\n\n" + _SOURCE
    sources = {"v2": shifted, "v1": _SOURCE, "v0": "def other():\n    pass\n"}
    monkeypatch.setattr(
        engine_facts,
        "_release_file",
        lambda owner, repo, release, path: sources[release],
    )
    events = [
        _frame("/root/sglang/python/sglang/srt/a.py(6): cached"),
        _frame("/root/sglang/python/sglang/srt/a.py(10): plain"),
    ]
    assert traced_engine_release("sglang", events, ["v2", "v1", "v0"]) == "v1"
    # No engine frames, or an unknown engine: nothing to say.
    assert traced_engine_release("sglang", [], ["v1"]) is None
    assert traced_engine_release("other", events, ["v1"]) is None


def test_a_tie_goes_to_the_newer_release(monkeypatch):
    monkeypatch.setattr(
        engine_facts, "_release_file", lambda owner, repo, release, path: _SOURCE
    )
    events = [_frame("/x/sglang/srt/a.py(10): plain")]
    assert traced_engine_release("sglang", events, ["v2", "v1"]) == "v2"


def test_the_trace_s_metadata_names_the_engine_s_source():
    from TraceLens.ModelUtils.engine_facts import recorded_engine_version

    assert (
        recorded_engine_version({"vllm_version": "0.30.1rc1.dev709+g21d93d0d8"}, "vllm")
        == "21d93d0d8"
    )
    assert recorded_engine_version({"sglang_version": "0.5.21"}, "sglang") == "0.5.21"
    assert recorded_engine_version({}, "sglang") is None
