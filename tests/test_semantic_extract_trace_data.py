###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/extract_trace_data.py extractors."""

import json
import os, sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)
import extract_trace_data


def _kernel(name, ts, dur, stream=None, cat="kernel", **extra):
    ev = {"name": name, "ts": ts, "dur": dur, "cat": cat}
    if stream is not None:
        ev["args"] = {"stream": stream}
    ev.update(extra)
    return ev


def test_load_trace_from_dict_sorts_and_skips_nondict():
    data_in = {
        "traceEvents": [
            {"name": "b", "cat": "kernel", "ts": 5},
            {"name": "a", "cat": "kernel", "ts": 1},
            "not-a-dict",
            {"name": "c", "cat": "cpu_op", "ts": 3},
        ]
    }
    data, by_cat = extract_trace_data.load_trace(data_in)
    assert data is data_in
    # Non-dict element skipped; kernels sorted by ts.
    assert [e["name"] for e in by_cat["kernel"]] == ["a", "b"]
    assert [e["name"] for e in by_cat["cpu_op"]] == ["c"]


def test_load_trace_from_path():
    payload = {"traceEvents": [{"name": "k", "cat": "kernel", "ts": 1}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp)
        path = tmp.name
    try:
        data, by_cat = extract_trace_data.load_trace(path)
        assert data["traceEvents"][0]["name"] == "k"
        assert [e["name"] for e in by_cat["kernel"]] == ["k"]
    finally:
        os.unlink(path)


def test_stamp_raw_uid():
    data = {"traceEvents": [{"name": "a"}, "skip", {"name": "b"}]}
    extract_trace_data._stamp_raw_uid(data)
    assert data["traceEvents"][0]["_gpu_op_uid"] == 0
    assert data["traceEvents"][2]["_gpu_op_uid"] == 2


def test_get_stream_id_variants():
    # Valid string stream -> int.
    assert extract_trace_data.get_stream_id({"args": {"stream": "3"}}) == 3
    # Unhashable stream value raises TypeError -> falls back to tid.
    assert extract_trace_data.get_stream_id({"args": {"stream": [1]}, "tid": 7}) == 7
    # Non-numeric stream string raises ValueError -> falls back to tid.
    assert extract_trace_data.get_stream_id({"args": {"stream": "x"}, "tid": 4}) == 4
    # No stream, tid present.
    assert extract_trace_data.get_stream_id({"tid": 5}) == 5
    # Invalid tid -> None.
    assert extract_trace_data.get_stream_id({"tid": "nope"}) is None
    # Nothing usable -> None.
    assert extract_trace_data.get_stream_id({}) is None


def test_filter_to_primary_stream_empty():
    by_cat = {}
    extract_trace_data.filter_to_primary_stream(by_cat)
    assert by_cat == {}


def test_filter_to_primary_stream_single_stream_noop():
    kernels = [_kernel("k", 1, 10, stream=0), _kernel("k", 2, 10, stream=0)]
    by_cat = {"kernel": kernels}
    extract_trace_data.filter_to_primary_stream(by_cat)
    assert len(by_cat["kernel"]) == 2


def test_filter_to_primary_stream_filters_minor_secondary():
    kernels = [_kernel("k", i, 10, stream=0) for i in range(9)]
    kernels.append(_kernel("k", 100, 1, stream=1))
    by_cat = {"kernel": kernels}
    extract_trace_data.filter_to_primary_stream(by_cat)
    # Secondary stream is <5% of time -> dropped, keeping primary only.
    assert all(k["args"]["stream"] == 0 for k in by_cat["kernel"])
    assert len(by_cat["kernel"]) == 9


def test_filter_to_primary_stream_keeps_significant_secondary():
    kernels = [_kernel("k", i, 10, stream=0) for i in range(5)]
    kernels.append(_kernel("k", 100, 10, stream=1))
    by_cat = {"kernel": kernels}
    extract_trace_data.filter_to_primary_stream(by_cat)
    # Secondary stream >5% of time -> keep all streams.
    assert len(by_cat["kernel"]) == 6


def test_extract_kernel_sequence():
    by_cat = {
        "kernel": [
            _kernel("gemm", 5, 3, stream=0, _gpu_op_uid=11),
            _kernel("relu", 1, 2, stream=0, gpu_op_uid=22),
        ],
        "gpu_memcpy": [_kernel("memcpy", 3, 1, cat="gpu_memcpy")],
    }
    seq = extract_trace_data.extract_kernel_sequence(by_cat)
    # Sorted by ts across kernels + memcpy.
    assert [k["name"] for k in seq] == ["relu", "memcpy", "gemm"]
    # _gpu_op_uid preferred, else gpu_op_uid.
    assert seq[2]["gpu_op_uid"] == 11
    assert seq[0]["gpu_op_uid"] == 22
    # memcpy has neither -> None.
    assert seq[1]["gpu_op_uid"] is None
    assert seq[1]["cat"] == "gpu_memcpy"


def test_detect_graph_mode():
    by_cat = {"cuda_runtime": [{"name": "hipGraphLaunch"}, {"name": "other"}]}
    is_graph, launches = extract_trace_data.detect_graph_mode(by_cat)
    assert is_graph is True
    assert len(launches) == 1

    is_graph2, launches2 = extract_trace_data.detect_graph_mode({})
    assert is_graph2 is False
    assert launches2 == []


def test_run_assertions_happy_path():
    data = {"traceEvents": [1]}
    by_cat = {"kernel": [], "cpu_op": []}
    kernels = [
        {"name": "a", "ts": 1, "dur": 2.0},
        {"name": "b", "ts": 3, "dur": 4.0},
    ]
    errors = extract_trace_data.run_assertions(data, by_cat, kernels, False)
    assert errors == []


def test_run_assertions_missing_trace_events_and_categories():
    data = {}
    by_cat = {"kernel": []}  # cpu_op missing under strict.
    kernels = [{"name": "a", "ts": 1, "dur": 2.0}]
    errors = extract_trace_data.run_assertions(data, by_cat, kernels, False)
    joined = " ".join(errors)
    assert "A1.1 FAIL" in joined
    assert "A1.2 FAIL" in joined


def test_run_assertions_no_kernels_and_zero_time():
    data = {"traceEvents": []}
    by_cat = {"kernel": [], "cpu_op": []}
    errors = extract_trace_data.run_assertions(data, by_cat, [], False)
    joined = " ".join(errors)
    assert "A1.3 FAIL" in joined
    assert "A1.5 FAIL" in joined


def test_run_assertions_nonpositive_duration():
    data = {"traceEvents": [1]}
    by_cat = {"kernel": [], "cpu_op": []}
    kernels = [{"name": "bad", "ts": 1, "dur": 0}]
    errors = extract_trace_data.run_assertions(data, by_cat, kernels, False)
    assert any("A3.2 FAIL" in e for e in errors)


def test_run_assertions_nonmonotonic_timestamps():
    data = {"traceEvents": [1]}
    by_cat = {"kernel": [], "cpu_op": []}
    kernels = [
        {"name": "a", "ts": 5, "dur": 1.0},
        {"name": "b", "ts": 1, "dur": 1.0},
    ]
    errors = extract_trace_data.run_assertions(data, by_cat, kernels, False)
    assert any("A3.1 FAIL" in e for e in errors)


def test_extract_and_build_result_with_and_without_region_meta():
    by_cat = {
        "kernel": [_kernel("gemm", 1, 3, stream=0, _gpu_op_uid=0)],
        "cpu_op": [],
    }
    result, kernels = extract_trace_data.extract_and_build_result(
        {"traceEvents": []}, by_cat, "trace.json"
    )
    assert result["source_file"] == "trace.json"
    assert result["metadata"]["total_kernels"] == 1
    assert result["metadata"]["total_kernel_time_us"] == 3.0
    assert "region_metadata" not in result
    assert len(kernels) == 1

    result2, _ = extract_trace_data.extract_and_build_result(
        {"traceEvents": []}, by_cat, "trace.json", region_metadata={"region": "steady"}
    )
    assert result2["region_metadata"] == {"region": "steady"}
