###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.TraceFusion."""

import gzip
import json

import pytest

from TraceLens.TraceFusion.trace_fuse import (
    TraceFuse,
    _default_filter_fn,
    _process_single_rank,
)

CUDA_LAUNCH = {
    "name": "cudaLaunchKernel",
    "ph": "X",
    "cat": "cuda_runtime",
    "pid": 10,
    "tid": 1,
    "ts": 100,
    "dur": 5,
    "args": {"correlation": 42},
}
KERNEL = {
    "name": "gemm_kernel",
    "ph": "X",
    "cat": "kernel",
    "pid": 11,
    "tid": 2,
    "ts": 110,
    "dur": 8,
    "args": {"correlation": 42},
}
TRACE_EVENT = {
    "name": "marker",
    "ph": "X",
    "cat": "Trace",
    "pid": 10,
    "ts": 0,
    "dur": 1,
}
PYFUNC_EVENT = {
    "name": "forward",
    "ph": "X",
    "cat": "python_function",
    "pid": 10,
    "ts": 50,
    "dur": 2,
}


def _write_trace(path, events):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


@pytest.mark.parametrize(
    "event,include_pyfunc,expected",
    [
        (TRACE_EVENT, False, False),
        (PYFUNC_EVENT, False, False),
        (PYFUNC_EVENT, True, True),
        (KERNEL, False, True),
    ],
)
def test_default_filter_fn(event, include_pyfunc, expected):
    assert _default_filter_fn(dict(event), include_pyfunc=include_pyfunc) is expected


def test_trace_fuse_default_filter_fn():
    assert TraceFuse.default_filter_fn(TRACE_EVENT) is False
    assert TraceFuse.default_filter_fn(KERNEL) is True


def test_trace_fuse_init_accepts_list_and_dict(tmp_path):
    rank0 = tmp_path / "rank0.json"
    rank1 = tmp_path / "rank1.json"
    for path in (rank0, rank1):
        _write_trace(path, [CUDA_LAUNCH, KERNEL])

    from_list = TraceFuse([str(rank0), str(rank1)])
    assert from_list.rank2filepath == {0: str(rank0), 1: str(rank1)}

    from_dict = TraceFuse({3: str(rank0), 7: str(rank1)})
    assert from_dict.rank2filepath == {3: str(rank0), 7: str(rank1)}


def test_set_linking_key_uses_correlation_when_launch_present(tmp_path):
    trace_path = tmp_path / "rank0.json"
    _write_trace(trace_path, [CUDA_LAUNCH, KERNEL])
    fuser = TraceFuse([str(trace_path)])
    assert fuser.linking_key == "correlation"


def test_set_linking_key_falls_back_to_external_id(tmp_path):
    trace_path = tmp_path / "rank0.json"
    _write_trace(
        trace_path,
        [
            {
                "name": "cpu_op",
                "ph": "X",
                "cat": "cpu_op",
                "pid": 1,
                "ts": 1,
                "dur": 1,
                "args": {"External id": 7},
            }
        ],
    )
    fuser = TraceFuse([str(trace_path)])
    assert fuser.linking_key == "External id"


def test_process_single_rank_applies_rank_and_offsets(tmp_path):
    trace_path = tmp_path / "rank0.json"
    _write_trace(trace_path, [CUDA_LAUNCH, KERNEL, TRACE_EVENT, PYFUNC_EVENT])

    fuser = TraceFuse([str(trace_path)])
    rank, events = _process_single_rank(
        rank=2,
        filepath=str(trace_path),
        filter_fn=None,
        include_pyfunc=False,
        offset_multiplier=fuser.offset_multiplier,
        linking_key=fuser.linking_key,
    )

    assert rank == 2
    assert len(events) == 2
    assert all(event["args"]["rank"] == 2 for event in events)
    launch = next(e for e in events if e["cat"] == "cuda_runtime")
    kernel = next(e for e in events if e["cat"] == "kernel")
    assert launch["args"]["correlation_raw"] == 42
    assert (
        launch["args"]["correlation"] == 42 + 2 * fuser.offset_multiplier["correlation"]
    )
    assert kernel["pid"] == 11 + 2 * fuser.offset_multiplier["pid"]


def test_process_single_rank_honors_custom_filter(tmp_path):
    trace_path = tmp_path / "rank0.json"
    _write_trace(trace_path, [CUDA_LAUNCH, KERNEL])

    fuser = TraceFuse([str(trace_path)])
    _, events = _process_single_rank(
        rank=0,
        filepath=str(trace_path),
        filter_fn=lambda event: event.get("cat") == "kernel",
        include_pyfunc=False,
        offset_multiplier=fuser.offset_multiplier,
        linking_key=fuser.linking_key,
    )
    assert len(events) == 1
    assert events[0]["cat"] == "kernel"


def test_generate_rank_metadata_labels_cpu_and_gpu():
    merged = [
        {
            "ph": "X",
            "pid": 10,
            "cat": "cpu_op",
            "args": {"rank": 0},
        },
        {
            "ph": "X",
            "pid": 11,
            "cat": "kernel",
            "args": {"rank": 0},
        },
        {
            "ph": "X",
            "pid": 20,
            "cat": "cpu_op",
            "args": {"rank": 1},
        },
    ]
    fuser = TraceFuse.__new__(TraceFuse)
    metadata = fuser._generate_rank_metadata(merged)

    names = {
        entry["args"]["name"] for entry in metadata if entry["name"] == "process_name"
    }
    assert names == {"RANK 0 - CPU", "RANK 0 - GPU", "RANK 1 - CPU"}


def test_merge_combines_ranks_in_order(tmp_path):
    rank0 = tmp_path / "rank0.json"
    rank1 = tmp_path / "rank1.json"
    _write_trace(rank0, [CUDA_LAUNCH, KERNEL])
    _write_trace(
        rank1,
        [
            {
                **CUDA_LAUNCH,
                "args": {"correlation": 99},
                "pid": 20,
            },
            {
                **KERNEL,
                "args": {"correlation": 99},
                "pid": 21,
            },
        ],
    )

    merged = TraceFuse([str(rank0), str(rank1)]).merge()
    non_metadata = [event for event in merged if event.get("ph") != "M"]
    metadata = [event for event in merged if event.get("ph") == "M"]

    assert len(non_metadata) == 4
    ranks = [event["args"]["rank"] for event in non_metadata]
    assert ranks[:2] == [0, 0]
    assert ranks[2:] == [1, 1]
    assert any(entry["name"] == "process_name" for entry in metadata)


def test_merge_and_save_writes_gzip_json(tmp_path):
    rank0 = tmp_path / "rank0.json"
    _write_trace(rank0, [CUDA_LAUNCH, KERNEL])

    output_file = tmp_path / "merged_trace.json"
    gz_path = TraceFuse([str(rank0)]).merge_and_save(str(output_file))

    assert gz_path == str(output_file) + ".gz"
    with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert len(payload["traceEvents"]) >= 2


def test_merge_multiprocessing_path(tmp_path):
    rank0 = tmp_path / "rank0.json"
    rank1 = tmp_path / "rank1.json"
    _write_trace(rank0, [CUDA_LAUNCH, KERNEL])
    _write_trace(
        rank1,
        [
            {**CUDA_LAUNCH, "args": {"correlation": 99}, "pid": 20},
            {**KERNEL, "args": {"correlation": 99}, "pid": 21},
        ],
    )

    merged = TraceFuse(
        [str(rank0), str(rank1)], use_multiprocessing=True, max_workers=2
    ).merge()
    non_metadata = [event for event in merged if event.get("ph") != "M"]
    ranks = [event["args"]["rank"] for event in non_metadata]
    assert ranks[:2] == [0, 0]
    assert ranks[2:] == [1, 1]
