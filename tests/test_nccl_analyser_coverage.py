###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only coverage for remaining TraceLens.NcclAnalyser paths."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from TraceLens.NcclAnalyser.nccl_analyser import (
    NcclAnalyser,
    _parse_split_sizes,
)


def _nccl_kernel(name="ncclKernel_AllReduce", external_id=42, ts=100, dur=10, **extra_args):
    args = {
        "External id": external_id,
        "Collective name": "allreduce",
        "Process Group Name": "default_pg",
        "Process Group Ranks": [0, 1],
        "Group size": 2,
        "dtype": "Float",
        "In msg nelems": 1024,
        "Out msg nelems": 1024,
        "In split size": "[]",
        "Out split size": "[]",
        "stream": 3,
    }
    args.update(extra_args)
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": args,
    }


def _write_trace(path, events):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


def _build_analyser(tmp_path, world_size, kernel_builder):
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        _write_trace(path, [kernel_builder(rank)])
        filepaths.append(str(path))
    return NcclAnalyser(filepaths, world_size)


@pytest.mark.parametrize(
    "value",
    [
        "not-a-list",
        "{bad",
        "[]",
    ],
)
def test_parse_split_sizes_unparseable(value):
    if value == "[]":
        assert _parse_split_sizes(value) == []
    else:
        assert _parse_split_sizes(value) is None


def test_parse_split_sizes_syntax_error():
    assert _parse_split_sizes("[1, 2,") is None


def test_build_df_implicit_sync_empty_df(tmp_path):
    path = tmp_path / "rank0.json"
    _write_trace(path, [{"ph": "X", "cat": "cpu_op", "name": "x", "ts": 1, "dur": 1, "args": {}}])
    analyser = NcclAnalyser([str(path)], world_size=1)
    df = analyser.build_df_nccl_implicit_sync_cat()
    assert df.empty


def test_build_df_implicit_sync_metadata_mismatch_warning(tmp_path):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        kernel = _nccl_kernel(
            external_id=100 + rank,
            ts=1000 + rank * 10,
            dur=50,
            dtype="Float" if rank == 0 else "Half",
        )
        _write_trace(path, [kernel])
        filepaths.append(str(path))

    analyser = NcclAnalyser(filepaths, world_size)
    analyser.build_df_long()
    df = analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
    assert not df.empty


def test_build_df_implicit_sync_strict_metadata_raises(tmp_path):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        kernel = _nccl_kernel(
            external_id=200 + rank,
            ts=2000 + rank,
            dur=40,
            dtype="Float" if rank == 0 else "Half",
        )
        _write_trace(path, [kernel])
        filepaths.append(str(path))

    analyser = NcclAnalyser(filepaths, world_size)
    analyser.build_df_long()
    with pytest.raises(ValueError, match="Metadata mismatch"):
        analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=True)


def test_build_df_implicit_sync_zero_comm_latency(tmp_path):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        kernel = _nccl_kernel(external_id=300 + rank, ts=3000, dur=0)
        _write_trace(path, [kernel])
        filepaths.append(str(path))

    analyser = NcclAnalyser(filepaths, world_size)
    analyser.build_df_long()
    df = analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
    assert pd.isna(df.iloc[0]["algo bw (GB/s)"])


def test_build_df_straggler_summary(tmp_path):
    analyser = _build_analyser(
        tmp_path,
        2,
        lambda rank: _nccl_kernel(external_id=400 + rank, ts=4000 + rank * 5, dur=30 + rank),
    )
    analyser.build_df_long()
    analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
    summary = analyser.build_df_straggler_summary(strict_metadata_check=False)
    assert not summary.empty
    assert "total_wait_time_us" in summary.columns


def test_build_df_straggler_summary_empty_when_no_implicit_sync(tmp_path):
    path = tmp_path / "rank0.json"
    _write_trace(path, [{"ph": "X", "cat": "cpu_op", "name": "x", "ts": 1, "dur": 1, "args": {}}])
    analyser = NcclAnalyser([str(path)], world_size=1)
    assert analyser.build_df_straggler_summary().empty


def test_infer_collective_name_in_build_df_long(tmp_path):
    path = tmp_path / "rank0.json"
    kernel = {
        "ph": "X",
        "cat": "kernel",
        "name": "vllm_cross_device_reduce_kernel",
        "ts": 100,
        "dur": 10,
        "args": {"External id": 1, "Collective name": None, "stream": 1},
    }
    _write_trace(path, [kernel])
    analyser = NcclAnalyser([str(path)], world_size=1)
    df = analyser.build_df_long()
    assert df.iloc[0]["Collective name"] == "allreduce"


def test_all2allv_heatmap_world_size_warnings(caplog):
    import logging

    analyser = NcclAnalyser.__new__(NcclAnalyser)
    analyser.logger = logging.getLogger("TraceLens.NcclAnalyser.nccl_analyser")

    analyser.world_size = 2048
    with caplog.at_level(logging.WARNING):
        assert analyser.build_df_all2allv_heatmap() is None
    assert "Skipping all2allv heatmap" in caplog.text

    analyser.world_size = 512
    analyser.df_per_rank_coll = pd.DataFrame()
    with caplog.at_level(logging.WARNING):
        result = analyser.build_df_all2allv_heatmap()
    assert result is None
    assert "all2allv heatmap will produce" in caplog.text


def test_all2allv_heatmap_skips_bad_metadata(tmp_path, caplog):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        kernel = _nccl_kernel(
            name="ncclKernel_AllToAllv",
            external_id=500 + rank,
            ts=5000 + rank,
            dur=20,
            **{
                "Collective name": "all_to_allv",
                "In split size": "bad-split",
                "Group size": 2,
            },
        )
        _write_trace(path, [kernel])
        filepaths.append(str(path))

    analyser = NcclAnalyser(filepaths, world_size)
    analyser.build_df_long()
    heatmap = analyser.build_df_all2allv_heatmap(strict_metadata_check=True)
    assert heatmap is None


def test_all2allv_summary_metadata_mismatch_warns(tmp_path):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        kernel = _nccl_kernel(
            name="ncclKernel_AllToAllv",
            external_id=600 + rank,
            ts=6000 + rank,
            dur=25,
            **{
                "Collective name": "all_to_allv",
                "In split size": "[512, 512]",
                "Group size": 2,
                "dtype": "Float" if rank == 0 else "Half",
            },
        )
        _write_trace(path, [kernel])
        filepaths.append(str(path))

    analyser = NcclAnalyser(filepaths, world_size)
    analyser.build_df_long()
    with pytest.warns(UserWarning, match="Metadata mismatch"):
        df = analyser.build_df_nccl_all2allv(detailed=False, strict_metadata_check=False)
    assert df is not None
