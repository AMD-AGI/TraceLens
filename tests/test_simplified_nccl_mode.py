###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the simplified inference-mode path in NcclAnalyser.

The simplified path activates when all NCCL kernels sit on a single stream
per rank and no Process Group metadata is present (typical for vLLM / SGLang
traces).  Collectives are matched across ranks by their temporal order on
the stream rather than by Process Group Name.
"""

import gzip
import json
import os
import shutil
import tempfile

import pandas as pd
import pytest

from TraceLens import NcclAnalyser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(rank, n_collectives, *, include_pg=False, multi_stream=False):
    """Create a minimal synthetic trace for one rank.

    Parameters
    ----------
    rank : int
    n_collectives : int
    include_pg : bool
        When True, add Process Group metadata (regular path).
    multi_stream : bool
        When True, alternate kernels between two streams.
    """
    events = []
    base_ts = 1_000_000 + rank * 50
    for i in range(n_collectives):
        ts = base_ts + i * 1000 + rank * (5 if i % 3 == 0 else 2)
        dur = 50 + rank * 3
        args = {
            "External id": 100 + i,
            "device": rank,
            "stream": 3 if not multi_stream else (3 + i % 2),
            "correlation": 50 + i,
            "kind": "Dispatch Kernel",
        }
        if include_pg:
            args.update(
                {
                    "Process Group Name": "default_pg",
                    "Process Group Ranks": list(range(4)),
                    "Collective name": "_allgather_base",
                    "Group size": 4,
                    "dtype": "BFloat16",
                    "In msg nelems": 40960,
                    "Out msg nelems": 163840,
                    "In split size": "[]",
                    "Out split size": "[]",
                }
            )
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void rcclGenericKernel<1, false>"
                "(ncclDevKernelArgsStorage<4096ul>)",
                "pid": rank,
                "tid": 3,
                "ts": ts,
                "dur": dur,
                "args": args,
            }
        )
    return {"traceEvents": events}


def _write_traces(tmpdir, world_size, n_collectives, **kwargs):
    """Write per-rank traces and return list of paths."""
    paths = []
    for rank in range(world_size):
        trace = _make_trace(rank, n_collectives, **kwargs)
        path = os.path.join(tmpdir, f"rank_{rank}.json.gz")
        with gzip.open(path, "wt") as f:
            json.dump(trace, f)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmpdir():
    d = tempfile.mkdtemp(prefix="nccl_simplified_test_")
    yield d
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimplifiedModeDetection:
    """Verify the auto-detection logic picks the right path."""

    def test_simplified_mode_activates(self, tmpdir):
        files = _write_traces(tmpdir, 4, 20, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        assert analyser._simplified_mode is True

    def test_regular_mode_with_pg(self, tmpdir):
        files = _write_traces(tmpdir, 4, 20, include_pg=True)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        assert not analyser._simplified_mode

    def test_regular_mode_multi_stream_no_pg(self, tmpdir):
        """Multiple streams but no PG → regular path (can't safely match)."""
        files = _write_traces(tmpdir, 4, 20, include_pg=False, multi_stream=True)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        assert analyser._simplified_mode is False


class TestSimplifiedMatching:
    """Verify collectives are correctly matched across ranks."""

    def test_collective_ids_match_all_ranks(self, tmpdir):
        files = _write_traces(tmpdir, 4, 10, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        df = analyser.build_df_long()
        for cid in df["collective_id"].unique():
            ranks = sorted(df.loc[df["collective_id"] == cid, "rank"].tolist())
            assert ranks == [0, 1, 2, 3], f"collective {cid} missing ranks: {ranks}"

    def test_index_in_group_sequential(self, tmpdir):
        files = _write_traces(tmpdir, 4, 15, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        df = analyser.build_df_long()
        for r in range(4):
            rank_df = df[df["rank"] == r].sort_values("ts")
            assert list(rank_df["index_in_group"]) == list(range(15))


class TestSimplifiedStragglerAnalysis:
    """Verify straggler analysis works without msg size metadata."""

    def test_straggler_columns_present(self, tmpdir):
        files = _write_traces(tmpdir, 4, 10, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        df = analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
        assert not df.empty
        for col in [
            "comm_latency",
            "skew in start time",
            "earliest arrival rank",
            "avg_wait_time",
            "skew in end time",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_bandwidth_columns_nan(self, tmpdir):
        """Without msg size metadata, bandwidth should be NaN."""
        files = _write_traces(tmpdir, 4, 10, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        df = analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
        assert df["algo bw (GB/s)"].isna().all()
        assert df["bus bw (GB/s)"].isna().all()

    def test_earliest_arrival_is_rank_0(self, tmpdir):
        """With the synthetic skew, rank 0 always starts first."""
        files = _write_traces(tmpdir, 4, 10, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        df = analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
        assert (df["earliest arrival rank"] == 0).all()

    def test_skew_is_positive(self, tmpdir):
        files = _write_traces(tmpdir, 4, 10, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        df = analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
        assert (df["skew in start time"] > 0).all()


class TestSimplifiedSummary:
    """Verify the summary table works in simplified mode."""

    def test_summary_not_empty(self, tmpdir):
        files = _write_traces(tmpdir, 4, 20, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
        summary = analyser.build_df_summary_nccl_implicit_sync_cat(
            strict_metadata_check=False
        )
        assert not summary.empty

    def test_summary_has_count(self, tmpdir):
        files = _write_traces(tmpdir, 4, 20, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
        summary = analyser.build_df_summary_nccl_implicit_sync_cat(
            strict_metadata_check=False
        )
        assert "count" in summary.columns
        assert summary["count"].sum() == 20

    def test_summary_omits_bw_when_all_nan(self, tmpdir):
        """Bandwidth agg columns should not appear when data is all NaN."""
        files = _write_traces(tmpdir, 4, 20, include_pg=False)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
        summary = analyser.build_df_summary_nccl_implicit_sync_cat(
            strict_metadata_check=False
        )
        bw_cols = [c for c in summary.columns if "bw" in c.lower()]
        assert len(bw_cols) == 0, f"Unexpected bw columns: {bw_cols}"


class TestRegularPathUnchanged:
    """Sanity-check that the regular path still produces bandwidth."""

    def test_regular_has_bandwidth(self, tmpdir):
        files = _write_traces(tmpdir, 4, 10, include_pg=True)
        analyser = NcclAnalyser(files, world_size=4)
        analyser.build_df_long()
        df = analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
        assert not df["algo bw (GB/s)"].isna().all()
        assert not df["bus bw (GB/s)"].isna().all()
