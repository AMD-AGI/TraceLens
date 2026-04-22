###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Synthetic tests for all2allv summary, per-instance metrics, and heatmap.

Since real MoE traces with all_to_allv events are not readily available,
these tests construct minimal synthetic trace files with known values and
verify that NcclAnalyser produces the expected metrics.
"""

import json
import math
import os
import shutil
import tempfile

import pandas as pd
import pytest

from TraceLens import NcclAnalyser

# ---------------------------------------------------------------------------
# Helpers to build minimal synthetic trace JSON
# ---------------------------------------------------------------------------


def _make_nccl_event(
    rank,
    collective_name,
    ts,
    dur,
    in_nelems,
    out_nelems,
    dtype="BFloat16",
    group_name="pg_default",
    group_ranks=None,
    group_size=4,
    in_split_size=None,
    out_split_size=None,
    stream=0,
    external_id=100,
):
    """Build a single NCCL kernel event in PyTorch trace JSON format."""
    if group_ranks is None:
        group_ranks = list(range(group_size))
    args = {
        "External id": external_id,
        "Process Group Name": group_name,
        "Process Group Ranks": group_ranks,
        "Collective name": collective_name,
        "Group size": group_size,
        "dtype": dtype,
        "In msg nelems": in_nelems,
        "Out msg nelems": out_nelems,
        "In split size": in_split_size,
        "Out split size": out_split_size,
        "stream": stream,
    }
    return {
        "ph": "X",
        "cat": "kernel",
        "name": f"ncclKernel_AllToAllv_{rank}",
        "ts": ts,
        "dur": dur,
        "args": args,
    }


def _write_trace(filepath, events):
    """Write a minimal PyTorch-format trace JSON file."""
    trace = {"traceEvents": events}
    with open(filepath, "w") as f:
        json.dump(trace, f)


def _build_analyser_from_events(rank_events_map, world_size):
    """Create an NcclAnalyser from a {rank: [events]} mapping.

    Writes temporary trace files and returns (analyser, tmpdir).
    The caller must clean up tmpdir.
    """
    tmpdir = tempfile.mkdtemp(prefix="test_all2allv_")
    filepaths = []
    for rank in range(world_size):
        fp = os.path.join(tmpdir, f"rank{rank}_trace.json")
        _write_trace(fp, rank_events_map.get(rank, []))
        filepaths.append(fp)
    analyser = NcclAnalyser(filepaths, world_size)
    return analyser, tmpdir


# ---------------------------------------------------------------------------
# Fixtures: a balanced 4-rank all2allv scenario
# ---------------------------------------------------------------------------


@pytest.fixture
def balanced_4rank():
    """4 ranks, 2 all2allv invocations, balanced splits (BFloat16 = 2 bytes)."""
    world_size = 4
    events = {}
    # Invocation 0: each rank sends 1024 elements to each peer = 4096 total per rank
    # Invocation 1: each rank sends 2048 elements to each peer = 8192 total per rank
    for rank in range(world_size):
        rank_events = []
        # Invocation 0
        rank_events.append(
            _make_nccl_event(
                rank=rank,
                collective_name="all_to_allv",
                ts=1000 + rank * 10,  # slight stagger
                dur=500,
                in_nelems=4096,
                out_nelems=4096,
                dtype="BFloat16",
                in_split_size=[1024, 1024, 1024, 1024],
                out_split_size=[1024, 1024, 1024, 1024],
                external_id=100 + rank,
            )
        )
        # Invocation 1
        rank_events.append(
            _make_nccl_event(
                rank=rank,
                collective_name="all_to_allv",
                ts=5000 + rank * 5,
                dur=800,
                in_nelems=8192,
                out_nelems=8192,
                dtype="BFloat16",
                in_split_size=[2048, 2048, 2048, 2048],
                out_split_size=[2048, 2048, 2048, 2048],
                external_id=200 + rank,
            )
        )
        events[rank] = rank_events

    analyser, tmpdir = _build_analyser_from_events(events, world_size)
    yield analyser, world_size
    shutil.rmtree(tmpdir)


@pytest.fixture
def imbalanced_4rank():
    """4 ranks, 1 invocation, imbalanced splits (rank 0 sends much more)."""
    world_size = 4
    events = {}
    # rank 0 sends 8000 elems total, others send ~1000
    split_configs = {
        0: [2000, 2000, 2000, 2000],  # 8000 total
        1: [250, 250, 250, 250],  # 1000 total
        2: [250, 250, 250, 250],  # 1000 total
        3: [250, 250, 250, 250],  # 1000 total
    }
    for rank in range(world_size):
        total_send = sum(split_configs[rank])
        events[rank] = [
            _make_nccl_event(
                rank=rank,
                collective_name="all_to_allv",
                ts=1000,
                dur=200 + rank * 100,  # rank 0 fastest, rank 3 slowest
                in_nelems=total_send,
                out_nelems=total_send,
                dtype="Float",  # 4 bytes
                in_split_size=split_configs[rank],
                out_split_size=split_configs[rank],
                external_id=300 + rank,
            )
        ]

    analyser, tmpdir = _build_analyser_from_events(events, world_size)
    yield analyser, world_size
    shutil.rmtree(tmpdir)


@pytest.fixture
def zero_data_rank():
    """4 ranks, 1 invocation, rank 2 sends zero data."""
    world_size = 4
    events = {}
    split_configs = {
        0: [500, 500, 0, 500],
        1: [500, 500, 0, 500],
        2: [0, 0, 0, 0],
        3: [500, 500, 0, 500],
    }
    for rank in range(world_size):
        total_send = sum(split_configs[rank])
        events[rank] = [
            _make_nccl_event(
                rank=rank,
                collective_name="all_to_allv",
                ts=1000,
                dur=100 if total_send > 0 else 10,
                in_nelems=total_send,
                out_nelems=total_send,
                dtype="Half",  # 2 bytes
                in_split_size=split_configs[rank],
                out_split_size=split_configs[rank],
                external_id=400 + rank,
            )
        ]

    analyser, tmpdir = _build_analyser_from_events(events, world_size)
    yield analyser, world_size
    shutil.rmtree(tmpdir)


@pytest.fixture
def no_all2allv():
    """4 ranks with only allreduce events (no all2allv)."""
    world_size = 4
    events = {}
    for rank in range(world_size):
        events[rank] = [
            _make_nccl_event(
                rank=rank,
                collective_name="allreduce",
                ts=1000,
                dur=100,
                in_nelems=1024,
                out_nelems=1024,
                dtype="Float",
                in_split_size=None,
                out_split_size=None,
                external_id=500 + rank,
            )
        ]
    analyser, tmpdir = _build_analyser_from_events(events, world_size)
    yield analyser, world_size
    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Tests: per-instance metrics
# ---------------------------------------------------------------------------


class TestPerInstanceMetrics:
    def test_wall_time_computed(self, balanced_4rank):
        analyser, _ = balanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=False)
        assert df is not None
        assert "wall_time (us)" in df.columns
        assert all(df["wall_time (us)"] > 0)

    def test_throughput_computed(self, balanced_4rank):
        analyser, _ = balanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=False)
        assert "throughput (GB/s)" in df.columns
        assert all(df["throughput (GB/s)"] > 0)

    def test_size_imbalance_balanced(self, balanced_4rank):
        """Balanced splits => size_imbalance == 1.0."""
        analyser, _ = balanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=False)
        assert "size_imbalance" in df.columns
        for val in df["size_imbalance"]:
            assert abs(val - 1.0) < 1e-6

    def test_size_imbalance_skewed(self, imbalanced_4rank):
        """Imbalanced splits => size_imbalance > 1.0."""
        analyser, _ = imbalanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=False)
        assert all(df["size_imbalance"] > 1.0)

    def test_throughput_formula(self, balanced_4rank):
        """Verify throughput = total_data_MB / 1024 / (wall_time / 1e6)."""
        analyser, _ = balanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=False)
        for _, row in df.iterrows():
            expected = (row["total data communicated (MB)"] / 1024) / (
                row["wall_time (us)"] / 1e6
            )
            assert abs(row["throughput (GB/s)"] - expected) < 1e-6

    def test_max_min_avg_dur(self, balanced_4rank):
        analyser, _ = balanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=True)
        for _, row in df.iterrows():
            rank_durs = [row[c] for c in df.columns if c.endswith("_dur")]
            assert row["max_rank_dur (us)"] == max(rank_durs)
            assert row["min_rank_dur (us)"] == min(rank_durs)
            assert (
                abs(row["avg_rank_dur (us)"] - sum(rank_durs) / len(rank_durs)) < 1e-6
            )

    def test_rank_throughput_with_zero_data(self, zero_data_rank):
        """Rank 2 sends zero -> excluded from per-rank throughput."""
        analyser, _ = zero_data_rank
        df = analyser.build_df_nccl_all2allv(detailed=False)
        assert not math.isnan(df.iloc[0]["max_rank_throughput (GB/s)"])
        assert not math.isnan(df.iloc[0]["min_rank_throughput (GB/s)"])

    def test_new_columns_present_in_non_detailed(self, balanced_4rank):
        """New metric columns should appear even without detailed=True."""
        analyser, _ = balanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=False)
        expected_cols = [
            "wall_time (us)",
            "max_rank_dur (us)",
            "min_rank_dur (us)",
            "avg_rank_dur (us)",
            "throughput (GB/s)",
            "max_rank_throughput (GB/s)",
            "min_rank_throughput (GB/s)",
            "size_imbalance",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Tests: summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_produced(self, balanced_4rank):
        analyser, _ = balanced_4rank
        summary = analyser.build_df_summary_nccl_all2allv()
        assert summary is not None
        assert not summary.empty

    def test_summary_count(self, balanced_4rank):
        """Two invocations in one PG+dtype group => count == 2."""
        analyser, _ = balanced_4rank
        summary = analyser.build_df_summary_nccl_all2allv()
        assert summary.iloc[0]["count"] == 2

    def test_summary_none_when_no_all2allv(self, no_all2allv):
        analyser, _ = no_all2allv
        summary = analyser.build_df_summary_nccl_all2allv()
        assert summary is None

    def test_summary_grouped_by_pg_and_dtype(self, balanced_4rank):
        analyser, _ = balanced_4rank
        summary = analyser.build_df_summary_nccl_all2allv()
        assert "Process Group Name" in summary.columns
        assert "dtype" in summary.columns

    def test_summary_total_wall_time_ms(self, balanced_4rank):
        analyser, _ = balanced_4rank
        summary = analyser.build_df_summary_nccl_all2allv()
        assert "Total wall_time (ms)" in summary.columns
        assert "Total wall_time (us)" in summary.columns
        for _, row in summary.iterrows():
            assert (
                abs(row["Total wall_time (ms)"] - row["Total wall_time (us)"] / 1000)
                < 1e-6
            )

    def test_summary_throughput_stats(self, balanced_4rank):
        analyser, _ = balanced_4rank
        summary = analyser.build_df_summary_nccl_all2allv()
        for metric in ["mean", "std", "min", "max"]:
            assert f"throughput (GB/s)_{metric}" in summary.columns

    def test_summary_imbalance_stats(self, imbalanced_4rank):
        analyser, _ = imbalanced_4rank
        summary = analyser.build_df_summary_nccl_all2allv()
        assert summary.iloc[0]["size_imbalance_max"] > 1.0
        assert summary.iloc[0]["size_imbalance_mean"] > 1.0


# ---------------------------------------------------------------------------
# Tests: heatmap
# ---------------------------------------------------------------------------


class TestHeatmap:
    def test_heatmap_produced(self, balanced_4rank):
        analyser, _ = balanced_4rank
        heatmap = analyser.build_df_all2allv_heatmap()
        assert heatmap is not None
        assert not heatmap.empty

    def test_heatmap_columns(self, balanced_4rank):
        analyser, _ = balanced_4rank
        heatmap = analyser.build_df_all2allv_heatmap()
        expected_cols = [
            "src_rank",
            "dst_rank",
            "total_sent_MB",
            "avg_sent_MB",
            "count",
        ]
        for col in expected_cols:
            assert col in heatmap.columns

    def test_heatmap_pair_count(self, balanced_4rank):
        """4 ranks => 16 src-dst pairs."""
        analyser, _ = balanced_4rank
        heatmap = analyser.build_df_all2allv_heatmap()
        assert len(heatmap) == 16

    def test_heatmap_total_sent_balanced(self, balanced_4rank):
        """Each rank sends to each peer in 2 invocations: 1024 + 2048 = 3072 elems.
        BFloat16 = 2 bytes => 3072 * 2 / 1024^2 MB per pair."""
        analyser, _ = balanced_4rank
        heatmap = analyser.build_df_all2allv_heatmap()
        expected_mb = 3072 * 2 / (1024**2)
        for _, row in heatmap.iterrows():
            assert abs(row["total_sent_MB"] - expected_mb) < 1e-6

    def test_heatmap_invocation_count(self, balanced_4rank):
        """2 invocations => count == 2 for each pair."""
        analyser, _ = balanced_4rank
        heatmap = analyser.build_df_all2allv_heatmap()
        for _, row in heatmap.iterrows():
            assert row["count"] == 2

    def test_heatmap_none_when_no_all2allv(self, no_all2allv):
        analyser, _ = no_all2allv
        heatmap = analyser.build_df_all2allv_heatmap()
        assert heatmap is None

    def test_heatmap_zero_data_pairs(self, zero_data_rank):
        """Rank 2 sends 0 to everyone => those pairs should have 0 MB."""
        analyser, _ = zero_data_rank
        heatmap = analyser.build_df_all2allv_heatmap()
        rank2_sends = heatmap[heatmap["src_rank"] == 2]
        for _, row in rank2_sends.iterrows():
            assert row["total_sent_MB"] == 0.0

    def test_heatmap_imbalanced(self, imbalanced_4rank):
        """Rank 0 sends 2000 elems to each peer, others send 250.
        Float = 4 bytes. rank 0 -> any: 2000 * 4 / 1024^2 MB."""
        analyser, _ = imbalanced_4rank
        heatmap = analyser.build_df_all2allv_heatmap()
        rank0_sends = heatmap[heatmap["src_rank"] == 0]
        expected_r0 = 2000 * 4 / (1024**2)
        for _, row in rank0_sends.iterrows():
            assert abs(row["total_sent_MB"] - expected_r0) < 1e-6

        rank1_sends = heatmap[heatmap["src_rank"] == 1]
        expected_r1 = 250 * 4 / (1024**2)
        for _, row in rank1_sends.iterrows():
            assert abs(row["total_sent_MB"] - expected_r1) < 1e-6


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_invocation(self, imbalanced_4rank):
        """Summary works with a single invocation."""
        analyser, _ = imbalanced_4rank
        summary = analyser.build_df_summary_nccl_all2allv()
        assert summary is not None
        assert summary.iloc[0]["count"] == 1

    def test_detailed_false_excludes_per_rank_cols(self, balanced_4rank):
        analyser, _ = balanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=False)
        rank_cols = [c for c in df.columns if c.startswith("rank_")]
        assert len(rank_cols) == 0

    def test_detailed_true_includes_per_rank_cols(self, balanced_4rank):
        analyser, _ = balanced_4rank
        df = analyser.build_df_nccl_all2allv(detailed=True)
        rank_cols = [c for c in df.columns if c.startswith("rank_")]
        assert len(rank_cols) > 0
