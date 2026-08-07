###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for generate_multi_rank_collective_report_pytorch."""

import glob
import os

import pandas as pd
import pytest

from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
    DEFAULT_RANK_REGEX,
    _resolve_trace_files_glob,
    find_trace_files,
    generate_collective_report,
    infer_world_size,
)

LLAMA_TRACE_DIR = "tests/traces/mi300/llama_70b_fsdp"


def test_find_trace_files(tmp_path):
    (tmp_path / "rank0_trace.json").write_text("{}")
    (tmp_path / "rank1_trace.json").write_text("{}")
    (tmp_path / "other.json").write_text("{}")

    found = find_trace_files(str(tmp_path))
    assert len(found) == 2
    assert all("rank" in os.path.basename(p) for p in found)


def test_infer_world_size():
    assert infer_world_size(["a", "b", "c"]) == 3


def test_resolve_trace_files_glob(tmp_path):
    sub = tmp_path / "run"
    sub.mkdir()
    (sub / "trace_rank_0.json").write_text("{}")
    (sub / "trace_rank_1.json").write_text("{}")

    paths = _resolve_trace_files_glob(
        str(sub / "trace_rank_*.json"),
        world_size=2,
        rank_regex=DEFAULT_RANK_REGEX,
    )
    assert len(paths) == 2


def test_resolve_trace_files_glob_missing_ranks(tmp_path):
    (tmp_path / "trace_rank_0.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Missing ranks"):
        _resolve_trace_files_glob(
            str(tmp_path / "trace_rank_*.json"),
            world_size=2,
        )


def test_resolve_trace_files_glob_no_matches(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files matched"):
        _resolve_trace_files_glob(str(tmp_path / "*.json"), world_size=1)


def test_generate_collective_report_requires_world_size():
    with pytest.raises(ValueError, match="world_size must be provided"):
        generate_collective_report(trace_dir="/tmp", world_size=None)


def test_generate_collective_report_exclusive_inputs():
    with pytest.raises(ValueError, match="exactly one"):
        generate_collective_report(
            trace_dir="/tmp",
            trace_glob="/tmp/*.json",
            world_size=1,
        )


def test_generate_collective_report_invalid_gpus_per_node(tmp_path):
    (tmp_path / "rank0_trace.json").write_text('{"traceEvents": []}')
    with pytest.raises(ValueError, match="gpus_per_node"):
        generate_collective_report(
            trace_dir=str(tmp_path),
            world_size=1,
            gpus_per_node=0,
            strict_world_size_check=False,
        )


def test_generate_collective_report_trace_pattern_missing_file(tmp_path):
    pattern = str(tmp_path / "trace_rank_*.json")
    (tmp_path / "trace_rank_0.json").write_text('{"traceEvents": []}')

    with pytest.raises(FileNotFoundError, match="Expected trace file not found"):
        generate_collective_report(trace_pattern=pattern, world_size=2)


def test_generate_collective_report_llama_traces(tmp_path):
    """Direct API call for coverage of generate_collective_report."""
    if not os.path.isdir(LLAMA_TRACE_DIR):
        pytest.skip(f"Trace dir not found: {LLAMA_TRACE_DIR}")

    pattern = os.path.join(LLAMA_TRACE_DIR, "rank*_trace_no_pyfn.json.gz")
    world_size = len(glob.glob(pattern))
    if world_size == 0:
        pytest.skip("No rank traces found")

    out_dir = str(tmp_path / "nccl_csvs")
    dfs = generate_collective_report(
        trace_pattern=pattern,
        world_size=world_size,
        output_csvs_dir=out_dir,
        detailed_analysis=False,
        gpus_per_node=8,
    )
    assert isinstance(dfs, dict)
    assert "nccl_summary_implicit_sync" in dfs
    assert isinstance(dfs["nccl_summary_implicit_sync"], pd.DataFrame)
    assert os.path.isfile(os.path.join(out_dir, "nccl_summary_implicit_sync.csv"))
