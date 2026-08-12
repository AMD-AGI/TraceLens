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


# --- migrated from test_coverage_95_final.py ---
import gzip
import importlib
import json
import os
import sys
from copy import deepcopy
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _extract_comparative_fusion_candidates,
)
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.Reporting import reporting_utils as ru
from TraceLens.Reporting.compare_traces_jax_llama import (
    Event,
    Summary,
    classify_stage_base,
    compute_stage_table,
    emit_report,
    extract_gpu_events,
    infer_params,
    is_loop_multiply_fusion,
    load_trace,
    mk_stats,
    percentile,
    summarize_one,
    token_start_times,
    top_stats_by_key,
)
from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
    _resolve_trace_files_glob,
    generate_collective_report,
)
from TraceLens.Reporting.rocprof_analysis import RocprofAnalyzer, _categorize_kernel
from TraceLens.Trace2Tree.extensions.pseudo_ops_registry import (
    apply_pseudo_op_extensions,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import align_streams
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import JaxTreePerfAnalyzer, TreePerfAnalyzer
from TraceLens.util import RocprofParser
from tests.fixtures.agent import _StubAnalyzer, _StubTree, _kernel_event
from tests.test_jax_analysis_report import _mock_side_inputs, _sample_averages_df
from tests.fixtures.reporting import _mk_ac2g, _mk_event
from tests.fixtures.treeperf import (
    _build_analyzer,
    _make_gpu_event,
    _mk_pytorch_trace,
)


class TestCollectiveReportErrors:
    def test_glob_rank_resolution_errors(self, tmp_path):
        (tmp_path / "bad.json").write_text("{}")
        with pytest.raises(ValueError, match="none matched"):
            _resolve_trace_files_glob(str(tmp_path / "*.json"), world_size=2)
        for rank in (0, 1):
            (tmp_path / f"trace_rank_{rank}.json").write_text("{}")
        paths = _resolve_trace_files_glob(str(tmp_path / "trace_rank_*.json"), 2)
        assert len(paths) == 2

    def test_collective_trace_pattern_and_all2allv(self, tmp_path):
        for rank in (0, 1):
            events = {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ncclKernel_AllReduce",
                        "pid": rank,
                        "tid": 3,
                        "ts": 1000,
                        "dur": 40,
                        "args": {
                            "External id": 10,
                            "Collective name": "allreduce",
                            "stream": 3,
                        },
                    }
                ]
            }
            (tmp_path / f"trace_{rank}_step.json").write_text(json.dumps(events))
        dfs = generate_collective_report(
            trace_pattern=str(tmp_path / "trace_*_step.json"),
            world_size=2,
            output_csvs_dir=str(tmp_path / "coll"),
            use_multiprocessing=False,
            strict_world_size_check=True,
            all2allv_heatmap=True,
        )
        assert isinstance(dfs, dict)

    def test_gpus_per_node_invalid(self, tmp_path):
        for rank in (0,):
            (tmp_path / f"rank{rank}_trace.json").write_text(
                json.dumps({"traceEvents": []})
            )
        with pytest.raises(ValueError, match="gpus_per_node"):
            generate_collective_report(
                trace_dir=str(tmp_path),
                world_size=1,
                gpus_per_node=0,
                strict_world_size_check=False,
            )


# --- migrated from test_coverage_push95.py::TestCoveragePush95Phase2.test_collective_report_main ---
import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _extract_comparative_fusion_candidates,
    _extract_standalone_fusion_candidates,
)
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.PerfModel.extensions import attention_perf_model_extensions as attn_ext
from TraceLens.PerfModel.extensions import rmsnorm_perf_model_extensions as rms_ext
from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch as generate_inference_report,
)
from TraceLens.Reporting.generate_perf_report_pftrace_hip_activity import (
    generate_perf_report_pftrace_hip_activity,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    _align_graph_to_capture_by_group,
    find_closest_batch_size,
    load_capture_folder,
    merge_capture_trace_into_graph,
    verify_subtree_events,
)
from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.agent import (
    _StubAnalyzer,
    _StubTree,
    _kernel_event,
    _write_minimal_orchestrator_csvs,
)
from tests.test_conv_backward_bytes import _conv_bias_bwd_event
from tests.fixtures.perfmodel import _ARCH, _GDN_ANNOTATION, _moe_unfused_event
from tests.fixtures.reporting import (
    _create_genesis_capture,
    _minimal_pftrace_events,
    _write_trace,
)
from tests.fixtures.treeperf import _build_analyzer


def test_collective_report_main(tmp_path):
    for rank in (0, 1):
        (tmp_path / f"trace_rank_{rank}.json").write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "ph": "X",
                            "cat": "kernel",
                            "name": "ncclKernel_AllReduce",
                            "pid": rank,
                            "tid": 3,
                            "ts": 1000,
                            "dur": 40,
                            "args": {
                                "External id": rank,
                                "Collective name": "allreduce",
                                "stream": 3,
                            },
                        }
                    ]
                }
            )
        )
    out = tmp_path / "coll.xlsx"
    mod = importlib.import_module(
        "TraceLens.Reporting.generate_multi_rank_collective_report_pytorch"
    )
    old_argv = sys.argv
    sys.argv = [
        "generate_multi_rank_collective_report_pytorch",
        "--trace_glob",
        str(tmp_path / "trace_rank_*.json"),
        "--world_size",
        "2",
        "--output_xlsx_path",
        str(out),
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert out.exists()


# --- migrated from test_reporting_cli_coverage.py ---
import importlib
import json
import os
import sys
import pytest
from tests.fixtures.reporting import (
    _minimal_pftrace_events,
    _write_trace,
)


def test_collective_report_main_trace_glob(tmp_path):
    for rank in (0, 1):
        events = {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "ncclKernel_AllReduce",
                    "pid": rank,
                    "tid": 3,
                    "ts": 1000,
                    "dur": 40,
                    "args": {
                        "External id": rank,
                        "Collective name": "allreduce",
                        "stream": 3,
                    },
                }
            ]
        }
        (tmp_path / f"trace_rank_{rank}.json").write_text(json.dumps(events))
    out = tmp_path / "coll.xlsx"
    import TraceLens.Reporting.generate_multi_rank_collective_report_pytorch as mod

    old_argv = sys.argv
    sys.argv = [
        "generate_multi_rank_collective_report_pytorch",
        "--trace_glob",
        str(tmp_path / "trace_rank_*.json"),
        "--world_size",
        "2",
        "--output_xlsx_path",
        str(out),
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert out.exists()
