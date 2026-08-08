###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Extra inference trace report runs for coverage (no CSV regression)."""

from __future__ import annotations

import os

import pytest

from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    generate_perf_report_pytorch,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    _align_capture_to_graph,
    _align_graph_to_capture_by_group,
    align_streams,
    capture_has_kernel_names,
    merge_capture_trace_into_graph,
)

INFERENCE_ROOT = os.path.join(os.path.dirname(__file__), "traces/inference")


def _discover_cases():
    if not os.path.isdir(INFERENCE_ROOT):
        return []
    cases = []
    for entry in sorted(os.listdir(INFERENCE_ROOT)):
        dirpath = os.path.join(INFERENCE_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        if not gz:
            continue
        capture = os.path.join(dirpath, "capture_traces")
        cases.append(
            pytest.param(
                dirpath,
                gz[0],
                capture if os.path.isdir(capture) else None,
                id=entry,
            )
        )
    return cases


@pytest.mark.parametrize("dirpath,trace_gz,capture_folder", _discover_cases())
def test_inference_report_extended_flags(dirpath, trace_gz, capture_folder, tmp_path):
    trace_path = os.path.join(dirpath, trace_gz)
    out = tmp_path / "out"
    generate_perf_report_pytorch(
        profile_json_path=trace_path,
        output_csvs_dir=str(out),
        output_xlsx_path=str(tmp_path / "report.xlsx"),
        collective_analysis=True,
        kernel_summary=True,
        short_kernel_study=True,
        include_overlap_info=True,
        group_by_parent_module=True,
        group_by_num_kernels=True,
        enable_pseudo_ops=True,
        micro_idle_thresh_us=1,
    )
    assert (out / "gpu_timeline.csv").exists()


@pytest.mark.parametrize("dirpath,trace_gz,capture_folder", _discover_cases())
def test_merge_capture_trace_integration(dirpath, trace_gz, capture_folder):
    if capture_folder is None:
        pytest.skip("no capture traces")
    metadata = os.path.join(capture_folder, "execution_details.json")
    if not os.path.isfile(metadata):
        pytest.skip("no execution_details.json")
    trace_path = os.path.join(dirpath, trace_gz)
    merged = merge_capture_trace_into_graph(
        capture_folder, metadata, trace_path
    )
    assert len(merged.events) > 0


class TestCaptureMergeHelpers:
    def test_align_capture_to_graph_memcpy(self):
        capture = [{"name": "cudaMemcpy", "args": {}}]
        graph = [{"name": "MemcpyHtoD", "args": {}}]
        aligned = _align_capture_to_graph(capture, graph)
        assert aligned is not None

    def test_align_capture_to_graph_mismatch(self):
        capture = [{"name": "hipLaunchKernel", "args": {"kernel": "a"}}]
        graph = [{"name": "b", "args": {}}]
        assert _align_capture_to_graph(capture, graph) is None

    def test_align_graph_to_capture_group_mismatch(self):
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
        ]
        graph = [{"name": "k1", "args": {}}]
        assert _align_graph_to_capture_by_group(capture, graph) is None

    def test_align_streams_and_capture_has_names(self):
        graph = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k2", "args": {"stream": 2}},
        ]
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
        ]
        assert capture_has_kernel_names(capture)
        aligned = align_streams(graph, capture)
        assert aligned is not None

    def test_capture_missing_kernel_name(self):
        capture = [{"name": "hipLaunchKernel", "args": {}}]
        assert capture_has_kernel_names(capture) is False
