###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer


def _minimal_events():
    return [
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::empty",
            "pid": 1,
            "tid": 1,
            "ts": 10,
            "dur": 5,
            "args": {},
        }
    ]


def test_trace_to_tree_defaults_to_empty_trace_metadata():
    tree = TraceToTree(_minimal_events())

    assert tree.trace_metadata == {}
    assert "trace_metadata" not in tree.events[0]


def test_trace_to_tree_preserves_trace_metadata():
    metadata = {
        "traceName": "example_trace.json",
        "record_shapes": 1,
        "deviceProperties": [{"name": "gfx942", "totalGlobalMem": 1}],
    }

    tree = TraceToTree(_minimal_events(), trace_metadata=metadata)

    assert tree.trace_metadata == metadata
    assert tree.trace_metadata is not metadata
    assert "traceEvents" not in tree.trace_metadata
    assert "trace_metadata" not in tree.events[0]


def test_tree_perf_analyzer_from_file_exposes_trace_metadata(tmp_path):
    trace = {
        "schemaVersion": 1,
        "traceName": "from_file_trace.json",
        "record_shapes": 1,
        "deviceProperties": [{"name": "gfx942"}],
        "traceEvents": _minimal_events(),
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    analyzer = TreePerfAnalyzer.from_file(trace_path.as_posix(), rebuild_tree=False)

    expected_metadata = {
        "schemaVersion": 1,
        "traceName": "from_file_trace.json",
        "record_shapes": 1,
        "deviceProperties": [{"name": "gfx942"}],
    }
    assert analyzer.trace_metadata == expected_metadata
    assert analyzer.tree.trace_metadata == expected_metadata
    assert "traceEvents" not in analyzer.trace_metadata
