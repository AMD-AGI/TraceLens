###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.util helpers."""

import os

import pytest

from TraceLens.util import (
    JaxProfileProcessor,
    TraceEventUtils,
    suppress_native_hlo_logs,
)

TK = TraceEventUtils.TraceKeys
TP = TraceEventUtils.TracePhases
MF = TraceEventUtils.MetadataFields
AN = TraceEventUtils.ArgNames
JST = TraceEventUtils.JaxSpecialThreads


def test_suppress_native_hlo_logs_filters_noise(monkeypatch):
    monkeypatch.delenv("TRACELENS_VERBOSE_NATIVE_LOGS", raising=False)
    writes = []
    real_write = os.write

    def capture_write(fd, data):
        if fd == 2:
            writes.append(data)
            return len(data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", capture_write)

    with suppress_native_hlo_logs():
        real_write(2, b"Instruction with id > INT_MAX\n")
        real_write(2, b"real message\n")

    assert any(b"real message" in w for w in writes)
    assert not any(b"INT_MAX" in w for w in writes)


def test_suppress_native_hlo_logs_disabled_when_verbose(monkeypatch):
    monkeypatch.setenv("TRACELENS_VERBOSE_NATIVE_LOGS", "1")
    writes = []
    real_write = os.write

    def capture_write(fd, data):
        writes.append((fd, data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", capture_write)

    with suppress_native_hlo_logs():
        os.write(2, b"Instruction with id > INT_MAX\n")

    assert any(fd == 2 and b"INT_MAX" in data for fd, data in writes)


@pytest.mark.parametrize(
    "line,expected",
    [
        ("", False),
        ("HloModule main", False),
        ("ROOT %x = f32[] constant(0)", False),
        ('metadata={op_name="foo"', True),
        ("%x = bf16[128,256]{1,0} parameter(0)", True),
        ("%x = s32[4]{0} constant({1,2,3,4})", True),
        ("%x = pred[] compare(...)", True),
        ("unrelated text without dtype hints", False),
    ],
)
def test_should_parse_hlo_graph_line(line, expected):
    assert JaxProfileProcessor._should_parse_hlo_graph_line(line) is expected


def test_get_operands_extracts_typed_tensor_tokens():
    operands = "custom-call(f32[128,256]{1,0} %lhs, f32[256,512]{1,0} %rhs),"
    result = JaxProfileProcessor.get_operands(operands)
    assert result == ["f32[128,256]{1,0}", "f32[256,512]{1,0}"]


def test_get_dict_parses_metadata_backend_config_and_custom_call():
    line = (
        "%gemm.1 = f32[128,256]{1,0} custom-call(f32[128,256]{1,0} %arg0, "
        'f32[256,512]{1,0} %arg1), custom_call_target="cublasLt", '
        'metadata={op_name="gemm"}, backend_config={"alpha":1.0}'
    )
    hlo_ops = {
        "%arg0": {"output": "f32[128,256]{1,0}"},
        "%arg1": {"output": "f32[256,512]{1,0}"},
    }
    key, parsed = JaxProfileProcessor.get_dict(hlo_ops, line)

    assert key == "%gemm.1"
    assert parsed["output"] == "f32[128,256]{1,0}"
    assert parsed["computation"] == "gemm"
    assert parsed["type"] == "f32"
    assert "metadata=" in parsed["metadata"]
    assert "backend_config=" in parsed["backend_config"]
    assert "custom_call_target=" in parsed["custom_call_target"]


def test_get_dict_parses_replica_groups():
    line = (
        "%all-gather-start = f32[8,128]{1,0} all-gather-start(f32[8,128]{1,0} %input), "
        "replica_groups={{0,1},{2,3}}"
    )
    key, parsed = JaxProfileProcessor.get_dict({}, line)
    assert key == "%all-gather-start"
    assert parsed["replica_groups"] == "{{0,1},{2,3}}"


def test_process_line_skips_unparseable_lines():
    hlo_ops = {}
    assert JaxProfileProcessor.process_line(hlo_ops, "HloModule main") is False
    assert hlo_ops == {}

    assert (
        JaxProfileProcessor.process_line(hlo_ops, "%x = bf16[4,8]{1,0} parameter(0)")
        is True
    )
    assert "%x" in hlo_ops


def test_resolve_operand_references_substitutes_output_types():
    hlo_ops = {
        "%src": {"output": "bf16[128,256]{1,0}", "operands": []},
        "%user": {"output": "bf16[128,256]{1,0}", "operands": ["%src"]},
    }
    JaxProfileProcessor._resolve_operand_references(hlo_ops)
    assert hlo_ops["%user"]["operands"] == ["bf16[128,256]{1,0}"]


def test_normalize_hlo_op_key_adds_percent_prefix():
    assert JaxProfileProcessor._normalize_hlo_op_key("gemm.1") == "%gemm.1"
    assert JaxProfileProcessor._normalize_hlo_op_key("%gemm.1") == "%gemm.1"


def test_build_collective_hlo_aliases_maps_numbered_runtime_ops():
    module_ops = {
        "%reduce-scatter-start": {"output": "f32[8,128]{1,0}"},
    }
    aliases = JaxProfileProcessor.build_collective_hlo_aliases(
        module_ops, ["reduce-scatter.12"]
    )
    assert aliases["%reduce-scatter.12"] == "%reduce-scatter-start"


def test_resolve_hlo_op_key_uses_aliases_and_direct_keys():
    module_ops = {"%gemm.0": {"output": "f32[1,1]{0}"}}
    aliases = {"%gemm.1": "%gemm.0"}

    assert JaxProfileProcessor.resolve_hlo_op_key("gemm.0", module_ops) == "%gemm.0"
    assert (
        JaxProfileProcessor.resolve_hlo_op_key("gemm.1", module_ops, aliases)
        == "%gemm.0"
    )
    assert JaxProfileProcessor.resolve_hlo_op_key("missing", module_ops) is None


def test_get_operand_type_from_prefix_or_lookup():
    hlo_ops = {"%arg": {"output": "bf16[4,8]{1,0}"}}
    assert JaxProfileProcessor.get_operand_type(hlo_ops, "bf16[4,8]{1,0}") == "bf16"
    assert JaxProfileProcessor.get_operand_type(hlo_ops, "%arg") == "bf16"


def test_trace_event_utils_split_by_field():
    k1 = {"cat": "kernel", "i": 1}
    c1 = {"cat": "cpu_op", "i": 2}
    k2 = {"cat": "kernel", "i": 3}
    grouped = TraceEventUtils.split_by_field([k1, c1, k2], "cat")
    assert {key: list(val) for key, val in grouped.items()} == {
        "kernel": [k1, k2],
        "cpu_op": [c1],
    }


def _metadata_event(pid, tid, field, value):
    arg_key = {
        MF.ProcessName: AN.Name,
        MF.ProcessLabels: AN.Labels,
        MF.ProcessSort: AN.SortIndex,
        MF.ThreadName: AN.Name,
        MF.ThreadSort: AN.SortIndex,
    }[field]
    return {
        TK.PID: pid,
        TK.TID: tid,
        TK.Phase: TP.Metadata,
        TK.Name: field,
        TK.Args: {arg_key: value},
    }


def test_trace_event_utils_split_event_list():
    events = [
        _metadata_event(1, None, MF.ProcessName, "host"),
        _metadata_event(1, 2, MF.ThreadName, JST.XlaOps),
        {TK.PID: 1, TK.TID: 2, TK.Phase: TP.Complete, TK.Name: "op"},
    ]
    metadata, rest = TraceEventUtils.split_event_list(events)
    assert metadata[1][None][MF.ProcessName] == "host"
    assert metadata[1][2][MF.ThreadName] == JST.XlaOps
    assert len(rest) == 1


def test_trace_event_utils_get_event_category():
    events = [
        _metadata_event(1, None, MF.ProcessName, "host"),
        _metadata_event(1, 7, MF.ThreadName, JST.XlaOps),
        {
            TK.PID: 1,
            TK.TID: 7,
            TK.Phase: TP.Complete,
            TK.Name: "my_op",
        },
    ]
    categorizer = TraceEventUtils.prepare_event_categorizer(events)
    assert categorizer(events[-1]) == "python function"


def test_trace_event_utils_sort_events_by_timestamp_duration():
    events = [
        {TK.TimeStamp: 20, TK.Duration: 1},
        {TK.TimeStamp: 10, TK.Duration: 5},
        {TK.TimeStamp: 10, TK.Duration: 2},
    ]
    TraceEventUtils.sort_events_by_timestamp_duration(events)
    assert [e[TK.TimeStamp] for e in events] == [10, 10, 20]
    assert [e[TK.Duration] for e in events] == [2, 5, 1]
