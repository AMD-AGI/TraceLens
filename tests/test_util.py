###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.util helpers."""

import contextlib
import gzip
import json
import os
import sys
import types
from unittest.mock import patch

import pytest

from TraceLens.util import (
    DataLoader,
    JaxProfileProcessor,
    PftraceParser,
    RocprofParser,
    TraceEventUtils,
    suppress_native_hlo_logs,
)

TK = TraceEventUtils.TraceKeys
TP = TraceEventUtils.TracePhases
MF = TraceEventUtils.MetadataFields
AN = TraceEventUtils.ArgNames
JST = TraceEventUtils.JaxSpecialThreads

_GEMM_BACKEND = (
    'backend_config={"gemm_backend_config":{"epilogue":"DEFAULT","beta":0,'
    '"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]}}'
)


def _install_mock_xprof_convert(return_value):
    mock_mod = types.ModuleType("xprof.convert.raw_to_tool_data")

    def xspace_to_tool_data(*args, **kwargs):
        return return_value

    def xspace_to_tool_names(*args, **kwargs):
        return None

    mock_mod.xspace_to_tool_data = xspace_to_tool_data
    mock_mod.xspace_to_tool_names = xspace_to_tool_names

    fake_convert = types.ModuleType("xprof.convert")
    fake_convert.raw_to_tool_data = mock_mod

    fake_xprof = types.ModuleType("xprof")
    fake_xprof.convert = fake_convert

    return {
        "xprof": fake_xprof,
        "xprof.convert": fake_convert,
        "xprof.convert.raw_to_tool_data": mock_mod,
    }


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


def test_suppress_native_hlo_logs_filters_hlo_instruction_noise(monkeypatch):
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
        real_write(2, b"hlo_instruction.cc: noisy line\n")
        real_write(2, b"kept message\n")

    assert any(b"kept message" in w for w in writes)
    assert not any(b"hlo_instruction.cc" in w for w in writes)


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


def test_get_operands_falls_back_to_comma_split():
    assert JaxProfileProcessor.get_operands("(%arg0,%arg1)") == ["%arg0", "%arg1"]


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


def test_get_dict_fp8_gemm_type():
    line = (
        "%gemm.1 = f8e5m2[128,256]{1,0} custom-call(f8e5m2[128,512]{1,0} %arg0, "
        'f8e5m2[512,256]{1,0} %arg1), custom_call_target="cublasLt_f8"'
    )
    key, parsed = JaxProfileProcessor.get_dict({}, line)
    assert key == "%gemm.1"
    assert parsed["type"] == "fp8"
    assert parsed["computation"] == "gemm"


def test_get_dict_operand_type_mismatch_raises():
    line = (
        "%gemm.1 = f32[128,256]{1,0} custom-call(bf16[128,512]{1,0} %arg0, "
        'f32[512,256]{1,0} %arg1), custom_call_target="cublasLt"'
    )
    with pytest.raises(Exception, match="Input operand type mismatch"):
        JaxProfileProcessor.get_dict({}, line)


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


def test_process_xla_file(tmp_path):
    hlo_path = tmp_path / "module.hlo.txt"
    hlo_path.write_text(
        "HloModule main\n"
        "%x = bf16[4,8]{1,0} parameter(0)\n"
        "%y = f32[4,8]{1,0} add(%x, %x)\n"
    )

    hlo_ops = JaxProfileProcessor.process_xla_file(str(hlo_path))

    assert "%x" in hlo_ops
    assert "%y" in hlo_ops


def test_resolve_operand_references_substitutes_output_types():
    hlo_ops = {
        "%src": {"output": "bf16[128,256]{1,0}", "operands": []},
        "%user": {"output": "bf16[128,256]{1,0}", "operands": ["%src"]},
    }
    JaxProfileProcessor._resolve_operand_references(hlo_ops)
    assert hlo_ops["%user"]["operands"] == ["bf16[128,256]{1,0}"]


def test_resolve_operand_references_skips_non_list_operands():
    hlo_ops = {"%x": {"output": "f32[1]{0}", "operands": "not-a-list"}}
    JaxProfileProcessor._resolve_operand_references(hlo_ops)
    assert hlo_ops["%x"]["operands"] == "not-a-list"


def test_resolve_operand_references_warns_on_unresolved(caplog):
    hlo_ops = {"%user": {"output": "f32[1]{0}", "operands": ["%missing"]}}
    JaxProfileProcessor._resolve_operand_references(hlo_ops)
    assert hlo_ops["%user"]["operands"] == ["%missing"]
    assert "Unable to resolve HLO operand reference" in caplog.text


def test_normalize_hlo_op_key_adds_percent_prefix():
    assert JaxProfileProcessor._normalize_hlo_op_key("gemm.1") == "%gemm.1"
    assert JaxProfileProcessor._normalize_hlo_op_key("%gemm.1") == "%gemm.1"


def test_collective_start_keys_prefers_done_when_no_start():
    module_ops = {
        "%all-to-all-done": {"output": "f32[1]{0}"},
    }
    keys = JaxProfileProcessor._collective_start_keys(module_ops, "all-to-all")
    assert keys == ["%all-to-all-done"]


def test_build_collective_hlo_aliases_maps_numbered_runtime_ops():
    module_ops = {
        "%reduce-scatter-start": {"output": "f32[8,128]{1,0}"},
    }
    aliases = JaxProfileProcessor.build_collective_hlo_aliases(
        module_ops, ["reduce-scatter.12"]
    )
    assert aliases["%reduce-scatter.12"] == "%reduce-scatter-start"


def test_build_collective_hlo_aliases_multiple_start_keys():
    module_ops = {
        "%reduce-scatter-start.0": {"output": "f32[8,128]{1,0}"},
        "%reduce-scatter-start.1": {"output": "f32[8,128]{1,0}"},
    }
    aliases = JaxProfileProcessor.build_collective_hlo_aliases(
        module_ops,
        ["reduce-scatter.0", "reduce-scatter.1", "reduce-scatter.2"],
    )
    assert aliases["%reduce-scatter.0"] == "%reduce-scatter-start.0"
    assert aliases["%reduce-scatter.1"] == "%reduce-scatter-start.1"
    assert aliases["%reduce-scatter.2"] == "%reduce-scatter-start.1"


def test_build_collective_hlo_aliases_skips_existing_and_non_numbered():
    module_ops = {
        "%all-gather-start": {"output": "f32[8,128]{1,0}"},
        "%all-gather.0": {"output": "f32[8,128]{1,0}"},
    }
    aliases = JaxProfileProcessor.build_collective_hlo_aliases(
        module_ops,
        ["all-gather.0", "all-gather.foo", "gemm.0"],
    )
    assert aliases == {}


def test_build_collective_hlo_aliases_no_start_keys():
    aliases = JaxProfileProcessor.build_collective_hlo_aliases(
        {},
        ["reduce-scatter.0"],
    )
    assert aliases == {}


def test_resolve_hlo_op_key_uses_aliases_and_direct_keys():
    module_ops = {"%gemm.0": {"output": "f32[1,1]{0}"}}
    aliases = {"%gemm.1": "%gemm.0"}

    assert JaxProfileProcessor.resolve_hlo_op_key("gemm.0", module_ops) == "%gemm.0"
    assert (
        JaxProfileProcessor.resolve_hlo_op_key("gemm.1", module_ops, aliases)
        == "%gemm.0"
    )
    assert JaxProfileProcessor.resolve_hlo_op_key("missing", module_ops) is None


def test_resolve_hlo_op_key_collective_numbered_fallback():
    module_ops = {"%all-gather-start": {"output": "f32[8,128]{1,0}"}}
    assert (
        JaxProfileProcessor.resolve_hlo_op_key("all-gather.5", module_ops)
        == "%all-gather-start"
    )


def test_get_operand_type_from_prefix_or_lookup():
    hlo_ops = {"%arg": {"output": "bf16[4,8]{1,0}"}}
    assert JaxProfileProcessor.get_operand_type(hlo_ops, "bf16[4,8]{1,0}") == "bf16"
    assert JaxProfileProcessor.get_operand_type(hlo_ops, "%arg") == "bf16"


def test_get_operand_type_fusion_prefix_and_none():
    hlo_ops = {
        "%arg": {"output": "s8[4,8]{1,0}"},
        "%typed": {"output": "bf16[4,8]{1,0}"},
    }
    assert JaxProfileProcessor.get_operand_type(hlo_ops, "fusion,%typed") == "bf16"
    assert JaxProfileProcessor.get_operand_type(hlo_ops, "%arg") is None


def test_process_gemm_ops_simple_f32():
    hlo_ops = {
        "%gemm.0": {
            "computation": "gemm",
            "type": "f32",
            "output": "(f32[128,256]{1,0})",
            "operands": ["f32[128,512]{1,0}", "f32[512,256]{1,0}"],
            "backend_config": _GEMM_BACKEND,
        }
    }
    gemm_dict = JaxProfileProcessor.process_gemm_ops(hlo_ops)
    assert gemm_dict["%gemm.0"] == {
        "Batch": 1,
        "M": 256,
        "N": 128,
        "K": 512,
        "Beta": 0,
        "Type": "f32",
        "Computation": "gemm",
    }


def test_process_gemm_ops_tuple_output_and_operand_lookup():
    hlo_ops = {
        "%lhs": {
            "computation": "rest",
            "output": "f8e5m2[128,512]{1,0}",
            "operands": [],
        },
        "%rhs": {
            "computation": "rest",
            "output": "f8e5m2[512,256]{1,0}",
            "operands": [],
        },
        "%gemm.0": {
            "computation": "gemm",
            "type": "fp8",
            "output": "(f8e5m2[128,256]{1,0}, f32[], s8[1024]{0})",
            "operands": ["%lhs", "%rhs"],
            "backend_config": _GEMM_BACKEND,
        },
    }
    gemm_dict = JaxProfileProcessor.process_gemm_ops(hlo_ops)
    assert gemm_dict["%gemm.0"]["Type"] == "fp8"
    assert gemm_dict["%gemm.0"]["K"] == 512


def test_process_gemm_ops_missing_backend_config_raises():
    hlo_ops = {
        "%gemm.0": {
            "computation": "gemm",
            "type": "f32",
            "output": "f32[1,1]{0,1}",
            "operands": ["f32[1,1]{0,1}", "f32[1,1]{0,1}"],
        }
    }
    with pytest.raises(ValueError, match="Gemm backend config"):
        JaxProfileProcessor.process_gemm_ops(hlo_ops)


def test_process_gemm_ops_batch_and_c_order():
    backend = (
        'backend_config={"gemm_backend_config":{"epilogue":"DEFAULT","beta":0,'
        '"lhs_contracting_dimensions":["2"],"rhs_contracting_dimensions":["1"]}}'
    )
    hlo_ops = {
        "%gemm.0": {
            "computation": "gemm",
            "type": "f32",
            "output": "(f32[2,128,256]{0,1,2})",
            "operands": ["f32[2,128,512]{0,1,2}", "f32[2,512,256]{0,1,2}"],
            "backend_config": backend,
        }
    }
    gemm_dict = JaxProfileProcessor.process_gemm_ops(hlo_ops)
    assert gemm_dict["%gemm.0"]["Batch"] == 2
    assert gemm_dict["%gemm.0"]["K"] == 512


def test_process_gemm_ops_beta_bias_and_invalid_cases():
    bias_backend = (
        'backend_config={"gemm_backend_config":{"epilogue":"BIAS","beta":1,'
        '"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]}}'
    )
    hlo_ops = {
        "%gemm.0": {
            "computation": "gemm",
            "type": "f32",
            "output": "(f32[128,256]{0,1})",
            "operands": [
                "f32[128,512]{0,1}",
                "f32[512,256]{0,1}",
                "f32[128,256]{0,1}",
                "f32[128,256]{0,1}",
            ],
            "backend_config": bias_backend,
        }
    }
    gemm_dict = JaxProfileProcessor.process_gemm_ops(hlo_ops)
    assert gemm_dict["%gemm.0"]["Beta"] == 1

    with pytest.raises(ValueError, match="Invalid operand list"):
        JaxProfileProcessor.process_gemm_ops(
            {
                "%gemm.1": {
                    **hlo_ops["%gemm.0"],
                    "operands": [],
                }
            }
        )

    with pytest.raises(ValueError, match="bias epilogue is nto set"):
        JaxProfileProcessor.process_gemm_ops(
            {
                "%gemm.2": {
                    **hlo_ops["%gemm.0"],
                    "backend_config": _GEMM_BACKEND,
                }
            }
        )

    with pytest.raises(ValueError, match="contracting dimension not matching"):
        JaxProfileProcessor.process_gemm_ops(
            {
                "%gemm.3": {
                    "computation": "gemm",
                    "type": "f32",
                    "output": "(f32[128,256]{1,0})",
                    "operands": ["f32[128,512]{1,0}", "f32[256,512]{1,0}"],
                    "backend_config": _GEMM_BACKEND,
                }
            }
        )


def test_process_gemm_ops_additional_error_paths(capsys):
    with pytest.raises(ValueError, match="tensor size is more than 3"):
        JaxProfileProcessor.process_gemm_ops(
            {
                "%gemm.4": {
                    "computation": "gemm",
                    "type": "f32",
                    "output": "(f32[1,2,3,4]{0,1,2,3})",
                    "operands": ["f32[1,2,3,4]{0,1,2,3}", "f32[1,2,3,4]{0,1,2,3}"],
                    "backend_config": _GEMM_BACKEND,
                }
            }
        )

    with pytest.raises(ValueError, match="Mistmatched parens"):
        JaxProfileProcessor.process_gemm_ops(
            {
                "%gemm.5": {
                    "computation": "gemm",
                    "type": "f32",
                    "output": "(f32[128,256]{1,0}",
                    "operands": ["f32[128,512]{1,0}", "f32[512,256]{1,0}"],
                    "backend_config": _GEMM_BACKEND,
                }
            }
        )

    with pytest.raises(ValueError, match="Did not find wide output"):
        JaxProfileProcessor.process_gemm_ops(
            {
                "%gemm.6": {
                    "computation": "gemm",
                    "type": "f32",
                    "output": "(f32[], s8[1024]{0})",
                    "operands": ["f32[128,512]{1,0}", "f32[512,256]{1,0}"],
                    "backend_config": _GEMM_BACKEND,
                }
            }
        )

    JaxProfileProcessor.process_gemm_ops(
        {
            "%gemm.7": {
                "computation": "gemm",
                "type": "f32",
                "output": "(f32[128,256]{1,0})",
                "operands": ["f32[128,512]{1,0}", "f32[512,256]{1,0}"],
                "backend_config": (
                    'backend_config={"gemm_backend_config":{"epilogue":"DEFAULT","beta":1,'
                    '"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]}}'
                ),
            }
        }
    )
    assert "onLy two operands found" in capsys.readouterr().out


def test_process_gemm_ops_skips_missing_hlo_operand_reference():
    hlo_ops = {
        "%gemm.8": {
            "computation": "gemm",
            "type": "f32",
            "output": "(f32[128,256]{1,0})",
            "operands": ["f32[128,512]{1,0}", "%missing", "f32[512,256]{1,0}"],
            "backend_config": _GEMM_BACKEND,
        }
    }
    gemm_dict = JaxProfileProcessor.process_gemm_ops(hlo_ops)
    assert gemm_dict["%gemm.8"]["K"] == 512


def test_process_gemm_ops_non_tuple_output_assigns_raw_output():
    hlo_ops = {
        "%gemm.9": {
            "computation": "gemm",
            "type": "f32",
            "output": "f32[128,256]{1,0}",
            "operands": ["f32[128,512]{1,0}", "f32[512,256]{1,0}"],
            "backend_config": _GEMM_BACKEND,
        }
    }
    with pytest.raises(TypeError):
        JaxProfileProcessor.process_gemm_ops(hlo_ops)


@patch("TraceLens.util.glob.glob")
@patch("TraceLens.util.suppress_native_hlo_logs")
def test_process_protobuf_file(mock_suppress, mock_glob, tmp_path):
    pb_path = tmp_path / "plugin.xplane.pb"
    pb_path.write_bytes(b"pb")
    hlo_pb = tmp_path / "main_jax.hlo_proto.pb"
    hlo_pb.write_bytes(b"hlo")

    mock_suppress.return_value = contextlib.nullcontext()
    mock_glob.side_effect = [
        [str(hlo_pb)],
        [str(hlo_pb)],
    ]

    graph_text = "%x = bf16[4,8]{1,0} parameter(0)\n"
    modules = _install_mock_xprof_convert((graph_text.encode("utf-8"), None))

    with patch.dict(sys.modules, modules):
        hlo_ops = JaxProfileProcessor.process_protobuf_file(str(pb_path), "main_jax")

    assert "%x" in hlo_ops


@patch("TraceLens.util.glob.glob")
@patch("TraceLens.util.suppress_native_hlo_logs")
def test_process_protobuf_file_no_hlo_match(mock_suppress, mock_glob, tmp_path, caplog):
    pb_path = tmp_path / "plugin.xplane.pb"
    pb_path.write_bytes(b"pb")
    mock_suppress.return_value = contextlib.nullcontext()
    mock_glob.side_effect = [[], []]

    modules = _install_mock_xprof_convert((b"", None))
    with patch.dict(sys.modules, modules):
        assert JaxProfileProcessor.process_protobuf_file(str(pb_path), "main_jax") == {}
    assert "No matching hlo_filenames" in caplog.text


@patch("TraceLens.util.glob.glob")
@patch("TraceLens.util.suppress_native_hlo_logs")
def test_process_protobuf_file_multiple_hlo_files(
    mock_suppress, mock_glob, tmp_path, caplog
):
    pb_path = tmp_path / "plugin.xplane.pb"
    pb_path.write_bytes(b"pb")
    hlo_a = tmp_path / "main_jax.a.hlo_proto.pb"
    hlo_b = tmp_path / "main_jax.b.hlo_proto.pb"
    hlo_a.write_bytes(b"a")
    hlo_b.write_bytes(b"b")

    mock_suppress.return_value = contextlib.nullcontext()
    mock_glob.return_value = [str(hlo_a), str(hlo_b)]

    graph_text = "%x = bf16[4,8]{1,0} parameter(0)\n"
    modules = _install_mock_xprof_convert((graph_text.encode("utf-8"), None))
    with patch.dict(sys.modules, modules):
        hlo_ops = JaxProfileProcessor.process_protobuf_file(str(pb_path), "main_jax")

    assert "%x" in hlo_ops
    assert "Multiple matching hlo_filenames" in caplog.text


@patch("TraceLens.util.glob.glob")
@patch("TraceLens.util.suppress_native_hlo_logs")
def test_process_protobuf_file_triggers_tool_names(mock_suppress, mock_glob, tmp_path):
    pb_path = tmp_path / "plugin.xplane.pb"
    pb_path.write_bytes(b"pb")
    hlo_pb = tmp_path / "main_jax.hlo_proto.pb"
    hlo_pb.write_bytes(b"hlo")

    mock_suppress.return_value = contextlib.nullcontext()
    mock_glob.side_effect = [[], [str(hlo_pb)]]

    calls = {"tool_names": 0, "tool_data": 0}
    graph_text = "%x = bf16[4,8]{1,0} parameter(0)\n"

    def xspace_to_tool_names(*args, **kwargs):
        calls["tool_names"] += 1
        return None

    def xspace_to_tool_data(*args, **kwargs):
        calls["tool_data"] += 1
        return (graph_text.encode("utf-8"), None)

    mock_mod = types.ModuleType("xprof.convert.raw_to_tool_data")
    mock_mod.xspace_to_tool_names = xspace_to_tool_names
    mock_mod.xspace_to_tool_data = xspace_to_tool_data
    fake_convert = types.ModuleType("xprof.convert")
    fake_convert.raw_to_tool_data = mock_mod
    fake_xprof = types.ModuleType("xprof")
    fake_xprof.convert = fake_convert

    with patch.dict(
        sys.modules,
        {
            "xprof": fake_xprof,
            "xprof.convert": fake_convert,
            "xprof.convert.raw_to_tool_data": mock_mod,
        },
    ):
        hlo_ops = JaxProfileProcessor.process_protobuf_file(str(pb_path), "main_jax")

    assert calls["tool_names"] == 1
    assert calls["tool_data"] == 1
    assert "%x" in hlo_ops


@patch("TraceLens.util.glob.glob")
@patch("TraceLens.util.suppress_native_hlo_logs")
def test_process_protobuf_file_tensorboard_fallback(
    mock_suppress, mock_glob, tmp_path, monkeypatch
):
    pb_path = tmp_path / "plugin.xplane.pb"
    pb_path.write_bytes(b"pb")
    hlo_pb = tmp_path / "main_jax.hlo_proto.pb"
    hlo_pb.write_bytes(b"hlo")

    mock_suppress.return_value = contextlib.nullcontext()
    mock_glob.side_effect = [[str(hlo_pb)], [str(hlo_pb)]]

    graph_text = "%x = bf16[4,8]{1,0} parameter(0)\n"
    tb_mod = types.ModuleType("tensorboard_plugin_profile.convert.raw_to_tool_data")
    tb_mod.xspace_to_tool_names = lambda *args, **kwargs: None
    tb_mod.xspace_to_tool_data = lambda *args, **kwargs: (
        graph_text.encode("utf-8"),
        None,
    )
    tb_convert = types.ModuleType("tensorboard_plugin_profile.convert")
    tb_convert.raw_to_tool_data = tb_mod
    tb_pkg = types.ModuleType("tensorboard_plugin_profile")
    tb_pkg.convert = tb_convert

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "xprof.convert":
            raise ImportError("xprof unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with patch.dict(
        sys.modules,
        {
            "tensorboard_plugin_profile": tb_pkg,
            "tensorboard_plugin_profile.convert": tb_convert,
            "tensorboard_plugin_profile.convert.raw_to_tool_data": tb_mod,
        },
    ):
        hlo_ops = JaxProfileProcessor.process_protobuf_file(str(pb_path), "main_jax")

    assert "%x" in hlo_ops


def test_dataloader_load_json(tmp_path):
    payload = {"traceEvents": [{"name": "kernel"}]}
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(payload))

    assert DataLoader.load_data(str(trace_path)) == payload


def test_dataloader_load_json_gz(tmp_path):
    payload = {"key": "value"}
    trace_path = tmp_path / "trace.json.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    assert DataLoader.load_data(str(trace_path)) == payload


def test_dataloader_save_preprocessed_json(tmp_path):
    payload = {"events": [1, 2, 3]}
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(payload))

    DataLoader.load_data(str(trace_path), save_preprocessed=True)

    assert json.loads(trace_path.read_text()) == payload


def test_dataloader_unknown_file_type():
    with pytest.raises(ValueError, match="Unknown file type"):
        DataLoader.load_data("/tmp/not-a-trace.xyz")


@patch("TraceLens.util.suppress_native_hlo_logs")
def test_dataloader_load_pb(mock_suppress, tmp_path):
    payload = {"traceEvents": []}
    trace_path = tmp_path / "trace.pb"
    trace_path.write_bytes(b"pb")

    mock_suppress.return_value = contextlib.nullcontext()
    modules = _install_mock_xprof_convert((json.dumps(payload).encode("utf-8"), None))

    with patch.dict(sys.modules, modules):
        result = DataLoader.load_data(str(trace_path))

    assert result == payload


@patch("TraceLens.util.suppress_native_hlo_logs")
def test_dataloader_load_pb_none_raises(mock_suppress, tmp_path):
    trace_path = tmp_path / "trace.pb"
    trace_path.write_bytes(b"pb")
    mock_suppress.return_value = contextlib.nullcontext()
    modules = _install_mock_xprof_convert((None, None))

    with patch.dict(sys.modules, modules):
        with pytest.raises(RuntimeError, match="returned None"):
            DataLoader.load_data(str(trace_path))


@patch("TraceLens.util.suppress_native_hlo_logs")
def test_dataloader_tensorboard_fallback(mock_suppress, tmp_path, monkeypatch):
    payload = {"traceEvents": []}
    trace_path = tmp_path / "trace.pb"
    trace_path.write_bytes(b"pb")
    mock_suppress.return_value = contextlib.nullcontext()

    tb_mod = types.ModuleType("tensorboard_plugin_profile.convert.raw_to_tool_data")

    def xspace_to_tool_data(*args, **kwargs):
        return (json.dumps(payload).encode("utf-8"), None)

    tb_mod.xspace_to_tool_data = xspace_to_tool_data
    tb_convert = types.ModuleType("tensorboard_plugin_profile.convert")
    tb_convert.raw_to_tool_data = tb_mod
    tb_pkg = types.ModuleType("tensorboard_plugin_profile")
    tb_pkg.convert = tb_convert

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "xprof.convert":
            raise ImportError("xprof unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with patch.dict(
        sys.modules,
        {
            "tensorboard_plugin_profile": tb_pkg,
            "tensorboard_plugin_profile.convert": tb_convert,
            "tensorboard_plugin_profile.convert.raw_to_tool_data": tb_mod,
        },
    ):
        assert DataLoader.load_data(str(trace_path)) == payload


def test_dataloader_orjson_fallback(tmp_path, monkeypatch):
    payload = {"value": 42}
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(payload))

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "orjson":
            raise ImportError("orjson unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert DataLoader.load_data(str(trace_path)) == payload


def test_trace_event_utils_split_by_field():
    events = [{"cat": "kernel"}, {"cat": "kernel"}, {"cat": "cpu_op"}]
    grouped = TraceEventUtils.split_by_field(events, "cat")
    assert set(grouped) == {"kernel", "cpu_op"}


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


def test_trace_event_utils_metadata_helpers():
    events = [
        _metadata_event(1, None, MF.ProcessName, "host"),
        {TK.PID: 1, TK.TID: 2, TK.Phase: TP.Complete, TK.Name: "op", "cat": "kernel"},
    ]
    metadata = TraceEventUtils.get_metadata(events)
    rest = TraceEventUtils.non_metadata_events(events)
    assert metadata[1][None][MF.ProcessName] == "host"
    assert len(rest) == 1
    assert TraceEventUtils.default_categorizer(rest[0]) == "kernel"


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


@pytest.mark.parametrize(
    "thread_name,event_name,expected",
    [
        (JST.FrameworkCallStack, "scope", "cpu_op"),
        ("py_xla_worker", "compile", "cpu_op"),
        ("Stream #7", "CopyHtoD", "memcpy"),
        ("Stream #7", "Memset32", "memset"),
        ("Stream #7", "my_kernel", "kernel"),
    ],
)
def test_trace_event_utils_get_event_category_branches(
    thread_name, event_name, expected
):
    events = [
        _metadata_event(1, None, MF.ProcessName, "host"),
        _metadata_event(1, 3, MF.ThreadName, thread_name),
        {
            TK.PID: 1,
            TK.TID: 3,
            TK.Phase: TP.Complete,
            TK.Name: event_name,
        },
    ]
    metadata = TraceEventUtils.get_metadata(events)
    category = TraceEventUtils.get_event_category(metadata, events[-1])
    assert category == expected


def test_trace_event_utils_get_event_category_metadata_and_unknown():
    metadata_event = {
        TK.Phase: TP.Metadata,
        TK.PID: 1,
        TK.TID: 1,
        TK.Name: MF.ThreadName,
        TK.Args: {AN.Name: "t"},
    }
    assert TraceEventUtils.get_event_category({}, metadata_event) == "metadata"
    assert (
        TraceEventUtils.get_event_category({}, {TK.Phase: TP.Complete, TK.Name: "x"})
        == "Unknown"
    )


def test_trace_event_utils_split_events_by_pid_tid():
    events = [
        _metadata_event(1, None, MF.ProcessName, "host"),
        {TK.PID: 1, TK.TID: 2, TK.Phase: TP.Complete, TK.Name: "a"},
        {TK.PID: 1, TK.TID: 2, TK.Phase: TP.Complete, TK.Name: "b"},
        {TK.PID: 1, TK.TID: 3, TK.Phase: TP.Complete, TK.Name: "c"},
    ]
    grouped = TraceEventUtils.split_events_by_pid_tid(events)
    assert len(grouped[1][2]) == 2
    assert len(grouped[1][3]) == 1


def test_trace_event_utils_sort_events_by_timestamp_duration():
    events = [
        {TK.TimeStamp: 20, TK.Duration: 1},
        {TK.TimeStamp: 10, TK.Duration: 5},
        {TK.TimeStamp: 10, TK.Duration: 2},
    ]
    TraceEventUtils.sort_events_by_timestamp_duration(events)
    assert [e[TK.TimeStamp] for e in events] == [10, 10, 20]
    assert [e[TK.Duration] for e in events] == [2, 5, 1]


def test_trace_event_utils_find_threads_and_end_times():
    metadata = {
        1: {
            10: {MF.ThreadName: "worker-a"},
            11: {MF.ThreadName: "worker-b"},
        }
    }
    threads = list(
        TraceEventUtils.find_threads_by_item_in_metadata(
            metadata[1],
            lambda item: item[1][MF.ThreadName].startswith("worker-"),
        )
    )
    assert threads == [10, 11]
    assert (
        TraceEventUtils.find_thread_by_item_in_metadata(
            metadata[1],
            lambda item: item[0] == 10,
        )
        == 10
    )

    events = [{TK.TimeStamp: 100, TK.Duration: 25}, {TK.TimeStamp: 50, TK.Duration: 5}]
    TraceEventUtils.compute_event_end_times(events)
    assert events[0][TK.TimeEnd] == 125
    assert events[1][TK.TimeEnd] == 55


def test_trace_event_utils_communication_helpers():
    default_regexes = TraceEventUtils.get_communication_regexes()
    assert len(default_regexes) >= 2

    custom = TraceEventUtils.get_communication_regexes([("my_allreduce", "allreduce")])
    assert any(p.search("my_allreduce_kernel") for p in custom)

    filters, rules = TraceEventUtils.build_collective_filter_and_inference_rules([])
    assert filters
    assert rules == []

    filters, rules = TraceEventUtils.build_collective_filter_and_inference_rules(
        [("custom_reduce", "allreduce")]
    )
    assert len(rules) == 1
    assert rules[0][1] == "allreduce"

    assert TraceEventUtils.is_communication_string("") is False
    assert TraceEventUtils.is_communication_string("ncclAllReduce") is True
    assert TraceEventUtils.is_communication_string("cross_device_reduce_0") is True


@pytest.mark.parametrize(
    "name,matcher,expected",
    [
        ("MEMORY_COPY_HOST_TO_DEVICE", TraceEventUtils.is_rocm_legacy_memcpy, True),
        ("__amd_rocclr_copyBuffer", TraceEventUtils.is_rocm_legacy_memcpy, True),
        ("__amd_rocclr_fillBuffer", TraceEventUtils.is_rocm_legacy_memset, True),
        ("regular_kernel", TraceEventUtils.is_rocm_legacy_memcpy, False),
    ],
)
def test_trace_event_utils_rocm_legacy_memory(name, matcher, expected):
    assert matcher(name) is expected
    assert matcher("") is False


def _minimal_rocprof_data():
    return {
        "rocprofiler-sdk-tool": [
            {
                "buffer_records": {
                    "kernel_dispatch": [
                        {
                            "start_timestamp": 100,
                            "end_timestamp": 250,
                            "stream_id": {"handle": 1},
                            "dispatch_info": {
                                "kernel_id": 7,
                                "grid_size": {"x": 2, "y": 1, "z": 1},
                                "workgroup_size": {"x": 256, "y": 1, "z": 1},
                                "dispatch_id": 42,
                                "agent_id": {"handle": 3},
                            },
                            "correlation_id": {"id": 99},
                            "thread_id": 5,
                        }
                    ],
                    "memory_copy": [
                        {
                            "start_timestamp": 50,
                            "end_timestamp": 80,
                            "kind": "H2D",
                            "operation": "copy",
                            "stream_id": {"handle": 0},
                        }
                    ],
                    "hip_api": [
                        {
                            "start_timestamp": 10,
                            "end_timestamp": 20,
                            "operation": "hipLaunchKernel",
                            "thread_id": 2,
                        }
                    ],
                    "hsa_api": [
                        {
                            "start_timestamp": 30,
                            "end_timestamp": 35,
                            "operation": "hsa_signal_wait",
                            "thread_id": 3,
                        }
                    ],
                },
                "kernel_symbols": [
                    {
                        "kernel_id": 7,
                        "truncated_kernel_name": "my_kernel",
                    }
                ],
                "metadata": {
                    "pid": 1234,
                    "init_time": 0,
                    "fini_time": 1000,
                    "node": {"hostname": "testhost"},
                    "command": ["./app"],
                },
                "agents": [{"id": 0}],
            }
        ]
    }


def test_rocprof_parser_synthetic_data(tmp_path):
    payload = _minimal_rocprof_data()
    trace_path = tmp_path / "rocprof.json"
    trace_path.write_text(json.dumps(payload))

    loaded = RocprofParser.load_rocprof_data(str(trace_path))
    assert loaded == payload

    kernels = RocprofParser.extract_kernel_events(loaded)
    assert len(kernels) == 1
    assert kernels[0]["name"] == "my_kernel"
    assert kernels[0]["dur"] == 150
    assert kernels[0]["grid"] == (2, 1, 1)

    memory = RocprofParser.extract_memory_events(loaded)
    assert len(memory) == 1
    assert memory[0]["kind"] == "H2D"

    api_events = RocprofParser.extract_api_events(loaded)
    assert len(api_events) == 2
    assert {event["type"] for event in api_events} == {"hip_api", "hsa_api"}

    metadata = RocprofParser.get_metadata(loaded)
    assert metadata["pid"] == 1234
    assert metadata["hostname"] == "testhost"


def test_rocprof_parser_invalid_file(tmp_path):
    trace_path = tmp_path / "not_rocprof.json"
    trace_path.write_text(json.dumps({"other": []}))
    with pytest.raises(ValueError, match="Not a valid rocprofv3 file"):
        RocprofParser.load_rocprof_data(str(trace_path))


def test_pftrace_parser_load_and_validate(tmp_path):
    events = [{"ph": "X", "name": "kernel"}]
    json_path = tmp_path / "trace.json"
    json_path.write_text(json.dumps({"traceEvents": events}))

    loaded = PftraceParser.load_pftrace_data(str(json_path))
    assert PftraceParser.get_events(loaded) == events

    gz_path = tmp_path / "trace.json.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)
    assert PftraceParser.load_pftrace_data(str(gz_path))["traceEvents"] == events


def test_pftrace_parser_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="expects .json or .json.gz"):
        PftraceParser.load_pftrace_data("/tmp/trace.pftrace")

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"other": []}))
    with pytest.raises(ValueError, match="missing 'traceEvents'"):
        PftraceParser.load_pftrace_data(str(bad_path))

    not_list_path = tmp_path / "not_list.json"
    not_list_path.write_text(json.dumps({"traceEvents": "nope"}))
    with pytest.raises(ValueError, match="must be a list"):
        PftraceParser.load_pftrace_data(str(not_list_path))


# --- migrated from test_coverage_95_phase13.py ---
import os
import pandas as pd
import pytest
from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.PerfModel.extensions import moe_perf_model_extensions as moe_ext
from TraceLens.Reporting import compare_traces_jax_llama as jax_cmp
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.reporting import _jax_llama_trace_events, _write_gz_trace


class TestAnalysisUtilsPhase13:
    def test_efficiency_memory_bound_and_fusion_map_empty(self, tmp_path):
        row = pd.Series(
            {
                "FLOPS/Byte": 0.5,
                "TFLOPS/s_mean": 10.0,
                "TB/s_mean": 2.0,
                "Roofline Bound": "MEMORY_BOUND",
                "Compute Spec": "matrix_fp16",
            }
        )
        eff = au.calculate_efficiency(
            row, peak_maf_or_maf_dict={"matrix_fp16": 100.0}, peak_hbm_bw=5300
        )
        assert eff["bound_type"] == "memory"
        assert au._load_fusion_map(str(tmp_path)) == {}


# --- migrated from test_coverage_95_phase7.py ---
import importlib
import json
import os
import sys
import pandas as pd
import pytest
from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.Agent.Analysis.category_analyses import kernel_fusion_analysis as kfa
from TraceLens.Reporting import compare_traces_jax_llama as jax_cmp
from TraceLens.Reporting.compare_perf_reports_pytorch import (
    generate_compare_perf_reports_pytorch,
)
from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (
    generate_collective_report,
)
from TraceLens.Reporting.pftrace_hip_activity_analysis import PftraceHipActivityAnalyzer
from TraceLens.Reporting.tracediff_comparison_extension import (
    tracediff_perf_summary_from_diff_stats,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)
from TraceLens.TraceDiff.trace_diff import TraceDiff
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.fixtures.reporting import _jax_llama_trace_events, _write_gz_trace
from tests.fixtures.reporting import (
    _minimal_pftrace_events,
    _mk_event,
    _write_trace,
)


class TestAnalysisUtilsPhase7:
    def test_efficiency_and_fusion_branches(self, tmp_path):
        row = pd.Series(
            {
                "FLOPS/Byte": 2.0,
                "TFLOPS/s_mean": 50.0,
                "TB/s_mean": 0.1,
                "Roofline Bound": "COMPUTE_BOUND",
                "Compute Spec": "matrix_fp16",
            }
        )
        eff = au.calculate_efficiency(
            row, peak_maf_or_maf_dict={"matrix_fp16": 100.0}, peak_hbm_bw=5300
        )
        assert eff["bound_type"] == "compute"

        cat_dir = tmp_path / "category_data"
        cat_dir.mkdir()
        (cat_dir / "kernel_fusion_metrics.json").write_text(
            json.dumps(
                {
                    "impact_estimates": [
                        {
                            "candidate_id": "c1",
                            "impact_score": 5.0,
                            "impact_score_low": 3.0,
                            "impact_score_high": 8.0,
                            "confidence": "high",
                        }
                    ]
                }
            )
        )
        loaded = au._load_fusion_map(str(tmp_path))
        assert isinstance(loaded, dict)

        ops = [
            {
                "kernel_names": ["a", "b"],
                "base_name": "Block",
                "instance_count": 3,
                "kernel_type_signature": ["GEMM", "elementwise"],
            }
        ]
        assert len(kfa._filter_and_dedup(ops)) >= 1


# --- migrated from test_coverage_95_phase8.py ---
import gzip
import json
import os
import pandas as pd
import pytest
from TraceLens.Agent.Analysis.category_analyses import analysis_utils as au
from TraceLens.PerfModel import perf_model
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    classify_graph_capture_trace,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
from tests.test_conv_backward_bytes import _conv_bias_fwd_event
from tests.fixtures.reporting import _mk_event


class TestAnalysisUtilsPhase8:
    def test_validate_efficiency_branches(self):
        assert au.validate_efficiency(50, 0, "TFLOPS")["is_anomaly"]
        assert au.validate_efficiency(None, 100, "TFLOPS")["value"] is None
        assert au.validate_efficiency(120, 100, "TFLOPS")["is_anomaly"]
        assert au.validate_efficiency(105, 100, "TFLOPS")["warning"] is not None
        assert au.validate_efficiency(80, 100, "TFLOPS")["value"] == 80.0

    def test_calculate_time_metrics_no_kernel_time(self):
        ops = pd.DataFrame({"name": ["aten::mm"], "operation_count": [3]})
        summary = au.calculate_time_metrics(
            ops, {"gpu_utilization": {"total_time_ms": 10}}
        )
        assert summary["total_time_ms"] == 0

    def test_calculate_efficiency_with_validation(self):
        out = au.calculate_efficiency_with_validation(50.0, 0.5, 100.0, 5300.0)
        assert "compute_efficiency_pct" in out

    def test_build_operation_metrics(self, tmp_path):
        cat_dir = tmp_path / "category_data"
        cat_dir.mkdir()
        (cat_dir / "gemm_metrics.json").write_text("{}")
        ops = pd.DataFrame(
            {
                "name": ["aten::mm"],
                "Kernel Time (µs)_sum": [50000.0],
                "TFLOPS/s_mean": [10.0],
                "TB/s_mean": [0.5],
                "FLOPS/Byte": [1.0],
                "Roofline Bound": ["COMPUTE_BOUND"],
                "Compute Spec": ["matrix_fp16"],
                "kernel_details_summary": ["[{'name': 'Cijk_a'}]"],
                "call_stack_full": ["['aten::mm']"],
            }
        )
        metrics = au.build_operation_metrics(
            ops,
            {
                "gpu_utilization": {"total_time_ms": 100.0},
                "peak_hbm_bw_tbs": 5.3,
                "max_achievable_tflops": {"matrix_fp16": 100.0},
            },
            {},
            comparison_scope="standalone",
        )
        assert isinstance(metrics, list)
