###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.Reporting.reporting_utils helpers."""

import json
import os
from unittest import mock

import pandas as pd
import pytest

from TraceLens.Reporting.reporting_utils import (
    add_gpu_arch_cli_args,
    add_node_span_columns,
    detect_gpus_per_node,
    export_data_df,
    request_install,
    resolve_gpu_arch,
)


def test_export_data_df_csv_and_xlsx(tmp_path):
    df = pd.DataFrame({"a": [1.23456, 2.0], "b": [3.0, 4.567]})
    export_data_df(
        df,
        tmp_path,
        "report",
        output_table_format=[".csv", ".xlsx"],
        suffix="_stats",
    )
    csv_path = tmp_path / "report_stats.csv"
    xlsx_path = tmp_path / "report_stats.xlsx"
    assert csv_path.exists()
    assert xlsx_path.exists()
    loaded = pd.read_csv(csv_path)
    assert list(loaded.columns) == ["a", "b"]
    assert loaded["a"].tolist() == [1.23, 2.0]


def test_export_data_df_verbose(tmp_path, capsys):
    df = pd.DataFrame({"x": [1]})
    export_data_df(
        df,
        tmp_path,
        "out",
        output_table_format=[".csv"],
        suffix="",
        verbose=1,
    )
    captured = capsys.readouterr()
    assert "Exporting data to" in captured.out


def test_request_install_declines_exits():
    with mock.patch("builtins.input", return_value="n"):
        with pytest.raises(SystemExit) as exc:
            request_install("openpyxl")
        assert exc.value.code == 1


def test_request_install_accepts_and_installs():
    with mock.patch("builtins.input", return_value="y"):
        with mock.patch("subprocess.check_call") as mock_install:
            request_install("openpyxl")
    mock_install.assert_called_once()


def test_request_install_failed_install_exits():
    import subprocess

    with mock.patch("builtins.input", return_value="y"):
        with mock.patch(
            "subprocess.check_call",
            side_effect=subprocess.CalledProcessError(1, "pip"),
        ):
            with pytest.raises(SystemExit) as exc:
                request_install("openpyxl")
            assert exc.value.code == 1


def test_add_node_span_columns_intra_node():
    df = pd.DataFrame(
        {
            "rank": [0, 1, 2, 3],
            "Process Group Ranks": ["[0, 1, 2, 3]"] * 4,
        }
    )
    out = add_node_span_columns(df, gpus_per_node=4, world_size=8)
    assert "node_id" in out.columns
    assert "node_span" in out.columns
    assert out["node_id"].tolist() == [0, 0, 0, 0]
    assert (out["node_span"] == "intra_node").all()


def test_add_node_span_columns_inter_node():
    df = pd.DataFrame(
        {
            "rank": [0, 4],
            "Process Group Ranks": [
                "[0, 1, 2, 3, 4, 5, 6, 7]",
                "[0, 1, 2, 3, 4, 5, 6, 7]",
            ],
        }
    )
    out = add_node_span_columns(df, gpus_per_node=4, world_size=8)
    assert (out["node_span"] == "inter_node").all()
    assert out["node_id"].tolist() == [0, 1]


def test_add_node_span_columns_pg_ranks_string_formats():
    df = pd.DataFrame({"Process Group Ranks": ["(0, 1)", "0, 2, 4"]})
    out = add_node_span_columns(df, gpus_per_node=2, world_size=8)
    assert set(out["node_span"].unique()) <= {"intra_node", "inter_node", "unknown"}


def test_add_node_span_columns_empty_or_missing_columns():
    empty = pd.DataFrame()
    assert add_node_span_columns(empty, gpus_per_node=4).empty

    no_cols = pd.DataFrame({"other": [1]})
    result = add_node_span_columns(no_cols, gpus_per_node=4)
    assert "node_id" not in result.columns


def test_add_node_span_columns_invalid_gpus_per_node():
    df = pd.DataFrame({"rank": [0]})
    with pytest.raises(ValueError, match="gpus_per_node"):
        add_node_span_columns(df, gpus_per_node=0)


def test_detect_gpus_per_node_from_trace():
    trace_path = os.path.join(
        "tests",
        "traces",
        "mi300",
        "Falconsai_nsfw_image_detection__1016002.json.gz",
    )
    if not os.path.isfile(trace_path):
        pytest.skip(f"Trace not found: {trace_path}")
    gpus = detect_gpus_per_node(trace_path)
    assert gpus is not None
    assert gpus > 0


def test_detect_gpus_per_node_invalid_file():
    assert detect_gpus_per_node("/nonexistent/trace.json") is None


def test_add_gpu_arch_cli_args_adds_mutually_exclusive_group():
    parser = __import__("argparse").ArgumentParser()
    add_gpu_arch_cli_args(parser)
    action_dests = {a.dest for a in parser._actions}
    assert "gpu_arch_json_path" in action_dests
    assert "gpu_arch_platform" in action_dests


def test_resolve_gpu_arch_json_roundtrip(tmp_path):
    arch = {"name": "test", "num_cus": 64}
    path = tmp_path / "arch.json"
    path.write_text(json.dumps(arch))
    assert resolve_gpu_arch(gpu_arch_json_path=str(path)) == arch
