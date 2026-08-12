###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.NcclAnalyser helpers and utilities."""

import json
import re

import pytest

from TraceLens.NcclAnalyser.nccl_analyser import (
    NcclAnalyser,
    _collective_filter,
    _infer_collective_name,
    _load_single_rank_process,
    _parse_split_sizes,
    list_to_tuple,
)
from TraceLens.NcclAnalyser.util.node_rank_to_protobuf_file_mapping import (
    extract_node_name_from_path,
    get_node_rank_protobuf_mapping,
    parse_log_file_for_node_rank,
)
from TraceLens.NcclAnalyser.util.xla_parser import XLACollectiveParser
from TraceLens.util import TraceEventUtils


def _nccl_kernel(name="ncclKernel_AllReduce", external_id=42, ts=100, dur=10):
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": {"External id": external_id, "Collective name": "allreduce"},
    }


def _write_trace(path, events):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ([1, 2, 3], [1, 2, 3]),
        ("[4, 5]", [4, 5]),
        ("not-a-list", None),
    ],
)
def test_parse_split_sizes(value, expected):
    assert _parse_split_sizes(value) == expected


def test_list_to_tuple_nested():
    assert list_to_tuple([1, [2, 3]]) == (1, (2, 3))


def test_infer_collective_name_matches_custom_rules():
    rules = [(re.compile(r"cross_device_reduce"), "custom_allreduce")]
    assert _infer_collective_name("vllm_cross_device_reduce_kernel", rules) == (
        "custom_allreduce"
    )
    assert _infer_collective_name("other_kernel", rules) is None


def test_collective_filter_requires_kernel_with_link_and_pattern():
    patterns = TraceEventUtils.get_communication_regexes()
    event = _nccl_kernel()
    assert _collective_filter(event, patterns) is True
    assert _collective_filter({**event, "cat": "cpu_op"}, patterns) is False
    assert _collective_filter({**event, "args": {}}, patterns) is False


def test_load_single_rank_process_extracts_nccl_events(tmp_path):
    trace_path = tmp_path / "rank0.json"
    _write_trace(
        trace_path,
        [
            _nccl_kernel(ts=100),
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "ignored",
                "ts": 1,
                "dur": 1,
                "args": {},
            },
        ],
    )
    rank, rank_dict = _load_single_rank_process(0, str(trace_path))
    assert rank == 0
    assert len(rank_dict) == 1
    assert rank_dict[0]["name"].startswith("ncclKernel")


def test_nccl_analyser_build_df_long_from_synthetic_trace(tmp_path):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        _write_trace(
            path,
            [
                _nccl_kernel(
                    external_id=100 + rank,
                    ts=1000 + rank * 100,
                    dur=50 + rank,
                )
            ],
        )
        filepaths.append(str(path))

    analyser = NcclAnalyser(filepaths, world_size)
    df_long = analyser.build_df_long()
    assert len(df_long) == world_size
    assert set(df_long["rank"]) == {0, 1}
    assert df_long["Collective name"].iloc[0] == "allreduce"


class TestXLACollectiveParser:
    def setup_method(self):
        self.parser = XLACollectiveParser({})

    def test_parse_replica_groups_explicit_groups(self):
        groups = self.parser._parse_replica_groups("{{0,1},{2,3}}")
        assert groups == [[0, 1], [2, 3]]

    def test_parse_device_assignment_iota_format(self):
        groups = self.parser._parse_replica_groups("[2,2]<=[4]")
        assert groups == [[0, 1], [2, 3]]

    def test_extract_collective_name(self):
        line = 'scheduling_name="all-reduce-start.1", replica_groups={{0,1}}'
        assert self.parser._extract_collective_name(line) == "all-reduce-start.1"

    def test_calculate_replica_group_size(self):
        assert self.parser._calculate_replica_group_size([[0, 1, 2, 3]]) == 4
        assert self.parser._calculate_replica_group_size([]) == 0

    def test_calculate_tensor_slice_and_data_bytes(self):
        tensor_spec = "bf16[8,16]{1,0}"
        replica_groups = [[0, 1]]
        tensor_slice = self.parser._calculate_tensor_slice(
            tensor_spec,
            split_dimension=0,
            replica_groups=replica_groups,
            collective_name="all-reduce",
            replica_group_size=2,
        )
        assert tensor_slice is not None
        assert tensor_slice[0]["bytes"] == 8 * 8 * 2
        assert self.parser._calculate_data_bytes(tensor_slice) == 8 * 8 * 2

    def test_parse_collectives_to_dataframe(self, tmp_path):
        xla_file = tmp_path / "node0.txt"
        xla_file.write_text(
            "HLO %x = bf16[4,8]{1,0} all-reduce(bf16[4,8]{1,0} %arg0), "
            'replica_groups={{0,1}}, scheduling_name="all-reduce", dimensions={0}\n',
            encoding="utf-8",
        )
        parser = XLACollectiveParser({"0": str(xla_file)})
        df = parser.parse_collectives_to_dataframe()
        assert not df.empty
        assert df.iloc[0]["collective_name"] == "all-reduce"
        assert df.iloc[0]["node"] == "0"

    def test_parse_collectives_missing_and_empty(self, tmp_path, capsys):
        missing = XLACollectiveParser({"0": str(tmp_path / "missing.txt")})
        assert missing.parse_collectives_to_dataframe().empty
        assert "Warning: File not found" in capsys.readouterr().out

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        assert (
            XLACollectiveParser({"0": str(empty_file)})
            .parse_collectives_to_dataframe()
            .empty
        )

    def test_parse_collectives_read_error(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "node.txt"
        path.write_text("x", encoding="utf-8")

        def boom(*args, **kwargs):
            raise OSError("read failed")

        monkeypatch.setattr("builtins.open", boom)
        assert (
            XLACollectiveParser({"0": str(path)}).parse_collectives_to_dataframe().empty
        )
        assert "Error reading file" in capsys.readouterr().out

    def test_extract_replica_groups_source_target_pairs(self):
        line = (
            "HLO %x = bf16[4,8]{1,0} collective-permute(bf16[4,8]{1,0} %arg0), "
            'source_target_pairs={{0,1},{1,0}}, scheduling_name="permute"'
        )
        replica_string, groups = self.parser._extract_replica_groups(line)
        assert replica_string == "{{0,1},{1,0}}"
        assert groups == [[0, 1], [1, 0]]

    def test_parse_all_gather_and_tuple_output(self, tmp_path):
        line = (
            "HLO %x = ((bf16[2,4]{1,0}, bf16[8,4]{1,0})) all-gather-start"
            "((bf16[2,4]{1,0}), (bf16[8,4]{1,0}) %arg0), "
            'replica_groups={{0,1}}, scheduling_name="all-gather-start.1", dimensions={0}'
        )
        path = tmp_path / "node.txt"
        path.write_text(line + "\n", encoding="utf-8")
        df = XLACollectiveParser({"0": str(path)}).parse_collectives_to_dataframe()
        assert not df.empty
        assert df.iloc[0]["collective_name"] == "all-gather-start.1"

    def test_device_assignment_with_transpose(self):
        groups = self.parser._parse_replica_groups("[2,2]<=T(1,0)[2,2]")
        assert groups == [[0, 2], [1, 3]]

    def test_tensor_spec_and_split_dimension_helpers(self):
        line = (
            "HLO %x = bf16[4,8]{1,0} all-reduce(bf16[4,8]{1,0} %arg0), "
            'dimensions={0,1}, scheduling_name="all-reduce"'
        )
        assert self.parser._extract_split_dimension(line) is None
        assert "bf16[4,8]{1,0}" in self.parser._extract_tensor_specs(line)
        assert self.parser._extract_output_tensor_from_tuple(
            "all-to-all", "((bf16[2,4]{1,0}), (bf16[8,4]{1,0}))"
        ).startswith("bf16[8,4]")

    def test_parse_replica_groups_empty_and_invalid(self):
        assert self.parser._parse_replica_groups("") == []
        assert self.parser._parse_replica_groups("none") == []
        assert self.parser._parse_replica_groups("{{0,1}}") == [[0, 1]]
        assert self.parser._parse_replica_groups("not-a-group") == []

    def test_parse_device_assignment_invalid(self):
        assert self.parser._parse_device_assignment("[2,2]<=") == []

    def test_extract_tensor_specs_fallback(self):
        line = "HLO %x = bf16[2,4]{1,0} all-reduce(bf16[2,4]{1,0} %arg0), channel_id=1"
        assert self.parser._extract_tensor_specs(line) == "bf16[2,4]{1,0}"

    def test_extract_split_dimension_single(self):
        line = 'HLO %x = bf16[4,8]{1,0} all-reduce(...), dimensions={0}'
        assert self.parser._extract_split_dimension(line) == 0

    def test_output_tensor_tuple_variants(self):
        parser = self.parser
        assert (
            parser._extract_output_tensor_from_tuple(
                "reduce-scatter", "bf16[2,4]{1,0}"
            )
            == "bf16[2,4]{1,0}"
        )
        nested = "((bf16[2,4]{1,0}), (bf16[8,4]{1,0}))"
        assert parser._extract_output_tensor_from_tuple(
            "all-gather-start", nested
        ).startswith("bf16[8,4]")
        simple = "(bf16[2,4]{1,0}, bf16[8,4]{1,0})"
        assert parser._extract_output_tensor_from_tuple(
            "all-to-all", simple
        ).startswith("bf16[8,4]")

    def test_calculate_tensor_slice_edge_cases(self):
        parser = self.parser
        assert parser._calculate_tensor_slice(None, 0, [[0, 1]], "all-reduce", 2) is None
        assert (
            parser._calculate_tensor_slice("bf16[4,8]{1,0}", 0, [], "all-reduce", 0)
            is None
        )
        assert (
            parser._calculate_tensor_slice("invalid", 0, [[0, 1]], "all-reduce", 2)
            is None
        )
        slices = parser._calculate_tensor_slice(
            "bf16[8,4]{1,0}",
            0,
            [[0, 1]],
            "all-reduce",
            2,
        )
        assert slices and slices[0]["bytes"] > 0
        assert parser._calculate_data_bytes({"bytes": 16}) == 16
        assert parser._calculate_data_bytes([{"bytes": 4}, {"bytes": 8}]) == 12
        assert parser._calculate_data_bytes("bad") == 0


def test_extract_node_name_from_path(tmp_path):
    pb_file = tmp_path / "nodeA.xplane.pb"
    pb_file.write_text("pb", encoding="utf-8")
    assert extract_node_name_from_path(pb_file) == "nodeA"
    assert extract_node_name_from_path(tmp_path / "file.dat") is None


def test_parse_log_file_for_node_rank_from_content(tmp_path):
    log_file = tmp_path / "worker.log"
    log_file.write_text(
        "export NODE_RANK=2\nhostname=nodeB\nnnodes=4\n",
        encoding="utf-8",
    )
    node_rank, node_name, nnodes = parse_log_file_for_node_rank(log_file)
    assert node_rank == 2
    assert node_name == "nodeB"
    assert nnodes == 4


def test_get_node_rank_protobuf_mapping(tmp_path):
    log_file = tmp_path / "launch.log"
    log_file.write_text("NODE_RANK=0\nhostname=node0\nnnodes=1\n", encoding="utf-8")
    pb_file = tmp_path / "node0.xplane.pb"
    pb_file.write_text("pb", encoding="utf-8")

    mapping, world_size = get_node_rank_protobuf_mapping(str(tmp_path))
    assert world_size == 1
    assert mapping["0"] == str(pb_file)


def test_get_node_rank_protobuf_mapping_missing_folder():
    mapping, world_size = get_node_rank_protobuf_mapping("/nonexistent/traces")
    assert mapping == {}
    assert world_size == 0


def test_get_node_rank_protobuf_mapping_finds_nested_logs(tmp_path):
    nested = tmp_path / "workers"
    nested.mkdir()
    log_file = nested / "worker.log"
    log_file.write_text("NODE_RANK=1\nhostname=node1\nnnodes=2\n", encoding="utf-8")
    pb_file = tmp_path / "node1.xplane.pb"
    pb_file.write_text("pb", encoding="utf-8")

    mapping, world_size = get_node_rank_protobuf_mapping(str(tmp_path))
    assert world_size == 2
    assert mapping["1"] == str(pb_file)


def test_get_node_rank_protobuf_mapping_warns_on_nnodes_mismatch(tmp_path, capsys):
    (tmp_path / "a.log").write_text("NODE_RANK=0\nhostname=node0\nnnodes=4\n")
    (tmp_path / "b.log").write_text("NODE_RANK=1\nhostname=node1\nnnodes=4\n")
    (tmp_path / "node0.xplane.pb").write_text("pb")
    (tmp_path / "node1.xplane.pb").write_text("pb")

    get_node_rank_protobuf_mapping(str(tmp_path))
    captured = capsys.readouterr().out
    assert "NODE_RANK count" in captured


def test_get_node_rank_protobuf_mapping_warns_on_unmapped_pb(tmp_path, capsys):
    (tmp_path / "node9.xplane.pb").write_text("pb")
    get_node_rank_protobuf_mapping(str(tmp_path))
    captured = capsys.readouterr().out
    assert "no corresponding NODE_RANK" in captured


def test_parse_log_file_for_node_rank_from_filename(tmp_path, monkeypatch):
    import importlib

    mapping_mod = importlib.import_module(parse_log_file_for_node_rank.__module__)
    log_file = tmp_path / "worker_node_rank_3.log"
    log_file.write_text("no vars here", encoding="utf-8")
    monkeypatch.setattr(
        mapping_mod,
        "extract_node_name_from_path",
        lambda _path: "worker",
    )
    node_rank, node_name, nnodes = parse_log_file_for_node_rank(log_file)
    assert node_rank == 3
    assert node_name == "worker"
    assert nnodes is None


def test_parse_log_file_for_node_rank_uses_path_when_hostname_missing(tmp_path):
    log_file = tmp_path / "nodeC.log"
    log_file.write_text("NODE_RANK=5\n", encoding="utf-8")
    node_rank, node_name, nnodes = parse_log_file_for_node_rank(log_file)
    assert node_rank == 5
    assert node_name == "nodeC"
    assert nnodes is None


def test_parse_log_file_for_node_rank_handles_read_error(tmp_path):
    missing = tmp_path / "missing.log"
    node_rank, node_name, nnodes = parse_log_file_for_node_rank(missing)
    assert (node_rank, node_name, nnodes) == (None, None, None)


def test_nccl_analyser_load_trace_data_multiprocessing(tmp_path):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        _write_trace(path, [_nccl_kernel(external_id=100 + rank, ts=1000 + rank)])
        filepaths.append(str(path))

    analyser = NcclAnalyser(
        filepaths, world_size, use_multiprocessing=True, max_workers=2
    )
    assert len(analyser.rank2trace_data) == world_size


def test_nccl_analyser_build_df_long_with_process_group_metadata(tmp_path):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        kernel = _nccl_kernel(external_id=100 + rank, ts=1000 + rank * 10, dur=50)
        kernel["args"].update(
            {
                "Process Group Name": "default_pg",
                "Process Group Ranks": [0, 1],
                "Collective name": "allreduce",
                "Group size": 2,
                "dtype": "Float",
                "In msg nelems": 1024,
                "Out msg nelems": 1024,
                "In split size": "[]",
                "Out split size": "[]",
                "stream": 3,
            }
        )
        _write_trace(path, [kernel])
        filepaths.append(str(path))

    analyser = NcclAnalyser(filepaths, world_size)
    df_long = analyser.build_df_long()
    assert len(df_long) == world_size
    assert not analyser._simplified_mode
    assert df_long["collective_id"].str.startswith("default_pg_").all()
    assert df_long["In msg size (MB)"].notna().all()


def test_nccl_analyser_build_df_long_empty_trace(tmp_path):
    path = tmp_path / "rank0.json"
    _write_trace(
        path,
        [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "ignored",
                "ts": 1,
                "dur": 1,
                "args": {},
            }
        ],
    )
    analyser = NcclAnalyser([str(path)], world_size=1)
    df_long = analyser.build_df_long()
    assert df_long.empty


def test_nccl_analyser_build_df_summary_long(tmp_path):
    world_size = 2
    filepaths = []
    for rank in range(world_size):
        path = tmp_path / f"rank{rank}.json"
        kernel = _nccl_kernel(external_id=100 + rank, ts=1000 + rank, dur=50 + rank)
        kernel["args"].update(
            {
                "Process Group Name": "default_pg",
                "Process Group Ranks": [0, 1],
                "Collective name": "allreduce",
                "Group size": 2,
                "dtype": "Float",
                "In msg nelems": 1024,
                "Out msg nelems": 1024,
                "In split size": "[]",
                "Out split size": "[]",
                "stream": 3,
            }
        )
        _write_trace(path, [kernel])
        filepaths.append(str(path))

    analyser = NcclAnalyser(filepaths, world_size)
    analyser.build_df_long()
    summary = analyser.build_df_summary_long()
    assert not summary.empty
    assert "dur_sum" in summary.columns
    assert "operation_count" in summary.columns


def test_nccl_analyser_instance_filter_fn():
    event = _nccl_kernel()
    analyser = NcclAnalyser.__new__(NcclAnalyser)
    analyser._filter_patterns = TraceEventUtils.get_communication_regexes()
    assert analyser._nccl_filter_event_fn(event) is True
    assert analyser._nccl_filter_event_fn({**event, "cat": "cpu_op"}) is False


# --- migrated from test_nccl_analyser_coverage.py ---
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import json

import pandas as pd
import pytest

from TraceLens.NcclAnalyser.nccl_analyser import (
    NcclAnalyser,
    _parse_split_sizes,
)


def _nccl_kernel(
    name="ncclKernel_AllReduce", external_id=42, ts=100, dur=10, **extra_args
):
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
    _write_trace(
        path, [{"ph": "X", "cat": "cpu_op", "name": "x", "ts": 1, "dur": 1, "args": {}}]
    )
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
        lambda rank: _nccl_kernel(
            external_id=400 + rank, ts=4000 + rank * 5, dur=30 + rank
        ),
    )
    analyser.build_df_long()
    analyser.build_df_nccl_implicit_sync_cat(strict_metadata_check=False)
    summary = analyser.build_df_straggler_summary(strict_metadata_check=False)
    assert not summary.empty
    assert "total_wait_time_us" in summary.columns


def test_build_df_straggler_summary_empty_when_no_implicit_sync(tmp_path):
    path = tmp_path / "rank0.json"
    _write_trace(
        path, [{"ph": "X", "cat": "cpu_op", "name": "x", "ts": 1, "dur": 1, "args": {}}]
    )
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
        df = analyser.build_df_nccl_all2allv(
            detailed=False, strict_metadata_check=False
        )
    assert df is not None
