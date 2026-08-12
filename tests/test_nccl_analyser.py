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

    analyser = NcclAnalyser(filepaths, world_size, use_multiprocessing=True, max_workers=2)
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
    _write_trace(path, [{"ph": "X", "cat": "cpu_op", "name": "ignored", "ts": 1, "dur": 1, "args": {}}])
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
