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
