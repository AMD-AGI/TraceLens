###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import unittest, pandas as pd, ast, os, pytest, importlib
from unittest.mock import MagicMock, patch
from TraceLens import JaxNcclAnalyser
from TraceLens.NcclAnalyser.jax_nccl_analyser import JaxNcclAnalyser


class TestJaxNcclAnalyserLoadTraceData(unittest.TestCase):
    """Unit tests for JaxNcclAnalyser.load_trace_data method."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.traces_dir = "/tmp/traces"
        self.world_size = 8
        self.node_to_pb_file_mapping = {
            0: "/tmp/traces/node_0/profile.pb",
            1: "/tmp/traces/node_1/profile.pb",
        }

    def test_load_trace_data_empty_node_mapping(self):
        """Test load_trace_data handles empty node mapping."""
        empty_mapping = {}

        # Prevent the constructor from calling methods we're testing
        with patch.object(JaxNcclAnalyser, "load_trace_data"), patch.object(
            JaxNcclAnalyser, "build_collectives_df_through_xla"
        ):

            # Create instance with empty mapping
            analyser = JaxNcclAnalyser(
                traces_dir=self.traces_dir,
                node_to_pb_file_mapping=empty_mapping,
                world_size=self.world_size,
            )

            # Call the method we're testing
            analyser.load_trace_data()

        # Verify node_to_trace_data is empty
        self.assertEqual(len(analyser.node_to_trace_data), 0)


class TestJaxNcclAnalyserGetBusBandwidthScaler(unittest.TestCase):
    """Unit tests for JaxNcclAnalyser.get_bus_bandwidth_scaler method."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.traces_dir = "/tmp/traces"
        self.world_size = 8
        self.node_to_pb_file_mapping = {
            0: "/tmp/traces/node_0/profile.pb",
            1: "/tmp/traces/node_1/profile.pb",
        }

        # Prevent the constructor from calling methods we're testing
        with patch.object(JaxNcclAnalyser, "load_trace_data"), patch.object(
            JaxNcclAnalyser, "build_collectives_df_through_xla"
        ):
            self.analyser = JaxNcclAnalyser(
                traces_dir=self.traces_dir,
                node_to_pb_file_mapping=self.node_to_pb_file_mapping,
                world_size=self.world_size,
            )

    def test_get_bus_bandwidth_scaler_basic(self):
        """Test basic bus bandwidth scaler functionality."""
        # Test all-reduce (should return 1.5)
        self.assertAlmostEqual(
            self.analyser.get_bus_bandwidth_scaler("all-reduce", 4), 1.5, places=2
        )

        # Test all-gather (should return 0.75)
        self.assertAlmostEqual(
            self.analyser.get_bus_bandwidth_scaler("all-gather", 4), 0.75, places=2
        )

        # Test all-to-all (should return 0.75)
        self.assertAlmostEqual(
            self.analyser.get_bus_bandwidth_scaler("all-to-all", 4), 0.75, places=2
        )

        # Test reduce-scatter (should return 0.75)
        self.assertAlmostEqual(
            self.analyser.get_bus_bandwidth_scaler("reduce-scatter", 4), 0.75, places=2
        )

        # Test collective-permute (should return 1.0)
        self.assertEqual(
            self.analyser.get_bus_bandwidth_scaler("collective-permute", 4), 1.0
        )

        # Test unknown collective (should return default 1.0)
        self.assertEqual(self.analyser.get_bus_bandwidth_scaler("unknown-op", 4), 1.0)


class TestJaxNcclAnalyserAnalyzeAllCollectivesFromDf(unittest.TestCase):
    """Unit tests for JaxNcclAnalyser.analyze_all_collectives_from_df method."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.traces_dir = "/tmp/traces"
        self.world_size = 8
        self.node_to_pb_file_mapping = {
            0: "/tmp/traces/node_0/profile.pb",
            1: "/tmp/traces/node_1/profile.pb",
        }

        # Prevent the constructor from calling methods we're testing
        with patch.object(JaxNcclAnalyser, "load_trace_data"), patch.object(
            JaxNcclAnalyser, "build_collectives_df_through_xla"
        ):
            self.analyser = JaxNcclAnalyser(
                traces_dir=self.traces_dir,
                node_to_pb_file_mapping=self.node_to_pb_file_mapping,
                world_size=self.world_size,
            )

    def create_mock_dataframe(self):
        """Create a comprehensive mock dataframe for testing using real data from CSV."""

        # Load the CSV file with all-gather data for a slice across 32 gpus
        # CSV contains made-up data for demonstrating representative calculations
        # No reference to any model or hardware.
        current_file_path = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(
            current_file_path,
            "test_data_jax_nccl_analyser",
            "all_gather_collective_mockup_df.csv",
        )
        df = pd.read_csv(csv_path, index_col=0)

        # Parse the replica_groups column from string to list
        # The CSV has it as a string representation of a list, so we need to evaluate it
        def parse_replica_groups(replica_groups_str):
            try:
                # Use ast.literal_eval to safely evaluate the string as a Python literal
                return ast.literal_eval(replica_groups_str)
            except (ValueError, SyntaxError):
                # If parsing fails, return empty list as fallback
                return []

        df["replica_groups"] = df["replica_groups"].apply(parse_replica_groups)

        return df

    @patch("builtins.print")  # Suppress print output during testing
    def test_analyze_all_collectives_from_df_basic(self, mock_print):
        """Test basic functionality of analyze_all_collectives_from_df."""

        # Create mock dataframe
        mock_df = self.create_mock_dataframe()

        # Call the method under test - use real bandwidth calculations
        results = self.analyser.analyze_all_collectives_from_df(mock_df)

        # Verify results structure
        self.assertIsInstance(results, dict)

        # Verify structure of individual results
        all_gather_result = results["all-gather"]
        self.assertIn("bandwidths", all_gather_result)
        self.assertIn("bus_bandwidths", all_gather_result)
        self.assertIn("avg_bandwidth", all_gather_result)
        self.assertIn("avg_bus_bandwidth", all_gather_result)
        self.assertIn("slice_info", all_gather_result)
        self.assertIn("num_slices", all_gather_result)
        self.assertIn("data_size_bytes", all_gather_result)

        # Verify group details exist and have correct structure
        slice_info = all_gather_result["slice_info"][0]
        group_details = slice_info["group_details"]
        self.assertEqual(len(group_details), 4)  # 4 replica groups

        # Check group 0 specific calculation numbers
        # Calculations can be checked from
        # TraceLens/tests/test_data_jax_nccl_analyser/all_gather_manual_bw_calculation.csv
        # Referred CSV contains made-up data for demonstrating representative calculations
        # No reference to any model or hardware.
        group_0 = group_details[0]
        self.assertEqual(group_0["group_idx"], 0)
        self.assertEqual(group_0["gpu_group"], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(group_0["gpus_in_data"], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertAlmostEqual(
            group_0["algorithmic_bandwidth_gbps"], 49.681, delta=0.001
        )
        self.assertAlmostEqual(group_0["bus_bandwidth_gbps"], 43.471, delta=0.001)
        self.assertAlmostEqual(group_0["bus_bandwidth_scaler"], 0.875, delta=0.001)
        self.assertAlmostEqual(group_0["duration_us"], 412.785, delta=0.001)
        self.assertEqual(group_0["fastest_gpu_rank"], 3)
        self.assertEqual(group_0["algorithmic_bytes"], 22020096)
        self.assertEqual(group_0["actual_group_size"], 8)
        self.assertEqual(group_0["participants_in_data"], 8)


if __name__ == "__main__":
    unittest.main()


_jax_nccl_mod = importlib.import_module("TraceLens.NcclAnalyser.jax_nccl_analyser")


@pytest.fixture
def jax_analyser():
    with patch.object(JaxNcclAnalyser, "load_trace_data"), patch.object(
        JaxNcclAnalyser, "build_collectives_df_through_xla"
    ):
        analyser = JaxNcclAnalyser(
            traces_dir="/tmp/traces",
            node_to_pb_file_mapping={0: "/tmp/traces/node0/profile.pb"},
            world_size=8,
        )
    return analyser


def _make_collective_row(
    *,
    node="0",
    gpu_rank=0,
    pid=1,
    ts=100.0,
    dur=100.0,
    collective_name="all-reduce",
    replica_string="{{0,1}}",
    replica_groups=None,
    data_bytes=1024,
    index_in_group=0,
):
    if replica_groups is None:
        replica_groups = [[0, 1]]
    collective_id = f"{collective_name}_{index_in_group}"
    return {
        "node": node,
        "gpu_rank": gpu_rank,
        "pid": pid,
        "ts": ts,
        "dur": dur,
        "collective_name": collective_name,
        "hlo_module": "module",
        "correlation_id": 1,
        "replica_string": replica_string,
        "replica_groups": replica_groups,
        "data(bytes)": data_bytes,
        "process_name": "gpu",
        "index_in_group": index_in_group,
        "collective_id": collective_id,
    }


def test_jax_nccl_event_filter(jax_analyser):
    assert jax_analyser._nccl_event_filter({"name": "ncclAllReduce"}) is True
    assert jax_analyser._nccl_event_filter({"name": "gemm_kernel"}) is False


@patch.object(_jax_nccl_mod, "JaxTraceToTree")
@patch.object(_jax_nccl_mod, "DataLoader")
@patch.object(_jax_nccl_mod.TraceEventUtils, "prepare_event_categorizer")
@patch.object(_jax_nccl_mod.TraceEventUtils, "split_event_list")
def test_jax_load_trace_data_extracts_nccl_events(
    mock_split, mock_categorizer, mock_loader, mock_tree_cls, tmp_path
):
    pb_path = tmp_path / "node0.xplane.pb"
    pb_path.write_text("pb", encoding="utf-8")
    mock_loader.load_data.return_value = {"traceEvents": [{"name": "evt"}]}
    mock_categorizer.return_value = lambda e: e.get("cat", "")
    mock_split.return_value = ([], [{"name": "ncclAllReduce"}])
    mock_tree = MagicMock()
    mock_tree.events = [
        {
            "name": "ncclAllReduce",
            "pid": 1,
            "ts": 100,
            "dur": 50,
            "args": {"hlo_op": "all-reduce", "correlation_id": 1},
        },
        {"name": "gemm", "pid": 1, "ts": 200, "dur": 10, "args": {}},
    ]
    mock_tree_cls.return_value = mock_tree

    with patch.object(JaxNcclAnalyser, "build_collectives_df_through_xla"):
        analyser = JaxNcclAnalyser(
            traces_dir=str(tmp_path),
            node_to_pb_file_mapping={"0": str(pb_path)},
            world_size=8,
        )

    assert "0" in analyser.node_to_trace_data
    assert len(analyser.node_to_trace_data["0"]) == 1
    assert analyser.node_to_trace_data["0"][0]["name"] == "ncclAllReduce"


def test_jax_build_collectives_df_with_provided_map(jax_analyser, tmp_path):
    xla_file = tmp_path / "node0.txt"
    xla_file.write_text(
        "HLO %x = bf16[4,8]{1,0} all-reduce(bf16[4,8]{1,0} %arg0), "
        'replica_groups={{0,1}}, scheduling_name="all-reduce", dimensions={0}\n',
        encoding="utf-8",
    )
    jax_analyser.node_to_xla_file_map = {"0": str(xla_file)}
    jax_analyser.build_collectives_df_through_xla()
    assert jax_analyser.df_collectives is not None
    assert not jax_analyser.df_collectives.empty


def test_jax_build_node_to_xla_file_map(jax_analyser, tmp_path):
    traces_dir = tmp_path / "traces"
    xla_dir = traces_dir / "node0" / "xla_dumps"
    xla_dir.mkdir(parents=True)
    pb_dir = traces_dir / "node0" / "profiles"
    pb_dir.mkdir(parents=True)
    xla_file = xla_dir / "jit_train_step.0_gpu_after_optimizations.txt"
    xla_file.write_text("noop\n", encoding="utf-8")
    pb_file = pb_dir / "profile.pb"
    pb_file.write_text("pb", encoding="utf-8")

    jax_analyser.traces_dir = str(traces_dir)
    jax_analyser.node_to_pb_file_mapping = {"0": str(pb_file)}
    mapping = jax_analyser._build_node_to_xla_file_map()
    assert mapping["0"] == str(xla_file)


def test_jax_build_node_to_xla_file_map_raises_when_missing(jax_analyser, tmp_path):
    jax_analyser.traces_dir = str(tmp_path)
    with pytest.raises(RuntimeError, match="No XLA files found"):
        jax_analyser._build_node_to_xla_file_map()


def test_jax_lookup_collective_info(jax_analyser):
    jax_analyser.df_collectives = pd.DataFrame(
        [
            {
                "node": "0",
                "collective_name": "all-reduce",
                "replica_string": "{{0,1}}",
                "replica_groups": [[0, 1]],
                "data(bytes)": 2048,
            }
        ]
    )
    replica_string, replica_groups, data_bytes = jax_analyser._lookup_collective_info(
        "0", "all-reduce"
    )
    assert replica_string == "{{0,1}}"
    assert replica_groups == [[0, 1]]
    assert data_bytes == 2048
    assert jax_analyser._lookup_collective_info("0", "missing") == (None, None, None)
    jax_analyser.df_collectives = pd.DataFrame()
    assert jax_analyser._lookup_collective_info("0", "all-reduce") == (None, None, None)


def test_jax_build_df_long_from_node_trace_data(jax_analyser):
    jax_analyser.node_to_trace_data = {
        "0": {
            0: {
                "pid": 1,
                "ts": 100,
                "dur": 50,
                "args": {
                    "hlo_op": "all-reduce",
                    "hlo_module": "m",
                    "correlation_id": 7,
                },
                "process": {"process_name": "gpu0"},
            },
            1: None,
        }
    }
    jax_analyser.df_collectives = pd.DataFrame(
        [
            {
                "node": "0",
                "collective_name": "all-reduce",
                "replica_string": "{{0,1}}",
                "replica_groups": [[0, 1]],
                "data(bytes)": 1024,
            }
        ]
    )
    df = jax_analyser.build_df_long()
    assert len(df) == 1
    assert df.iloc[0]["collective_name"] == "all-reduce"
    assert df.iloc[0]["gpu_rank"] == 0
    assert df.iloc[0]["data(bytes)"] == 1024


@patch("builtins.print")
def test_jax_analyze_all_collectives_for_multiple_types(mock_print, jax_analyser):
    rows = [
        _make_collective_row(
            gpu_rank=0, ts=100, dur=100, collective_name="all-reduce", data_bytes=1024
        ),
        _make_collective_row(
            gpu_rank=1, ts=100, dur=200, collective_name="all-reduce", data_bytes=1024
        ),
        _make_collective_row(
            gpu_rank=0,
            ts=300,
            dur=100,
            collective_name="all-gather",
            data_bytes=512,
            index_in_group=0,
        ),
        _make_collective_row(
            gpu_rank=1,
            ts=300,
            dur=150,
            collective_name="all-gather",
            data_bytes=512,
            index_in_group=0,
        ),
    ]
    df = pd.DataFrame(rows)
    results = jax_analyser.analyze_all_collectives_from_df(df)
    assert "all-reduce" in results
    assert "all-gather" in results
    assert results["all-reduce"]["num_slices"] == 1


@patch("builtins.print")
def test_jax_analyze_collective_types_and_display_summary(mock_print, jax_analyser):
    bandwidth_results = {
        "all-reduce.1": {
            "bandwidths": [10.0, 12.0],
            "bus_bandwidths": [15.0, 18.0],
            "avg_bandwidth": 11.0,
            "avg_bus_bandwidth": 16.5,
            "slice_info": [{"collective_id": "all-reduce.1_0"}],
            "num_slices": 1,
            "data_size_bytes": 1024,
        },
        "all-gather.1": {
            "bandwidths": [8.0],
            "bus_bandwidths": [6.0],
            "avg_bandwidth": 8.0,
            "avg_bus_bandwidth": 6.0,
            "slice_info": [{"collective_id": "all-gather.1_0"}],
            "num_slices": 1,
            "data_size_bytes": 512,
        },
        "custom-op": {
            "bandwidths": [5.0],
            "bus_bandwidths": [5.0],
            "avg_bandwidth": 5.0,
            "avg_bus_bandwidth": 5.0,
            "slice_info": [{"collective_id": "custom-op_0"}],
            "num_slices": 1,
            "data_size_bytes": 256,
        },
    }
    collective_types, summary_data = jax_analyser.analyze_collective_types_from_df(
        bandwidth_results
    )
    assert "all-reduce" in collective_types
    assert "all-gather" in collective_types
    assert "other" in collective_types
    assert summary_data
    jax_analyser.display_summary_table(summary_data)


@patch("builtins.print")
def test_jax_bandwidth_calculation_edge_cases(mock_print, jax_analyser):
    empty_df = pd.DataFrame(
        columns=[
            "collective_name",
            "collective_id",
            "gpu_rank",
            "dur",
            "replica_string",
            "replica_groups",
            "data(bytes)",
        ]
    )
    result = jax_analyser._calculate_collective_bandwidth_from_df(
        empty_df, "all-reduce"
    )
    assert result[0] == []

    mismatch_df = pd.DataFrame(
        [
            _make_collective_row(replica_string="{{0,1}}"),
            _make_collective_row(gpu_rank=1, replica_string="{{0,2}}"),
        ]
    )
    bw, _, _ = jax_analyser._calculate_bandwidth_per_replica_group(
        mismatch_df, "all-reduce", "all-reduce_0"
    )
    assert bw == []

    no_groups_df = pd.DataFrame([{**_make_collective_row(), "replica_groups": None}])
    bw, _, _ = jax_analyser._calculate_bandwidth_per_replica_group(
        no_groups_df, "all-reduce", "all-reduce_0"
    )
    assert bw == []


@patch("builtins.print")
@pytest.mark.parametrize(
    "collective_name",
    ["reduce-scatter", "all-to-all", "collective-permute", "unknown-op"],
)
def test_jax_bandwidth_for_collective_variants(
    mock_print, jax_analyser, collective_name
):
    rows = [
        _make_collective_row(
            gpu_rank=0,
            collective_name=collective_name,
            data_bytes=1024,
            dur=100,
            ts=100,
        ),
        _make_collective_row(
            gpu_rank=1,
            collective_name=collective_name,
            data_bytes=1024,
            dur=110,
            ts=100,
        ),
    ]
    df = pd.DataFrame(rows)
    bw, _, details = jax_analyser._calculate_bandwidth_per_replica_group(
        df, collective_name, f"{collective_name}_0"
    )
    assert len(bw) == 1
    assert len(details) == 1


@patch("builtins.print")
def test_jax_analyze_all_collectives_builds_df_when_missing(mock_print, jax_analyser):
    jax_analyser.node_to_trace_data = {
        "0": {
            0: {
                "pid": 1,
                "ts": 100,
                "dur": 50,
                "args": {"hlo_op": "all-reduce"},
                "process": {"process_name": "gpu0"},
            }
        }
    }
    jax_analyser.df_collectives = pd.DataFrame(
        [
            {
                "node": "0",
                "collective_name": "all-reduce",
                "replica_string": "{{0,1}}",
                "replica_groups": [[0, 1]],
                "data(bytes)": 1024,
            }
        ]
    )
    results = jax_analyser.analyze_all_collectives_from_df()
    assert "all-reduce" in results


@patch("builtins.print")
def test_jax_build_collectives_df_auto_xla_map(mock_print, jax_analyser, tmp_path):
    traces_dir = tmp_path / "traces"
    xla_dir = traces_dir / "node0" / "xla_dumps"
    xla_dir.mkdir(parents=True)
    pb_dir = traces_dir / "node0" / "profiles"
    pb_dir.mkdir(parents=True)
    xla_file = xla_dir / "jit_train_step.0_gpu_after_optimizations.txt"
    xla_file.write_text(
        "HLO %x = bf16[4,8]{1,0} all-reduce(bf16[4,8]{1,0} %arg0), "
        'replica_groups={{0,1}}, scheduling_name="all-reduce", dimensions={0}\n',
        encoding="utf-8",
    )
    pb_file = pb_dir / "profile.pb"
    pb_file.write_text("pb", encoding="utf-8")

    jax_analyser.traces_dir = str(traces_dir)
    jax_analyser.node_to_pb_file_mapping = {"0": str(pb_file)}
    jax_analyser.node_to_xla_file_map = None
    jax_analyser.build_collectives_df_through_xla()
    assert jax_analyser.df_collectives is not None
    assert not jax_analyser.df_collectives.empty


@patch("builtins.print")
def test_jax_analyze_collective_types_all_variants(mock_print, jax_analyser):
    empty_result = {
        "bandwidths": [],
        "bus_bandwidths": [],
        "avg_bandwidth": 0.0,
        "avg_bus_bandwidth": 0.0,
        "slice_info": [],
        "num_slices": 0,
        "data_size_bytes": 0,
    }
    bandwidth_results = {
        "all-to-all.1": {**empty_result, "bandwidths": [4.0], "bus_bandwidths": [3.0]},
        "reduce-scatter.1": {
            **empty_result,
            "bandwidths": [6.0],
            "bus_bandwidths": [4.5],
        },
        "collective-permute.1": {
            **empty_result,
            "bandwidths": [7.0],
            "bus_bandwidths": [7.0],
        },
        "empty-type": empty_result,
    }
    collective_types, summary_data = jax_analyser.analyze_collective_types_from_df(
        bandwidth_results
    )
    assert "all-to-all" in collective_types
    assert "reduce-scatter" in collective_types
    assert "collective-permute" in collective_types
    assert "empty-type" not in {entry["type"] for entry in summary_data}


@patch("builtins.print")
def test_jax_bandwidth_more_edge_cases(mock_print, jax_analyser):
    missing_slice = jax_analyser._calculate_bandwidth_per_replica_group(
        pd.DataFrame([_make_collective_row()]),
        "all-reduce",
        "missing_slice",
    )
    assert missing_slice[0] == []

    no_replica_df = pd.DataFrame([{**_make_collective_row(), "replica_string": None}])
    bw, _, _ = jax_analyser._calculate_bandwidth_per_replica_group(
        no_replica_df, "all-reduce", "all-reduce_0"
    )
    assert bw == []

    multi_group_df = pd.DataFrame(
        [
            _make_collective_row(
                gpu_rank=0,
                replica_groups=[[99, 100], [0, 1]],
                replica_string="{{0,1}}",
            )
        ]
    )
    bw, _, _ = jax_analyser._calculate_bandwidth_per_replica_group(
        multi_group_df, "all-reduce", "all-reduce_0"
    )
    assert len(bw) == 1

    invalid_slice_df = pd.DataFrame(
        [_make_collective_row(gpu_rank=0, replica_groups=[[]], replica_string="{{}}")]
    )
    _, _, _, _, slice_info = jax_analyser._calculate_collective_bandwidth_from_df(
        invalid_slice_df, "all-reduce"
    )
    assert slice_info == []
