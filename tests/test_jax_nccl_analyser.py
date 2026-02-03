###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import unittest
from unittest.mock import patch
from TraceLens import JaxNcclAnalyser


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
        import pandas as pd
        import ast

        # Load the CSV file with all-gather data for a slice across 32 gpus
        # CSV contains made-up data for demonstrating representative calculations
        # No reference to any model or hardware.
        csv_path = "TraceLens/tests/test_data_jax_nccl_analyser/all_gather_collective_mockup_df.csv"
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
        import pandas as pd
        import numpy as np

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
