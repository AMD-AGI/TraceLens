# MIT License

# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
from collections import defaultdict
import os
import json
import pandas as pd
import warnings
import gzip
from glob import glob
import numpy as np
import math
from TraceLens.NcclAnalyser.util.xla_parser import XLACollectiveParser
from TraceLens.Trace2Tree.trace_to_tree import JaxTraceToTree
from TraceLens.util import DataLoader,TraceEventUtils
from functools import reduce
from operator import mul
import ast

class JaxNcclAnalyser:
    def __init__(self, traces_dir, node_to_pb_file_mapping, world_size):
        self.traces_dir = traces_dir
        self.node_to_pb_file_mapping = node_to_pb_file_mapping
        self.world_size = world_size

        # Internal storage
        self.node_to_trace_data = {}  # Stores per-node data
        self.df_per_rank_coll = None  # Will store the dataframe
        self.df_collectives = None # Store collective info parsed from xla

        self.load_trace_data()
        self.build_collectives_df_through_xla()

    # Filter communication events
    def _nccl_event_filter(self,event):     

        is_coll_comm_event = event.get("gpu_kernel_op_cat","") == "Communication rccl/nccl" 
        return is_coll_comm_event

    
    def load_trace_data(self):
        """Uses JaxTraceToTree to extracts relevant events."""
        
        self.node_to_trace_data.clear()

        for node, protobuf_filepath in self.node_to_pb_file_mapping.items():

            print(f"Loading node {node} from {protobuf_filepath}")
            pb_data = DataLoader.load_data(protobuf_filepath, save_preprocessed=False)
            trace_events = pb_data["traceEvents"]            
            linking_key = 'correlation_id'
            categorizer =  TraceEventUtils.prepare_event_categorizer(trace_events)
            non_metadata_events = TraceEventUtils.non_metadata_events(trace_events)
            tree = JaxTraceToTree(non_metadata_events, event_to_category=categorizer, linking_key=linking_key)
            metadata = TraceEventUtils.get_metadata(trace_events)
            tree.build_tree(metadata=metadata, pb_file_name=protobuf_filepath)
            nccl_events = [event for event in tree.events if self._nccl_event_filter(event)]
            
            # Build a dictionary with event data
            node_dict = {idx: evt for idx, evt in enumerate(nccl_events)}
            self.node_to_trace_data[node] = node_dict
  

    def lookup_collective_info(self, node, collective_name):
        """
        Lookup replica_groups and data(bytes) from df_collectives for given node and collective_name
        
        Args:
            node: The node identifier
            collective_name: The collective operation name
            
        Returns:
            tuple: (replica_string, replica_groups, data_bytes) or (None, None, None) if not found
        """
        if self.df_collectives is None or self.df_collectives.empty:
            return None, None, None
        
        # Filter df_collectives for matching node and collective_name
        matches = self.df_collectives[
            (self.df_collectives['node'] == node) & 
            (self.df_collectives['collective_name'] == collective_name)
        ]
        
        if matches.empty:
            return None, None, None
        
        # Get the first match (assuming duplicates have same values)
        match_row = matches.iloc[0]
        
        replica_groups = match_row.get('replica_groups')
        data_bytes = match_row.get('data(bytes)')    
        replica_string = match_row.get('replica_string')    
        
        return replica_string, replica_groups, data_bytes

    # Build a data frame through parsing a XLA file per node        
    def build_collectives_df_through_xla(self):

        # Try to get xla dump file for every node
        node_to_xla_file_map = {}    

        # Find all XLA files in traces directory first
        xla_pattern = os.path.join(self.traces_dir, "**", "xla_dumps", "*jit_train_step.gfx942_gpu_after_optimizations.txt")
        all_xla_files = glob(xla_pattern, recursive=True)
        
        # Normalize traces_dir path
        traces_dir_normalized = os.path.normpath(self.traces_dir)
        
        for node, pb_file_path in self.node_to_pb_file_mapping.items():

            # Remove traces_dir from protobuf file path
            pb_relative = os.path.relpath(pb_file_path, traces_dir_normalized)
            
            best_match = None
            max_common_length = 0
            
            # Try to find XLA file with most common relative path
            for xla_file in all_xla_files:
                # Remove traces_dir from XLA file path
                xla_relative = os.path.relpath(xla_file, traces_dir_normalized)
                
                # Find common prefix length
                common_length = 0
                min_len = min(len(pb_relative), len(xla_relative))
                
                for i in range(min_len):
                    if pb_relative[i] == xla_relative[i]:
                        common_length += 1
                    else:
                        break
                
                if common_length > max_common_length:
                    max_common_length = common_length
                    best_match = xla_file
            
            if best_match:
                node_to_xla_file_map[node] = best_match

        # Constructing dataframe        

        if node_to_xla_file_map:
            parser = XLACollectiveParser(node_to_xla_file_map)
            self.df_collectives = parser.parse_collectives_to_dataframe()       
        

    def build_df_long(self):
        """Constructs a long table where each row is a collective event on a gpu rank."""      

        rows = []
        for node, node_events in self.node_to_trace_data.items():
            for event in node_events.values():
                if not event:
                    continue
                    
                # Build row with guaranteed fields
                row = {
                    "node": node,
                    "gpu_rank": int(node) * 8 + int(event["pid"]) - 1,
                    "pid": event["pid"],
                    "ts": event["ts"],
                    "dur": event["dur"]
                }
                
                # Extract args fields safely
                if args := event.get("args"):
                    row.update({
                        "collective_name": args.get("hlo_op"),
                        "hlo_module": args.get("hlo_module"),
                        "correlation_id": args.get("correlation_id")
                    })

                # Get replica_groups and data(bytes) from df_collectives via lookup
                collective_name = row.get("collective_name")
                replica_string, replica_groups, data_bytes = None, None, None

                if collective_name:
                    replica_string, replica_groups, data_bytes = self.lookup_collective_info(node, collective_name)
                
                row["replica_string"] = replica_string
                row["replica_groups"] = replica_groups
                row["data(bytes)"] = data_bytes
                
                # Extract process information safely
                if process := event.get("process"):
                    row["process_name"] = process.get("process_name")
                
                rows.append(row)
                
        df_long = pd.DataFrame(rows)
        df_long = df_long.reset_index(drop=True)

        # Assign an index within each process group and rank
        df_long['collective_name'] = df_long['collective_name'].fillna('unknown')
        df_long['index_in_group'] = df_long.groupby(['collective_name', 'pid', 'node'])['ts'].rank(method='first').astype(int) - 1

        # Create a composite collective ID (process group + index)
        df_long['collective_id'] = df_long['collective_name'] + '_' + df_long['index_in_group'].astype(str)

        self.df_per_rank_coll = df_long

        return df_long

    def calculate_bandwidth_per_replica_group(self, df, collective_name, collective_id):
        """
        Calculate bandwidth for each replica group within a collective slice
        """
        slice_data = df[(df["collective_name"] == collective_name) & 
                        (df["collective_id"] == collective_id)].copy()
        
        if len(slice_data) == 0:
            return [], []
        
        # Validate replica string consistency within the collective slice
        unique_replica_groups = slice_data["replica_string"].nunique()
        if unique_replica_groups == 0:
            print(f"Error: No replica groups found for collective {collective_id}")
            return [], []
        elif unique_replica_groups > 1:
            print(f"Warning: Collective {collective_id} has {unique_replica_groups} different replica group configurations. Expected exactly 1.")
            return [], []
        
        # Get gpu groups 
        gpu_groups = slice_data.iloc[0]["replica_groups"]      
        
        if not gpu_groups:
            print(f"    No valid groups found")
            return [], []
        
        group_bandwidths = []
        group_details = []
        
        # Get available GPU ranks in this slice
        available_gpu_ranks = set(slice_data["gpu_rank"].unique())
        print(f"    Available GPU ranks in data: {sorted(available_gpu_ranks)}")
        
        # Calculate bandwidth for each replica group
        for group_idx, gpu_group in enumerate(gpu_groups):
            print(f"      Analyzing group {group_idx}: {gpu_group}")
            
            # Find intersection between this group and available GPU ranks
            group_gpus_in_data = [gpu_id for gpu_id in gpu_group if gpu_id in available_gpu_ranks]
            
            if not group_gpus_in_data:
                print(f"        No matching GPUs found in data")
                continue
            
            print(f"        Matching GPUs in data: {group_gpus_in_data}")
            
            # Get data for GPUs in this group
            group_data = slice_data[slice_data["gpu_rank"].isin(group_gpus_in_data)]
            
            if len(group_data) == 0:
                continue
           
            # Find the GPU that completes fastest (shortest duration)
            fastest_gpu_idx = group_data['dur'].idxmin()
            fastest_gpu_info = group_data.loc[fastest_gpu_idx]
            
            # Duration is determined by the fastest GPU's duration
            fastest_gpu_duration_us = fastest_gpu_info['dur']
            fastest_gpu_duration_s = fastest_gpu_duration_us / 1e6
            
            if fastest_gpu_duration_s > 0:
                # Calculate algorithmic bytes based on collective type
                actual_group_size = len(gpu_group)
                
                # Get data size for this specific group from the fastest GPU's data
                group_data_size_bytes = fastest_gpu_info['data(bytes)']
                
                if collective_name.startswith('all-gather'):
                    # All-gather: each GPU sends its data to (group_size - 1) other GPUs
                    # and receives data from (group_size - 1) other GPUs
                    algorithmic_bytes = 2 * group_data_size_bytes * (actual_group_size - 1)
                elif collective_name.startswith('all-reduce'):
                    # All-reduce: Each GPU provides an array of size N 
                    # and get a reduced (reduction operation could be sum, min, max) array of size N
                    algorithmic_bytes = 2 * group_data_size_bytes
                elif collective_name.startswith('reduce-scatter'):
                    # Reduce-scatter: each GPU sends its tensor to participate in collective 
                    # and receives reduced result
                    algorithmic_bytes = group_data_size_bytes + (group_data_size_bytes) / actual_group_size
                elif collective_name.startswith('all-to-all'):
                    # All-to-all: each GPU exchanges data with every other GPU
                    algorithmic_bytes = 2 * group_data_size_bytes * (actual_group_size - 1) / actual_group_size
                elif collective_name.startswith('collective-permute'):
                    # Collective-permute: point-to-point communication
                    algorithmic_bytes = 2 * group_data_size_bytes
                else:
                    # Default case
                    algorithmic_bytes = group_data_size_bytes

                bandwidth_gbps = (algorithmic_bytes / (1024**3)) / fastest_gpu_duration_s
                group_bandwidths.append(bandwidth_gbps)
                
                group_details.append({
                    'group_idx': group_idx,
                    'gpu_group': gpu_group,
                    'gpus_in_data': group_gpus_in_data,
                    'bandwidth_gbps': bandwidth_gbps,
                    'duration_us': fastest_gpu_duration_us,
                    'fastest_gpu_rank': int(fastest_gpu_info['gpu_rank']),
                    'algorithmic_bytes': algorithmic_bytes,
                    'group_data_size_bytes': group_data_size_bytes,
                    'actual_group_size': actual_group_size,
                    'participants_in_data': len(group_data)
                })
                
                print(f"        Bandwidth: {bandwidth_gbps:.2f} GB/s")
                print(f"        Fastest GPU: rank {int(fastest_gpu_info['gpu_rank'])}, duration: {float(fastest_gpu_info['dur']):.1f} μs")
                print(f"        Group data size: {group_data_size_bytes:,} bytes ({group_data_size_bytes/(1024**3):.3f} GB)")
                print(f"        Algorithmic bytes: {algorithmic_bytes:,} ({algorithmic_bytes/(1024**3):.3f} GB)")
        
        return group_bandwidths, group_details

    def calculate_collective_bandwidth_from_df(self, df, collective_name):
        """
        Calculate bandwidth for a collective operation by analyzing each replica group
        """
        collective_data = df[df["collective_name"] == collective_name].copy()
        
        if collective_data.empty:
            return [], 0.0, []
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {collective_name}")
       
        # Get unique collective IDs (slices)
        unique_collective_ids = collective_data["collective_id"].unique()
        print(f"Number of slices: {len(unique_collective_ids)}")
        
        all_slice_bandwidths = []
        slice_info = []
        
        for collective_id in unique_collective_ids:
            print(f"\n  Slice: {collective_id}")
            
            # Calculate bandwidth for each replica group in this slice
            group_bandwidths, group_details = self.calculate_bandwidth_per_replica_group(
                df, collective_name, collective_id
            )
            
            if group_bandwidths:
                # For the collective slice, use average across groups
                slice_avg_bandwidth = np.mean(group_bandwidths)
                slice_min_bandwidth = np.min(group_bandwidths)
                slice_max_bandwidth = np.max(group_bandwidths)
                
                all_slice_bandwidths.append(slice_avg_bandwidth)
                
                slice_info.append({
                    'collective_id': collective_id,
                    'group_bandwidths': group_bandwidths,
                    'group_details': group_details,
                    'slice_avg_bandwidth': slice_avg_bandwidth,
                    'slice_min_bandwidth': slice_min_bandwidth,
                    'slice_max_bandwidth': slice_max_bandwidth,
                    'num_groups': len(group_bandwidths)
                })
                
                print(f"    Slice summary: avg={slice_avg_bandwidth:.2f} GB/s, range=[{slice_min_bandwidth:.2f}, {slice_max_bandwidth:.2f}] GB/s")
            else:
                print(f"    No valid bandwidth calculations for slice {collective_id}")
        
        # Overall collective bandwidth (average across slices)
        overall_avg_bandwidth = np.mean(all_slice_bandwidths) if all_slice_bandwidths else 0.0
        
        print(f"\nOverall Results:")
        print(f"  Average bandwidth: {overall_avg_bandwidth:.2f} GB/s")
        if all_slice_bandwidths:
            print(f"  Range: {min(all_slice_bandwidths):.2f} - {max(all_slice_bandwidths):.2f} GB/s")
            print(f"  Std deviation: {np.std(all_slice_bandwidths):.2f} GB/s")
        
        return all_slice_bandwidths, overall_avg_bandwidth, slice_info

    def analyze_all_collectives_from_df(self, df=None):
        """
        Analyze bandwidth for all collective operations using dataframe data directly
        """
        if df is None:
            df = self.df_per_rank_coll
            if df is None:
                df = self.build_df_long()
        
        results = {}
        
        # Get unique collective names
        unique_collectives = df["collective_name"].unique()
        
        print("Analyzing bandwidth for all collective operations from dataframe...")
        print("=" * 80)
        
        for collective_name in unique_collectives:
            bandwidths, avg_bandwidth, slice_info = self.calculate_collective_bandwidth_from_df(df, collective_name)
            
            if slice_info:
                results[collective_name] = {
                    'bandwidths': bandwidths,
                    'avg_bandwidth': avg_bandwidth,
                    'slice_info': slice_info,
                    'num_slices': len(slice_info),
                    'data_size_bytes': df[df["collective_name"] == collective_name].iloc[0]['data(bytes)']
                }
        
        return results

    def analyze_collective_types_from_df(self, bandwidth_results):
        """
        Combine all collective operations by type and calculate aggregate statistics
        """
        collective_types = {}
        
        # Group results by collective type
        for collective_name, result in bandwidth_results.items():
            # Extract collective type from name
            if collective_name.startswith('all-gather'):
                col_type = 'all-gather'
            elif collective_name.startswith('all-reduce'):
                col_type = 'all-reduce'
            elif collective_name.startswith('all-to-all'):
                col_type = 'all-to-all'
            elif collective_name.startswith('reduce-scatter'):
                col_type = 'reduce-scatter'
            elif collective_name.startswith('collective-permute'):
                col_type = 'collective-permute'
            else:
                col_type = 'other'
            
            if col_type not in collective_types:
                collective_types[col_type] = {
                    'all_bandwidths': [],
                    'operations': [],
                    'total_slices': 0,
                    'total_data_bytes': 0
                }
            
            # Collect all bandwidth measurements from this collective
            collective_types[col_type]['all_bandwidths'].extend(result['bandwidths'])
            collective_types[col_type]['operations'].append(collective_name)
            collective_types[col_type]['total_slices'] += result['num_slices']
            collective_types[col_type]['total_data_bytes'] += result['data_size_bytes']
        
        # Calculate aggregate statistics for each type
        print("\nCOMBINED COLLECTIVE TYPE ANALYSIS")
        print("=" * 80)
        
        summary_data = []
        
        for col_type, data in collective_types.items():
            if not data['all_bandwidths']:
                continue
                
            bandwidths = np.array(data['all_bandwidths'])
            
            stats = {
                'type': col_type,
                'num_operations': len(data['operations']),
                'total_slices': data['total_slices'],
                'total_measurements': len(bandwidths),
                'avg_bandwidth_gbps': np.mean(bandwidths),
                'median_bandwidth_gbps': np.median(bandwidths),
                'min_bandwidth_gbps': np.min(bandwidths),
                'max_bandwidth_gbps': np.max(bandwidths),
                'std_bandwidth_gbps': np.std(bandwidths),
                'p25_bandwidth_gbps': np.percentile(bandwidths, 25),
                'p75_bandwidth_gbps': np.percentile(bandwidths, 75),
                'avg_data_size_mb': data['total_data_bytes'] / len(data['operations']) / (1024**2)
            }
            
            summary_data.append(stats)
        
        return collective_types, summary_data

    def display_summary_table(self, summary_data):
        """
        Display a summary table of all collective types
        """
        print("\n\nSUMMARY TABLE - ALL COLLECTIVE TYPES")
        print("=" * 120)
        
        header = f"{'Type':<15} {'Ops':<4} {'Slices':<7} {'Avg(GB/s)':<10} {'Med(GB/s)':<10} {'Min(GB/s)':<10} {'Max(GB/s)':<10} {'StdDev':<8} {'DataSize(MB)':<12}"
        print(header)
        print("-" * 120)
        
        # Sort by average bandwidth descending
        sorted_data = sorted(summary_data, key=lambda x: x['avg_bandwidth_gbps'], reverse=True)
        
        for stats in sorted_data:
            row = f"{stats['type']:<15} {stats['num_operations']:<4} {stats['total_slices']:<7} {stats['avg_bandwidth_gbps']:<10.1f} {stats['median_bandwidth_gbps']:<10.1f} {stats['min_bandwidth_gbps']:<10.1f} {stats['max_bandwidth_gbps']:<10.1f} {stats['std_bandwidth_gbps']:<8.1f} {stats['avg_data_size_mb']:<12.1f}"
            print(row)