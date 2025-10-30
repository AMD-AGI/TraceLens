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
from TraceLens.Trace2Tree.trace_to_tree import JaxTraceToTree
from TraceLens.util import DataLoader,TraceEventUtils
from functools import reduce
from operator import mul

class JaxNcclAnalyser:
    def __init__(self, traces_dir, node_to_pb_file_mapping, world_size):
        self.traces_dir = traces_dir
        self.node_to_pb_file_mapping = node_to_pb_file_mapping
        self.world_size = world_size

        # Byte sizes per dtype
        self.dtype_to_bytes = {
            'bf16': 2,
            'f16': 2, 
            'half': 2,
            'f32': 4,
            'float': 4,
            'f64': 8,
            'double': 8,
            's8': 1,
            's16': 2,
            's32': 4,
            's64': 8,
            'u8': 1,
            'u16': 2, 
            'u32': 4,
            'u64': 8,
            'pred': 1,  # predicate/boolean
            'c64': 8,   # complex64
            'c128': 16, # complex128
        }

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

    # Calculate data bytes from tensor output of a collective event
    def _parse_tensor_spec_to_bytes(self,output_string): 
            
            if not output_string or output_string.strip() == "":
                return 0

            # Regex pattern to match tensor specifications
            # Matches: dtype[dim1,dim2,...]{layout} or just dtype[dim1,dim2,...]
            tensor_pattern = r'([a-zA-Z]+[0-9]*)\[([0-9,]+)\](?:\{[0-9,]+\})?'    
            total_bytes = 0
            matches = re.findall(tensor_pattern, output_string)
            
            for dtype_str, dimensions_str in matches:
                # Get bytes per element
                bytes_per_element = self.dtype_to_bytes.get(dtype_str.lower(), 0)  # Default to 0 bytes
                if bytes_per_element == 0:
                    print("Datatype can not be parsed.")

                # Parse dimensions and calculate total elements
                dimensions = [int(d.strip()) for d in dimensions_str.split(',') if d.strip()]
                
                if dimensions:
                    total_elements = reduce(mul, dimensions, 1)
                    tensor_bytes = total_elements * bytes_per_element
                    total_bytes += tensor_bytes
            
            return int(total_bytes)
    
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

    # Extract collective name from the xla dump file
    def _extract_collective_name(self, line: str):    
                
        match = re.search(r'scheduling_name="([^"]+)"', line)
        if match:
            return match.group(1)
        return None
    
    # Extract collective's replica_groups from the xla dump file
    def _extract_replica_groups(self, line: str):
   
        pattern = r"replica_groups=(?P<replica_string>(?:\{(?:\{[0-9]+(?:,[0-9]+)*\}(?:,\{[0-9]+(?:,[0-9]+)*\})*)\}|\[[0-9]+(?:,[0-9]+)*\]<=\[[0-9]+(?:,[0-9]+)*\])(?:T\([0-9,]+\)\s+dimensions=\{[0-9,]*\})?)"
        match = re.search(pattern, line)
        if match:
            return match.group('replica_string')
        
        # If no replica_groups found, try to match source_target_pairs (for collective-permute)
        source_target_pattern = r"source_target_pairs=(?P<source_target_string>\{(?:\{[0-9]+,[0-9]+\}(?:,\{[0-9]+,[0-9]+\})*)\})"
        match = re.search(source_target_pattern, line)
        if match:
            return match.group('source_target_string')
        
        return None

    # Extract collective's tensors from the xla dump file
    def _extract_tensor_specs(self, line: str):

        # Look for tensor spec after '= ' and before the collective operation call
        # Format: ROOT? operation_name = tensor_spec operation_call(params), metadata...
        
        # Match everything after '= ' until we hit a space followed by an operation name
        equals_match = re.search(r'=\s+(.+?)\s+(?:all-to-all|all-gather|all-reduce|reduce-scatter|collective-permute)\(', line)
        if equals_match:
            return equals_match.group(1).strip()
        
        # Fallback: match everything after '= ' until end of meaningful tensor content
        # This handles cases where the operation pattern might be different
        equals_fallback = re.search(r'=\s+(.+?)(?:\s+[a-z-]+\(|,\s+channel_id|,\s+replica_groups|$)', line)
        if equals_fallback:
            tensor_spec = equals_fallback.group(1).strip()
            # Clean up any trailing commas or metadata
            tensor_spec = re.sub(r',?\s*$', '', tensor_spec)
            return tensor_spec
        
        return None
    

    def _parse_collectives_to_dataframe(self, file_path, node):
        """Parse collective operations and return a pandas DataFrame"""
        collective_ops = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and any(pattern in line for pattern in [
                'all-to-all-start', 'all-gather-start', 'all-reduce-start', 
                'reduce-scatter', 'collective-permute-start',
                'all-reduce =', 'all-gather =', 'all-to-all =',
                'all-to-all.', 'reduce-scatter.'
            ]):
                collective_name = self._extract_collective_name(line)
                replica_groups = self._extract_replica_groups(line)
                tensors = self._extract_tensor_specs(line)

                data = self._parse_tensor_spec_to_bytes(tensors)

                collective_ops.append({
                    'node': node,
                    'collective_name': collective_name,
                    'replica_groups': replica_groups,
                    'tensors': tensors,
                    'data(bytes)': data
                })
        
        return collective_ops

    def lookup_collective_info(self, node, collective_name):
        """
        Lookup replica_groups and data(bytes) from df_collectives for given node and collective_name
        
        Args:
            node: The node identifier
            collective_name: The collective operation name
            
        Returns:
            tuple: (replica_groups, data_bytes) or (None, None) if not found
        """
        if self.df_collectives is None or self.df_collectives.empty:
            return None, None
        
        # Filter df_collectives for matching node and collective_name
        matches = self.df_collectives[
            (self.df_collectives['node'] == node) & 
            (self.df_collectives['collective_name'] == collective_name)
        ]
        
        if matches.empty:
            return None, None
        
        # Get the first match (assuming duplicates have same values)
        match_row = matches.iloc[0]
        
        replica_groups = match_row.get('replica_groups')
        data_bytes = match_row.get('data(bytes)')
        
        # Convert NaN to None for consistency
        replica_groups = None if pd.isna(replica_groups) else replica_groups
        data_bytes = None if pd.isna(data_bytes) else data_bytes
        
        return replica_groups, data_bytes

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
            collective_ops = []
            for node, xla_file_path in node_to_xla_file_map.items():
                rows = self._parse_collectives_to_dataframe(xla_file_path,node)
                collective_ops.extend(rows)
            
            if collective_ops:
                if self.df_collectives is None:
                    self.df_collectives = pd.DataFrame(collective_ops)        
        

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

                # Initialize replica_groups and data(bytes) as None
                replica_groups = None
                data_bytes = None
                
                # Extract metadata fields safely
                if metadata := event.get("metadata"):
                    if "replica_groups" in metadata:
                        replica_groups = metadata["replica_groups"]
                    if "output" in metadata:
                        data_bytes = self._parse_tensor_spec_to_bytes(metadata["output"])                        

                # Check if replica_groups or data(bytes) are None/empty and fill from df_collectives
                collective_name = row.get("collective_name")

                # Check if we need to lookup missing values
                needs_replica_groups = replica_groups is None or (isinstance(replica_groups, str) and replica_groups.strip() == "")
                needs_data_bytes = data_bytes is None or data_bytes == 0

                if (needs_replica_groups or needs_data_bytes) and collective_name:
                    lookup_replica_groups, lookup_data_bytes = self.lookup_collective_info(node, collective_name)
                    
                    # Fill replica_groups if missing
                    if needs_replica_groups and lookup_replica_groups is not None:
                        replica_groups = lookup_replica_groups
                        
                    # Fill data(bytes) if missing
                    if needs_data_bytes and lookup_data_bytes is not None:
                        data_bytes = lookup_data_bytes
                
                row["replica_groups"] = replica_groups
                row["data(bytes)"] = data_bytes
                        

                # Ensure these fields exist in the row even if None
                if "replica_groups" not in row:
                    row["replica_groups"] = None
                if "data(bytes)" not in row:
                    row["data(bytes)"] = None
                
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

    def parse_replica_groups(self, replica_groups_str):
        """
        Parse replica groups and return list of lists
        """
        if not replica_groups_str or str(replica_groups_str).strip().lower() in ['nan', 'none', '']:
            return []            
        
        replica_groups_str = str(replica_groups_str).strip()
        
        # Handle explicit groups: {{0,1,2,3},{4,5,6,7},...}
        if replica_groups_str.startswith('{{'):
            # Remove outer braces first
            inner_content = replica_groups_str[2:-2]  # Remove {{ and }}
            # Split by },{
            group_strings = inner_content.split('},{')
            
            gpu_groups = []
            for group_str in group_strings:
                group = [int(x.strip()) for x in group_str.split(',')]
                gpu_groups.append(group)

            return gpu_groups
        
        # Handle IotaTileAssignment: [dims]<=[total]
        elif '<=' in replica_groups_str:
            # Example parse of a string like: [8,4]<=[4,8]T(1,0) dimensions={0}
            parts = replica_groups_str.split('<=')
            
            # Get dims: [8,4] 
            dims_match = re.search(r'\[([^\]]+)\]', parts[0])
            if not dims_match:
                return []
            dims_str = dims_match.group(1)
            dims = [int(x.strip()) for x in dims_str.split(',')]
            
            # Get reshape: [4,8]
            reshape_match = re.search(r'\[([^\]]+)\]', parts[1])
            if not reshape_match:
                return []
            reshape_str = reshape_match.group(1)
            reshape_dims = [int(x.strip()) for x in reshape_str.split(',')]
            
            # Get transpose if exists: T(1,0)
            transpose_perm = None
            if 'T(' in replica_groups_str:
                transpose_match = re.search(r'T\(([^)]+)\)', replica_groups_str)
                if transpose_match:
                    transpose_perm = [int(x.strip()) for x in transpose_match.group(1).split(',')]
            
            # Using IOTA logic to build the device array
            total_elements = math.prod(dims)
            arr = np.arange(total_elements)

            # Reshape to intermediate shape
            if reshape_dims != [total_elements]:
                arr = arr.reshape(reshape_dims)

            # Apply transpose if specified
            if transpose_perm:
                arr = arr.transpose(transpose_perm)
            
            # Reshape to final dimensions
            arr = arr.reshape(dims)

            # Create groups
            gpu_groups = []                                        
            for row in arr:
                gpu_groups.append([int(x) for x in row])
            
            return gpu_groups
        
        return []

    def calculate_bandwidth_per_replica_group(self, df, collective_name, collective_id):
        """
        Calculate bandwidth for each replica group within a collective slice
        """
        slice_data = df[(df["collective_name"] == collective_name) & 
                        (df["collective_id"] == collective_id)].copy()
        
        if len(slice_data) == 0:
            return [], []
        
        # Validate replica group consistency within the collective slice
        unique_replica_groups = slice_data["replica_groups"].nunique()
        if unique_replica_groups == 0:
            print(f"Error: No replica groups found for collective {collective_id}")
            return [], []
        elif unique_replica_groups > 1:
            print(f"Warning: Collective {collective_id} has {unique_replica_groups} different replica group configurations. Expected exactly 1.")
            return [], []
        
        # Get replica groups and data size from the first row
        replica_groups_str = slice_data.iloc[0]["replica_groups"]
        data_size_bytes = slice_data.iloc[0]["data(bytes)"]
        
        print(f"    Analyzing slice: {collective_id}")
        
        # Parse replica groups to get actual GPU groupings
        gpu_groups = self.parse_replica_groups(replica_groups_str)
        
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
           
            # Find the GPU that takes longest to complete (slowest GPU)
            slowest_gpu_idx = group_data['dur'].idxmax()
            slowest_gpu_info = group_data.loc[slowest_gpu_idx]
            
            # Duration is determined by the slowest GPU's duration
            slowest_gpu_duration_us = slowest_gpu_info['dur']
            slowest_gpu_duration_s = slowest_gpu_duration_us / 1e6
            
            if slowest_gpu_duration_s > 0:
                # Calculate algorithmic bytes based on collective type
                actual_group_size = len(gpu_group)
                
                # Get data size for this specific group from the slowest GPU's data
                group_data_size_bytes = slowest_gpu_info['data(bytes)']
                
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

                bandwidth_gbps = (algorithmic_bytes / (1024**3)) / slowest_gpu_duration_s
                group_bandwidths.append(bandwidth_gbps)
                
                group_details.append({
                    'group_idx': group_idx,
                    'gpu_group': gpu_group,
                    'gpus_in_data': group_gpus_in_data,
                    'bandwidth_gbps': bandwidth_gbps,
                    'duration_us': slowest_gpu_duration_us,
                    'slowest_gpu_rank': int(slowest_gpu_info['gpu_rank']),
                    'algorithmic_bytes': algorithmic_bytes,
                    'group_data_size_bytes': group_data_size_bytes,
                    'actual_group_size': actual_group_size,
                    'participants_in_data': len(group_data)
                })
                
                print(f"        Bandwidth: {bandwidth_gbps:.2f} GB/s")
                print(f"        Slowest GPU: rank {int(slowest_gpu_info['gpu_rank'])}, duration: {float(slowest_gpu_info['dur']):.1f} μs")
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

