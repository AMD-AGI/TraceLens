###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import os
from functools import reduce
from operator import mul
import pandas as pd
import numpy as np
import math

class XLACollectiveParser:
    def __init__(self, node_to_file_mapping):
        """Initialize parser with node-to-file mapping and data type size definitions."""
        self.dtype_to_bytes = {
            'bf16': 2, 'f16': 2, 'half': 2, 'f32': 4, 'float': 4, 'f64': 8, 'double': 8,
            's8': 1, 's16': 2, 's32': 4, 's64': 8, 'u8': 1, 'u16': 2, 'u32': 4, 'u64': 8,
            'pred': 1, 'c64': 8, 'c128': 16,
        }
        self.node_to_file_mapping = node_to_file_mapping

    # Public methods
    def parse_collectives_to_dataframe(self):
        """Parse multiple XLA dump files and extract collective operations into a DataFrame."""
        all_collective_ops = []
        
        for node, file_path in self.node_to_file_mapping.items():
            if not os.path.exists(file_path):
                print(f"Warning: File not found for node {node}: {file_path}")
                continue
            
            collective_ops = []
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Warning: Error reading file for node {node}: {e}")
                continue
            
            for line in lines:
                line = line.strip()
                if line and re.search(r'scheduling_name="[^"]*(?:all-to-all|all-gather|all-reduce|reduce-scatter|collective-permute)[^"]*"', line):
                    collective_name = self._extract_collective_name(line)               
                    replica_string, replica_groups = self._extract_replica_groups(line)
                    tensors = self._extract_tensor_specs(line)
                    split_dimension = self._extract_split_dimension(line)
                    replica_group_size = self._calculate_replica_group_size(replica_groups)
                    tensor_slice = self._calculate_tensor_slice(tensors, split_dimension, replica_groups, collective_name, replica_group_size)
                    data_bytes = self._calculate_data_bytes(tensor_slice)
                    
                    op_data = {
                        'node': node,
                        'collective_name': collective_name,
                        'replica_string': replica_string,
                        'replica_groups': replica_groups,
                        'replica_group_size': replica_group_size,
                        'tensors': tensors,
                        'split_dimension': split_dimension,
                        'tensor_slice': tensor_slice,
                        'data(bytes)': data_bytes,                        
                    }
                    
                    collective_ops.append(op_data)
            
            if collective_ops:
                df_node = pd.DataFrame(collective_ops)
                df_node['split_dimension'] = df_node['split_dimension'].astype('Int64')
                all_collective_ops.append(df_node)
        
        if not all_collective_ops:
            return pd.DataFrame()
        
        # Combine all dataframes
        df_combined = pd.concat(all_collective_ops, ignore_index=True)
        return df_combined

    # Private methods
    def _extract_collective_name(self, line):
        """Extract collective operation name from scheduling_name attribute."""    
        match = re.search(r'scheduling_name="([^"]+)"', line)
        return match.group(1) if match else None
    
    def _extract_replica_groups(self, line):
        """Extract replica groups or source-target pairs."""
        # Try replica_groups first
        pattern = r"replica_groups=(?P<replica_string>(?:\{(?:\{[0-9]+(?:,[0-9]+)*\}(?:,\{[0-9]+(?:,[0-9]+)*\})*)\}|\[[0-9]+(?:,[0-9]+)*\]<=\[[0-9]+(?:,[0-9]+)*\]))"
        match = re.search(pattern, line)
        if match:
            replica_groups_str = match.group('replica_string')
            return replica_groups_str, self._parse_replica_groups(replica_groups_str)
        
        # Try source_target_pairs for collective-permute
        source_target_pattern = r"source_target_pairs=(?P<source_target_string>\{(?:\{[0-9]+,[0-9]+\}(?:,\{[0-9]+,[0-9]+\})*)\})"
        match = re.search(source_target_pattern, line)
        if match:
            source_target_str = match.group('source_target_string')
            return source_target_str, self._parse_replica_groups(source_target_str)
        
        return None, None

    def _parse_replica_groups(self, replica_groups_str):
        """Parse replica groups from string format into list of device groups."""
        if not replica_groups_str or str(replica_groups_str).strip().lower() in ['nan', 'none', '']:
            return []            
        
        replica_groups_str = str(replica_groups_str).strip()
        
        # Handle explicit groups: {{0,1,2,3},{4,5,6,7},...}
        if replica_groups_str.startswith('{{'):
            inner_content = replica_groups_str[2:-2]
            group_strings = inner_content.split('},{')
            return [[int(x.strip()) for x in group_str.split(',')] for group_str in group_strings]
        
        # Handle Replica groups format following IotaTileAssignment: [dims]<=[total]
        elif '<=' in replica_groups_str:
            return self._parse_device_assignment(replica_groups_str)
        
        return []

    def _parse_device_assignment(self, replica_groups_str):
        """Parse device assignment from IotaTileAssignment format."""
        parts = replica_groups_str.split('<=')
        
        dims_match = re.search(r'\[([^\]]+)\]', parts[0])
        reshape_match = re.search(r'\[([^\]]+)\]', parts[1])
        
        if not dims_match or not reshape_match:
            return []
        
        dims = [int(x.strip()) for x in dims_match.group(1).split(',')]
        reshape_dims = [int(x.strip()) for x in reshape_match.group(1).split(',')]
        
        transpose_perm = None
        transpose_match = re.search(r'T\(([^)]+)\)', replica_groups_str)
        if transpose_match:
            transpose_perm = [int(x.strip()) for x in transpose_match.group(1).split(',')]
        
        total_elements = math.prod(dims)
        arr = np.arange(total_elements).reshape(reshape_dims)
        
        if transpose_perm:
            arr = arr.transpose(transpose_perm)
        
        arr = arr.reshape(dims)
        return [row.tolist() for row in arr]

    def _extract_tensor_specs(self, line):
        """Extract tensor specifications."""
        equals_match = re.search(r'=\s+(.+?)\s+(?:all-to-all|all-gather|all-reduce|reduce-scatter|collective-permute)\(', line)
        if equals_match:
            return equals_match.group(1).strip()
        
        equals_fallback = re.search(r'=\s+(.+?)(?:\s+[a-z-]+\(|,\s+channel_id|,\s+replica_groups|$)', line)
        if equals_fallback:
            return re.sub(r',?\s*$', '', equals_fallback.group(1).strip())
        
        return None

    def _extract_split_dimension(self, line):
        """Extract split dimension from dimensions attribute."""
        match = re.search(r"dimensions=\{([0-9,]+)\}", line)
        if not match:
            return None
        
        dimensions = match.group(1)
        return int(dimensions) if ',' not in dimensions else None

    def _extract_output_tensor_from_tuple(self, collective_name, tensor_spec):
        """Extract output tensor from tuple format based on collective operation type."""
        if not tensor_spec:
            return tensor_spec
        
        if collective_name and ('reduce-scatter' in collective_name or 'collective-permute' in collective_name):
            return tensor_spec
        
        if collective_name and ('all-gather-start' in collective_name or 'all-to-all' in collective_name):
            # Handle nested tuples: ((input), (output)) or ((input), output)
            if tensor_spec.startswith('((') and '), (' in tensor_spec:
                parts = tensor_spec.split('), (')
                if len(parts) >= 2:
                    return parts[1].rstrip(')')
            elif tensor_spec.startswith('((') and '), ' in tensor_spec:
                parts = tensor_spec.split('), ')
                if len(parts) >= 2:
                    return parts[1].rstrip(')')
            
            # Handle simple tuples: (input, output)
            elif tensor_spec.startswith('(') and ', ' in tensor_spec and tensor_spec.count('(') == 1:
                inner_content = tensor_spec[1:-1]
                parts = inner_content.split(', ')
                if len(parts) >= 2:
                    return parts[1]
        
        return tensor_spec

    def _calculate_tensor_slice(self, tensor_spec, split_dimension, replica_groups, collective_name, replica_group_size):
        """Calculate tensor slice information including dimensions and bytes per replica."""
        if tensor_spec is None or replica_groups is None:
            return None    

        output_tensor = self._extract_output_tensor_from_tuple(collective_name, tensor_spec)
        
        if replica_group_size == 0:
            return None   

        tensor_pattern = r'([a-zA-Z]+[0-9]*)\[([0-9,]+)\](?:\{[0-9,]+\})?'
        matches = re.findall(tensor_pattern, output_tensor)
        
        if not matches:
            return None
        
        def calculate_bytes(dtype, dimensions):
            bytes_per_element = self.dtype_to_bytes.get(dtype.lower(), 0)
            if bytes_per_element == 0 or not dimensions:
                return 0
            return reduce(mul, dimensions, 1) * bytes_per_element
        
        results = []
        for dtype_str, dimensions_str in matches:
            dimensions = [int(d.strip()) for d in dimensions_str.split(',') if d.strip()]
            
            if pd.isna(split_dimension):
                slice_info = {
                    'dtype': dtype_str,
                    'dimensions': tuple(dimensions),
                    'bytes': calculate_bytes(dtype_str, dimensions)
                }
            elif split_dimension >= len(dimensions) or dimensions[split_dimension] % replica_group_size != 0:
                continue
            else:
                sliced_dimensions = dimensions.copy()
                sliced_dimensions[split_dimension] = dimensions[split_dimension] // replica_group_size
                slice_info = {
                    'dtype': dtype_str,
                    'dimensions': tuple(sliced_dimensions),
                    'bytes': calculate_bytes(dtype_str, sliced_dimensions)
                }
            
            results.append(slice_info)
        
        return results

    def _calculate_data_bytes(self, tensor_slice):
        """Calculate total data bytes from tensor slice information."""
        if tensor_slice is None:
            return 0
        elif isinstance(tensor_slice, dict):
            return tensor_slice.get('bytes', 0)
        elif isinstance(tensor_slice, list):
            return sum(slice_info.get('bytes', 0) for slice_info in tensor_slice if isinstance(slice_info, dict))
        return 0

    def _calculate_replica_group_size(self, replica_groups):
        """Calculate the size of the first replica group."""
        return len(replica_groups[0]) if replica_groups and len(replica_groups) > 0 else 0