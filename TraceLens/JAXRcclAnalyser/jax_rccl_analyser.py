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


import os
import json
import pandas as pd
import warnings
import gzip
from TraceLens.Trace2Tree.trace_to_tree import JaxTraceToTree
from TraceLens.util import DataLoader,TraceEventUtils
from util import rccl_complete_analysis as rca

def list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(list_to_tuple(item) for item in obj)
    return obj

class RcclAnalyser:
    def __init__(self, traces_dir, rank_to_pb_file_mapping, world_size):
        self.traces_dir = traces_dir
        self.rank_to_pb_file_mapping = rank_to_pb_file_mapping
        self.world_size = world_size

        # Byte sizes per dtype
        self.dtype2bytes = {
            "Float": 4, "Int": 4, "Long": 8, "BFloat16": 2, "Bool": 1,
            "Byte": 1, "Double": 8, "Half": 2, "Short": 2
        }

        # Scaling factors for recognized collectives
        self.collective2scaling_factor = {
            'allreduce':     lambda n: 2 * (n - 1) / n,
            'reducescatter': lambda n: (n - 1) / n,
            'allgather':     lambda n: (n - 1) / n,
            'alltoall':      lambda n: (n - 1) /  n,
        }

        # Known names => "category"
        self.collective_type2name = {
            'allreduce':     ['allreduce', 'allreduce_coalesced'],
            'reducescatter': ['reducescatter', '_reduce_scatter_base', 'reduce_scatter_tensor_coalesced'],
            'allgather':     ['allgather', 'all_gather', '_allgather_base', 'all_gather_into_tensor_coalesced', 'allgather_into_tensor_coalesced'],
            'alltoall':      ['all_to_all'],
            'alltoallv':     ['all_to_allv'],
        }

        self.collective_name2type = {
            name: cat for cat, names in self.collective_type2name.items()
            for name in names
        }
        self.implicit_sync_cat = {'allreduce', 'reducescatter', 'allgather', 'alltoall'}

        # Internal storage
        self.collectives_size_map = {} # Stores collectives hloop as key and sizes, data-type, and replica-groups as values
        self.parse_xla_dumps()
        self.rank2trace_data = {}  # Stores per-rank data
        self.load_trace_data()


    def _rccl_event_filter(self,event):     

        is_coll_comm_event = event.get("gpu_kernel_op_cat","") == "Communication rccl/nccl"   
        # check_hlo_module = event.get("args") is not None and event.get("args").get("hlo_module") == 'jit_train_step'        
        
        # return is_coll_comm_event and 
        
        return is_coll_comm_event

    def load_trace_data(self):
        """Uses JaxTraceToTree to extracts relevant events."""
        print(f"Make sure the rank to file mapping is correct as incorrect mapping may lead to unexpected results.")
        print('Also note that we need all ranks for the analysis. We will add a fallback soon for lesser features for single rank or partial data.')
        self.rank2trace_data.clear()
        for rank, protobuf_filepath in self.rank_to_pb_file_mapping.items():
            print(f"Loading rank {rank} from {protobuf_filepath}")
            pb_data = DataLoader.load_data(protobuf_filepath, save_preprocessed=False)
            trace_events = pb_data["traceEvents"]            
            linking_key = 'correlation_id'
            categorizer =  TraceEventUtils.prepare_event_categorizer(trace_events)
            non_metadata_events = TraceEventUtils.non_metadata_events(trace_events)
            tree = JaxTraceToTree(non_metadata_events, event_to_category=categorizer, linking_key=linking_key)
            metadata = TraceEventUtils.get_metadata(trace_events)
            tree.build_tree(metadata=metadata, pb_file_name=protobuf_filepath)
            rccl_events = [event for event in tree.events if self._rccl_event_filter(event)]
            
            # Build a dictionary with event data
            rank_dict = {idx: evt for idx, evt in enumerate(rccl_events)}
            self.rank2trace_data[rank] = rank_dict

    def parse_xla_dumps(self):

        # parse the xla dumps file to get a dictionary with key as rccl/nccl hlo op and value
        # as a list containing replica groups, combined size of tensors involved in collective (bytes)
        # and the datatype of tensor

        # Out of all the xla dumps generated, we are interested in the largest size buffer assignmnent file. It could be
        # from any of the participating nodes.  
        buffer_assignment_file = rca.detect_xla_file(self.traces_dir)
        messages, replica_groups = rca.read_xla_dump(buffer_assignment_file)
        self.collectives_size_map = rca.combine_xla_dump(messages,replica_groups)

    # ------------------------------------------------------------------------
    # Step 1: Build a long table where each row is a collective event on a rank
    # ------------------------------------------------------------------------
    def build_df_long(self):
        """Constructs a long table where each row is a collective event on a gpu rank."""
        rows = []
        for rank in self.rank2trace_data:
            for _ , event in self.rank2trace_data[rank].items():                
                if event:
                    row = {}                
                    row["rank"] = rank # Node 
                    row["gpu_rank"] = int(rank) * 8 + int(event["pid"]) - 1 # GPU rank
                    row["pid"] = event["pid"]
                    row["ts"] = event["ts"]
                    row["dur"] = event["dur"]
                    if event.get("args") is not None:
                        args = event.get("args")
                        row["collective_name"] = args.get("hlo_op")
                        row["hlo_module"] = args.get("hlo_module")
                        row["correlation_id"] = args.get("correlation_id")
                    if event.get("process") is not None:
                        row["process_name"] = event.get("process").get("process_name")

                rows.append(row)

        df_long = pd.DataFrame(rows)
        df_long = df_long.reset_index(drop=True)

        # Assign an index within each process group and rank
        df_long['collective_name'] = df_long['collective_name'].fillna('unknown')
        df_long['index_in_group'] = df_long.groupby(['collective_name', 'pid', 'rank'])['ts'].rank(method='first').astype(int) - 1

        # Create a composite collective ID (process group + index)
        df_long['collective_id'] = df_long['collective_name'] + '_' + df_long['index_in_group'].astype(str)


        self.df_per_rank_coll = df_long
        return df_long
    
    # ------------------------------------------------------------------------
    # Step 2: Build a wide table for implicit sync class
    # where each row is a collective operation
    # ------------------------------------------------------------------------
    def build_df_nccl_implicit_sync_cat(self, detailed=False, strict_metadata_check=True):
        """
        Builds a single DF with one row *per collective ID*, including per-rank ts/dur + metadata.
        Ensures metadata consistency across ranks.
        """
        if not hasattr(self, 'df_per_rank_coll'):
            self.build_df_long()

        df = self.df_per_rank_coll

        metadata_fields = ["collective_name","hlo_module",'collective_id',"index_in_group"]
        collective_ids = df['collective_id'].unique()
        rows = []

        for cid in collective_ids:
            rank_events = df[df['collective_id'] == cid]
            rank_events = rank_events.set_index('gpu_rank')

            # # Skip if the collective type is not in the implicit sync category
            # collective_name = rank_events.iloc[0]['Collective name']
            # if self.collective_name2type.get(collective_name) not in self.implicit_sync_cat:
            #     continue

            # **Metadata Consistency Check**
            ref_metadata = {field: rank_events.iloc[0][field] for field in metadata_fields}
            for field in metadata_fields:
                unique_values = rank_events[field].unique()
                if len(unique_values) > 1:
                    if strict_metadata_check:
                        raise ValueError(f"Metadata mismatch in '{field}' for collective {cid}: {unique_values}")
                    warnings.warn(f"Metadata mismatch in '{field}' for collective {cid}: {unique_values}")

            row = {'collective_id': cid, **ref_metadata}

            # Compute per-rank timestamps and durations
            for r in rank_events.index:
                row[f'rank_{r}_ts'] = rank_events.loc[r, 'ts']
                row[f'rank_{r}_dur'] = rank_events.loc[r, 'dur']

            # Compute communication latency
            latest_start = max(row.get(f'rank_{r}_ts', 0) for r in rank_events.index)
            earliest_end = min(row.get(f'rank_{r}_ts', 0) + row.get(f'rank_{r}_dur', 0) for r in rank_events.index)
            row['comm_latency'] = min(row[f'rank_{r}_dur'] for r in rank_events.index)

            # Compute per-rank wait time
            for r in rank_events.index:
                row[f'rank_{r}_wait_time'] = latest_start - row.get(f'rank_{r}_ts', 0)

            # Compute max wait time and rank
            max_wait, max_wait_rank = max((row[f'rank_{r}_wait_time'], r) for r in rank_events.index)
            row['skew in start time'] = max_wait
            row['earliest arrival rank'] = max_wait_rank
            row['avg_wait_time'] = sum(row[f'rank_{r}_wait_time'] for r in rank_events.index) / len(rank_events.index)

            # Compute end time spread
            latest_end = max(row.get(f'rank_{r}_ts', 0) + row.get(f'rank_{r}_dur', 0) for r in rank_events.index)
            row['skew in end time'] = latest_end - earliest_end

            # # Compute algorithmic and bus bandwidth
            # c_type = self.collective_name2type.get(row['Collective name'])
            # row['Full msg size (MB)'] = row['Out msg size (MB)'] if c_type == 'allgather' else row['In msg size (MB)']
            # row['algo bw (GB/s)'] = (row['Full msg size (MB)']/1024) / (row['comm_latency'] / 1e6)
            # scaling_factor = self.collective2scaling_factor[c_type](row['Group size'])
            # row['bus bw (GB/s)'] = row['algo bw (GB/s)'] * scaling_factor

            rows.append(row)

        df = pd.DataFrame(rows).reset_index(drop=True)

        # Separate per-rank columns
        per_rank_cols = [col for col in df.columns if col.startswith('rank_')]
        # Define explicit order for general (non-rank) columns
        general_cols = [
            # Collective Identifier & Metadata
            "collective_name","hlo_module","collective_id","index_in_group",

            # High-Level Performance Metrics
            "comm_latency", "skew in start time", "earliest arrival rank",
            "avg_wait_time", "skew in end time"
        ]

        # Reorder columns: General metadata + performance metrics + per-rank details
        ordered_cols = general_cols + per_rank_cols
        df = df[ordered_cols]

        self.df_implicit_sync_cat_detailed = df
        self.df_implicit_sync_cat = df.drop(columns=per_rank_cols)

        return self.df_implicit_sync_cat if not detailed else self.df_implicit_sync_cat_detailed




    



