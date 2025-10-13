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
from TraceLens.JAXRcclAnayser.util import rccl_complete_analysis as rca

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
        # from any of the participating nodes. That is why 
        buffer_assignment_file = rca.detect_xla_file(self.traces_dir)
        messages, replica_groups = rca.read_xla_dump(buffer_assignment_file)
        self.collectives_size_map = rca.combine_xla_dump(messages,replica_groups)




    # ------------------------------------------------------------------------
    # Step 1: Build a long table where each row is a collective event on a rank
    # ------------------------------------------------------------------------
    def build_df_long(self):
        """Constructs a long table where each row is a collective event on a rank."""
        rows = []
        for rank in self.rank2trace_data:
            for _ , event in self.rank2trace_data[rank].items():
                row = {}                
                row["rank"] = rank
                if event:
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

