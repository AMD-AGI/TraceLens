import os
import json
import pandas as pd 

class NcclAnalyser:
    # in the init we create a dict from rank to trace data
    # also we have the trace data in dict format key as event['args']['External id'] and value as event
    def __init__(self, list_profile_filepaths, world_size):
        self.list_profile_filepaths = list_profile_filepaths
        self.world_size = world_size
        assert len(self.list_profile_filepaths) == self.world_size
        self.filter_event_fn = self.nccl_filter_event_fn
        self.metadata_fields = ['Collective name', 'In msg nelems', 'Out msg nelems',
                            'Group size', 'dtype', 'Process Group Name',
                            'Process Group Description']

        self.dtype2bytes = {
                    "Float": 4,
                    "Int": 4,
                    "Long": 8,
                    "BFloat16": 2,
                    "Bool": 1,
                    "Byte": 1,
                    "Double": 8,
                    "Half": 2,
                    "Short": 2,
                }

        # current support for allreduce, reducescatter, allgather
        self.collective2scaling_factor = {
            'allreduce': lambda n: 2 * (n - 1) / n,
            'reducescatter': lambda n: (n - 1) / n,
            'allgather': lambda n: (n - 1) / n,
            # 'broadcast': lambda n: 1,
            # 'reduce': lambda n: 1,
            # 'alltoall': lambda n: (n - 1) / n,
        }

        self.collective_cat2name = {
            'allreduce': ['allreduce', 'allreduce_coalesced'],
            'reducescatter': ['reducescatter', '_reduce_scatter_base', 'reduce_scatter_tensor_coalesced'],
            'allgather': ['allgather', 'all_gather', '_allgather_base', 'all_gather_into_tensor_coalesced'],
        }
        self.collective_name2cat = {name: cat for cat, names in self.collective_cat2name.items() for name in names}

    def assign_stream_index_id(self, nccl_events):
        """
        Iterates through a list of NCCL events, groups them by their stream,
        sorts them by time, and assigns a tuple (stream, order_id) to each event
        in the event['args'] dictionary under the key 'stream_index_id'.
        
        Args:
            nccl_events (list): List of NCCL event dictionaries.
        """
        # Group events by stream
        events_by_stream = {}
        for event in nccl_events:
            # Ensure that the necessary keys exist
            stream = event['args'].get('stream')
            if stream is None:
                raise ValueError("Event missing 'stream' in args")
            events_by_stream.setdefault(stream, []).append(event)
        
        # Process each group of events corresponding to a single stream
        for stream, events in events_by_stream.items():
            # Sort events by timestamp
            events.sort(key=lambda e: e['ts'])
            
            # Assign an order id to each event in this stream
            for order_id, event in enumerate(events):
                event['args']['stream_index_id'] = (stream, order_id)

    def nccl_filter_event_fn(self, event):
            
        if 'cat' not in event or 'args' not in event:
            return False
        if event['cat'] != 'kernel':
            return False
        if 'nccl' not in event['name']:
            return False
        #TODO raise warning if collective name not in the list
        #TODO think about how to handle collectives like broadcast,
        # that dont obey the sync property
        if event['args']['Collective name'] not in self.collective_name2cat:
            return False
        return True
    
    def load_trace_data(self):
        self.rank2trace_data = {}
        for rank in range(self.world_size):
            print(f'Processing rank {rank}')
            file_path = self.list_profile_filepaths[rank]
            with open(file_path, 'r') as f:
                data = json.load(f)
            dict_collective_id2event = {}
            nccl_events = []
            for event in data['traceEvents']:
                if self.nccl_filter_event_fn(event):
                    nccl_events.append(event)
            self.assign_stream_index_id(nccl_events)
            self.rank2trace_data[rank] = dict_collective_id2event
            for event in nccl_events:
                collective_id = event['args']['stream_index_id']
                dict_collective_id2event[collective_id] = event
            self.rank2trace_data[rank] = dict_collective_id2event

    def build_df_nccl(self):

        self.load_trace_data()
        list_of_nccl_events = []
        for collective_id in self.rank2trace_data[0].keys():
            # Find the minimum duration of the event across all ranks
            # this is important for getting communication time
            comm_latency = float('inf')
            for rank in range(self.world_size):
                comm_latency = min(comm_latency, self.rank2trace_data[rank][collective_id]['dur'])
            df_row = {'collective_id': collective_id, 'comm latency (µs)': comm_latency}

            # Get metadata fields from rank 0
            rank0_event = self.rank2trace_data[0][collective_id]
            for field in self.metadata_fields:
                df_row[field] = rank0_event['args'][field]

            df_row['In msg size (MB)'] = df_row['In msg nelems'] * self.dtype2bytes[df_row['dtype']] / (1024*1024)
            # algorithm bandwidth = size / time in GB/s
            df_row['algo bw (GB/s)'] = (df_row['In msg size (MB)']/1024) / (comm_latency/1e6)
            # bus bandwidth = algo bw * correction factor
            # df_row['bus bw (GB/s)'] = df_row['algo bw (GB/s)'] * self.collective2scaling_factor[df_row['Collective name']](df_row['Group size'])
            df_row['bus bw (GB/s)'] = df_row['algo bw (GB/s)'] * self.collective2scaling_factor[self.collective_name2cat[df_row['Collective name']]](df_row['Group size'])
            list_of_nccl_events.append(df_row)
        list_of_nccl_events.sort(key=lambda x: x['In msg size (MB)'], reverse=True)

        df_nccl = pd.DataFrame(list_of_nccl_events)
        df_nccl = df_nccl[['collective_id'] + self.metadata_fields + ['comm latency (µs)', 'In msg size (MB)', 'algo bw (GB/s)', 'bus bw (GB/s)']]
        return df_nccl
    
    @staticmethod
    def summarize_df_nccl(df_nccl, agg_metrics=['mean', 'std']):

        agg_logic = {
            # Metadata fields (take the first value in the group)
            'In msg nelems': 'first',
            'Out msg nelems': 'first',
            'Group size': 'first',
            'Process Group Name': 'first',
            'Process Group Description': 'first',
            'comm latency (µs)': agg_metrics + ['size', lambda x: x.sum() / 1000],  # Size and sum (convert to ms)
            'algo bw (GB/s)': agg_metrics,
            'bus bw (GB/s)': agg_metrics,
        }

        agg_result = df_nccl.groupby(['Collective name', 'In msg size (MB)', 'dtype']).agg(agg_logic)
        agg_result.columns = [
            f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
            for col in agg_result.columns
        ]
        column_renames = {
            'comm latency (µs)_<lambda_0>': 'Total latency (ms)',
            'comm latency (µs)_size': 'count',
        }
        agg_result.rename(columns=column_renames, inplace=True)
        summary_df = agg_result.reset_index()
        summary_df = summary_df.sort_values(by='Total latency (ms)', ascending=False)

        return summary_df