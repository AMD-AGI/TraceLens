import os
import json
import pandas as pd 

class NcclAnalyser:
    # in the init we create a dict from rank to trace data
    # also we have the trace data in dict format key as event['args']['External id'] and value as event
    def __init__(self, profiles_root_dir, world_size):
        self.profiles_root_dir = profiles_root_dir
        self.world_size = world_size
        self.filter_event_fn = self.nccl_filter_event_fn
        self.fn_rank2file = self.default_fn_rank2file
        self.metadata_fields = ['Collective name', 'In msg nelems', 'Out msg nelems',
                            'Group size', 'dtype', 'Process Group Name',
                            'Process Group Description']

        self.dtype2bytes = {'Float': 4, 'Int': 4, 'Long': 8}

        # Initialize the scaling factor for bus bandwidth
        self.collective2scaling_factor = {
            'allreduce': lambda n: 2 * (n - 1) / n,
            'reducescatter': lambda n: (n - 1) / n,
            'allgather': lambda n: (n - 1) / n,
            'broadcast': lambda n: 1,
            'reduce': lambda n: 1,
            'alltoall': lambda n: (n - 1) / n,
        }

    def nccl_filter_event_fn(self, event):
            
        if 'cat' not in event or 'args' not in event:
            return False
        if event['cat'] != 'kernel':
            return False
        if 'nccl' not in event['name']:
            return False
        # current support for allreduce, reducescatter, allgather
        if event['args']['Collective name'] not in ['allreduce', 'reducescatter', 'allgather']:
            return False

        return True

    def default_fn_rank2file(self, rank):
        """Default function to map rank to file path."""
        return os.path.join(self.profiles_root_dir, str(rank), 'pytorch_profile.json')

    def set_rank2file_fn(self, fn_rank2file):
        """Set a custom rank-to-file mapping function."""
        self.fn_rank2file = fn_rank2file
    
    def load_trace_data(self):
        self.rank2trace_data = {}
        for rank in range(self.world_size):
            print(f'Processing rank {rank}')
            file_path = self.fn_rank2file(rank)
            with open(file_path, 'r') as f:
                data = json.load(f)
            dict_external_id2event = {}
            for event in data['traceEvents']:
                if not self.filter_event_fn(event):
                    continue
                dict_external_id2event[event['args']['External id']] = event
            self.rank2trace_data[rank] = dict_external_id2event

    def build_df_nccl(self):

        self.load_trace_data()
        print('Building df_nccl')

        list_of_nccl_events = []
        for external_id in self.rank2trace_data[0].keys():
            # Find the minimum duration of the event across all ranks
            # this is important for getting communication time
            comm_latency = float('inf')
            for rank in range(self.world_size):
                comm_latency = min(comm_latency, self.rank2trace_data[rank][external_id]['dur'])
            df_row = {'external_id': external_id, 'comm latency (µs)': comm_latency}

            # Get metadata fields from rank 0
            rank0_event = self.rank2trace_data[0][external_id]
            for field in self.metadata_fields:
                df_row[field] = rank0_event['args'][field]

            df_row['In msg size (MB)'] = df_row['In msg nelems'] * self.dtype2bytes[df_row['dtype']] / (1024*1024)
            # algorithm bandwidth = size / time in GB/s
            df_row['algo bw (GB/s)'] = (df_row['In msg size (MB)']/1024) / (comm_latency/1e6)
            # bus bandwidth = algo bw * correction factor
            df_row['bus bw (GB/s)'] = df_row['algo bw (GB/s)'] * self.collective2scaling_factor[df_row['Collective name']](df_row['Group size'])
            
            list_of_nccl_events.append(df_row)
        list_of_nccl_events.sort(key=lambda x: x['In msg size (MB)'], reverse=True)

        df_nccl = pd.DataFrame(list_of_nccl_events)
        df_nccl = df_nccl[['external_id'] + self.metadata_fields + ['comm latency (µs)', 'In msg size (MB)', 'algo bw (GB/s)', 'bus bw (GB/s)']]
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