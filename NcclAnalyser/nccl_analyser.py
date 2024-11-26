import os
import json
import csv
# i did not want any dependencies but here we are :p
import pandas as pd 

class NcclAnalyser:
    # in the init we create a dict from rank to trace data
    # also we have the trace data in dict format key as event['args']['External id'] and value as event
    def __init__(self, profiles_root_dir, output_db_path, world_size):
        self.profiles_root_dir = profiles_root_dir
        self.output_db_path = output_db_path
        self.world_size = world_size
        self.filter_event_fn = self.nccl_filter_event_fn
        self.fn_rank2file = self.default_fn_rank2file
        self.debug_cnt = 10
        self.metadata_fields = ['Collective name', 'In msg nelems', 'Out msg nelems',
                            'Group size', 'dtype', 'Process Group Name',
                            'Process Group Description']

        self.dtype2bytes = {'Float': 4, 'Int': 4, 'Long': 8}

    def nccl_filter_event_fn(self, event):
            
        if 'cat' not in event or 'args' not in event:
            return False
        if event['cat'] != 'kernel':
            return False
        if 'nccl' not in event['name']:
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

    def get_bus_bw(self, db_row):
        
        # Summary
        # To obtain a bus bandwidth which should be independent of the number of ranks n, we apply a correction factor to the algorithm bandwidth :

        # AllReduce : 2*(n-1)/n
        # ReduceScatter : (n-1)/n
        # AllGather : (n-1)/n
        # Broadcast : 1
        # Reduce : 1
        # AlltoAll: (n-1)/n    
        n = db_row['Group size']
        dict_collective2bus_bw = {'allreduce': 2*(n-1)/n, 'reducescatter': (n-1)/n, 'allgather': (n-1)/n, 'broadcast': 1, 'reduce': 1, 'alltoall': (n-1)/n}
        return db_row['algo bw (GB/s)'] * dict_collective2bus_bw[db_row['Collective name']]

    def build_db(self):

        self.load_trace_data()
        print('Building db')

        list_of_nccl_events = []
        for external_id in self.rank2trace_data[0].keys():
            # Find the minimum duration of the event across all ranks
            # this is important for getting communication time
            comm_latency = float('inf')
            for rank in range(self.world_size):
                comm_latency = min(comm_latency, self.rank2trace_data[rank][external_id]['dur'])
            db_row = {'external_id': external_id, 'comm latency (µs)': comm_latency}

            # Get metadata fields from rank 0
            rank0_event = self.rank2trace_data[0][external_id]
            for field in self.metadata_fields:
                db_row[field] = rank0_event['args'][field]

            db_row['In msg size (MB)'] = db_row['In msg nelems'] * self.dtype2bytes[db_row['dtype']] / (1024*1024)
            db_row['In msg size (MB)'] = round(db_row['In msg size (MB)'], 3)

            # algorithm bandwidth = size / time in GB/s
            db_row['algo bw (GB/s)'] = (db_row['In msg size (MB)']/1024) / (comm_latency/1e6)
            db_row['algo bw (GB/s)'] = round(db_row['algo bw (GB/s)'], 3)

            # bus bandwidth = algo bw * correction factor
            db_row['bus bw (GB/s)'] = self.get_bus_bw(db_row)
            db_row['bus bw (GB/s)'] = round(db_row['bus bw (GB/s)'], 3)
            
            list_of_nccl_events.append(db_row)
        list_of_nccl_events.sort(key=lambda x: x['In msg size (MB)'], reverse=True)

        # create a pandas dataframe
        self.df_nccl_db = pd.DataFrame(list_of_nccl_events)
        self.df_nccl_db = self.df_nccl_db[['external_id'] + self.metadata_fields + ['comm latency (µs)', 'In msg size (MB)', 'algo bw (GB/s)', 'bus bw (GB/s)']]
        
    def build_and_save_db(self):
        self.build_db()
        # Write to csv
        filepath = self.output_db_path
        # check extension
        if not filepath.endswith('.csv'):
            raise ValueError('Output file must be a csv file')
        self.df_nccl_db.to_csv(filepath, index=False)
    
    def build_summary_db(self):

        if not hasattr(self, 'df_nccl_db'):
            self.build_db()

        # Group by 'Collective name' and 'In msg size (MB)'
        # Aggregate metadata and metrics
        self.df_summary_db = self.df_nccl_db.groupby(['Collective name', 'In msg size (MB)']).agg(
            # Metadata fields (first element in the group)
            In_msg_nelems=('In msg nelems', 'first'),
            Out_msg_nelems=('Out msg nelems', 'first'),
            Group_size=('Group size', 'first'),
            dtype=('dtype', 'first'),
            Process_Group_Name=('Process Group Name', 'first'),
            Process_Group_Description=('Process Group Description', 'first'),
            # Count and total comm latency
            count=('comm latency (µs)', 'size'),
            total_latency_ms=('comm latency (µs)', lambda x: x.sum() / 1000),  # Convert to ms
            # Metrics (aggregated statistics)
            min_dur_mean=('comm latency (µs)', 'mean'),
            min_dur_max=('comm latency (µs)', 'max'),
            min_dur_min=('comm latency (µs)', 'min'),
            min_dur_std=('comm latency (µs)', 'std'),
            algo_bw_mean=('algo bw (GB/s)', 'mean'),
            algo_bw_max=('algo bw (GB/s)', 'max'),
            algo_bw_min=('algo bw (GB/s)', 'min'),
            algo_bw_std=('algo bw (GB/s)', 'std'),
            bus_bw_mean=('bus bw (GB/s)', 'mean'),
            bus_bw_max=('bus bw (GB/s)', 'max'),
            bus_bw_min=('bus bw (GB/s)', 'min'),
            bus_bw_std=('bus bw (GB/s)', 'std'),
        ).reset_index()

        # Rename total latency column to include units
        self.df_summary_db.rename(columns={'total_latency_ms': 'Total latency (ms)'}, inplace=True)
        # Round all numerical fields to 3 decimal places
        self.df_summary_db = self.df_summary_db.round(3)
        
    def build_and_save_summary_db(self, filepath):
        if not hasattr(self, 'df_summary_db'):
            self.build_summary_db()
        # Write to csv
        # check extension
        if not filepath.endswith('.csv'):
            raise ValueError('Output file must be a csv file')
        self.df_summary_db.to_csv(filepath, index=False)