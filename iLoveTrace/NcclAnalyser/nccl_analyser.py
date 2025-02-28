import os
import json
import pandas as pd

class NcclAnalyser:
    def __init__(self, list_profile_filepaths, world_size):
        self.list_profile_filepaths = list_profile_filepaths
        self.world_size = world_size
        assert len(self.list_profile_filepaths) == self.world_size

        # Basic fields that must match across ranks
        self.metadata_fields = [
            'Collective name', 
            'In msg nelems', 'Out msg nelems',
            'Group size', 'dtype', 'Process Group Name',
            'Process Group Description'
        ]

        # Byte sizes per dtype
        self.dtype2bytes = {
            "Float":    4,
            "Int":      4,
            "Long":     8,
            "BFloat16": 2,
            "Bool":     1,
            "Byte":     1,
            "Double":   8,
            "Half":     2,
            "Short":    2,
        }

        # Scaling factors for recognized collectives
        self.collective2scaling_factor = {
            'allreduce':     lambda n: 2 * (n - 1) / n,
            'reducescatter': lambda n: (n - 1) / n,
            'allgather':     lambda n: (n - 1) / n,
        }

        # Known names => "category"
        self.collective_type2name = {
            'allreduce':     ['allreduce', 'allreduce_coalesced'],
            'reducescatter': ['reducescatter', '_reduce_scatter_base', 'reduce_scatter_tensor_coalesced'],
            'allgather':     ['allgather', 'all_gather', '_allgather_base', 'all_gather_into_tensor_coalesced'],
            'alltoallv':      ['alltoallv'],
        }
        
        self.collective_name2type = {
            name: cat for cat, names in self.collective_type2name.items()
            for name in names
        }
        self.implicit_sync_cat = {'allreduce', 'reducescatter', 'allgather'}
        # Filter function: keep only kernel events with "nccl" in the name
        self.filter_event_fn = self._nccl_filter_event_fn

        # Internal storage
        self.rank2trace_data = {}  # rank => dict of (stream_index_id => event)
        self.load_trace_data()

    # ------------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------------
    def _nccl_filter_event_fn(self, event):
        """Return True if this is a kernel event with 'nccl' in the name."""
        is_nccl_kernel = event.get('cat') == 'kernel' and 'nccl' in event.get('name', '').lower()
        is_linked = event.get('args', {}).get('External id') is not None
        return is_nccl_kernel and is_linked
    
    def _assign_stream_index_id(self, nccl_events):
        """
        Sort by timestamp within each 'stream' and assign a unique (stream, order_index).
        This becomes our 'collective_id' for that rank's events.
        """
        events_by_stream = {}
        for e in nccl_events:
            stream = e['args']['stream']
            if stream is None:
                raise ValueError("Event missing 'stream' in args")
            events_by_stream.setdefault(stream, []).append(e)

        for stream, evts in events_by_stream.items():
            evts.sort(key=lambda x: x['ts'])
            for i, e in enumerate(evts):
                e['args']['stream_index_id'] = (stream, i)

    # ------------------------------------------------------------------------
    # Step 1: Build a "raw" DF from all ranks
    # ------------------------------------------------------------------------
    def load_trace_data(self):
        """Load JSON files, filter relevant events, assign IDs, store in rank2trace_data."""
        self.rank2trace_data.clear()
        for rank in range(self.world_size):
            print(f"Loading rank {rank} from {self.list_profile_filepaths[rank]}")
            with open(self.list_profile_filepaths[rank], 'r') as f:
                raw_data = json.load(f)

            # Filter
            nccl_events = [e for e in raw_data['traceEvents'] if self.filter_event_fn(e)]
            # Assign unique ID
            self._assign_stream_index_id(nccl_events)

            # Build dict: (stream_index_id) => event
            rank_dict = {}
            for evt in nccl_events:
                cid = evt['args']['stream_index_id']
                rank_dict[cid] = evt

            self.rank2trace_data[rank] = rank_dict

    def build_df_nccl_implicit_sync_cat(self, detailed=False):
        """
        Builds a single DF with one row *per collective ID from rank=0*, 
        including per-rank ts/dur columns + metadata. 
        
        Returns a DF with columns:
            [
              'collective_id',
              # metadata_fields (e.g. 'Collective name', 'In msg nelems', etc.),
              'rank_0_ts', 'rank_0_dur',
              'rank_1_ts', 'rank_1_dur',
              ...
              'start time (µs)',
            ]
        """

        # Our reference set: all IDs from rank=0
        collective_ids = list(self.rank2trace_data[0].keys())

        rows = []
        for cid in collective_ids:
            # Gather that ID from all ranks
            rank_events = []
            for r in range(self.world_size):
                evt = self.rank2trace_data[r][cid]
                if evt is None:
                    raise ValueError(f"Missing collective ID {cid} in rank {r}")
                rank_events.append(evt)

            rank0 = rank_events[0]
            # skip if not in implicit_sync_cat
            c_name = rank0['args']['Collective name']
            c_type = self.collective_name2type.get(c_name)
            if c_type not in self.implicit_sync_cat:
                continue

            # Check metadata consistency, rely on rank0 as canonical
            for field in self.metadata_fields:
                val0 = rank0['args'][field]
                for r in range(1, self.world_size):
                    val_r = rank_events[r]['args'][field]
                    if val_r != val0:
                        raise ValueError(
                            f"Metadata mismatch for '{field}'.\n"
                            f"Collective ID: {cid}\n"
                            f"Rank0 => {val0},  Rank{r} => {val_r}"
                        )

            # Build row
            row = {'collective_id': cid}
            row['stream'] = rank0['args']['stream']
            # metadata
            for field in self.metadata_fields:
                row[field] = rank0['args'][field]
            # msg size in MB
            in_nelems = row['In msg nelems']
            dtype = row['dtype']
            if in_nelems and dtype in self.dtype2bytes:
                in_bytes = in_nelems * self.dtype2bytes[dtype]
                in_mb = in_bytes / (1024*1024)
                row['In msg size (MB)'] = in_mb
            else:
                raise ValueError(f"Missing or invalid dtype for {cid}")

            # Per-rank columns
            for r, evt in enumerate(rank_events):
                ts = evt['ts']
                dur = evt['dur']
                row[f'rank_{r}_ts'] = ts
                row[f'rank_{r}_dur'] = dur

            # communication latency is latest start to earliest end
            latest_start = max(row[f'rank_{r}_ts'] for r in range(self.world_size))
            earliest_end = min(row[f'rank_{r}_ts'] + row[f'rank_{r}_dur'] for r in range(self.world_size))
            latest_end = max(row[f'rank_{r}_ts'] + row[f'rank_{r}_dur'] for r in range(self.world_size))
            row['comm latency'] = earliest_end - latest_start

            # wait time for each rank is the time from a ranks start to the latest start
            for r in range(self.world_size):
                row[f'rank_{r}_wait_time'] = latest_start - row[f'rank_{r}_ts']
            
            # now we add max wait time, max wait rank and avg wait time
            max_wait, max_wait_rank = max((row[f'rank_{r}_wait_time'], r) for r in range(self.world_size))
            row['avg wait time'] = sum(row[f'rank_{r}_wait_time'] for r in range(self.world_size)) / self.world_size
            row['max wait time'] = max_wait
            row['max wait rank'] = max_wait_rank

            # spread of end time 
            row['spread of end time'] = latest_end - earliest_end

            # algo bw and bus bw are computed based on in_msg_size and comm latency
            row['algo bw (GB/s)'] = (row['In msg size (MB)']/1024) / (row['comm latency'] / 1e6)
            row['bus bw (GB/s)'] = row['algo bw (GB/s)'] * self.collective2scaling_factor[c_type](row['Group size'])

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.reset_index(drop=True)
        self.df_implicit_sync_cat_detailed = df
        self.df_implicit_sync_cat = df.drop(columns=[c for c in df.columns if c.startswith('rank_')])

        if not detailed:
            return self.df_implicit_sync_cat
        else:
            return self.df_implicit_sync_cat_detailed

    def build_df_summary_nccl_implicit_sync(self, agg_metrics=['mean', 'std']):
        """
        Builds a summary DF with one row per collective name, dtype, and msg size.
        Aggregates across all collectives and ranks.
        """
        if not hasattr(self, 'df_implicit_sync_cat'):
            self.df_implicit_sync_cat = self.build_df_nccl_implicit_sync_cat()
        
        df = self.df_implicit_sync_cat
        # we group by collective name, dtype and in msg size
        # we agg the cols comm latency, algo bw, bus bw, max wait time
        # we also count the number of collectives in the group
        agg_logic = {
            'Out msg nelems': 'first',
            'In msg size (MB)': 'first',
            'comm latency': agg_metrics + ['size', lambda x: x.sum() / 1000],  # Size and sum (convert to ms)
            'max wait time': agg_metrics,
            'avg wait time': agg_metrics,
            'spread of end time': agg_metrics,
            'algo bw (GB/s)': agg_metrics,
            'bus bw (GB/s)': agg_metrics,
        }
        agg_result = df.groupby(['Collective name', 'dtype', 'In msg nelems']).agg(agg_logic)
        # flatten the column names
        agg_result.columns = [
            f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
            for col in agg_result.columns
        ]
        column_renames = {
            'comm latency_<lambda_0>': 'Total latency (ms)',
            'comm latency_size': 'count',
            'Out msg nelems_first': 'Out msg nelems',
            'In msg size (MB)_first': 'In msg size (MB)',
        }
        agg_result.rename(columns=column_renames, inplace=True)
        summary_df = agg_result.reset_index()
        summary_df = summary_df.sort_values(by='Total latency (ms)', ascending=False)
        # Dynamically build the column ordering
        # Grouping columns remain the same
        group_cols = ['Collective name', 'dtype', 'In msg nelems', 'Out msg nelems', 'In msg size (MB)']
        # Define the metric groups (for which we computed agg metrics)
        metric_groups = ['algo bw (GB/s)', 'bus bw (GB/s)','comm latency', 'max wait time', 'avg wait time', 'spread of end time',]
        columns_order = group_cols.copy()
        # For the other groups, add their corresponding aggregated columns
        for group in metric_groups:
            for agg in agg_metrics:
                columns_order.append(f"{group}_{agg}")
        # Finally, append the special renamed columns from the comm latency group
        columns_order.extend(['count', 'Total latency (ms)'])
        summary_df = summary_df[columns_order]
        return summary_df
