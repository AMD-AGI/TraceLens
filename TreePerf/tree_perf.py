import pandas as pd
from torch_op_mapping import *

class TreePerfAnalyzer:
    def __init__(self, trace_to_tree):
        self.tree = trace_to_tree

    def agg_kernels_in_subtree(self, event_UID, filter_func=None, verbose=False):
        events_by_uid = self.tree.events_by_uid
        event = events_by_uid[event_UID]
        if filter_func is None:
            filter_func = lambda x: True
        if event.get('cat') == 'kernel':
            if not filter_func(event):
                return 0, []
            if verbose:
                print(f"Found kernel event, duration: {event['dur']}, name: {event['name']}")
            return event['dur'], [event_UID]
        total_dur = 0
        list_kernels = []
        for child_UID in event.get('children', []):
            child_total_dur, child_list_kernels = self.agg_kernels_in_subtree(child_UID, filter_func, verbose)
            total_dur += child_total_dur
            list_kernels.extend(child_list_kernels)
        return total_dur, list_kernels

    def loop_and_aggregate_kernels(self, event_UIDs, filter_func=None, verbose=False):
        total_kernel_time = 0
        list_kernels = []
        for event_UID in event_UIDs:
            this_total_kernel_time, this_list_kernels = self.agg_kernels_in_subtree(event_UID, filter_func, verbose=False)
            total_kernel_time += this_total_kernel_time
            list_kernels.extend(this_list_kernels)
        return total_kernel_time, list_kernels

    @staticmethod
    def non_data_mov_filter(event):
        DATA_MOVEMENT_PATTERNS = ['at::native::direct_copy_kernel_cuda', 'transpose_']
        return not any(pattern in event['name'] for pattern in DATA_MOVEMENT_PATTERNS)

    def compute_perf_metrics(self, event, bwd=False, bytes_per_element=2, non_data_mov=False):
        # Select the appropriate dictionary for FLOPS and memory functions
        perf_model_class = op_to_perf_model_class_map[event['name']]
        perf_model = perf_model_class(event)

        gflops = (perf_model.flops() if not bwd else perf_model.flops_bwd())/ 1e9  
        # Handle kernel aggregation
        if bwd:
            if not event.get('bwd_events'):
                self.tree.link_bwd_events(event['UID']) 
            cpu_op_uids = event['bwd_events']
        else:
            cpu_op_uids = [event['UID']]
        total_kernel_time, _ = self.loop_and_aggregate_kernels(cpu_op_uids)
        total_non_data_mov_time, _ = self.loop_and_aggregate_kernels(cpu_op_uids, filter_func=self.non_data_mov_filter)

        tflops_per_s = (gflops / 1e3) / (total_kernel_time / 1e6)
        non_data_mov_tflops_per_s = (gflops / 1e3) / (total_non_data_mov_time / 1e6)
        bytes_moved = perf_model.bytes(bytes_per_element) if not bwd else perf_model.bytes_bwd(bytes_per_element)

        # Return metrics
        dict_metrics = {
            'GFLOPS': gflops,
            'Kernel Time (µs)': total_kernel_time,
            'TFLOPS/s': tflops_per_s,
        }
        if non_data_mov:
            dict_metrics['Non-Data-Mov Kernel Time (µs)'] = total_non_data_mov_time
            dict_metrics['Non-Data-Mov TFLOPS/s'] = non_data_mov_tflops_per_s
        if bytes_moved is not None:
            dict_metrics['FLOPS/Byte'] = (gflops * 1e9) / bytes_moved
        
        for key, value in perf_model.param_details.items():
            dict_metrics[f"param: {key}"] = value

        return dict_metrics

    def compute_fwd_perf_metrics(self, event, non_data_mov=False):
        return self.compute_perf_metrics(event, bwd=False, non_data_mov=non_data_mov)
    def compute_bwd_perf_metrics(self, event, non_data_mov=False):
        return self.compute_perf_metrics(event, bwd=True, non_data_mov=non_data_mov)
    
    def build_df_perf_metrics(self, event_UIDs, bwd, non_data_mov=False):
        rows = []
        for event_uid in event_UIDs:
            event = self.tree.events_by_uid[event_uid]
            metrics_event = {'cat': event['cat'], 'name': event['name'],
                        'pid': event['pid'], 'tid': event['tid'],
                        'external_id': event['args']['External id']}
            dict_perf_metrics = self.compute_perf_metrics(event, bwd=bwd, non_data_mov=non_data_mov)
            if dict_perf_metrics is not None:
                metrics_event.update(dict_perf_metrics)
            rows.append(metrics_event)

        df_perf_metrics = pd.DataFrame(rows)
        return df_perf_metrics
    
    def build_df_fwd_perf_metrics(self, event_UIDs):
        return self.build_df_perf_metrics(event_UIDs, bwd=False)
    def build_df_bwd_perf_metrics(self, event_UIDs):
        return self.build_df_perf_metrics(event_UIDs, bwd=True)
    

    @staticmethod
    def summarize_df_perf_metrics(df_perf_metrics, agg_metrics=['mean', 'std']):
        dict_agg = {}
        # first element for GFLOPS and FLOPS/Byte
        dict_agg['GFLOPS'] = 'first'
        if 'FLOPS/Byte' in df_perf_metrics.columns:
            dict_agg['FLOPS/Byte'] = 'first'
        dict_agg['TFLOPS/s'] = agg_metrics
        if 'Non-Data-Mov TFLOPS/s' in df_perf_metrics.columns:
            dict_agg['Non-Data-Mov TFLOPS/s'] = agg_metrics
        if 'Non-Data-Mov Kernel Time (µs)' in df_perf_metrics.columns:
            dict_agg['Non-Data-Mov Kernel Time (µs)'] = ['sum']
        dict_agg['Kernel Time (µs)'] = ['sum']
        dict_agg['name'] = 'count'  # Use the 'name' column as a proxy for counting rows

        # Identify parameter columns for grouping
        param_cols = [col for col in df_perf_metrics.columns if col.startswith('param: ')]

        # Perform the aggregation
        df_perf_metrics_summary = (
            df_perf_metrics
            .groupby(['name'] + param_cols)
            .agg(dict_agg)
        )
        df_perf_metrics_summary.columns = ['_'.join(col).strip() for col in df_perf_metrics_summary.columns.values]
        df_perf_metrics_summary.reset_index(inplace=True)

        df_perf_metrics_summary.sort_values(by='Kernel Time (µs)_sum', ascending=False, inplace=True)

        return df_perf_metrics_summary

    def get_kernel_launchers(self):
        # This method traverses the event tree to identify CPU operations that serve as 
        # "kernel launchers." These are operations that result in GPU kernel 
        # execution without further cpu op calls. 
        # Note that kernels are called through runtime events.
        # This is why, this method identifies such cases 
        # by checking if grandchildren of CPU operations are kernel events.
        kernel_launchers = []
        for event in self.tree.events:
            if event.get('cat') != 'cpu_op':
                continue
            kernel_launcher = False
            total_direct_kernel_time = 0
            direct_kernel_count = 0
            for child_UID in event.get('children', []):
                child = self.tree.events_by_uid[child_UID]
                for grand_child_UID in child.get('children', []):
                    grand_child = self.tree.events_by_uid[grand_child_UID]
                    if grand_child.get('cat') == 'kernel':
                        kernel_launcher = True
                        total_direct_kernel_time += grand_child['dur']
                        direct_kernel_count += 1
            if kernel_launcher:
                event['total_direct_kernel_time'] = total_direct_kernel_time
                event['direct_kernel_count'] = direct_kernel_count
                kernel_launchers.append(event)
        return kernel_launchers

    def get_df_kernel_launchers(self, id_cols=False):

        def list_to_tuple(obj):
            if isinstance(obj, list):
                return tuple(list_to_tuple(item) for item in obj)
            return obj
        
        kernel_launchers = self.get_kernel_launchers()
        rows = []
        for event in kernel_launchers:
            metrics_event = {'name': event['name'],
                        'Input Dims': list_to_tuple(event['args']['Input Dims']),
                        'total_direct_kernel_time': event['total_direct_kernel_time'],
                        'direct_kernel_count': event['direct_kernel_count']}
            if id_cols:
                metrics_event['pid'] = event['pid']
                metrics_event['tid'] = event['tid']
                metrics_event['external_id'] = event['args']['External id']
            rows.append(metrics_event)
        df = pd.DataFrame(rows)
        return df
    
    @staticmethod
    def get_df_kernel_launchers_summary(df_kernel_launchers):
        df_temp = df_kernel_launchers.copy()
        df_agg = df_temp.groupby('name').agg({'total_direct_kernel_time': ['sum', 'count']})
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(columns={'total_direct_kernel_time_count': 'Count'}, inplace=True)
        df_agg.sort_values(by='total_direct_kernel_time_sum', ascending=False, inplace=True)
        df_agg['total_direct_kernel_time_ms'] = df_agg['total_direct_kernel_time_sum'] / 1000
        total_duration_ms = df_agg['total_direct_kernel_time_ms'].sum()
        df_agg['Percentage (%)'] = (df_agg['total_direct_kernel_time_ms'] / total_duration_ms) * 100
        df_agg['Cumulative Percentage (%)'] = df_agg['Percentage (%)'].cumsum()
        
        return df_agg

    #separate out name wise perf breakdown and shape wise perf breakdown for a given name
    @staticmethod
    def get_df_kernel_launchers_summary_by_shape(df_kernel_launchers, name):
        df_temp = df_kernel_launchers.copy()
        df_temp = df_temp[df_temp['name'] == name]
        dict_agg = {'total_direct_kernel_time': ['sum', 'count', 'mean', 'std'],
                    'direct_kernel_count': ['max', 'min']}
        df_agg = df_temp.groupby(['Input Dims']).agg(dict_agg)
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        df_agg.reset_index(inplace=True)
        df_agg.rename(columns={'total_direct_kernel_time_sum': 'Total Kernel Time (µs)',
                               'total_direct_kernel_time_count': 'Count',
                               'total_direct_kernel_time_mean': 'Mean Kernel Time (µs)',
                               'total_direct_kernel_time_std': 'Std Kernel Time (µs)',
                               'direct_kernel_count_max': 'Max Direct Kernel Count',
                               'direct_kernel_count_min': 'Min Direct Kernel Count'}, inplace=True)
        df_agg.sort_values(by='Total Kernel Time (µs)', ascending=False, inplace=True)
        df_agg['Total Kernel Time (ms)'] = df_agg['Total Kernel Time (µs)'] / 1000
        total_duration_ms = df_agg['Total Kernel Time (ms)'].sum()
        df_agg['Percentage (%)'] = (df_agg['Total Kernel Time (ms)'] / total_duration_ms) * 100
        df_agg['Cumulative Percentage (%)'] = df_agg['Percentage (%)'].cumsum()
        
        return df_agg