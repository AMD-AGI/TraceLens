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

    def compute_perf_metrics(self, event, bwd=False, bytes_per_element=2):
        # Select the appropriate dictionary for FLOPS and memory functions
        flops_func = (op_to_bwd_flops_func_map if bwd else op_to_flops_func_map)[event['name']]
        param_details_func = op_to_param_details_func_map[event['name']]
        bytes_func = (op_to_bwd_bytes_func_map if bwd else op_to_bytes_func_map).get(event['name'])

        param_details = param_details_func(event)
        gflops = flops_func(param_details) / 1e9  
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
        bytes_moved = bytes_func(param_details, bytes_per_element) if bytes_func else None

        # Return metrics
        dict_metrics = {
            'GFLOPS': gflops,
            'Kernel Time (µs)': total_kernel_time,
            'TFLOPS/s': tflops_per_s,
            'Non-Data-Mov Kernel Time (µs)': total_non_data_mov_time,
            'Non-Data-Mov TFLOPS/s': non_data_mov_tflops_per_s,
        }
        if bytes_moved is not None:
            dict_metrics['FLOPS/Byte'] = (gflops * 1e9) / bytes_moved
        
        for key, value in param_details.items():
            dict_metrics[f"param: {key}"] = value

        return dict_metrics

    def compute_fwd_perf_metrics(self, event):
        return self.compute_perf_metrics(event, bwd=False)
    def compute_bwd_perf_metrics(self, event):
        return self.compute_perf_metrics(event, bwd=True)
    
    def build_df_perf_metrics(self, event_UIDs, bwd):
        rows = []
        for event_uid in event_UIDs:
            event = self.tree.events_by_uid[event_uid]
            metrics_event = {'cat': event['cat'], 'name': event['name'],
                        'pid': event['pid'], 'tid': event['tid'],
                        'external_id': event['args']['External id']}
            dict_perf_metrics = self.compute_perf_metrics(event, bwd=bwd)
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
        if 'Non-Data-Mov Kernel Time (µs)' in df_perf_metrics.columns:
            dict_agg['Non-Data-Mov Kernel Time (µs)'] = ['sum']
        dict_agg['Kernel Time (µs)'] = ['sum']

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
