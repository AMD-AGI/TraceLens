import gzip
import json
import pandas as pd
from bokeh.models import ColumnDataSource

class TraceDataProcessor:
    """Handles loading and processing of trace data."""
    
    @staticmethod
    def extract_kernel_data(trace_path):
        """Extract kernel data from a trace file."""
        with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
            trace_data = json.load(f)
        
        events = trace_data.get("traceEvents", [])
        kernel_events = []
        
        for event in events:
            if event.get("ph") == "X" and "kernel" in event.get("cat", "").lower():
                kernel_name = event.get("name", "")
                start = event.get("ts", 0)
                duration = event.get("dur", 0)
                end = start + duration
                kernel_events.append((kernel_name, start, duration, end))
        
        if kernel_events:
            base_time = kernel_events[0][1]
            parsed = [{
                "Kernel Index": idx,
                "Kernel Name": name,
                "Start (us)": round(start - base_time, 3),
                "Duration (us)": round(duration, 3),
                "End (us)": round(end - base_time, 3),
            } for idx, (name, start, duration, end) in enumerate(kernel_events)]
            return pd.DataFrame(parsed)
        else:
            return pd.DataFrame(columns=["Kernel Index", "Kernel Name", "Start (us)", "Duration (us)", "End (us)"])

    @staticmethod
    def create_top_n_data(df, n=30):
        """Create top N kernels by total latency and counts."""
        kernel_stats = df.groupby('Kernel Name').agg({
            'Duration (us)': ['sum', 'count', 'mean']
        }).round(3)
        kernel_stats.columns = ['Total Duration (us)', 'Count', 'Avg Duration (us)']
        kernel_stats = kernel_stats.sort_values('Total Duration (us)', ascending=False).head(n)
        kernel_stats = kernel_stats.reset_index()
        return kernel_stats

    @staticmethod
    def create_sorted_latency_data(df):
        """Create sorted kernel data by latency for individual kernels."""
        sorted_df = df.sort_values('Duration (us)', ascending=False).reset_index(drop=True)
        return sorted_df

class DataSourceManager:
    """Manages Bokeh data sources for charts and tables."""
    
    def __init__(self, df_gpu_a, df_gpu_b, default_window_size=100):
        self.df_gpu_a = df_gpu_a
        self.df_gpu_b = df_gpu_b
        self.default_window_size = default_window_size
        self._create_all_sources()
    
    def _create_all_sources(self):
        """Create all data sources needed for the visualization."""
        # Main data sources
        self.source_gpu_a = ColumnDataSource(self.df_gpu_a)
        self.source_gpu_b = ColumnDataSource(self.df_gpu_b)
        
        # Filtered sources for sliding window
        self.source_gpu_a_filtered = ColumnDataSource(self.df_gpu_a.head(self.default_window_size))
        self.source_gpu_b_filtered = ColumnDataSource(self.df_gpu_b.head(self.default_window_size))
        
        # Sorted filtered sources
        initial_gpu_a_sorted = self.df_gpu_a.head(self.default_window_size).sort_values('Duration (us)', ascending=False).reset_index(drop=True)
        initial_gpu_b_sorted = self.df_gpu_b.head(self.default_window_size).sort_values('Duration (us)', ascending=False).reset_index(drop=True)
        
        self.source_sorted_gpu_a_filtered = ColumnDataSource(initial_gpu_a_sorted)
        self.source_sorted_gpu_b_filtered = ColumnDataSource(initial_gpu_b_sorted)
        
        # Combined view sources
        self.source_gpu_a_combined_filtered = ColumnDataSource(self.df_gpu_a.head(self.default_window_size))
        self.source_gpu_b_combined_filtered = ColumnDataSource(self.df_gpu_b.head(self.default_window_size))
        
        self.source_sorted_gpu_a_combined_filtered = ColumnDataSource(initial_gpu_a_sorted)
        self.source_sorted_gpu_b_combined_filtered = ColumnDataSource(initial_gpu_b_sorted)
        
        # Top N data sources
        self._create_top_n_sources()
    
    def _create_top_n_sources(self):
        """Create top N data sources."""
        top_n_gpu_a = TraceDataProcessor.create_top_n_data(self.df_gpu_a)
        top_n_gpu_b = TraceDataProcessor.create_top_n_data(self.df_gpu_b)
        top_n_both = TraceDataProcessor.create_top_n_data(pd.concat([self.df_gpu_a, self.df_gpu_b], ignore_index=True))
        
        self.source_top_gpu_a = ColumnDataSource(top_n_gpu_a)
        self.source_top_gpu_b = ColumnDataSource(top_n_gpu_b)
        self.source_top_both = ColumnDataSource(top_n_both)