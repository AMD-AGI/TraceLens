import pandas as pd
from bokeh.plotting import figure
from bokeh.models import DataTable, TableColumn, NumberFormatter, CustomJS, Spinner, Slider, Div
from bokeh.layouts import column, row
from bokeh.models import Tabs, TabPanel
from src.data import TraceDataProcessor, DataSourceManager

class ChartBuilder:
    """Builds charts and visualizations."""
    
    @staticmethod
    def create_bar_chart(title, source_filtered, color, width=800):
        """Create a bar chart for kernel latency data."""
        p = figure(
            title=title,
            x_axis_label="Kernel Index",
            y_axis_label="Duration (us)",
            width=width,
            height=400,
            tools="reset,tap"
        )
        bars = p.vbar(x='Kernel Index', top='Duration (us)', width=0.8, 
                     source=source_filtered, color=color, alpha=0.7)
        return p, bars


class TableBuilder:
    """Builds data tables for kernel information."""
    
    @staticmethod
    def create_kernel_table_columns():
        """Create standard columns for kernel data tables."""
        return [
            TableColumn(field="Kernel Index", title="Index", width=80),
            TableColumn(field="Kernel Name", title="Kernel Name", width=600),
            TableColumn(field="Start (us)", title="Start (μs)", width=100, 
                       formatter=NumberFormatter(format="0,0.000")),
            TableColumn(field="Duration (us)", title="Duration (μs)", width=120, 
                       formatter=NumberFormatter(format="0,0.000")),
            TableColumn(field="End (us)", title="End (μs)", width=100, 
                       formatter=NumberFormatter(format="0,0.000")),
        ]
    
    @staticmethod
    def create_top_n_table_columns():
        """Create columns for top N kernel tables."""
        return [
            TableColumn(field="Kernel Name", title="Kernel Name", width=680),
            TableColumn(field="Total Duration (us)", title="Total Duration (μs)", width=120,
                       formatter=NumberFormatter(format="0,0.000")),
            TableColumn(field="Count", title="Count", width=60),
            TableColumn(field="Avg Duration (us)", title="Avg Duration (μs)", width=120,
                       formatter=NumberFormatter(format="0,0.000")),
        ]
    
    @staticmethod
    def create_kernel_table(source, width=2000):
        """Create a data table for kernel information."""
        columns = TableBuilder.create_kernel_table_columns()
        return DataTable(source=source, columns=columns, width=width, height=600, index_position=None)
    
    @staticmethod
    def create_top_n_table(source, width=2000):
        """Create a data table for top N kernels."""
        columns = TableBuilder.create_top_n_table_columns()
        return DataTable(source=source, columns=columns, width=width, height=600, index_position=None)


class ControlsBuilder:
    """Builds UI controls like sliders and spinners."""
    
    @staticmethod
    def create_window_size_spinner(default_value=100):
        """Create a spinner for window size control."""
        return Spinner(title="Window Size:", low=10, high=1000, step=10, value=default_value, width=150)
    
    @staticmethod
    def create_slider(df_length, window_size, gpu_name, width=1000):
        """Create a slider for sliding window control."""
        return Slider(
            start=0, 
            end=max(0, df_length - window_size), 
            value=0, 
            step=window_size, 
            title=f"{gpu_name} Kernel Index Window (showing {window_size} at a time)", 
            width=width
        )


class CallbackManager:
    """Manages JavaScript callbacks for interactive functionality."""
    
    @staticmethod
    def create_sorted_data_js():
        """JavaScript function to create sorted data from filtered data."""
        return """
        function createSortedData(filtered_data) {
            const indices = [];
            const length = filtered_data['Kernel Index'].length;
            for (let i = 0; i < length; i++) {
                indices.push(i);
            }
            
            // Sort indices by Duration (us) in descending order
            indices.sort((a, b) => filtered_data['Duration (us)'][b] - filtered_data['Duration (us)'][a]);
            
            const sorted_data = {};
            for (let key in filtered_data) {
                sorted_data[key] = indices.map(i => filtered_data[key][i]);
            }
            return sorted_data;
        }
        """
    
    @staticmethod
    def create_window_size_callback(sources, controls, gpu_names):
        """Create callback for window size changes."""
        return CustomJS(
            args=dict(**sources, **controls, **gpu_names),
            code=CallbackManager.create_sorted_data_js() + """
            const window_size = spinner.value;
            
            // Update slider properties
            slider_gpu_a.step = window_size;
            slider_gpu_b.step = window_size;
            slider_gpu_a.end = Math.max(0, source_gpu_a.data['Kernel Index'].length - window_size);
            slider_gpu_b.end = Math.max(0, source_gpu_b.data['Kernel Index'].length - window_size);
            slider_gpu_a.title = `${gpu_name_a} Kernel Index Window (showing ${window_size} at a time)`;
            slider_gpu_b.title = `${gpu_name_b} Kernel Index Window (showing ${window_size} at a time)`;
            
            // Update filtered data
            const start_gpu_a = slider_gpu_a.value;
            const end_gpu_a = Math.min(start_gpu_a + window_size, source_gpu_a.data['Kernel Index'].length);
            
            const start_gpu_b = slider_gpu_b.value;
            const end_gpu_b = Math.min(start_gpu_b + window_size, source_gpu_b.data['Kernel Index'].length);
            
            // Update GPU A filtered data
            const gpu_a_filtered = {};
            for (let key in source_gpu_a.data) {
                gpu_a_filtered[key] = source_gpu_a.data[key].slice(start_gpu_a, end_gpu_a);
            }
            source_gpu_a_filtered.data = gpu_a_filtered;
            
            // Update GPU B filtered data
            const gpu_b_filtered = {};
            for (let key in source_gpu_b.data) {
                gpu_b_filtered[key] = source_gpu_b.data[key].slice(start_gpu_b, end_gpu_b);
            }
            source_gpu_b_filtered.data = gpu_b_filtered;
            
            // Update sorted filtered data
            source_sorted_gpu_a_filtered.data = createSortedData(gpu_a_filtered);
            source_sorted_gpu_b_filtered.data = createSortedData(gpu_b_filtered);
            
            source_gpu_a_filtered.change.emit();
            source_gpu_b_filtered.change.emit();
            source_sorted_gpu_a_filtered.change.emit();
            source_sorted_gpu_b_filtered.change.emit();
            """
        )
    
    @staticmethod
    def create_slider_callback(sources, controls):
        """Create callback for slider changes."""
        return CustomJS(
            args=dict(**sources, **controls),
            code=CallbackManager.create_sorted_data_js() + """
            const start = slider.value;
            const window_size = spinner.value;
            const end = Math.min(start + window_size, source.data['Kernel Index'].length);
            
            const filtered_data = {};
            for (let key in source.data) {
                filtered_data[key] = source.data[key].slice(start, end);
            }
            source_filtered.data = filtered_data;
            
            // Create sorted version of the current window
            source_sorted_filtered.data = createSortedData(filtered_data);
            
            source_filtered.change.emit();
            source_sorted_filtered.change.emit();
            """
        )
    
    @staticmethod
    def create_tap_callback():
        """Create callback for bar chart tap events."""
        return CustomJS(
            args={},  # Will be set when creating specific instances
            code="""
            const indices = source.selected.indices;
            if (indices.length > 0) {
                table.source.selected.indices = indices;
                sorted_table.source.selected.indices = indices;
                const row_height = 25;
                const scroll_top = indices[0] * row_height;
                table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
                sorted_table.view.el.querySelector('.slick-viewport').scrollTop = scroll_top;
            }
            """
        )


class GPUTraceDashboard:
    """Main class that orchestrates the creation of the GPU trace dashboard."""
    
    def __init__(self, trace_path1, trace_path2, gpu_name_a="GPU_A", gpu_name_b="GPU_B"):
        self.trace_path1 = trace_path1
        self.trace_path2 = trace_path2
        self.gpu_name_a = gpu_name_a
        self.gpu_name_b = gpu_name_b
        self.default_window_size = 100
        
        # Load and process data
        self.df_gpu_a = TraceDataProcessor.extract_kernel_data(trace_path1)
        self.df_gpu_b = TraceDataProcessor.extract_kernel_data(trace_path2)
        
        # Initialize managers
        self.data_sources = DataSourceManager(self.df_gpu_a, self.df_gpu_b, self.default_window_size)
        
    def _create_charts(self):
        """Create all charts for the dashboard."""
        # Individual charts
        self.chart_gpu_a, self.bars_gpu_a = ChartBuilder.create_bar_chart(
            f"{self.gpu_name_a} Kernel Latency", 
            self.data_sources.source_gpu_a_filtered, 
            "blue", 
            width=2000
        )
        
        self.chart_gpu_b, self.bars_gpu_b = ChartBuilder.create_bar_chart(
            f"{self.gpu_name_b} Kernel Latency", 
            self.data_sources.source_gpu_b_filtered, 
            "red", 
            width=2000
        )
        
        # Combined charts
        self.chart_gpu_a_combined, self.bars_gpu_a_combined = ChartBuilder.create_bar_chart(
            f"{self.gpu_name_a} Kernel Latency", 
            self.data_sources.source_gpu_a_combined_filtered, 
            "blue", 
            width=1000
        )
        
        self.chart_gpu_b_combined, self.bars_gpu_b_combined = ChartBuilder.create_bar_chart(
            f"{self.gpu_name_b} Kernel Latency", 
            self.data_sources.source_gpu_b_combined_filtered, 
            "red", 
            width=1000
        )
    
    def _create_tables(self):
        """Create all tables for the dashboard."""
        # Individual tables
        self.table_gpu_a = TableBuilder.create_kernel_table(self.data_sources.source_gpu_a_filtered)
        self.table_gpu_b = TableBuilder.create_kernel_table(self.data_sources.source_gpu_b_filtered)
        
        # Sorted tables
        self.sorted_table_gpu_a = TableBuilder.create_kernel_table(self.data_sources.source_sorted_gpu_a_filtered)
        self.sorted_table_gpu_b = TableBuilder.create_kernel_table(self.data_sources.source_sorted_gpu_b_filtered)
        
        # Combined tables
        self.table_gpu_a_combined = TableBuilder.create_kernel_table(self.data_sources.source_gpu_a_combined_filtered, width=1000)
        self.table_gpu_b_combined = TableBuilder.create_kernel_table(self.data_sources.source_gpu_b_combined_filtered, width=1000)
        
        # Combined sorted tables
        self.sorted_table_gpu_a_combined = TableBuilder.create_kernel_table(self.data_sources.source_sorted_gpu_a_combined_filtered, width=1000)
        self.sorted_table_gpu_b_combined = TableBuilder.create_kernel_table(self.data_sources.source_sorted_gpu_b_combined_filtered, width=1000)
        
        # Top N tables
        self.top_table_gpu_a = TableBuilder.create_top_n_table(self.data_sources.source_top_gpu_a)
        self.top_table_gpu_b = TableBuilder.create_top_n_table(self.data_sources.source_top_gpu_b)
        self.top_table_both = TableBuilder.create_top_n_table(self.data_sources.source_top_both, width=800)
        
        # Combined top N tables
        self.top_table_gpu_a_combined = TableBuilder.create_top_n_table(self.data_sources.source_top_gpu_a, width=1000)
        self.top_table_gpu_b_combined = TableBuilder.create_top_n_table(self.data_sources.source_top_gpu_b, width=1000)
    
    def _create_controls(self):
        """Create all controls for the dashboard."""
        # Window size controls
        self.window_size_spinner = ControlsBuilder.create_window_size_spinner(self.default_window_size)
        self.window_size_spinner_combined = ControlsBuilder.create_window_size_spinner(self.default_window_size)
        
        # Sliders
        self.slider_gpu_a = ControlsBuilder.create_slider(len(self.df_gpu_a), self.default_window_size, self.gpu_name_a)
        self.slider_gpu_b = ControlsBuilder.create_slider(len(self.df_gpu_b), self.default_window_size, self.gpu_name_b)
        self.slider_gpu_a_combined = ControlsBuilder.create_slider(len(self.df_gpu_a), self.default_window_size, self.gpu_name_a)
        self.slider_gpu_b_combined = ControlsBuilder.create_slider(len(self.df_gpu_b), self.default_window_size, self.gpu_name_b)
    
    def _attach_callbacks(self):
        """Attach all JavaScript callbacks to controls."""
        # Window size callbacks for individual tabs
        window_size_sources = {
            'spinner': self.window_size_spinner,
            'slider_gpu_a': self.slider_gpu_a,
            'slider_gpu_b': self.slider_gpu_b,
            'source_gpu_a': self.data_sources.source_gpu_a,
            'source_gpu_b': self.data_sources.source_gpu_b,
            'source_gpu_a_filtered': self.data_sources.source_gpu_a_filtered,
            'source_gpu_b_filtered': self.data_sources.source_gpu_b_filtered,
            'source_sorted_gpu_a_filtered': self.data_sources.source_sorted_gpu_a_filtered,
            'source_sorted_gpu_b_filtered': self.data_sources.source_sorted_gpu_b_filtered,
        }
        
        gpu_names = {'gpu_name_a': self.gpu_name_a, 'gpu_name_b': self.gpu_name_b}
        
        window_size_callback = CallbackManager.create_window_size_callback(
            window_size_sources, {}, gpu_names
        )
        
        self.window_size_spinner.js_on_change('value', window_size_callback)
        
        # Window size callbacks for combined tab
        window_size_sources_combined = {
            'spinner': self.window_size_spinner_combined,
            'slider_gpu_a': self.slider_gpu_a_combined,
            'slider_gpu_b': self.slider_gpu_b_combined,
            'source_gpu_a': self.data_sources.source_gpu_a,
            'source_gpu_b': self.data_sources.source_gpu_b,
            'source_gpu_a_filtered': self.data_sources.source_gpu_a_combined_filtered,
            'source_gpu_b_filtered': self.data_sources.source_gpu_b_combined_filtered,
            'source_sorted_gpu_a_filtered': self.data_sources.source_sorted_gpu_a_combined_filtered,
            'source_sorted_gpu_b_filtered': self.data_sources.source_sorted_gpu_b_combined_filtered,
        }
        
        window_size_callback_combined = CallbackManager.create_window_size_callback(
            window_size_sources_combined, {}, gpu_names
        )
        
        self.window_size_spinner_combined.js_on_change('value', window_size_callback_combined)
        
        # Slider callbacks for individual tabs
        slider_callback_gpu_a = CallbackManager.create_slider_callback(
            {
                'source': self.data_sources.source_gpu_a,
                'source_filtered': self.data_sources.source_gpu_a_filtered,
                'source_sorted_filtered': self.data_sources.source_sorted_gpu_a_filtered,
            },
            {
                'slider': self.slider_gpu_a,
                'spinner': self.window_size_spinner
            }
        )
        
        slider_callback_gpu_b = CallbackManager.create_slider_callback(
            {
                'source': self.data_sources.source_gpu_b,
                'source_filtered': self.data_sources.source_gpu_b_filtered,
                'source_sorted_filtered': self.data_sources.source_sorted_gpu_b_filtered,
            },
            {
                'slider': self.slider_gpu_b,
                'spinner': self.window_size_spinner
            }
        )
        
        self.slider_gpu_a.js_on_change('value', slider_callback_gpu_a)
        self.slider_gpu_b.js_on_change('value', slider_callback_gpu_b)
        
        # Slider callbacks for combined tab
        slider_callback_gpu_a_combined = CallbackManager.create_slider_callback(
            {
                'source': self.data_sources.source_gpu_a,
                'source_filtered': self.data_sources.source_gpu_a_combined_filtered,
                'source_sorted_filtered': self.data_sources.source_sorted_gpu_a_combined_filtered,
            },
            {
                'slider': self.slider_gpu_a_combined,
                'spinner': self.window_size_spinner_combined
            }
        )
        
        slider_callback_gpu_b_combined = CallbackManager.create_slider_callback(
            {
                'source': self.data_sources.source_gpu_b,
                'source_filtered': self.data_sources.source_gpu_b_combined_filtered,
                'source_sorted_filtered': self.data_sources.source_sorted_gpu_b_combined_filtered,
            },
            {
                'slider': self.slider_gpu_b_combined,
                'spinner': self.window_size_spinner_combined
            }
        )
        
        self.slider_gpu_a_combined.js_on_change('value', slider_callback_gpu_a_combined)
        self.slider_gpu_b_combined.js_on_change('value', slider_callback_gpu_b_combined)
        
        # Tap callbacks for individual tabs
        tap_callback_gpu_a = CallbackManager.create_tap_callback()
        tap_callback_gpu_a.args = dict(
            source=self.data_sources.source_gpu_a_filtered,
            table=self.table_gpu_a,
            sorted_table=self.sorted_table_gpu_a
        )
        
        tap_callback_gpu_b = CallbackManager.create_tap_callback()
        tap_callback_gpu_b.args = dict(
            source=self.data_sources.source_gpu_b_filtered,
            table=self.table_gpu_b,
            sorted_table=self.sorted_table_gpu_b
        )
        
        self.bars_gpu_a.data_source.selected.js_on_change('indices', tap_callback_gpu_a)
        self.bars_gpu_b.data_source.selected.js_on_change('indices', tap_callback_gpu_b)
        
        # Tap callbacks for combined tab
        tap_callback_gpu_a_combined = CallbackManager.create_tap_callback()
        tap_callback_gpu_a_combined.args = dict(
            source=self.data_sources.source_gpu_a_combined_filtered,
            table=self.table_gpu_a_combined,
            sorted_table=self.sorted_table_gpu_a_combined
        )
        
        tap_callback_gpu_b_combined = CallbackManager.create_tap_callback()
        tap_callback_gpu_b_combined.args = dict(
            source=self.data_sources.source_gpu_b_combined_filtered,
            table=self.table_gpu_b_combined,
            sorted_table=self.sorted_table_gpu_b_combined
        )
        
        self.bars_gpu_a_combined.data_source.selected.js_on_change('indices', tap_callback_gpu_a_combined)
        self.bars_gpu_b_combined.data_source.selected.js_on_change('indices', tap_callback_gpu_b_combined)
    
    def _create_layouts(self):
        """Create the layout for each tab."""
        # GPU A layout
        self.gpu_a_layout = column(
            Div(text=f"<h2>{self.gpu_name_a} Kernel Analysis</h2>"),
            row(self.window_size_spinner, self.slider_gpu_a),
            self.chart_gpu_a,
            Div(text="<h3>Kernel Details</h3>"),
            self.table_gpu_a,
            Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>"),
            self.sorted_table_gpu_a,
            Div(text="<h3>Top 10 Kernels by Total Duration</h3>"),
            self.top_table_gpu_a
        )
        
        # GPU B layout
        self.gpu_b_layout = column(
            Div(text=f"<h2>{self.gpu_name_b} Kernel Analysis</h2>"),
            row(self.window_size_spinner, self.slider_gpu_b),
            self.chart_gpu_b,
            Div(text="<h3>Kernel Details</h3>"),
            self.table_gpu_b,
            Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>"),
            self.sorted_table_gpu_b,
            Div(text="<h3>Top 10 Kernels by Total Duration</h3>"),
            self.top_table_gpu_b
        )
        
        # Combined layout
        self.both_layout = column(
            Div(text="<h2>Side by Side Comparison</h2>"),        
            row(self.window_size_spinner_combined),
            row(self.slider_gpu_a_combined, self.slider_gpu_b_combined),
            row(self.chart_gpu_a_combined, self.chart_gpu_b_combined),
            Div(text="<h3>Kernel Details</h3>"),
            row(
                column(Div(text=f"<h4>{self.gpu_name_a} Kernels</h4>"), self.table_gpu_a_combined),
                column(Div(text=f"<h4>{self.gpu_name_b} Kernels</h4>"), self.table_gpu_b_combined)
            ),
            Div(text="<h3>Kernels Sorted by Latency (Current Window)</h3>"),
            row(
                column(Div(text=f"<h4>{self.gpu_name_a} Sorted by Latency</h4>"), self.sorted_table_gpu_a_combined),
                column(Div(text=f"<h4>{self.gpu_name_b} Sorted by Latency</h4>"), self.sorted_table_gpu_b_combined)
            ),
            Div(text="<h3>Top 30 Kernels Comparison</h3>"),
            row(
                column(Div(text=f"<h4>{self.gpu_name_a} Top Kernels</h4>"), self.top_table_gpu_a_combined),
                column(Div(text=f"<h4>{self.gpu_name_b} Top Kernels</h4>"), self.top_table_gpu_b_combined)
            ),
        )
    
    def create_visualization(self):
        """Create the complete visualization dashboard."""
        self._create_charts()
        self._create_tables()
        self._create_controls()
        self._attach_callbacks()
        self._create_layouts()
        
        # Create tabs
        tab1 = TabPanel(child=self.gpu_a_layout, title=self.gpu_name_a)
        tab2 = TabPanel(child=self.gpu_b_layout, title=self.gpu_name_b) 
        tab3 = TabPanel(child=self.both_layout, title="Side-by-Side Comparison")
        
        tabs = Tabs(tabs=[tab1, tab2, tab3])
        
        return column(tabs)