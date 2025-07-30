import argparse
import pandas as pd
from bokeh.plotting import figure, save, output_file
from src.chart import GPUTraceDashboard
import datetime

def main():
    parser = argparse.ArgumentParser(description='Generate GPU Kernel Profiling Dashboard')
    parser.add_argument('--trace1', required=True, help='Path to first trace file (example: ./trace1.pt.trace.json.gz)')
    parser.add_argument('--trace2', required=True, help='Path to second trace file (example: ./trace2.pt.trace.json.gz)')
    parser.add_argument('--name1', default="Trace_A", help='Name for first trace (default: Trace_A)')
    parser.add_argument('--name2', default="Trace_B", help='Name for second trace (default: Trace_B)')    
    parser.add_argument('--output', default="tm.html",  help='Output HTML file name (default: tm_{timestamp}.html)')
    
    args = parser.parse_args()
    
    # Add timestamp to output filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = args.output.replace('.html', f'_{timestamp}.html')
    
   # Create and save the visualization
    output_file(output_filename, title="GPU Kernel Profiling Dashboard")
    dashboard = GPUTraceDashboard(args.trace1, args.trace2, args.name1, args.name2)
    layout = dashboard.create_visualization()
    save(layout)
    print(f"Dashboard saved to {output_filename}")

if __name__ == "__main__":
    main()