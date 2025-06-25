import argparse
import json
import os
import sys
import pandas as pd

from pathlib import Path

from TraceLens.TreePerf import JaxAnalyses

from TraceLens.Reporting.reporting_utils import export_data_df

def calculate_gpu_event_statistics(profile_xplane_pb_path: str) -> tuple:

    averages, categorized, xla_events = JaxAnalyses.summarize_gpu_events(profile_xplane_pb_path)

    overlapped_comm = averages[averages['type']=='total_comm_time']['time ms'].iloc[0] - averages[averages['type']=='exposed_comm_time']['time ms'].iloc[0]
    averages.loc[len(averages)] = ['overlapped_comm', overlapped_comm, overlapped_comm/averages[averages['type']=='total_time']['time ms'].iloc[0]*100]

    new_order = [0, 1, 2, 3, 4, 5, 8, 6, 7]
    df_gpu_events_averages = averages.reindex(new_order)

    df_gpu_events_categorized_mean = categorized.copy()
    df_gpu_events_categorized_mean = df_gpu_events_categorized_mean.reset_index().rename(columns={'index': 'name'})

    # categorize xla events by stripping digits and underscores
    xla_grouped = xla_events.groupby(xla_events.index.str.replace(r'\d+|_+$', '', regex=True)).sum(numeric_only=True)
    xla_grouped = xla_grouped.reset_index().rename(columns={'index': 'short_name_grouped'})
    df_xla_grouped = xla_grouped.sort_values(by='percent', ascending=False)

    return df_gpu_events_averages, df_gpu_events_categorized_mean, df_xla_grouped


    
def main():
    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate performance report tables.')
    parser.add_argument("--profile_xplane_pb_path", type=str, required=True, help="Path to the profile.xplane.pb file (e.g., '/path/to/profile.xplane.pb')")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--output_table_formats", type=str, nargs="+", default=[".xlsx", ".csv"], choices=[".xlsx", ".csv"], help="Output table save formats. You can select one or both formats: .xlsx and/or .csv.")
    parser.add_argument("--num_cus", type=str, default=304, help="Number of compute units, MI300X - 304; MI210: 104")
    parser.add_argument("--output_filename", type=str, default="trace_analysis_results", help="Base name for output files")
    
    args = parser.parse_args()

    num_cus = {"num_cus": int(args.num_cus)}


    output_folder = Path(args.output_path)
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Insufficient permissions to create the directory at {args.output_path}.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: The specified path {args.output_path} is invalid.", file=sys.stderr)
        sys.exit(1)

    output_filename = args.output_filename

    df_gpu_events_averages, df_gpu_events_categorized_mean, df_xla_grouped = calculate_gpu_event_statistics(args.profile_xplane_pb_path)

    gemms = JaxAnalyses.summarize_gpu_gemm_events_from_pb(args.profile_xplane_pb_path)
    gemms = gemms.reset_index().rename(columns={'index': 'name'})

    gemms_detailed = JaxAnalyses.gemm_performance_from_pb(args.profile_xplane_pb_path, arch = num_cus)
    
    export_data_df(
            df_gpu_events_averages,
            output_folder,
            output_filename,
            output_table_format=args.output_table_formats,
            suffix="_gpu_events_averages",
    )

    export_data_df(
            df_gpu_events_categorized_mean,
            output_folder,
            output_filename,
            output_table_format=args.output_table_formats,
            suffix="_gpu_events_categorized_mean",
    )

    export_data_df(
            df_xla_grouped,
            output_folder,
            output_filename,
            output_table_format=args.output_table_formats,
            suffix="_xla_grouped",
    )

    export_data_df(
            gemms,
            output_folder,
            output_filename,
            output_table_format=args.output_table_formats,
            suffix="_gemms",
    )

    export_data_df(
            gemms_detailed,
            output_folder,
            output_filename,
            output_table_format=args.output_table_formats,
            suffix="_gemms_detailed",
    )

    print(f"DataFrames successfully written to {args.output_path}")

if __name__ == "__main__":
    main()
