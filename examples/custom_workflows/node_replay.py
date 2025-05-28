"""
Using TraceLens, and .json traces produced by torch profiler, this script provides a comparison on
how much the workload-specific metrics of most important GEMMs and CONVS
differ when microbenchmarking the same ops using torch and low-level benchmarking tools.
Currently works with hipblaslt-bench and MIOpenDriver, if they are available in the environment.
Tested with main branch @ 8f40079.

Note that the trace .json files are located recursively - this script does not depend on a certain directory structure.

Example directory tree:

profiling_fa_cudnn
    results_013_profile_nv_cudnn
        traces
            huvideo_traces_rank_0_step_3.json
            ...
            huvideo_traces_rank_7_step_3.json

    results_013_profile_nv_fa
        traces
            huvideo_traces_rank_0_step_3.json
            ...
            huvideo_traces_rank_7_step_3.json
    results_013_profile_rocm_fa
        traces
            huvideo_traces_rank_0_step_3.json
            ...
            huvideo_traces_rank_7_step_3.json

Example usage:

Analyze all .json trace files containing rocm_fa and step_3 in their full path.

Trace files include rank number using the pattern rank_0, rank_1, ..., rank_n.
Script uses this pattern to recognize rank numbers, thus we give -p rank_

python analyze_traces_tracelens.py -b profiling_fa_cudnn -f rocm_fa step_3

Targeting:

profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_0_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_1_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_2_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_3_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_4_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_5_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_6_step_3.json
profiling_fa_cudnn\results_013_profile_rocm_fa\traces\huvideo_traces_rank_7_step_3.json

NOTE: add -d flag for launching a dry run (check if correct files will be targeted).
      add -e flag for specifying extension. json and gz are currently supported, json is used by default
"""

import argparse
import gc
import os
import os.path as osp
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from perf_report_configs import conv_ops, gemm_ops
from perf_report_utils import collect_df_perf_metrics_per_group, parse_traces
from TraceLens import EventReplayer, TreePerfAnalyzer

summarize_df_perf_metrics = TreePerfAnalyzer.summarize_df_perf_metrics

group2ops = {
    "gemm": gemm_ops,
    "conv": conv_ops,
}

df_cols_gemm = [
    "param: M",
    "param: N",
    "param: K",
    "param: bias",
    "param: stride_A",
    "param: stride_B",
    "param: dtype_A_B",
    "param: transpose",
    "kernel_names_first",
]

df_cols_conv = [
    "param: convNd",
    "param: input_shape",
    "param: filter_shape",
    "param: dtype_input_weight",
    "param: input_stride",
    "param: weight_stride",
    "param: bias",
    "param: stride",
    "param: padding",
    "param: dilation",
    "param: transposed_conv",
    "param: output_padding",
    "param: groups",
    "kernel_names_first",
]

df_cols_common = [
    "name",
    "Kernel Time (µs)_mean",
    "kernel_time_sum_cum_pct",
]

group2dfcols = {
    "gemm": df_cols_common + df_cols_gemm,
    "conv": df_cols_common + df_cols_conv,
}


def benchmark_func(
    func: Callable,
    device: str,
    envvars: Optional[Dict[str, str]] = None,
    warmup: int = 1000,
    active_steps: int = 1000
) -> Optional[float]:
    """
    Benchmark a function with warmup and average steps.
    Disclaimer: This method would be innacurate for very short ops.
    Args:
        func (callable): The function to benchmark.
        device (str): GPU device.
        envvars (dict): Pass environment variables for possible logging purposes.
        warmup (int): Number of warmup iterations.
        active_steps (int): Number of iterations to average over.
    Returns:
        float: Average time taken per iteration in microseconds.
    """
    import torch

    if envvars:
        for key, val in envvars.items():
            os.environ[key] = val

        if "MIOPEN_ENABLE_LOGGING_CMD" in envvars:
            # MIOpen writes to stderr using C/C++ system calls (instead of Python's sys.stderr object)
            # So we need to redirect at the system level:
            # Save original stderr
            # Read and write permission, create file if required, truncate to zero length if file exists
            # Redirect stderr at OS level
            # Restore stderr and clean up
            orig_stderr_fd = os.dup(2)
            log_fd = os.open(envvars["MIOPEN_LOG_FILE"], os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(log_fd, 2)
            func()
            os.dup2(orig_stderr_fd, 2)
            os.close(orig_stderr_fd)
            os.close(log_fd)
        else:
            func()

        return None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup phase
    for _ in range(warmup):
        func()

    # Benchmarking phase
    torch.cuda.synchronize(device)

    start.record()

    for _ in range(active_steps):
        func()

    end.record()
    torch.cuda.synchronize(device)

    elapsed_time_ms = start.elapsed_time(end)
    avg_time_ms = elapsed_time_ms / active_steps
    avg_time_us = avg_time_ms * 1e3

    torch.cuda.empty_cache()

    return avg_time_us


def benchmark_func_wrapper(args: Tuple) -> Optional[float]:
    avg_time_us = benchmark_func(*args)
    return avg_time_us


def run_subprocess_cmd(cmd: Union[str, List[str]], shell: bool = False) -> Optional[str]:
    """Run a subprocess command and return the stdout as string"""
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        return None


def get_logging_envvars(group: str, i: int, base_path: str) -> Dict[str, str]:
    logging_envvars = None

    if group == "gemm":
        logging_envvars = {
            "HIPBLASLT_LOG_MASK": "32",
            "HIPBLASLT_LOG_FILE": osp.join(base_path, f"hipblaslt-bench-{i}.log"),
        }

    elif group == "conv":
        logging_envvars = {
            "MIOPEN_ENABLE_LOGGING_CMD": "1",
            "MIOPEN_LOG_FILE": osp.join(base_path, f"miopen-driver-{i}.log"),
        }

    return logging_envvars


def prepare_hipblaslt_bench_cmd(log_file: str):
    """Prepare hipblaslt-bench command for microbenchmarking."""
    modify = [
        [r"s/--algo_method index/--algo_method heuristic/g", log_file],
        [r"s/--solution_index [0-9]*/--requested_solution 1/", log_file],
        [r"s/--cold_iters [0-9]*/--cold_iters 100/g", log_file],
        [r"s/--iters [0-9]*/--iters 100/g", log_file],
        [r"s/--rotating [0-9]*/--rotating 512/g", log_file],
        [r"s/$/ --flush --initialization trig_float --use_gpu_timer --print_kernel_info/", log_file],
        # hipBLASLT 0.15 bug, remove aux_type flag due to warning: --use_e not set but --aux_type is provided
        [r"s/--aux_type f32_r//g", log_file],
    ]

    for mod in modify:
        _ = run_subprocess_cmd(["sed", "-i"] + mod)


def prepare_miopen_driver_cmd(log_file: str):
    """Prepare hipblaslt-bench command for microbenchmarking."""
    modify = [
        [r"/\[LogCmdConvolution\]/!d", log_file],
        [r"s/.*\.\/bin\///", log_file],
    ]

    for mod in modify:
        _ = run_subprocess_cmd(["sed", "-i"] + mod)


def run_single_hipblaslt_bench(log_file: str) -> float:
    """Run single hipblaslt-bench command and return microseconds from output."""
    with open(log_file, "r") as file:
        hipblaslt_bench_cmd = file.read().strip()

    result_stdout = run_subprocess_cmd(hipblaslt_bench_cmd, shell=True) # shell=True is a potential security risk (!)
    result_stdout_lines = result_stdout.split("\n")
    target_line = result_stdout_lines[-5]
    # ...,hipblaslt-Gflops,hipblaslt-GB/s,us
    bench_time_mean = float(target_line.split(",")[-1])

    return bench_time_mean


def run_single_miopen_driver(log_file: str) -> float:
    """Run single MIOpenDriver command and return microseconds from output"""
    with open(log_file, "r") as file:
        miopen_driver_cmd = file.read().strip()

    result_stdout = run_subprocess_cmd(miopen_driver_cmd, shell=True) # shell=True is a potential security risk (!)
    result_stdout_lines = result_stdout.split("\n")
    target_line = result_stdout_lines[-3]
    # ..., GFLOPs, GB/s, timeMs
    bench_time_mean = float(target_line.split(",")[-1]) * 1e3

    return bench_time_mean


def run_microbenchmarking(group: str, logging_envvars: Dict[str, str]) -> float:
    if group == "gemm":
        # Modify hipblaslt-bench.log command for microbenchmarking
        log_file = logging_envvars["HIPBLASLT_LOG_FILE"]
        prepare_hipblaslt_bench_cmd(log_file)

        # hipblaslt-bench time
        bench_time_mean = run_single_hipblaslt_bench(log_file)

    elif group == "conv":
        # Modify miopen-driver.log command for microbenchmarking
        log_file = logging_envvars["MIOPEN_LOG_FILE"]
        prepare_miopen_driver_cmd(log_file)

        # MIOpenDriver time
        bench_time_mean = run_single_miopen_driver(log_file)

    return bench_time_mean


def run_node_replay(group, df_ops_summary, base_path):
    if df_ops_summary is None:
        print(f"No events to replay from group {group}")
        return

    df_ops_summary_99pct = df_ops_summary[df_ops_summary["kernel_time_sum_cum_pct"] < 0.99]

    if df_ops_summary_99pct.empty:
        df_ops_summary_99pct = df_ops_summary

    print(df_ops_summary_99pct.loc[:, group2dfcols[group]].to_string(max_colwidth=75))

    df_results = None
    device = "cuda"
    args_cols = ["Input Dims_first", "Input type_first", "Input Strides_first", "Concrete Inputs_first"]

    for i in range(len(df_ops_summary_99pct)):
        event = dict(args=dict())
        row = dict(df_ops_summary_99pct.iloc[i])

        for col in row.keys():
            if col in args_cols:
                event["args"][col.split("_")[0]] = row[col]
            else:
                event[col] = row[col]

        replayer = EventReplayer(event, device="cuda")

        # Get group-specific logging environment variables
        logging_envvars = get_logging_envvars(group, i, base_path)

        # Log hipblaslt-bench/MIOpenDriver commands (pass envvars, no warmup, 1 active step)
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            args = [(replayer.replay, device, logging_envvars, 0, 1)]
            _ = list(executor.map(benchmark_func_wrapper, args))

        # Run microbenchmarking with hipblaslt-bench/MIOpenDriver
        bench_time_mean = run_microbenchmarking(group, logging_envvars)

        # Node replay time (no envvars, 1000 warmup and 1000 active steps)
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            args = [(replayer.replay, device, None, 1000, 1000)]
            result = list(executor.map(benchmark_func_wrapper, args))

        replay_time_mean = result[0]

        # Workload time from profile trace
        profile_time_mean = row["Kernel Time (µs)_mean"]

        # Record results
        diff_profile = replay_time_mean - profile_time_mean
        diff_bench = bench_time_mean - profile_time_mean

        percent_diff_profile = (replay_time_mean - profile_time_mean) / profile_time_mean * 100
        percent_diff_bench = (bench_time_mean - profile_time_mean) / profile_time_mean * 100

        result_dict = {
            f"{group} no.": i,
            f"Avg. {group}-bench time (us)": bench_time_mean,
            "Avg. replay time (us)": replay_time_mean,
            "Avg. profile time (us)": profile_time_mean,
            "Diff profile vs. replay (us)": diff_profile,
            f"Diff profile vs. {group}-bench (us)": diff_bench,
            "Diff profile vs. replay (pct)": percent_diff_profile,
            f"Diff profile vs. {group}-bench (pct)": percent_diff_bench,
        }

        df_results = pd.concat([df_results, pd.DataFrame([result_dict])]) if df_results is not None else pd.DataFrame([result_dict])

        del replayer
        gc.collect()

        time.sleep(1)

    df_results.set_index(f"{group} no.", inplace=True)
    return df_results


def run_standalone_node_replay(base_dirpath, rank_pattern="rank_", ext="json", include_only=["rank_0"], dry_run=False, xlsx_path=None):
    all_traces_grouped = parse_traces(base_dirpath, ext, include_only, rank_pattern)

    if all_traces_grouped is None:
        return

    for parent_dirpath, filenames in all_traces_grouped.items():
        if xlsx_path is not None and len(all_traces_grouped) > 1:
            print("Multiple parent directories with traces found, give a more specific base path for the report with custom Excel path.")
            return
        
        dfs_all = {group: None for group in group2ops}
        
        parent_dirname = osp.basename(parent_dirpath)
        prefix = "_".join([parent_dirname, *include_only])

        xlsx_path = xlsx_path or osp.join(parent_dirpath, f"{prefix}_node_replay_report.xlsx")

        print("==================== Creating node replay report ====================")
        print(f"Parent directory: {parent_dirpath}")
        print(f"Excel file: {osp.basename(xlsx_path)}")
        print(f"Filters: {', '.join(include_only)}")
        print("Traces:")
        print(*filenames, sep="\n")

        if dry_run:
            print("==================== End of dry run ====================")
            xlsx_path = None
            continue

        if osp.exists(xlsx_path):
            print(f"Excel file already exists: ({xlsx_path})")
            print("Terminating...")
            return

        for filename in filenames:
            filepath = osp.join(parent_dirpath, filename)

            print(f"Starting TreePerfAnalyzer with {filename}")
            perf_analyzer = TreePerfAnalyzer.from_file(filepath)

            # Collect group-specific perf metrics from single rank
            dfs_per_group = collect_df_perf_metrics_per_group(perf_analyzer, group2ops)
            for group, df in dfs_per_group.items():
                dfs_all[group] = pd.concat([dfs_all[group], df]) if dfs_all[group] is not None else df

            del perf_analyzer
            gc.collect()

        for group, df_ops in dfs_all.items():
            df_ops_summary = summarize_df_perf_metrics(df_ops, agg_metrics=["mean", "std"])
            df_ops_summary["kernel_time_sum_cum_pct"] = df_ops_summary["Kernel Time (µs)_sum"].cumsum() / df_ops_summary["Kernel Time (µs)_sum"].sum()
            df_results = run_node_replay(group, df_ops_summary, parent_dirpath)
            print(df_results.to_string())
            
            with pd.ExcelWriter(xlsx_path, mode="a" if osp.exists(xlsx_path) else "w") as writer:
                df_results.to_excel(writer, sheet_name=f"{group}_node_replay", index=True)
                
        xlsx_path = None
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Parse and summarize traces produced by torch profiler using TraceLens.")
    parser.add_argument("-b", type=str, required=True, help="Path to base directory which contains profiling experiments as subdirectories.")
    parser.add_argument("-p", type=str, default="rank_", help="Pattern to use for finding the rank of a trace from filename. Supports <string><sep> where separator can be empty, - or _.")
    parser.add_argument("-e", type=str, default="json", help="Extension to use for identifying trace files. json and gz are supported.")
    parser.add_argument("-f", type=str, nargs='+', default=["rank_0"], help="Select files containing given substring(s) in their name.")
    parser.add_argument("-d", action="store_true", help="Dry run for checking if correct trace paths found.")
    parser.add_argument("-o", type=str, default=None, help="Filepath to save the Excel node replay report. Note that this works only with a single base/parent directory containing one set of traces.")

    args = parser.parse_args()

    run_standalone_node_replay(args.b, args.e, args.f, args.d, args.o)


if __name__ == "__main__":
    main()