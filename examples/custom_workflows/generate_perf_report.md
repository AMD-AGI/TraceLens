# Enhanced Performance Report Generator

[generate_perf_report.py](./generate_perf_report.py) script produces a performance report in Excel format from traces produced by torch profiler. Works for both single- and multi-GPU traces.

Traces are located recursively and by default, the outputs are saved in the same directory where the traces are located.

## Arguments

```
Flag    Type        Required    Default      Description
-b      str         True                     Path to base directory which contains profiling experiments as subdirectories
-p      str         False       "rank_"      Pattern to use for finding the rank of a trace from filename. Supports <string><sep> where separator can be empty, - or _
-e      str         False       "json"       Extension to use for identifying trace files. json and gz are supported
-f      list<str>   False       ["rank_0"]   Select files containing given substring(s) in their full filepaths
-r      bool        False       False        Run node replay for GEMMs and CONVs that contribute 99pct to group-specific execution time
-d      bool        False       False        Dry run for checking if correct trace paths are found
-a      bool        False       False        Save all individual kernels from all ranks (sheets kernels_0 ... kernels_n). Produces a lot of data
-o      str         False       None         Filepath to save the Excel performance report. Note that this works only with a single base/parent directory containing one set of traces
```

## How to use

Following directory tree is an example from a 8-GPU distributed inference setup where traces have been produced for 4 different configurations.

```
profiles
    013_profile_544p_bs1
        traces_rank_0_step_8.json
        traces_rank_1_step_8.json
        ...
        traces_rank_7_step_8.json
    013_profile_720p_bs1
        traces_rank_0_step_8.json
        traces_rank_1_step_8.json
        ...
        traces_rank_7_step_8.json
    014_profile_544p_bs1
        traces_rank_0_step_8.json
        traces_rank_1_step_8.json
        ...
        traces_rank_7_step_8.json
    014_profile_720p_bs1
        traces_rank_0_step_8.json
        traces_rank_1_step_8.json
        ...
        traces_rank_7_step_8.json
```

Generate performance reports for all configurations in one go using traces from all ranks:

```
python <path to generate_perf_report.py> -b results -f step_8
```

Then following files will be generated:

```
profiles/013_profile_544p_bs1/013_profile_544p_bs1_step_8_performance_report.xlsx
profiles/013_profile_720p_bs1/013_profile_720p_bs1_step_8_performance_report.xlsx
profiles/014_profile_544p_bs1/014_profile_544p_bs1_step_8_performance_report.xlsx
profiles/014_profile_720p_bs1/014_profile_720p_bs1_step_8_performance_report.xlsx
```

Generate performance report for one configuration using traces from all ranks:

```
# User-defined excel paths can be passed via -o flag

# Narrow down search using base path
python <path to generate_perf_report.py> -b results/013_profile_544p_bs1 -f step_8

# Narrow down search using filters
python <path to generate_perf_report.py> -b results -f 013_profile_544p_bs1 step_8
```

Generate performance reports for single ranks:

```
python <path to generate_perf_report.py> -b results -f rank_0 step_8
```

Then following files will be generated:

```
profiles/013_profile_544p_bs1/013_profile_544p_bs1_rank_0_step_8_performance_report.xlsx
profiles/013_profile_720p_bs1/013_profile_720p_bs1_rank_0_step_8_performance_report.xlsx
profiles/013_profile_544p_bs1/013_profile_544p_bs1_rank_0_step_8_performance_report.xlsx
profiles/013_profile_720p_bs1/013_profile_720p_bs1_rank_0_step_8_performance_report.xlsx
```

## Additional use cases


### Node replay

The script also supports the node replay feature for GEMMs and CONVs via `-r` flag, in which case:

- Node replay is used for running high-level microbenchmarking with torch for GEMMs and CONVs that contribute 99pct to the group-specific execution time
- hipblaslt-bench is used for running low-level microbenchmarking for the GEMMS
- MIOpenDriver is used for running low-level microbenchmarking for the CONVs
- The high- and low-level microbenchmarking results are collected and compared against metrics calculated from the actual workload
- The identified and benchmarked GEMMs and CONVs are saved to their individual files for later possible use, e.g. tuning