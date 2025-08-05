# TraceDiff

TraceDiff is a Python API and tool within TraceLens for comparing two PyTorch Kineto trace files. It enables users to identify, visualize, and analyze the differences between two execution traces at the operation and kernel level. TraceDiff is especially useful for regression analysis, performance debugging, and understanding the impact of code or environment changes on GPU workloads.

---

## Key Features

- **Automated Tree Comparison**: Builds hierarchical event trees from two traces and identifies points of difference (PODs) using a recursive diff algorithm.
- **Tree Diff Visualization**: Produces a diff output file that highlights matched and unmatched operations between traces.
- **Detailed and Summary Reports**: Generates CSV reports with kernel time statistics and aggregated summaries for each operation.
- **UID Mapping**: Provides a mapping between event UIDs in the two traces, enabling cross-referencing and deeper analysis.
- **Seamless Integration**: Designed to work with TraceLens's TraceToTree objects and PyTorch profiler JSON traces.

---

## Quick Start


### Example: Compare Two Traces and Generate Reports

```python
from TraceLens import TraceToTree, TraceDiff
import json

# Load two trace files
trace_file1 = '/path/to/trace1.json'
trace_file2 = '/path/to/trace2.json'
with open(trace_file1, 'r') as f:
    trace_data1 = json.load(f)
with open(trace_file2, 'r') as f:
    trace_data2 = json.load(f)

# Build event trees
events1 = trace_data1['traceEvents']
events2 = trace_data2['traceEvents']
tree1 = TraceToTree(events1)
tree1.build_tree()
tree2 = TraceToTree(events2)
tree2.build_tree()

# Compare and analyze
td = TraceDiff(tree1, tree2)
td.generate_tracediff_report()  # Generates DataFrames, does NOT write files
td.print_tracediff_report_files('rprt_diff')  # Writes all reports to files in 'rprt_diff/'
```



**Output files:**
- `rprt_diff/merged_tree_output.txt`: Text visualization of the merged tree, showing matched and unmatched nodes.
- `rprt_diff/diff_stats.csv`: Detailed kernel and op statistics for each operation (see below for example and explanation).
- `rprt_diff/diff_stats_summary.csv`: Aggregated summary statistics by op name and input shape (see below for example and explanation).

---


## Output File Examples and Interpretation

### diff_stats_summary.csv

This file provides a summary of kernel time and statistics for each operation, grouped by op name and input shape. It is useful for quickly comparing performance between two traces at a high level.

**Example (first two rows):**

| name           | input_shape_trace1                                      | input_shape_trace2                                      | total_kernel_time_trace1 | avg_kernel_time_trace1 | total_kernel_time_trace2 | avg_kernel_time_trace2 | kernel_names_trace1                | kernel_names_trace2                |
|----------------|--------------------------------------------------------|--------------------------------------------------------|-------------------------|------------------------|-------------------------|------------------------|-------------------------------------|-------------------------------------|
| FlashAttnFunc  | [[5,2048,32,64],[5,11040,32,64],...]                   | [[5,2048,32,64],[5,11040,32,64],...]                   | 3655.85                | 3655.85                | 2431.72                | 2431.72                | void flash_fwd_kernel<...           | void ck_tile::kentry<...           |
| FlashAttnFunc  | [[5,2048,32,64],[5,2048,32,64],...]                    | [[5,2048,32,64],[5,2048,32,64],...]                    | 16351.01               | 681.29                 | 11369.84                | 473.74                 | void flash_fwd_kernel<...;...        | void ck_tile::kentry<...;...        |

**Key columns:**
- `name`: The operation name (e.g., FlashAttnFunc)
- `input_shape_trace1/2`: Input shapes for the op in each trace
- `total_kernel_time_trace1/2`: Total time spent in kernels for this op in each trace (microseconds)
- `avg_kernel_time_trace1/2`: Average kernel time per instance
- `kernel_names_trace1/2`: Semicolon-separated kernel names launched by this op

**How to use:**
- Compare `total_kernel_time` and `avg_kernel_time` between traces to spot regressions or improvements.
- Compare how arguments such as input shapes and data types change across backends.
- Differences in `kernel_names` may indicate kernel fusion, codegen, or backend changes.

### diff_stats.csv

This file contains detailed statistics for every op instance, including input shapes, types, kernel times, and kernel names. It is useful for fine-grained analysis and debugging.

**How to use:**
- Drill down to individual op instances to investigate outliers or mismatches.
- Use the detailed input and kernel info to correlate with model code or trace events.

---

---


## Accessing DataFrames and UID Mapping

TraceDiff provides methods to access the detailed and summary DataFrames directly, as well as a `merged_uid_map` to cross-reference events between the two traces. This is useful for linking statistics or visualizations.

### Accessing DataFrames

```python
# After running td.generate_tracediff_report():
df = td.get_diff_stats_df()  # Detailed per-op DataFrame
df_summary = td.get_diff_stats_summary_df()  # Summary DataFrame
if df is not None:
    print(df.head())
if df_summary is not None:
    print(df_summary.head())
```

### UID Mapping Example

```python
# Get the corresponding UID in tree2 for a given UID in tree1
uid1 = next(iter(td.baseline.cpu_root_nodes))
uid2 = td.get_corresponding_uid(1, uid1)
if uid2 != -1:
    print(f"Tree1 UID {uid1} corresponds to Tree2 UID {uid2}")
else:
    print("No match found for this UID in tree2.")
```

---


## Use Cases

- **Performance Regression Analysis**: Quickly identify which operations or kernels have changed between two runs.
- **Debugging and Optimization**: Pinpoint new bottlenecks or regressions introduced by code or environment changes.
- **Cross-Trace Linking**: Map and compare specific events or kernels between two traces for deeper investigation.

---


## Notes

- TraceDiff is designed for PyTorch profiler JSON traces and requires TraceToTree objects as input.
- For more advanced usage, see the example notebook: `examples/trace_diff_example.ipynb`.
- Output folder and file names can be customized via the API.
- The API now separates report generation (`generate_tracediff_report`) from file output (`print_tracediff_report_files`).
- DataFrames are only available after running `generate_tracediff_report()`.
