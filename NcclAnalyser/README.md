# NCCL Analyser

In distributed deep learning, analyzing the performance of collective communication operations is crucial for diagnosing and optimizing **perfromance as we scale**. NCCL Analyser is a Python SDK designed to parse and analyze NCCL kernel events from trace files generated during distributed training or inference. By computing key metrics like **communication latency, message sizes, algorithm bandwidth, and bus bandwidth**, NCCL Analyser helps you gain insights into the communication patterns and potential bottlenecks in your distributed setups.

---

## Key Features

- **Detailed NCCL Event Analysis**: Extracts NCCL kernel events from trace files across multiple ranks. Computes communication latency for all-to-all events by taking the **minimum duration across ranks to eliminate waiting time**. Using this communication latency and message size, the tool computes algorithm bandwidth. Additionally, it calculates bus bandwidth based on the collective operation.

- **Summary Statistics**: Aggregates per-event data into summarized statistics grouped by **message size**, **dtype** and **collective operation**. These include summary metrics such as total latency, mean/min/max for communication duration, algorithm bandwidth, and bus bandwidth. 


- **PyTorch Support**: Currently built for PyTorch trace files, with the potential to extend support to other frameworks.

---

### Example

The NCCL Analyser can generate two types of dataframes: **detailed** and **summarized**.

1. **Detailed Dataframe**: Contains records of each NCCL kernel event, including metadata and computed metrics such as communication latency, algorithm bandwidth, and bus bandwidth for every event across all ranks.

#### Example Detailed Dataframe

| external_id | Collective name      | dtype    | comm latency (µs) | In msg size (MB) | algo bw (GB/s) | bus bw (GB/s) |
|-------------|----------------------|----------|-------------------|------------------|----------------|---------------|
| 1954        | _allgather_base       | BFloat16 | 6031.856          | 204.0039         | 33.0284        | 28.8999       |
| 20483       | _allgather_base       | BFloat16 | 5916.243          | 204.0039         | 33.6738        | 29.4646       |
| 36421       | _reduce_scatter_base  | Float    | 11649.693         | 3264.0625        | 273.6176       | 239.4154      |
| 49991       | _reduce_scatter_base  | Float    | 9934.89           | 3264.0625        | 320.8451       | 280.7395      |
| 38421       | _reduce_scatter_base  | Float    | 11638.397         | 3264.0625        | 273.8832       | 239.6478      |
| 11483       | _allgather_base       | BFloat16 | 5913.518          | 204.0039         | 33.6893        | 29.4782       |
| 2536        | _allgather_base       | BFloat16 | 6131.004          | 204.0039         | 32.4943        | 28.4325       |
| 43676       | _allgather_base       | BFloat16 | 6012.867          | 204.0039         | 33.1327        | 28.9911       |
| 16283       | _allgather_base       | BFloat16 | 5881.311          | 204.0039         | 33.8738        | 29.6396       |
| 13083       | _allgather_base       | BFloat16 | 6092.666          | 204.0039         | 32.6988        | 28.6114       |


2. **Summarized Dataframe**: Groups events by collective type, message size, and data type to compute aggregated metrics, including the mean values for communication latency, algorithm bandwidth, and bus bandwidth.

#### Example Summarized Dataframe

| Collective name      | In msg size (MB) | dtype    | comm latency (µs)_mean | count | Total latency (ms) | algo bw (GB/s)_mean | bus bw (GB/s)_mean |
|----------------------|------------------|----------|------------------------|-------|---------------------|---------------------|---------------------|
| _allgather_base      | 204.0039         | BFloat16 | 6041.878               | 318   | 1921.3172           | 33.0018             | 28.8766             |
| _reduce_scatter_base | 3264.0625        | Float    | 11662.7668             | 160   | 1866.0427           | 273.4303            | 239.2515            |
| _reduce_scatter_base | 8016.0312        | Float    | 22988.503              | 2     | 45.977              | 340.5255            | 297.9598            |
| _allgather_base      | 501.002          | BFloat16 | 11920.8385             | 2     | 23.8417             | 41.0427             | 35.9124             |
| allreduce            | 0                | Float    | 18.5802                | 6     | 0.1115              | 0.0002              | 0.0004              |

Note that the last row in msg size is just rounded down to 0. 


--- 

## Quick Start

Follow these steps to use NCCL Analyser for analyzing NCCL kernel events:

### Example: Build and Save Detailed and Summary Dataframes

```python
from nccl_analyser import NcclAnalyser
import os

# Initialize NCCL Analyser
root_dir = '/path/to/dir'
world_size = 8
# Modify the following to pass your list of filepaths for the profiles
# We need all ranks for Nccl Analysis
list_profile_filepaths = [os.path.join(root_dir, f'rank{i}_trace.json') for i in range(world_size)]
output_db_path = os.path.join(root_dir, 'nccl_events_df.csv')
summary_db_path = os.path.join(root_dir, 'nccl_summary_df.csv')
nccl_analyser = NcclAnalyser(list_profile_filepaths, world_size)

# Build and save the detailed dataframe
df_nccl = nccl_analyser.build_df_nccl()
df_nccl.to_csv(output_db_path, index=False)

# Build and save the summary dataframe
df_nccl_summary = nccl_analyser.summarize_df_nccl(df_nccl)
df_nccl_summary.to_csv(summary_db_path, index=False)
```
**Modify the filepaths for your profiles and the output filepaths in example.py and get analysis instantly!**
