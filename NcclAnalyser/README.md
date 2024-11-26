# NCCL Analyser

In distributed deep learning, analyzing the performance of collective communication operations is crucial for diagnosing and optimizing **perfromance as we scale**. NCCL Analyser is a Python SDK designed to parse and analyze NCCL kernel events from trace files generated during distributed training or inference. By computing key metrics like **communication latency, message sizes, algorithm bandwidth, and bus bandwidth**, NCCL Analyser helps you gain insights into the communication patterns and potential bottlenecks in your distributed setups.

---

## Key Features

- **Detailed NCCL Event Analysis**: Extracts NCCL kernel events from trace files across multiple ranks. Computes communication latency for all-to-all events by taking the **minimum duration across ranks to eliminate waiting time**. Using this communication latency and message size, the tool computes algorithm bandwidth. Additionally, it calculates bus bandwidth based on the collective operation.

- **Summary Statistics**: Aggregates per-event data into summarized statistics grouped by **message size**, **dtype** and **collective operation**. These include summary metrics such as total latency, mean/min/max for communication duration, algorithm bandwidth, and bus bandwidth. 

- **Customizable File Mapping**: Offers flexible rank-to-file mapping to match your directory structure and naming conventions, making integration seamless.

- **PyTorch Support**: Currently built for PyTorch trace files, with the potential to extend support to other frameworks.

- **Lightweight and Simple**: Minimal dependencies and a straightforward codebase ensure the tool is easy to use and extend.

---

### Example Database

The NCCL Analyser can generate two types of databases: **detailed** and **summarized**.

1. **Detailed Database**: Contains records of each NCCL kernel event, including metadata and computed metrics such as communication latency, algorithm bandwidth, and bus bandwidth for every event across all ranks.

#### Example Detailed Database

| external_id | Collective name | In msg nelems | Out msg nelems | Group size | dtype | Process Group Name | comm latency (µs) | In msg size (MB) | algo bw (GB/s) | bus bw (GB/s) |
|-------------|-----------------|---------------|----------------|------------|-------|---------------------|-------------------|------------------|----------------|---------------|
| 14996150    | allreduce       | 16785408      | 16785408       | 32         | Float | default_pg          | 2996.022          | 64.031           | 20.871         | 40.438        |
| 14996653    | allreduce       | 16785408      | 16785408       | 32         | Float | default_pg          | 1425.172          | 64.031           | 43.876         | 85.010        |
| 14997156    | allreduce       | 16785408      | 16785408       | 32         | Float | default_pg          | 1367.510          | 64.031           | 45.726         | 88.594        |
| 14998162    | allreduce       | 16785408      | 16785408       | 32         | Float | default_pg          | 1420.680          | 64.031           | 44.014         | 85.277        |
| 15040802    | allreduce       | 10416000      | 10416000       | 32         | Float | default_pg          | 893.604           | 39.734           | 43.423         | 84.132        |

2. **Summarized Database**: Groups events by collective type, message size, and data type to compute aggregated metrics, including the mean values for communication latency, algorithm bandwidth, and bus bandwidth.

#### Example Summarized Database

| Collective name | In msg size (MB) | In_msg_nelems | Out_msg_nelems | Group size | dtype | count | Total latency (ms) | min_dur_mean | algo_bw_mean | bus_bw_mean |
|-----------------|------------------|---------------|----------------|------------|-------|-------|--------------------|--------------|--------------|-------------|
| allreduce       | 0.01            | 2530          | 2530           | 32         | Int   | 1     | 0.102              | 101.529      | 0.096        | 0.186       |
| allreduce       | 0.289           | 75808         | 75808          | 32         | Float | 1     | 0.210              | 210.321      | 1.342        | 2.600       |
| allreduce       | 25.142          | 6590752       | 6590752        | 32         | Float | 9     | 6.007              | 667.422      | 36.830       | 71.358      |
| allreduce       | 32.016          | 8392704       | 8392704        | 32         | Float | 26    | 26.585             | 1022.505     | 30.765       | 59.607      |
| broadcast       | 2.971           | 778752        | 778752         | 32         | Float | 1     | 0.026              | 25.943       | 111.836      | 111.836     |


--- 

## Quick Start

Follow these steps to use NCCL Analyser for analyzing NCCL kernel events:

### Example: Build and Save Detailed and Summary Databases

```python
from nccl_analyser import NcclAnalyser
import os

# Initialize NCCL Analyser
profiles_root_dir = '/path/to/profiles/'
world_size = 32
nccl_analyser = NcclAnalyser(profiles_root_dir, world_size)

# Define custom rank-to-file mapping (optional)
def rank2file(rank):
    return os.path.join(profiles_root_dir, f'pytorch_profile_rank{rank}_step120.json')
nccl_analyser.set_rank2file_fn(rank2file)

# Build and save the detailed database
output_db_path = os.path.join(profiles_root_dir, 'nccl_events_db.csv')
nccl_analyser.build_and_save_db(output_db_path)

# Build and save the summary database
summary_db_path = os.path.join(profiles_root_dir, 'nccl_summary_db.csv')
nccl_analyser.build_and_save_summary_db(summary_db_path)
```