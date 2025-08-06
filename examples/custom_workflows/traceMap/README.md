## Overview

This tool generates interactive HTML dashboards for analyzing PyTorch profiler trace files (`.pt.trace.json.gz`) from vLLM workloads. It provides visualization capabilities for GPU kernel profiling and performance analysis.
<img width="2022" height="1265" alt="image" src="https://github.com/user-attachments/assets/d5571795-8e0d-4d57-988a-759aba2d96c6" />

### Features
- Interactive HTML Dashboard: Generate a standalone HTML report to zoom, pan, and inspect individual kernel execution events.
- Side-by-Side Trace Comparison: Compare two trace files to easily spot regressions or improvements in kernel-level performance.
- Lightweight and Portable: Outputs a self-contained HTML file viewable in any modern browser. You can open it in your phone, laptop and tablet. 


## Prerequisites

- **OS**: Linux 
- **Python**: 3.10 - 3.12
- vLLM profiling refer to https://docs.vllm.ai/en/v0.5.5/dev/profiling/profiling_index.html
## Environment Setup

### Option 1: Using venv (Recommended)

1. **Create virtual environment:**
   ```bash
   python3 -m venv tracemap
   source tracemap/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install bokeh pandas numpy
   ```

## Usage

### Basic Usage

Run the profiling dashboard generator with default trace files:

```bash
python3 main.py
```

### Custom Trace Files

Specify your own trace files for comparison:

```bash
python3 main.py \
    --trace1 /path/to/first_trace.pt.trace.json.gz \
    --trace2 /path/to/second_trace.pt.trace.json.gz \
    --name1 "trace1 name" \
    --name2 "trace2 name" \
    --output custom_dashboard.html
```

### Command Line Arguments

- `--trace1`: Path to first trace file (default: `./trace_file/examples/trace1.pt.trace.json.gz`)
- `--trace2`: Path to second trace file (default: `./trace_file/examples/trace2.pt.trace.json.gz`)
- `--name1`: Name for first trace (default: `Trace_A`)
- `--name2`: Name for second trace (default: `Trace_B`)
- `--output`: Output HTML file name (default: `gpu_trace_profiling.html`)

## Output

The tool generates an interactive HTML dashboard that includes:
- GPU kernel execution timelines
- Performance comparisons between different traces
- Interactive visualizations for detailed analysis
- Summary statistics and profiling metrics

Open the generated HTML file in your web browser to explore the profiling results.
