# ComparativeMode: Getting Started Guide

> **ComparativeMode** is an AI-powered GPU performance analysis tool that provides comprehensive gap analysis for different GPU architectures (e.g., AMD MI300, MI325, MI350 ... ) with actionable optimization recommendations.

---

## 📋 Table of Contents
0. [Cluster Access](#Cluster)
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [Running Jarvis Analysis](#running-jarvis-analysis)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)

---

## Cluster

For now, access to AMD Network GPU Cluster (ALOLA Cluster) is required to run the LLM based analysis.

Follow this link to get access to the ALOLA Cluster: https://amd.atlassian.net/wiki/spaces/MLSE/pages/897801231/AGS+cluster+user+resources

## LLM Gateway Access

You need Gateway api key to run LLM. If you don't have one, feel free to reach out to me.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Git** installed
- **pip** or **uv** package installer
- Access to GPU trace files (Kineto JSON format)
- **API Key** for AI analysis (OpenAI/Azure/Anthropic)
  - Set environment variable: `export AMD_LLM_API_KEY='your-api-key-here'`

---

## Environment Setup

### Option 1: Using Python Virtual Environment (Recommended)

```bash
# Navigate to the project directory
cd TraceLens/AgenticMode/Comparative

# Create a Python virtual environment
python3 -m venv gpu_analysis_env

# Activate the virtual environment
source gpu_analysis_env/bin/activate

# Verify activation (should show the venv path)
which python3
```

### Option 2: Using Docker

```bash
# Start a Docker container
bash start_docker.sh

# Or manually start a plain Docker container
docker run -it --rm -v $(pwd):/workspace python:3.10 bash
cd /workspace
```

---

## Installation

### Step 1: Install Dependencies

```bash
# Run the requirements installation script
bash requirements.sh
```

**Important** 

Set your AMD-LLM-gateway API key in jarvis-analysis.sh.


**Verify installation:**

```bash
python3 -c "import TraceLens; print('TraceLens installed successfully!')"
```

---

## Running Jarvis Analysis

### Quick Test Run

```bash
# Run the test script (uses sample traces)
bash jarvis-analysis.sh
```

### Custom Analysis

Modify jarvis-analysis.sh accordingly to run on your custom traces.

### Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--target-gpu-kineto` | Kineto trace file for target GPU | ✅ Yes |
| `--target-gpu-et` | Execution trace file for target GPU | ✅ Yes |
| `--baseline-gpu-kineto` | Kineto trace file for baseline GPU | ✅ Yes |
| `--baseline-gpu-et` | Execution trace file for baseline GPU | ✅ Yes |
| `--target-gpu-name` | Name/label for Target GPU (e.g., "H200") | ✅ Yes |
| `--baseline-gpu-name` | Name/label for Baseline GPU (e.g., "MI300") | ✅ Yes |
| `--api-key` | API key for AI analysis | ✅ Yes |
| `--output-dir` | Directory for output reports and plots | Optional (default: current dir) |
| `--enable-plots` | Generate visualization plots | Optional |
| `--disable-critical-path ` | If don't have ET traces, disable the critical path analysis and use Timeline based analysis | Optional | * Currently cricital path is a work in progress, so it's best to keep it disabled.

---

## Understanding the Output

After running Jarvis analysis, you'll find the following outputs:

### Generated Files

The output directory contains multiple files with analysis of different levels of granularity. 
To start with, we recommend downloading three files from the output directory specified in jarvis-analysis.sh:

- output_dir/cumulative_projection_parent_module.png
- output_dir/optimization_opportunities_interactive.html
and 
- output_dir/lca_analysis/kernel_optimization_interactive.html

The first two files will give a broad overview of optimization opportunities when the projection is done on the nn.Module level.
The system identifies pytorch nn.Modules that lag behind the target and gives extra information on the kernels involved, and provides additional AI-based comments.

LCA (least common ancestor) analysis provides a much more fine-grained prediction, based on structural matching of the two traces. This results in more optimistic prediction and a more detailed breakdown of which kernels should be optimized.

The goal is to identify the mapping between the smallest functionally equivalent groups of kernels between the two traces, accommodating for corner cases like fusion etc.

One note of caution: the LCA analysis uses slightly different aggregation that at present doesn't fully account for parallel execution, and hence the overall time might differ between module-based and LCA analysis results. Still, we believe the optimization opportunities identified with LCA analysis should be the most reliable.  


## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Ensure TraceLens is installed
pip install -e TraceLens/

# Verify installation
python3 -c "import TraceLens"
```

### Issue: API key errors

**Solution:**
```bash
Follow instructions in jarvis-analysis.sh to properly set the AMD Gateway LLM api Key.
```

### Issue: Trace files not found

**Solution:**
```bash
# Verify trace files exist
ls -lh complete_traces/crit_path_traces/H200/
ls -lh complete_traces/crit_path_traces/MI300/

# Use absolute paths in the command
--gpu1-kineto /full/path/to/trace.json
```

### Issue: Permission denied when running scripts

**Solution:**
```bash
# Make scripts executable
chmod +x jarvis-analysis.sh
chmod +x requirements.sh

# Run with bash explicitly
bash jarvis-analysis.sh
```

---

## Advanced Usage

### Using Custom Traces

The simplest way is to modify jarvis-analysis.sh and provide sharepoint links to your baseline and target traces. However local paths should work as well. 

---

## Next Steps

After successful setup:

1. ✅ Run the test analysis: `bash test-jarvis-analysis.sh`
2. ✅ Review the generated reports in `jarvis_test_plots/`
3. ✅ Try analyzing your own GPU traces
4. ✅ Explore optimization recommendations
5. ✅ Implement suggested optimizations and compare results

**Happy Analyzing! 🚀**

## Upcoming Features

In the future the user will be able to get the following features:

1. Using Roofline as Target and Single GPU traces as Baseline. This part is in progress and the support will be added soon
2. MIDAS Roofline as GEMM Target
3. Local LLM for report generations to avoid cluster bounding and gatway-api key access.

**Note**
This tool is veru much work in progress. In case something goes wrong, you feedback will be highly appreciated.
