# Roofline Analysis Tool

A functional programming-oriented Python application for performing roofline analysis on GEMM kernels. This tool ingests Excel files containing kernel performance data, performs calculations to determine memory and compute bounds, and exports the results with visualizations.

## Overview

The Roofline Model is a visually intuitive performance model used to provide performance estimates of applications running on multicore, manycore, or accelerator processor architectures. This tool helps analyze the performance of GEMM (General Matrix Multiplication) kernels by:

1. Reading kernel performance data from Excel files
2. Calculating roofline model parameters
3. Determining if kernels are memory or compute bound
4. Visualizing the results with a roofline plot
5. Exporting the analysis to a new Excel file

## Installation

### Requirements

- Python 3.8 or higher
- Dependencies listed in pyproject.toml

### Installation Steps

1. Clone this repository:
```bash
git clone <repository-url>
cd path/to/roofline-analysis
```
2. Install the package
```bash
pip install -e .
```
## Configuration
The tool requires a configuration file in TOML format. Create a config.toml file with the following sections:
```toml
[accelerator]
max_memory_bandwidth = 1.5  # TB/s
max_compute_teraflops = 19.5  # TFLOPS
max_achievable_teraflops = 16.0  # TFLOPS

[excel]
sheet_name = "gemm"
flops_per_byte_column = "FLOPS/byte"
performance_column = "Non-Data-Mov TFLOPS/s_mean"

[output]
prefix = "export-roofline"
```

### Configuration Options

* **accelerator**: Hardware parameters
    * **max_memory_bandwidth**: Maximum memory bandwidth in TB/s
    * **max_compute_teraflops**: Maximum theoretical compute throughput in TFLOPS
    * **max_achievable_teraflops**: Maximum achievable compute throughput in TFLOPS
* **excel**: Excel file configuration
    * **sheet_name**: The name of the worksheet containing kernel data
    * **flops_per_byte_column**: Column name containing arithmetic intensity values
    * **performance_column**: Column name containing kernel performance values
* **output**: Output configuration
    * **prefix**: Prefix for the output Excel file name

## Usage
Basic Command
```bash
python main.py <excel-file> <config-file>
```
Command-line Options
```bash
Options:
  --plot-output PATH  Path to save a separate copy of the roofline plot
  --skip-plot         Skip generating the roofline plot
  --help              Show this message and exit.
```
IRL Example
```bash
python main.py mi300x_013_profile_output_5_steps_step_10_performance_report.xlsx config.toml
```
### Input Requirements
The Excel file should contain a worksheet (default: "gemm") with at least the following columns:

1. A column for arithmetic intensity (FLOPS/byte)
2. A column for kernel performance (TFLOPS/s)

The exact column names are specified in the config.toml file.

### Output Description
The tool generates an Excel file with the following content:

1. {sheet_name}_analyzed: The original data with additional calculated columns:
    * **kernel_memory_roofline**: Memory bandwidth limit for each kernel
    * **bound_type_maf**: Whether the kernel is "memory" or "compute" bound
* **bound_distance**: Distance to the nearest roofline
* **bound_distance_pct**: Percentage distance to the nearest roofline
2. **{sheet_name}_original**: A copy of the original data

3. **ScalarValues**: Key calculated values including:
    * **AI_ridge_mtf**: The arithmetic intensity ridge point based on max theoretical FLOPS
    * **AI_ridge_maf**: The arithmetic intensity ridge point based on max achievable FLOPS
4. **RooflinePlot**: A visual representation of the roofline model with:
    * Memory bandwidth roofline
    * Max theoretical compute roofline
    * Max achievable compute roofline
    * Kernel data points

## Project Structure
The project follows a functional programming approach with the following structure:

* **main.py**: The main entry point
* **src/cli.py**: Command line interface and config handling
* **src/xlsx.py**: Excel file processing functions
* **src/roofline.py**: Roofline analysis calculations
* **pyproject.toml**: Project metadata and dependencies
* **config.toml**: Example configuration

## License
Copyright AMD 2025.