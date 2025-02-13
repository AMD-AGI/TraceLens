# iLoveTrace
iLoveTrace is a Python library focused on **automating analysis from trace files** and enabling rich performance insights. Designed with **simplicity and extensibility** in mind, this library provides tools to simplify the process of profiling and debugging complex distributed training and inference systems.

🚨 **Alpha Release**: iLoveTrace is currently in its Alpha stage. This means the core features are functional, but the software may still have bugs. Feedback is highly encouraged to improve the tool for broader use cases!

### Overview

The library currently includes three tools:

- **TraceFusion** : Merges distributed trace files for a global view of events across ranks.
- **NcclAnalyser**: Analyzes collective communication operations to extract key metrics like communication latency and bandwidth.
- **Trace2Tree**: Parses trace files into a hierarchical tree intermediate representation (IR) that maps CPU operations to GPU kernels. 
- **TreePerf**: Uses the tree IR from Trace2Tree to compute detailed performance metrics such as TFLOPS/s, FLOPS, FLOPS/Byte, and GPU execution times. 

## Installation


1. (Optional) Create virtual environment: `python3 -m venv .venv`
2. (Optional) Activate the virtual environment: `source .venv/bin/activate`
3. Install the package `pip install .`


### Quick start
Each tool in iLoveTrace is modular, with its own documentation and usage instructions. To get started with any tool navigate to the respective tool's directory and follow the detailed README in the tool's directory for usage instructions.
