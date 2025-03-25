# TraceLens
TraceLens is a Python library focused on **automating analysis from trace files** and enabling rich performance insights. Designed with **simplicity and extensibility** in mind, this library provides tools to simplify the process of profiling and debugging complex distributed training and inference systems.

🚨 **Alpha Release**: TraceLens is currently in its Alpha stage. This means the core features are functional, but the software may still have bugs. Feedback is highly encouraged to improve the tool for broader use cases!

### Overview

The library currently includes three tools:

- **Trace2Tree and TreePerf**: 
    - Trace2Tree Parses trace files into a hierarchical **call stack tree** intermediate representation (IR) that maps CPU operations to GPU kernels. 
    - TreePerf uses the tree IR from Trace2Tree to compute detailed **performance metrics** such as TFLOPS/s, FLOPS, FLOPS/Byte, and GPU execution times. 
- **NcclAnalyser**: Analyzes collective communication operations to extract key metrics like communication latency, bandwidth and sync metrics.
- **TraceFusion** : Merges distributed trace files for a global view of events across ranks in PerfettoUI. 

## Installation


1. (Optional) Create virtual environment: `python3 -m venv .venv`
2. (Optional) Activate the virtual environment: `source .venv/bin/activate`
3. Install the package `pip install .`


### Quick start
Each tool in TraceLens has documentation and examples. To get started with any tool navigate to the respective tool's docs markdown file and then to the example. 

### What's New in v0.2.x
- **NCCL Analyzer Upgrade**: Improved robustness, supports collectives on subset of world size and handles asymmetric activity across ranks
- **API Changes**:  
  - `TreePerfAnalyser.from_file(file)` replaces `TreePerfAnalyser(file)`.  
  - NCCL Analyzer API updated for clarity (see docs & notebooks).  

### What's New in v0.3.x
- **NN Module View**: Visualize the nn module hierarchy with the GPU time spent in each module. This is useful for performance aware architecture design.
- **Perf Model**: New ops support including unary and binary elementwise ops.
- **Jax Support for GPUEventAnalyser**: Get compute-communication-memcpy and overlap metrics for Jax profiles. Thanks to @gabeweisz for the contribution!

Bug Fixes: Allgather incorrect msg size calculation fixed.

Check out the example notebooks for details! 🚀  

