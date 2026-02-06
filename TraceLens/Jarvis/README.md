# TraceLens Agentic Mode

> **⚠️ Experimental**: This feature is under active development and may change.

TraceLens Agentic Mode is a Cursor-based AI-powered performance analysis tool that uses TraceLens to analyze PyTorch profiler traces and generate actionable optimization recommendations.

---

## Prerequisites

### 1. Clone and checkout the Jarvis branch

```bash
git clone https://github.com/AMD-AGI/TraceLens.git
cd TraceLens
git checkout experimental/cursor-jarvis-orchestrator
```

### 2. Install TraceLens inside your container

SSH into your node and exec into the container where your traces reside:

```bash
ssh <node>
docker exec -it <container> bash
```

Install TraceLens:

```bash
cd /path/to/TraceLens
pip install -e .
```

---

## Quick Start - How to Use

### To run performance analysis:

1. **In a Cursor chat, invoke:**
   ```
   @standalone-analysis-orchestrator
   ```

2. **Provide when prompted:**
   - Trace file path
   - Platform (MI300X/MI325X/MI355X/MI400)
   - Node name
   - Container name
   - Output directory (optional)

3. **Get results:**
   - `standalone_analysis.md` - Stakeholder report with prioritized recommendations
   - `category_findings/` - Per-category detailed analysis

---

## Output Files

```
analysis_output/
├── standalone_analysis.md          # Stakeholder report
├── category_manifest.json          # Category metadata and GPU utilization
├── perf_report.xlsx                # Excel performance report
├── perf_report_csvs/               # CSV exports (gpu_timeline, ops_summary, etc.)
├── category_data/                  # Per-category CSVs, metrics JSONs, tree data
│   ├── *_ops.csv
│   ├── *_metrics.json
│   └── *_tree_data.json
├── category_findings/              # Per-category analysis (markdown)
│   └── *_findings.md
├── metadata/                       # Category metadata JSONs
│   └── *_metadata.json
└── replay_packages/                # Kernel replay artifacts (optional)
```

---

## Architecture

### Orchestrator
The **Standalone Analysis Orchestrator** skill coordinates the entire analysis workflow.
It queries user inputs, runs TraceLens to pre-compute trace data, and invokes category-specific sub-agents in parallel. Finally, it aggregates findings, generates a prioritized stakeholder report.

### Workflow Steps

```
0.  Query User Inputs (Platform, Trace Path, Node, Container)
1.  Generate Performance Report (perf_report.xlsx + CSVs)
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Category Filtering)
6.  Invoke Category-Specific Subagents (PARALLEL)
7.  Aggregate Results and Determine Optimization Recommendations
8.  Generate Replay Artifacts (optional)
9.  Generate Final Report (standalone_analysis.md)
```

### Continual Learning
After an analysis run, if you identify a missed issue, invoke the **Continual Learning** skill to update the relevant sub-agent's pattern library. It proposes minimal, append-only additions to the "Common Patterns" section of the appropriate analyzer so future runs catch similar issues automatically.

### Sub-Agents
| Agent | Purpose |
|-------|---------|
| `cpu-idle-analyzer` | Analyzes GPU idle time and CPU bottlenecks |
| `gemm-analyzer` | Analyzes matrix multiplication operations |
| `sdpa-analyzer` | Analyzes scaled dot-product attention |
| `elementwise-analyzer` | Analyzes elementwise operations |
| `reduce-analyzer` | Analyzes reduction operations |
| `triton-analyzer` | Analyzes Triton-compiled kernels |
| `moe-analyzer` | Analyzes MoE operations |
| `batchnorm-analyzer` | Analyzes batch normalization |
| `convolution-analyzer` | Analyzes convolution operations |
| `generic-op-analyzer` | Analyzes uncategorized operations |