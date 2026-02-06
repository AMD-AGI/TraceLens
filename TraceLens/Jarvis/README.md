# TraceLens Jarvis - Agentic Analysis Framework

> **⚠️ Experimental**: This feature is under active development and may change.

JARVIS is an AI-powered performance analysis agent that uses TraceLens to analyze PyTorch profiler traces and generate actionable optimization recommendations. 

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
   - Cluster name
   - Container name
   - Output directory (optional)

3. **Get results:**
   - `standalone_analysis.md` - Stakeholder report
   - Category-specific findings in `category_findings/`

---

## Architecture

### Orchestrator
The **Standalone Analysis Orchestrator** skill coordinates the entire analysis workflow.
It queries user inputs, runs TraceLens to pre-compute trace data, and invokes category-specific sub-agents in parallel. Finally, it aggregates findings and generates a prioritized stakeholder report with optimization recommendations.

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