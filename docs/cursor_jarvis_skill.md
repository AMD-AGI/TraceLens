# JARVIS Performance Analysis - Cursor AI Skill

> **⚠️ EXPERIMENTAL - NOT OFFICIAL**
> 
> This is a highly experimental feature and is **not officially supported**. Use at your own risk. The skill methodology, output format, and behavior may change significantly without notice. Feedback and contributions are welcome!

JARVIS is an AI-powered performance analysis agent that uses TraceLens to analyze PyTorch profiler traces and generate actionable optimization recommendations.

## What is a Cursor Skill?

[Cursor](https://cursor.com) is an AI-powered code editor. **Skills** are modular instruction files that teach Cursor's AI agent domain-specific workflows. When you ask Cursor to analyze a trace, the JARVIS skill provides the methodology, commands, and best practices automatically.

## Requirements

- **Cursor IDE** version 2.4 or later (Skills feature added Jan 22, 2026)
- **TraceLens** installed in your Python environment

## Quick Setup

### Option 1: Copy to Your Workspace (Recommended)

If you're working in a specific project directory:

```bash
# Create the skills directory in your workspace
mkdir -p .cursor/skills

# Copy the JARVIS skill
cp /path/to/TraceLens/.cursor/skills/jarvis-perf-analysis.md .cursor/skills/
```

### Option 2: Global Installation

To make JARVIS available across all your workspaces:

```bash
# Create global skills directory
mkdir -p ~/.cursor/skills

# Copy the JARVIS skill
cp /path/to/TraceLens/.cursor/skills/jarvis-perf-analysis.md ~/.cursor/skills/
```

### Option 3: One-Liner from TraceLens Repo

```bash
# From anywhere, if TraceLens is installed
TRACELENS_PATH=$(python -c "import TraceLens; import os; print(os.path.dirname(os.path.dirname(TraceLens.__file__)))")
mkdir -p ~/.cursor/skills && cp "$TRACELENS_PATH/.cursor/skills/jarvis-perf-analysis.md" ~/.cursor/skills/
```

## Usage

Once installed, open Cursor and start a new chat. The skill activates when you mention:
- "analyze trace"
- "performance analysis"  
- "jarvis"
- "tracelens"
- "pytorch profiling"

### Example Prompts

**Standalone Analysis:**
```
Analyze this PyTorch trace on MI300X: /path/to/trace.json.gz
```

**Comparative Analysis:**
```
Compare performance between H100 and MI300X:
- Target (MI300X): /path/to/mi300x_trace.json.gz
- Reference (H100): /path/to/h100_trace.json.gz
```

### What JARVIS Does

1. Runs `TraceLens_generate_perf_report_pytorch` to create performance reports
2. Analyzes GPU utilization, top operations, and efficiency metrics
3. Identifies compute-bound vs memory-bound bottlenecks
4. Provides optimization recommendations (both algorithmic and kernel-level)
5. Generates replay artifacts for kernel team investigation
6. Outputs structured reports (`*_fair.md` for stakeholders, `*_rough.md` for detailed notes)

## Skill Contents

The skill file (`.cursor/skills/jarvis-perf-analysis.md`) contains:

| Section | Description |
|---------|-------------|
| Analysis Workflows | Step-by-step for standalone and comparative analysis |
| Hardware Reference | Specs for MI300X, MI325X, H100, H200, A100 |
| TraceLens Commands | CLI commands and Python API snippets |
| Common Patterns | Attention, BatchNorm, tiny GEMMs, NCHW→NHWC, etc. |
| Output Templates | Report structure for fair and rough documents |
| Key Principles | What can/cannot be inferred from traces |

## Troubleshooting

### Skill Not Being Picked Up

1. **Check Cursor version**: Must be 2.4+ (`Help > About`)
2. **Restart Cursor** after adding the skill file
3. **Try explicit invocation**: Type `/` in chat to see available skills
4. **Verify file location**: Must be in `.cursor/skills/` directory

### TraceLens Not Found

Ensure TraceLens is installed in the Python environment Cursor uses:

```bash
pip install TraceLens
# or
pip install -e /path/to/TraceLens
```

## Updating the Skill

To get the latest JARVIS skill:

```bash
cd /path/to/TraceLens
git pull origin main
cp .cursor/skills/jarvis-perf-analysis.md ~/.cursor/skills/
```

## Contributing

Found an issue or want to improve the JARVIS skill? Edit `.cursor/skills/jarvis-perf-analysis.md` and submit a PR!

## See Also

- [TraceLens Documentation](../README.md)
- [Performance Report Columns](./perf_report_columns.md)
- [TreePerf API](./TreePerf.md)
- [EventReplay](./EventReplay.md)
