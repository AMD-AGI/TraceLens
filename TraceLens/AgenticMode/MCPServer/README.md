<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens MCP Server

TraceLens MCP Server is a GPU performance analysis service built on MCP (Model Context Protocol), enabling AI Agents (e.g., Cursor) to perform GPU trace analysis via natural language.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Cursor IDE)                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ HTTP (Streamable MCP)
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TraceLens MCP Server                        │
│                                                                  │
│  Tools:                                                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐    │
│  │check_trace_   │  │run_full_      │  │run_comparative_   │    │
│  │file           │  │standalone_    │  │analysis           │    │
│  │               │  │analysis       │  │                   │    │
│  └───────────────┘  └───────────────┘  └───────────────────┘    │
│                                                                  │
│  Resources: platform-specs    Prompts: standalone_analysis      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Invoke TraceLens Core
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  TraceLens Core: Standalone | Comparative | TreePerf | ...      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Read trace files
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Shared NFS Storage                         │
│                        /shared_nfs/...                           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Deploy the Server

```bash
git clone git@github.com:AMD-AGI/TraceLens-internal.git
cd TraceLens-internal/TraceLens/AgenticMode/MCPServer
./run.sh
```

The server starts at `http://0.0.0.0:8000`.

### 2. Configure Cursor

Add the MCP Server in Cursor Settings:

```json
{
  "mcpServers": {
    "tracelens": {
      "url": "http://<server-ip>:8000/mcp"
    }
  }
}
```

### 3. Start Analyzing

Use natural language in Cursor to request analysis:

```
Analyze this trace file: /shared_nfs/<your_directory>/prof_rank-0.json.gz
Platform is MI355X
```

## Available Tools

### 1. check_trace_file

Verifies that a trace file exists.

**Parameters:**
- `trace_path`: Absolute path to the trace file

**Returns:**
- Whether the file exists, its size, and type

### 2. run_full_standalone_analysis

Runs the complete single-file analysis pipeline.

**Parameters:**
- `trace_path`: Path to the trace file
- `platform`: GPU platform (MI300X, MI325X, MI350X, MI355X, MI400)
- `trace_type`: Trace type (pytorch, jax, rocprof), defaults to pytorch
- `cleanup`: Auto-delete intermediate files, defaults to true

**Returns:**
- GPU utilization, per-category analysis metrics, optimization recommendations

### 3. run_comparative_analysis

Compares two trace files.

**Parameters:**
- `gpu1_kineto`: Path to the first trace file
- `gpu2_kineto`: Path to the second trace file
- `gpu1_name`: Display name for the first trace (optional)
- `gpu2_name`: Display name for the second trace (optional)
- `cleanup`: Auto-delete intermediate files, defaults to true

**Returns:**
- Comparison report (markdown format)

## Workflow

### Standalone Analysis Flow

```
┌──────────────────────────────────────┐
│  User: "Analyze /path/to/trace.json, │
│         platform MI355X"             │
└──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  1. check_trace_file                 │
│     Verify file exists               │
└──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  2. run_full_standalone_analysis     │
│     ┌─────────────────────────────┐  │
│     │ Generate perf_report        │  │
│     │      ↓                      │  │
│     │ prepare_agentic             │  │
│     │      ↓                      │  │
│     │ Category analysis (parallel)│  │
│     │      ↓                      │  │
│     │ Collect metrics             │  │
│     │      ↓                      │  │
│     │ Cleanup (auto)              │  │
│     └─────────────────────────────┘  │
└──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  3. Return metrics + report          │
│     instructions                     │
└──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  4. AI generates                     │
│     standalone_analysis.md           │
└──────────────────────────────────────┘
```

### Comparative Analysis Flow

```
┌──────────────────────────────────────┐
│  User: "Compare trace1.json and      │
│         trace2.json"                 │
└──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  run_comparative_analysis            │
│  ┌─────────────────────────────────┐ │
│  │ Decompress .gz (if needed)      │ │
│  │        ↓                        │ │
│  │ ┌──────────┐  ┌──────────┐     │ │
│  │ │Process   │  │Process   │     │ │
│  │ │GPU1      │  │GPU2      │     │ │
│  │ └────┬─────┘  └────┬─────┘     │ │
│  │      └──────┬───────┘           │ │
│  │             ↓                   │ │
│  │      Compare & generate report  │ │
│  │             ↓                   │ │
│  │      Cleanup (auto)             │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Return comparison_markdown          │
└──────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  AI generates comparison report      │
└──────────────────────────────────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACELENS_HOST` | 0.0.0.0 | Listen address |
| `TRACELENS_PORT` | 8000 | Listen port |

## Supported Platforms

| Platform | HBM Bandwidth | Memory | BF16 Peak TFLOPS |
|----------|--------------|--------|-------------------|
| MI300X | 5300 GB/s | 192 GB | 708 |
| MI325X | 6000 GB/s | 256 GB | 843 |
| MI350X | 6000 GB/s | 288 GB | 843 |
| MI355X | 8000 GB/s | 288 GB | 1686 |
| MI400 | 19600 GB/s | 432 GB | 2500 |

## Supported Trace Formats

- PyTorch Profiler (`*.json`, `*.json.gz`)
- JAX XPlane (`*.pb`)
- rocprofv3 (`*.json`)

## Health Check

```bash
curl http://<server-ip>:8000/health
```

Response:
```json
{
  "status": "ok",
  "cached_traces": 0
}
```

## File Structure

```
MCPServer/
├── __init__.py                    # Package init
├── __main__.py                    # Entry point
├── config.py                      # Server configuration
├── server.py                      # Server launcher
├── mcp_app.py                     # MCP application (tools, resources, prompts)
├── standalone_tools.py            # Standalone analysis pipeline
├── comparative_tools.py           # Comparative analysis pipeline
├── run.sh                         # Deployment script
├── requirements.txt               # Dependencies
└── cursor_mcp_config.example.json # Cursor MCP config example
```

## FAQ

### Q: Where are analysis results stored?

A: Intermediate files are automatically cleaned up (`cleanup=True`). Final results are returned directly to the AI Agent with zero disk footprint.

### Q: Is there a trace file size limit?

A: No hard limit, but large files (>1 GB) will take longer to process.

### Q: How do I view supported GPU platform specs?

A: Use the MCP resource `tracelens://platform-specs` to get the full platform specifications table.
