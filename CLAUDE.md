<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- When compacting, preserve: modified file list, last test command run, any failing test names -->

## Setup & Commands

```bash
pip install -e .
pytest tests/                                        # all tests
pytest tests/test_flash_attention_backward.py -v     # single file
pip install black==26.1.0 && black .                 # format (CI pins 26.1.0)
```

All Python files must have an AMD copyright header — [tests/test_copyright_headers.py](tests/test_copyright_headers.py) enforces this in CI.

## Architecture

@docs/html/index.html covers the full pipeline with examples. At a high level:

```
Trace file (.json.gz / .pb / .pftrace)
  → util.py       (load raw events)
  → Trace2Tree/   (flat events → nested CPU/GPU tree)
  → TreePerf/     (tree + PerfModel → DataFrames with FLOPs, bytes, roofline)
  → Reporting/    (DataFrames → Excel/CSV/Markdown)
```

Key entry points:
- `TraceToTree.build_host_call_stack_tree()` — CPU nesting by timestamp containment
- `add_gpu_ops_to_tree()` — attaches GPU kernels via `args["External id"]` (correlation ID)
- `link_all_fwd_bwd_events()` — connects fwd↔bwd via autograd flow events (`ph: 's'/'f'`)
- Op-to-class mapping: [PerfModel/torch_op_mapping.py](TraceLens/PerfModel/torch_op_mapping.py) → `op_to_perf_model_class_map`
- Input parsing: each perf model class `__init__` parses `kwargs["Concrete Inputs"]`

See [PerfModel/CLAUDE.md](TraceLens/PerfModel/CLAUDE.md) for perf model authoring conventions.

## Key Terms

- **pseudo-op**: synthetic tree node injected between a CPU op and its GPU kernel children to enable per-kernel roofline analysis (see `Trace2Tree/extensions/`)
- **correlation ID**: integer in `args["External id"]` linking a CPU runtime event to its GPU kernel
- **exposed comm / memcpy**: collective/memcpy time not overlapped by compute, computed via interval set-subtraction in `GPUEventAnalyser`
- **roofline**: performance bound — `min(flops/peak_flops, bytes/peak_bw)` expressed as TFLOPS/s utilization

## Verification

After adding or modifying a perf model class:
1. `pytest tests/test_perf_model.py -v` (or the relevant model-specific test)
2. Run the report generator on a sample trace and confirm the TFLOPS column is non-zero

After modifying Trace2Tree:
1. `pytest tests/test_tree_construction.py -v`
2. Check tree depth and GPU kernel correlation counts haven't regressed

## User Extension System

Pass `--extension_file my_ext.py` to any report generator CLI. The file may define:
- `tree_postprocess_extension(trace_to_tree)` — modify the tree after build
- `perf_model_extension = {"op_name": PerfModelClass, ...}` — add/override op mappings
- `dict_cat2names_extension = {"Category": ["ClassName"], ...}` — add report categories

See [examples/example_megatron_extension.py](examples/example_megatron_extension.py) for a complete example.

## Branch & Commit Convention

Branches: `<type>/<scope>/<short-description>` — e.g. `feat/perfmodel/aiter-fav3`.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `feat(perfmodel): add perf model for aiter fav3`.

## Local Overrides

Create `CLAUDE.local.md` (gitignored) for machine-specific paths, local trace locations, or personal preferences.
