# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Commands

```bash
# Install (editable)
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_flash_attention_backward.py -v

# Format code (Black is pinned at 26.1.0 in CI)
pip install black==26.1.0
black .
black path/to/file.py
```

All Python files must have an AMD copyright header — `tests/test_copyright_headers.py` enforces this in CI.

## Architecture

The core pipeline is a four-stage transformation:

```
Trace file (.json.gz / .pb / .pftrace)
  → util.py          (load raw events)
  → Trace2Tree/      (flat events → nested CPU call-stack tree with GPU kernels attached)
  → TreePerf/        (tree + PerfModel → DataFrames with FLOPs, bytes, roofline)
  → Reporting/       (DataFrames → Excel/CSV/Markdown)
```

### How Trace2Tree works

`TraceToTree.build_host_call_stack_tree()` sorts CPU events by timestamp and nests them by temporal containment — four layers: Python frontend → PyTorch op (`cpu_op`) → HIP/CUDA runtime → GPU kernel leaf.

`add_gpu_ops_to_tree()` then attaches GPU kernels to their CPU launchers via the `correlation` / `External id` field in each event's `args` dict. `link_all_fwd_bwd_events()` uses autograd flow events (`ph: 's'/'f'`) to connect forward ops to their backward counterparts.

### How PerfModel works

`PerfModel/torch_op_mapping.py` contains `op_to_perf_model_class_map` — a dict from PyTorch op name strings (e.g. `"aten::mm"`) to perf model classes. Each class implements `flops()` and `bytes()` based on tensor shapes parsed from the trace's `Concrete Inputs` field. `TreePerfAnalyzer.compute_perf_metrics(event, arch)` does the lookup and instantiation.

The 62 base model classes live in `PerfModel/perf_model.py` (GEMM, SDPA/flash-attention, Conv, Elementwise, Reduce, RMSNorm, MoE, MambaSSD, etc.). Another 40+ inference-specific classes live in `PerfModel/extensions/`.

### Pseudo-Op Extensions (Trace2Tree/extensions/)

When a single CPU op launches multiple GPU kernels that need individual roofline entries (e.g. fused MoE with separate up/down GEMMs), `apply_pseudo_op_extensions(tree)` auto-detects known patterns and injects synthetic "pseudo-op" nodes between the CPU op and its GPU kernel children. The pseudo-op names map to their own perf model classes via `PerfModel/extensions/pseudo_ops_perf_utils.py`.

### User Extension System

Pass `--extension_file my_ext.py` to any report generator. The file may define:
- `tree_postprocess_extension(trace_to_tree)` — modify the tree after build
- `perf_model_extension = {"op_name": PerfModelClass, ...}` — add/override op mappings
- `dict_cat2names_extension = {"Category": ["ClassName"], ...}` — add report categories

See `examples/example_megatron_extension.py` for a complete example.

### GPU Timeline Calculation

`GPUEventAnalyser` uses interval arithmetic: it merges overlapping GPU kernel intervals, then computes *exposed* communication (NCCL/RCCL time not overlapped by compute) and *exposed* memcpy via set-subtraction. This gives the true serialized overhead rather than raw collective time.

## Branch & Commit Convention

Branches: `<type>/<scope>/<short-description>` — e.g. `feat/perfmodel/aiter-fav3`, `fix/tracediff/diff-reporting-bug`.

Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `feat(perfmodel): add perf model for aiter fav3`.
