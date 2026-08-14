<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# TraceLens release notes
```{meta}
:description: Release notes for TraceLens. Learn what features and SDK modules are available in each version, and see links to the compatibility matrix.
:keywords: TraceLens, release notes, changelog, ROCm, GPU trace analysis, PyTorch, JAX, rocprofv3, SDK, performance report
```

This topic summarizes the features available in each TraceLens release. For the
hardware and software versions validated for a release, see the
[Compatibility matrix](../reference/compatibility.md).

## TraceLens 1.0.0

Marks the first stable release: the Analysis Agent hardened for public use, continued perf-model and inference-framework expansion, a TraceDiff matching overhaul, and production-grade test coverage.

### TraceLens Analysis Agent

- Agent specs migrated to a portable `skills/` layout (#860).
- Graph-capture trace comparison wired through the agent end-to-end (#860).
- Comparative graph-capture analysis: comparative mode accepts a capture folder per trace, merging call-stack and shape info into both replay trees (#797).
- Model identification now reads `nn.Module` names for more robust model detection (#799).
- Kernel Path / Kernel Name columns check propagate into comparative-mode tables (#799).
- Compute and system tiers emit default "no findings" messages instead of empty sections (#799).
- Roofline classification deferred to TraceLens Core, removing a duplicated peak-resolution path (#860).
- Local eval support: repeatability tests can run against a local Pi/vLLM/SGLang server for offline eval loops (#828).

### Performance models

- **DeepSeek-V4** sparse paged-decode modes (SWA/CSA/HCA), the mHC operator family, and `aiter::pa_sparse_prefill_opus_fwd` (#802).
- `aten::upsample_nearest` 1d/2d/3d modeled as a purely HBM-bound copy (#931).
- `aten::_efficient_attention` classification (#719).
- FP8 MoE precision fix for fp8/bf16 inputs (#737).
- `mx_available()` helper for detecting block-scaled MXFP4/MXFP6 GEMM paths (#810).

### TraceDiff

- Name disambiguation added to Wagner-Fischer alignment so same-named sibling nodes no longer mismatch on position (#800).
- Aggressive final matching pass in phase 5 for slightly-differing Python-function names (#820).
- Rewrite of `_disambiguate_same_name_candidates` fixing three correctness issues in the candidate pool (#835).
- Cleanup pass removing ~1200 lines of duplication/dead code, utilities moved to `util.py`, plus new regression tests (#840).

### Frameworks and profiling

- **xDiT** diffusion profiling via the same capture-merge infrastructure as vLLM/SGLang (#848).
- SGLang patches through v0.5.17 (#853, #930, #936).
- vLLM support through v0.26.0+, switching to the `--profiler-config.capture_torch_profiler` flag and stock upstream images for v0.26.0+ (#933).
- **Genesis**: new `TraceLens_generate_perf_report_genesis` CLI for Genesis/Taichi physics-sim profiling, isolating the steady-state window from JIT/build overhead (#866).
