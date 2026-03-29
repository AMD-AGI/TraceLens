<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

### ⚠️ Disclaimer

The traces provided in this directory (`tests/traces/h100` and `tests/traces/mi300`) are **for testing, validation, and educational purposes only**.
They are derived from **open-source models** and are intended to help contributors and users verify TraceLens functionality, experiment with analysis workflows, and learn about performance behavior across hardware.

These traces **do not represent**:

* Official or benchmarked performance numbers.
* Any internal, production, or customer environment.
* Validated hardware configurations, tuning parameters, or training setups.

Performance metrics observed from these traces may differ substantially from real workloads. They should **not** be interpreted as indicative of actual device performance or official results.

---

### 📝 JAX trace naming (`mi300/`)

| Directory | JAX version | Description |
|-----------|-------------|-------------|
| `jax_conv_minimal_legacy` | ~0.6 | Older trace; `.hlo_proto.pb` sidecars are generated at runtime by xprof (ignored via `.gitignore`). Full perf model (FLOPS, bytes, shapes) works when sidecars are present. Used by `test_jax_conv_analysis.py`. |
| `jax_conv_minimal_08` | 0.8 | Comparable minimal 3D conv trace (same params). For E2E smoke tests; perf model metadata limited until #425. |

Both use the same convolution parameters (16→5120 channels, 1×2×2 kernel, stride 1×2×2, bf16). Generated via `jax-minimal/traces/generate_conv_trace.py`.

---

### 📝 Note on `mi300/llama_70b_fsdp`

Python function events have been removed from these traces to reduce test data size while maintaining identical NCCL analysis results.
