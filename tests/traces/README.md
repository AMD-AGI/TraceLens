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

### 📝 Note on `mi300/llama_70b_fsdp`

Python function events have been removed from these traces to reduce test data size while maintaining identical NCCL analysis results.
