<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

## Operator

`<op>`: what it computes and the kernels it launches. It had no performance model.

## Changes

- Performance model: `<class>` in `<extension file>`, a subclass of `<base class>`. FLOPs are `<formula>`. Bytes are `<terms>`.
- Category mapping: `<op>` in `get_pseudo_op_mappings()` (`pseudo_ops_perf_utils.py`).
- Category mapping: `<op>` → `<category>` in `get_pseudo_op_category_only_mappings()` (`pseudo_ops_perf_utils.py`). Category only: <why no perf model, e.g. a `(Synthetic Op)` name>.
- Validator harness: `<name>` in `harnesses.py`.
- Tests: `<test class>` in `tests/test_perfmodel_extensions.py`.

## Validation

Measured with rocprofv3 hardware counters. Ratios are measured over predicted. Every row ran and matches or has an explained deviation.

| Shape | Predicted FLOPs | Predicted Bytes | Measured Read Bytes | Measured Write Bytes | FLOPs Ratio | Bytes Ratio |
|---|---|---|---|---|---|---|
| <named dims and dtype> | <predicted_flops> | <predicted_bytes> | <rocprof_read_bytes> | <rocprof_write_bytes> | <flops_ratio> | <bytes_ratio> |

## Ratio Deviations

- <ratio header> <value> (<shape>), <ratio header> <value> (<shape>): <cause>.
