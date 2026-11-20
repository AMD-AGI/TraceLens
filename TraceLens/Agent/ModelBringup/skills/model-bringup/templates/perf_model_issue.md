<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

<!--
Omit Pseudo Op Check when no pseudo op is proposed, and Validation when there are no validation rows.
-->

## Operator

`<op>`: what it computes, the framework call that launches it, and its kernels.

## Observed Shapes

| Input Dims | Input Type | Concrete Inputs |
|---|---|---|
| <Input Dims> | <Input type> | <Concrete Inputs> |

## Blocking Reason

- <reason>

## Pseudo Op Check

| Pseudo Op | Brittleness | Paths Checked | Differing Paths | Complexity | Events Added |
|---|---|---|---|---|---|
| `<name>` | stable / fragile | <count and list> | <paths, or none> | O(...) | <count> |

## Validation

Measured with rocprofv3 hardware counters. Ratios are measured over predicted.

| Shape | Predicted FLOPs | Predicted Bytes | Measured Read Bytes | Measured Write Bytes | FLOPs Ratio | Bytes Ratio |
|---|---|---|---|---|---|---|
| <named dims and dtype> | <predicted_flops> | <predicted_bytes> | <rocprof_read_bytes> | <rocprof_write_bytes> | <flops_ratio> | <bytes_ratio> |
