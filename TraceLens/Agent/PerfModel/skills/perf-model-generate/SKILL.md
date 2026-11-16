---
name: perf-model-generate
description: >-
  Authors TraceLens perf models for cpu_ops missing coverage. Starts from a
  unified_perf_summary.csv top-ops ranking, a manually named op, or a static scan
  of a vendor framework (aiter, vLLM, SGLang, ATOM). For each op it writes a full
  perf model or a category-only mapping, integrated into TraceLens/PerfModel/
  extensions or as a standalone extension file, then hands off to the
  validate-perf-model skill. Use when the user asks to add or fix a perf model,
  cover ops without a perf model, or follow the perf-model authoring skill.
license: MIT
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Perf-model authoring

Write a perf model (FLOPs, bytes, compute precision) for an op TraceLens does not
yet model, register it, and hand it to validation. A generated class is a draft:
confirm it against the kernel, several trace events, and any model-specific config
the trace does not contain.

## Full procedure

Follow [reference.md](reference.md). Read each section when its step needs it; the
workflow index below names the section for every step.

## Inputs

- An op: its profiler `name`, kernel names and call stack, from a `unified_perf_summary.csv` (EP1), the user (EP2) or a framework scan (EP3).
- The vendor framework source the op comes from.
- A TraceLens checkout to edit, and a Python environment that imports it.

## Outputs

The perf model for the op in the chosen depth and mode, and a regenerated report. For a full model the op has `has_perf_model` true; a categorize-only op keeps it false.

## Workflow index

| Step | What | reference.md section |
|---|---|---|
| 0 | Pick an entry point: CSV ranking, manual op, or framework scan | Entry points |
| 1 | Confirm the candidates, then full or categorize-only, and integrate or extension-only | Per-op choices |
| 2 | Locate the binding and kernel; find the vendor roofline reference | Step 1, Step 2 |
| 3 | Map `Input Dims` to the signature; pick the base class | Step 3, Step 4 |
| 4 | Implement FLOPs, bytes, precision and annotation use; write the docstring | Step 5, Perf model class docstring template |
| 5 | Register the op; `py_compile`; regenerate the report and check it | Step 6, Step 7, Output modes |
| 6 | Hand off to hardware-counter validation | Validate with hardware counters |

A pseudo op is never part of this flow by default: see "Pseudo ops" in reference.md.

## Rules

- Model the operation, not the kernel: padding, tiling and other kernel artifacts stay out of FLOPs and bytes.
- `get_param_details` returns every value `flops()` and `bytes()` use, including annotation-derived ones; validation runs from them.
- Every registered class has the docstring from the template.
- No perf model class for a `(Synthetic Op)` name; categorize it only.
- Never add a pseudo op on your own judgment; run its stability and cost check and let the user decide.
- Check reference.md's Pitfalls before finishing.

## Error handling

- The kernel source cannot be found: stop and report the op and the paths searched. Do not guess FLOPs or bytes.
- The wrapper, the kernel and the vendor formula disagree and the cause is unresolved: stop and report the disagreement.

## Unattended use

A caller without a user, such as the model-bringup skill, supplies the op and the paths. Ask nothing and answer as below. Where an answer needs information the caller did not give, stop and report that question as the reason.

| Ask | Answer |
|---|---|
| Confirm the candidate list (EP1, EP3) | The caller's op is the candidate |
| Name, kernel and call stack (EP2) | The op's rows in the caller's CSV |
| Depth and output mode (Per-op choices) | full and integrate; categorize-only for a `(Synthetic Op)` name |
| Python environment | The caller's command prefix |
| Validate with hardware counters | Always validate |
| Whether to add a pseudo op (stability and cost check) | Never decide. Do not add the pseudo op; report the check's table to the caller as a decision for the user |
