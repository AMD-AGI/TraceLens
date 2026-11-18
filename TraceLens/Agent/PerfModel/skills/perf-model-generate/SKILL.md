---
name: perf-model-generate
description: >-
  Authors a TraceLens perf model for a cpu_op that has none: locates its binding
  and kernel, writes FLOPs, bytes and compute precision (or a category-only
  mapping), registers it in TraceLens/PerfModel/extensions or a standalone
  extension file, regenerates the report, and hands off to the validate-perf-model
  skill. Runs from the inputs it is given and stops with a reason when one is
  missing; standalone, it can also find candidate ops from a unified_perf_summary.csv
  or a framework scan. Use to add or fix a perf model, cover ops without one, or
  follow the perf-model-generate skill.
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

| Input | Required | Default |
|---|---|---|
| The op: profiler `name`, kernel names and call stack (its rows in a `unified_perf_summary.csv`, or given directly) | yes | — |
| The vendor framework source the op comes from | yes | — |
| A TraceLens checkout to edit, and a Python environment (or command prefix) that imports it | yes | — |
| Depth: full model or categorize-only | no | full; categorize-only for a `(Synthetic Op)` name |
| Output mode: integrate or extension-only | no | integrate |
| Validate | no | yes: the validate-perf-model skill takes the Outputs below as its Inputs |

## Outputs

The perf model for the op in the chosen depth and mode, the regenerated report's `perf_report_csvs` directory, and the op name; the last two are the validate-perf-model skill's Inputs. For a full model the op is covered (`has_perf_model` true with non-zero GFLOPS or Data Moved); a categorize-only op keeps `has_perf_model` false.

## Workflow index

| Step | What | reference.md section |
|---|---|---|
| 1 | Apply the depth and output mode | Per-op choices |
| 2 | Locate the binding and kernel; find the vendor roofline reference | Step 1, Step 2 |
| 3 | Map `Input Dims` to the signature; pick the base class | Step 3, Step 4 |
| 4 | Implement FLOPs, bytes, precision and annotation use; write the docstring | Step 5, Perf model class docstring template |
| 5 | Register the op; `py_compile`; regenerate the report and check it | Step 6, Step 7, Output modes |
| 6 | Hand the report directory and op name to the validate-perf-model skill | Hand off to validation |

A pseudo op is never part of this flow by default: see "Pseudo ops" in reference.md.

## Rules

- Model the operation, not the kernel: padding, tiling and other kernel artifacts stay out of FLOPs and bytes.
- `get_param_details` returns every value `flops()` and `bytes()` use, including annotation-derived ones; validation runs from them.
- Every registered class has the docstring from the template.
- No perf model class for a `(Synthetic Op)` name; categorize it only.
- Check reference.md's Pitfalls before finishing.

## Stop rules

Stop and report the reason, rather than guess, when:
- a required input is missing;
- the kernel source cannot be found: report the op and the paths searched. Do not guess FLOPs or bytes;
- the wrapper, the kernel and the vendor formula disagree and the cause is unresolved: report the disagreement;
- the op needs a pseudo op: never add one yourself. Run its stability and cost check ("Pseudo ops" in reference.md) and report the table as a decision for the user.

## Standalone use

When a user runs this skill directly, without the inputs above, the entry points
in reference.md find the op, and these are the only questions to ask:
- **CSV ranking (EP1) or framework scan (EP3):** confirm the candidate list.
- **Manual op (EP2):** the profiler `name`, and optionally the kernel name and call stack.
- **Any entry point:** depth and output mode, if the user wants other than the defaults; the Python environment; whether to validate; and whether to add a pseudo op after its stability and cost check.
