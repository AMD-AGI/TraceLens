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
| The op: profiler `name`, kernel names and call stack (its rows in a `unified_perf_summary.csv`, or given directly) | yes | none |
| The vendor framework source the op comes from | yes | none |
| A TraceLens checkout to edit, and a Python environment (or command prefix) that imports it | yes | none |
| Depth: full model or categorize-only | no | full; categorize-only for a `(Synthetic Op)` name |
| Output mode: integrate or extension-only | no | integrate |
| Validate with hardware counters | no | yes, with the validate-perf-model skill |
| Model config and launch settings | no | none |

## Outputs

The perf model for the op in the chosen depth and mode, the regenerated report's `perf_report_csvs` directory, and the op name; the last two are the validate-perf-model skill's Inputs. For a full model the op is covered (`has_perf_model` true with non-zero GFLOPS or Data Moved); a categorize-only op keeps `has_perf_model` false.

## Workflow index

| What | reference.md section |
|---|---|
| Apply the depth and output mode | [Per-op choices](reference.md#per-op-choices) |
| Locate the binding and kernel; find the vendor roofline reference | [Locate](reference.md#locate), [Vendor reference](reference.md#vendor-reference) |
| Map `Input Dims` to the signature; pick the base class | [Map Input Dims](reference.md#map-input-dims), [Base class](reference.md#base-class) |
| Implement FLOPs, bytes, precision and annotation use; write the docstring | [Implement](reference.md#implement), [Perf model class docstring template](reference.md#perf-model-class-docstring-template) |
| Register the op; `py_compile`; regenerate the report and check it | [Register](reference.md#register), [Regenerate and check](reference.md#regenerate-and-check), [Output modes](reference.md#output-modes) |
| Hand the report directory and op name to the validate-perf-model skill | [Hand off to validation](reference.md#hand-off-to-validation) |

## Rules

- Model the operation, not the kernel: padding, tiling and other kernel artifacts stay out of FLOPs and bytes.
- `get_param_details` returns every value `flops()` and `bytes()` use, including annotation-derived ones; validation runs from them.
- Every registered class has the docstring from the template.
- No perf model class for a `(Synthetic Op)` name; categorize it only.
- Model the op once, at the most stable name that carries its shapes, not per kernel or template variant ("Locate" in reference.md).

## Stop rules

Stop and report the reason, rather than guess, when:
- a required input is missing;
- the kernel source cannot be found: report the op and the paths searched. Do not guess FLOPs or bytes;
- the wrapper, the kernel and the vendor formula disagree and the cause is unresolved: report the disagreement;
- the op needs a pseudo op: never add one yourself. Run its stability and cost check ([Pseudo ops](reference.md#pseudo-ops)) and report the table as a decision for the caller.

## Standalone use

When a user runs this skill directly, without the inputs above, the entry points
in reference.md find the op, and these are the only questions to ask:
- **CSV ranking (EP1) or framework scan (EP3):** confirm the candidate list.
- **Manual op (EP2):** the profiler `name`, and optionally the kernel name and call stack.
- **Any entry point:** depth and output mode, if the user wants other than the defaults; the Python environment; whether to validate; and whether to add a pseudo op after its stability and cost check.
