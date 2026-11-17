---
name: validate-perf-model
description: >-
  Validates TraceLens perf model predictions (FLOPs and bytes) against
  hardware counters from rocprofv3 --pmc on an AMD GPU. Runs a standalone
  harness per registered op, from CLI dimensions or from traced cases
  (--cases, built from a perf report with each row's shapes and perf_params),
  and writes a per-op report plus a summary CSV with a run status and match status per
  row. Covers GEMM, MoE, attention, norm, quant/activation and RoPE ops from
  aiter, vLLM, SGLang and ATOM. Use when the user asks to validate a perf model,
  check its FLOPs or bytes against hardware counters, follow the
  validate-perf-model skill, or when the perf-model-generate skill hands off
  for validation.
license: MIT
---


<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Perf-model validation with rocprofv3 hardware counters

This validation runs a TraceLens perf model's analytical FLOPs and bytes moved
against hardware counters to see how well they align. A gap between the
analytical and measured values has one of two causes:
- the analytical model is wrong;
- the kernel does extra work the operation does not need: padding or tiling, or
  extra loads and stores because data does not fit in a cache level.

The goal is to find the gap and, where there is one, read the kernel source to
explain it. The counters show that values differ; they do not say why. Only
change the perf model when the analytical model itself is wrong. Kernel
artifacts are not part of the operation, so they stay out of the model and are
recorded as the cause of the deviation. When tiling or padding dominates a
small shape, also validate larger shapes or multiples of the tile size.

## Full procedure

Follow [reference.md](reference.md). Read each section when its step needs it; the
workflow index below names the section for every step.

## Inputs

- A cases JSON (`--cases`) built from a perf report, or an op key with dimensions.
- A GPU, its `--arch`, and an output directory.
- Where to run it, given by the user: a Python environment (conda env or
  virtual environment), a Docker image, or the name of a running container.
  It must meet the prerequisites in [reference.md](reference.md); the serving image of the framework that
  owns the op usually does. Do not pick one yourself; if none is given, ask.
  For a Docker image, start a container from it with the GPU devices passed
  through, and run the validator inside it.

## Outputs

`validation_report_<timestamp>.csv` with one row per op or shape, and a directory per op. Each row has two statuses:

- `run_status`: whether the op could be measured. `OK`, `MODEL_ONLY` (predicted only, not run) or `FAIL: <reason>`. A `--cases` run exits 1 when any row is `FAIL`.
- `match_status`: how the prediction compares, from the ratio bands in `TraceLens/Agent/PerfModel/validate_perf_model/ratio_bands.py`. Empty unless `run_status` is `OK`; otherwise:
  - `match`: both ratios inside their ranges;
  - `over-predicted`: a ratio below its range, so the model predicts more than was measured. Never acceptable;
  - `deviation-suspect`: a ratio above its range, not yet explained. The validator writes this;
  - `deviation-explained`: a ratio above its range whose cause was found in the code (see "Investigating a deviation" in reference.md). Only the investigating agent sets this;
  - `no prediction`: the perf model returned nothing.

Per-ratio bands are in `flops_band` and `bytes_band`. Both ranges start at 1.0, since a model must never predict more than was measured: FLOPs 1.0–1.05, bytes 1.0–1.2.

## Workflow index

| Step | What | reference.md section |
|---|---|---|
| 0 | Have every input above, and check the prerequisites | Prerequisites |
| 1 | Pick the ops: a cases file from a perf report, an op key, a category, or all | How a run works |
| 2 | Run the validator | Running |
| 3 | Read `run_status` and `match_status` for every row | Counters |
| 4 | Investigate each `deviation-suspect` row and record the result in the CSV | Investigating a deviation |
| 5 | Cover the variants that change the work | What to validate |

An op with no harness yet: see "Adding an op: where things go".

## Rules

- Change the perf model only when the analytical model is wrong, never just to match a counter.
- `over-predicted` is never acceptable.
- Edit the validation CSV only as "Investigating a deviation" allows: one row's `match_status` and `comment`, after the investigation.
- The harness must run the same kernel as the trace; check the kernel name before trusting a ratio.
- Run only where the user said, with the GPU and output directory they gave.

## Error handling

A `FAIL: <reason>` row: the reason and the per-op `validation_report.txt` say why. Fix the registration or the formula for a stated cause and rerun. Do not change a formula only to match a counter.

## Unattended use

A caller without a user, such as the model-bringup skill, supplies the target machine, where to run (a Python environment, Docker image or container, usually as a command prefix), the output directory and the ops (`--cases` with `--only-ops`). Ask nothing; where one of them is missing, stop and report it as the reason.
