---
name: model-bringup
description: >-
  Brings a new model up on TraceLens without stopping for input: profiles it with Magpie
  (fixed-sequence or AgentX), gates profile and report health, runs the analysis
  orchestrator, finds ops without a perf model, and drafts one validated perf-model PR or
  issue per op. Publishes approved drafts on a later prompt. Use when the user asks to
  bring up a model, follow the model-bringup skill, run it from a bringup_config.yaml, or
  publish a finished bringup.
license: MIT
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model bringup

## Full procedure

Read [reference.md](reference.md) before Step 0 and run every step from it.

## Workflow index

| Step | Name | Gate |
|---|---|---|
| 0 | Environment and Input Sanity Checks | every command succeeds |
| 1 | Profile (Magpie sub-agent) | `result.json` reports outcome `profiled` |
| 2 | Profile health | `prepare` and `report-health --phase profile` exit 0 |
| 3 | Baseline analysis (orchestrator sub-agent) | `analysis-result` exits 0; a failure is relaunched once |
| 4 | Report health | `report-health --phase report` exits 0 |
| 5 | Find gaps | `gaps` exits 0 |
| 6 | Perf-model sub-agents | `op-result` exits 0 per op; a failing op is relaunched once, then excluded |
| 7 | Coverage | `coverage` exits 0 |
| 8 | Summary | always runs |
| Publish | Separate prompt after review | only the named op slugs |

## Rules

- Never ask the user after Step 0. Where a step or an owned skill would ask, stop with `INPUT_REQUIRED`.
- Stop means: run Step 8 with the stop's step and code, end. Step 0 failures end without Step 8.
- Never push, run `gh`, or open an issue or PR during a run. Only Publish does, and only
  on `AMD-AGI/TraceLens` for the op slugs its prompt names.
- Never commit in `<tracelens_dir>`. Commits go only on `<op_branch>` in `<op_worktree>`.
- Run git in your own shell, never through `<prefix>`.
- Read only the files each step names, such as `result.json`, `gates/*.json` and `gaps.json`, never the instruction file of a skill or sub-agent a step launches, or a PerfModel guide.
- Never analyze a trace or author or validate a perf model by hand to work around a failed step.

## Primary outputs

`<bringup_dir>/bringup_summary.md`, `<bringup_dir>/analysis/analysis.md`,
`<bringup_dir>/perf_models/<op_slug>/` (draft, validation CSV, `result.json`).
