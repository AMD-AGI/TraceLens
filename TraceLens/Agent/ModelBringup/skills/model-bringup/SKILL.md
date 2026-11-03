---
name: model-bringup
description: >-
  Brings up a new model in TraceLens end to end: profiles a model and image with Magpie,
  checks trace quality, runs the analysis orchestrator, checks the perf report and graph
  capture merge, builds and validates perf models for uncovered ops, reruns the analysis
  for coverage, and generates and checks kernel_candidates.json. Use when the user asks
  for model bringup, to bring up a model or image in TraceLens, or to close perf-model
  gaps for a new model.
license: MIT
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model bringup

This skill orchestrates existing skills and tools; it adds no scripts. Fill in
[templates/bringup_config.yaml](templates/bringup_config.yaml) first. `<ws>` is its
`workspace`, `<out>` the analysis output dir, `<tools>` the tooling worktree.

## Rules

- **Stop on failure.** If a gate fails, stop and report its output to the user. Never open
  an issue for a gate failure.
- **Where to file.** PRs and issues go to `AMD-AGI/TraceLens`.
- **Number reporting in PRs or issues.** Report only TFLOPs and bytes moved per shape:
  predicted, measured, and their ratio. Leave out kernel time, duration, % of GPU time,
  TFLOPS/s and TB/s.
- **Review gate.** Write every title and body to `<ws>/drafts/<op_slug>_{pr,issue}.md`
  and show it to the user. Run `git push`, `gh pr create` or `gh issue create` only after
  the user approves that text.
- **Commits.** Do not commit in the user's checkout. Commit only on the per-op branches
  this skill creates.

## Workspaces

```bash
git fetch origin main feat/perf_model_validation_skill
git worktree add --detach <ws>/tl_perfmodel_tools origin/feat/perf_model_validation_skill
git worktree add -b perf_model/<op_slug> <ws>/pm_<op_slug> origin/main   # per op, Stage 7
```

`<tools>` is read-only. `PM=<tools>/TraceLens/Agent/PerfModel` holds the perf-model
guides and scripts used below.

## Stages

| # | Stage | Run | Gate |
|---|---|---|---|
| 1 | Configure | Fill in the template. After Stage 2, run `python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference --output_dir <out> --job trace_path=<mixed split>,platform=<platform>,capture_folder=<captures>` | `build` passes its check |
| 2 | Profile | [magpie-benchmark-profiling](../../../Profiling/skills/magpie-benchmark-profiling/SKILL.md) Steps 0-6, including the trace split | `torch_trace/` and `trace_split/` exist |
| 3 | Trace quality | `python3 -m TraceLens.Agent.Profiling.check_torch_trace <bench_dir> --json-out <ws>/trace_check.json` | No FAIL. On FAIL, fix the profiling and redo Stage 2, up to `max_profile_retries` times |
| 4 | Baseline analysis | [analysis-orchestrator](../../../Analysis/skills/analysis-orchestrator/SKILL.md) from `run_config.yaml`. Tee its Step 1 output to `<out>/logs/step1_perf_report.log` | Orchestrator Step 2 gate |
| 5 | Report health | `python3 -m TraceLens.Agent.Analysis.triage.runner --run-dir <out> --stream-file <out>/logs/step1_perf_report.log --detailed` | None of `1c`, `1d`, `2b`, `3b`, `3c`, `3i`, `4a` fires. Show the user `triage_diags.txt` and the `graph_merge_check.json` mapping either way |
| 6 | Find gaps | `python3 $PM/unified_perf_model_extension/run_other_bucket_triage.py <out>/perf_report_csvs/unified_perf_summary.csv --mode top-ops --threshold <min_share>`, saved to `<ws>/stage6/top_ops.txt` | Drop communication ops. The user confirms the list |
| 7 | Perf models | One [perf-model-worker](agents/perf-model-worker.md) subagent per op, in parallel | Each op validated, or an issue drafted |
| 8 | Coverage | Merge the approved op branches into `<ws>/pm_integration`, then rerun Stages 4-6 with `PYTHONPATH=<ws>/pm_integration` | Fewer ops listed in Stage 6; `csv_quality` missing-perf-model share went down |
| 9 | Kernel candidates | Hyperloom `analysis_md_harness.py <out>/analysis.md --finalize --csv-dir <out>/perf_report_csvs -o <out>/kernel_candidates/kernel_candidates.json`, inside the serving container so sources resolve | File written |
| 10 | Candidates quality | `python3 -m TraceLens.Agent.Analysis.triage.verify_kernel_candidates --analysis-dir <out> --kernel-candidates <out>/kernel_candidates/kernel_candidates.json --json-out <ws>/verify_kc.json` | All checks pass; `candidate_kernels` lists none absent or unresolved |
| 11 | Handoff | Write `<ws>/bringup_summary.md` | None |

## Notes

- **Stage 4:** Step 1 of the orchestrator is
  `TraceLens_generate_perf_report_from_config --config <out>/cache/run_config.yaml`.
  Append `2>&1 | tee <out>/logs/step1_perf_report.log` to it.
- **Stage 5:** the merge mapping has one row per annotation root and capture trace. Each
  row lists its roots, how many merged, and `rank_mismatch`. `3i` fires when the replay
  trace merged with captures from another rank, which happens when the capture folder
  holds every rank. Point `capture_folder` at the replay rank's captures.
- **Stage 6:** use `--also-global-pareto` to see the ops that cover `min_share` of the
  total, and `--check-mapping` to see which ops already map to a perf model.
- **Stage 11:** the summary links every stage output: `trace_check.json`,
  `triage_details.csv`, `graph_merge_check.json`, the Stage 6 lists from before and after
  Stage 8, the validation CSVs, approved PR and issue links, `kernel_candidates.json`, and
  `verify_kc.json`.
