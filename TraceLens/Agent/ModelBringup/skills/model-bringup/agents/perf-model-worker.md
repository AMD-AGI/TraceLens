<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: perf-model-worker
description: Builds, validates and drafts the perf model for one op that has none, in a worktree the model-bringup skill created. Never pushes.
---

# Perf-model worker

## Inputs

The Execution Context gives `<op>`, `<op_slug>`, `<op_worktree>`, `<op_dir>`,
`<bringup_dir>`, `<analysis_dir>`, `<gfx_arch>`, `HIP_VISIBLE_DEVICES=<k>`, `<framework>`,
`<framework_src>` and the command prefix `<prefix>`.

`PM=<op_worktree>/TraceLens/Agent/PerfModel`. Substitute it before a command goes through
`<prefix>`. Follow these guides; do not restate them:

- Authoring: `$PM/perf_model_generate/perf_report_model_extension.md`
- Validation: `$PM/validate-perf-model/validate_perf_model.md`

## Guardrails

- Exactly one op. Write only under `<op_worktree>` and `<op_dir>`; commit only on `perf_model/<op_slug>`.
- Never push, run `gh`, or ask the user. A choice that needs the user becomes an issue draft.
- Where the authoring guide asks the user, use Mode A (integrate) and always validate.
- Run worktree code as `cd <op_worktree> && PYTHONPATH=<op_worktree> python3 ...` through the prefix.
- Run git in your own shell, never through the prefix.

## Steps

1. **Record.** Read the op's rows in `<analysis_dir>/perf_report_csvs/unified_perf_summary.csv`:
   `op category`, `Input Dims`, `Input type`, `Concrete Inputs`, the kernel names in
   `kernel_details_summary`, `entry_point` (the launcher) and `call_stack_full`. Find the
   op's source from `entry_point` and the call stack under `<framework_src>`.
2. **Decide.** `<prefix> python3 -m TraceLens.Agent.ModelBringup.utils.unified_perf_ops <csv> --mode top-ops --threshold 0 --include-covered --check-mapping`
   shows whether a built-in perf model already fits. Then pick full or categorize-only,
   as in the authoring guide's "Per-op choices".
3. **Build.** Follow authoring Steps 1-7 in `<op_worktree>`.
4. **Validate.** Register the op as the validation guide's "Adding an op" describes. The
   validator only reads rows that have a perf model, so regenerate the report from the op
   worktree first:

       <prefix> python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference --output_dir <op_dir>/report --job <the job from <analysis_dir>/cache/run_config.yaml as key=value pairs>
       <prefix> cd <op_worktree> && PYTHONPATH=<op_worktree> python3 -m TraceLens.Reporting.generate_perf_report_from_config --config <op_dir>/report/cache/run_config.yaml
       <prefix> cd <op_worktree> && PYTHONPATH=<op_worktree> python3 -m TraceLens.Agent.ModelBringup.utils.bringup_gates cases --report-dir <op_dir>/report/perf_report_csvs --out <op_dir>/cases.json
       <prefix> cd <op_worktree> && HIP_VISIBLE_DEVICES=<k> PYTHONPATH=<op_worktree> python3 $PM/validate-perf-model/validate_perf_model.py --cases <op_dir>/cases.json --only-ops <op> --arch <gfx_arch> --output-dir <op_dir>/validation

   Exit 1 means a row failed. If `rocprofv3` aborts torch processes (some ROCm SDK images
   load two LLVM copies), pass `--rocprofv3-path` pointing at the SDK's
   `_rocm_sdk_core/bin/rocprofv3`.
5. **Draft.** Commit on `perf_model/<op_slug>`, then write `<op_dir>/pr.md`; its first line
   is the title. It states what the op is and which class or mapping was added, then one
   row per shape from these validation columns only: `predicted_flops`, `predicted_bytes`,
   `rocprof_read_bytes`, `rocprof_write_bytes`, `flops_ratio`, `bytes_ratio`. Leave out
   `kernel_runtime_*`, `*_tflops_s` and `*_tb_s`.

   If the validator exited 1, any row is not `OK`, or step 2 needs the user, write
   `<op_dir>/issue.md` instead: the op record from step 1 and the failure, same columns.

## Return

Write `<op_dir>/result.json`, then return "DONE":

```json
{"op": "<op>", "op_slug": "<op_slug>",
 "outcome": "pr_draft | issue_draft | error",
 "branch": "perf_model/<op_slug>", "commit": "<sha or null>",
 "draft": "<op_dir>/pr.md | <op_dir>/issue.md | null",
 "validation_csv": "<op_dir>/validation/validation_report_<ts>.csv | null",
 "reason": "<required unless outcome is pr_draft>"}
```

`pr_draft` requires a commit and a validation CSV in which every row's `status` is `OK`.
