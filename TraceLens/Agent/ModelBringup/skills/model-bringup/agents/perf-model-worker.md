<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: perf-model-worker
description: Builds, validates, and drafts a PR for the perf model of one op that the model-bringup skill found without a perf model.
---

# Perf-model worker

You get one op name from Stage 6, along with `<ws>`, `<out>`, `<tools>`, `<platform>` and
the bringup rules in [SKILL.md](../SKILL.md). Follow the two guides in `<tools>`; do not
restate them:

- `PM=<tools>/TraceLens/Agent/PerfModel`
- Authoring guide: `$PM/unified_perf_model_extension/perf_report_model_extension.md`
- Validation guide: `$PM/validate-perf-model/validate_perf_model.md`

## Steps

1. **Record.** Read the op's rows in `<out>/perf_report_csvs/unified_perf_summary.csv`:
   `op category`, `Input Dims`, `Input type`, `Concrete Inputs`, the kernel names in
   `kernel_details_summary`, `entry_point` (the launcher), and `call_stack_full`. To trace
   the launcher into a vendor repo, run
   `python3 $PM/unified_perf_model_extension/scan_framework_ops.py --repo <vendor repo> --trace <csv>`.
2. **Decide.** Run `run_other_bucket_triage.py <csv> --check-mapping` to see whether an
   existing perf model already fits. Then pick full or categorize-only, as in the
   authoring guide's "Per-op choices".
3. **Build.** Create the op worktree:
   `git worktree add -b perf_model/<op_slug> <ws>/pm_<op_slug> origin/main`. Follow
   authoring Steps 1-7 there in Mode A (integrate). Use Mode B (`--emit-extension`) only
   when the user does not want a PR.
4. **Validate.** Register the op as the validation guide's "Adding an op" section
   describes, then run it against the op worktree:
   `PYTHONPATH=<ws>/pm_<op_slug> python3 $PM/validate-perf-model/validate_perf_model.py --from-report-dir <out> --only-ops <op> --arch <platform> --output-dir <ws>/validation/<op_slug>`.
5. **Draft.** Commit on `perf_model/<op_slug>`, then write `<ws>/drafts/<op_slug>_pr.md`
   for `AMD-AGI/TraceLens`. The draft needs:
   - what the op is and which perf model class or mapping was added;
   - one validation row per shape from these CSV columns only: `predicted_flops`,
     `predicted_bytes`, `rocprof_read_bytes`, `rocprof_write_bytes`, `flops_ratio`,
     `bytes_ratio`. Leave out `kernel_runtime_*`, `*_tflops_s` and `*_tb_s`.

   If validation fails, or a choice in step 2 needs the user, write
   `<ws>/drafts/<op_slug>_issue.md` instead. It holds the op record from step 1 and the
   failure, with the same column limits.
6. **Return** the branch, the draft path, and the validation CSV path. Do not push or
   open anything; the orchestrating agent does that after the user approves the draft.
