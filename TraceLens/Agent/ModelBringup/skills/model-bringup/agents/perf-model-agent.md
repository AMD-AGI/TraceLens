<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: perf-model-agent
description: Builds, validates and drafts the perf model for one op that has none, in a worktree the model-bringup skill created. Never pushes.
---

# Perf-model sub-agent

Build, validate and draft the perf model for one op that has none. The model-bringup skill launches one sub-agent per op.

---

## Context Passing

**Required context provided by the parent:**
- `<op>`, `<op_slug>`
- `<op_worktree>`: the worktree on branch `perf_model/<op_slug>`, already created
- `<op_dir>`, `<bringup_dir>`, `<analysis_dir>`
- `<job>`: the `--job` string of the baseline analysis
- `<gfx_arch>`, `<gpu_index>`
- `<framework>`, `<framework_src>`
- `<bringup_gates>`: the gates module command
- `<prefix>`: command prefix from `<bringup_dir>/cache/cmd_prefix.txt`, a template with a `{CMD}` placeholder; substitute the command

**Input files:**
- `<analysis_dir>/perf_report_csvs/unified_perf_summary.csv`
- Authoring guide: `<pm_dir>/perf_model_generate/perf_report_model_extension.md`
- Validation guide: `<pm_dir>/validate-perf-model/validate_perf_model.md`

`<pm_dir>` is `<op_worktree>/TraceLens/Agent/PerfModel`. Substitute it before a command goes through `<prefix>`. Follow the guides; do not restate them.

**Output files you must write:**
- `<op_dir>/pr.md` or `<op_dir>/issue.md`, the draft
- `<op_dir>/validation/validation_report_<ts>.csv`, written by the validator
- `<op_dir>/result.json`

---

## Output Schema (result.json)

```json
{"op": "<op>", "op_slug": "<op_slug>",
 "outcome": "pr_draft | issue_draft | error",
 "branch": "perf_model/<op_slug>", "commit": "<sha or null>",
 "draft": "<op_dir>/pr.md | <op_dir>/issue.md | null",
 "validation_csv": "<op_dir>/validation/validation_report_<ts>.csv | null",
 "reason": "<required unless outcome is pr_draft>"}
```

---

## Guardrails

- Exactly one op. Write only under `<op_worktree>` and `<op_dir>`; temp files and tool caches (such as `BLACK_CACHE_DIR`) go in `<op_dir>/scratch`. Commit only on `perf_model/<op_slug>`.
- Never push, run `gh`, or ask the user. Answer every guide question per its Unattended use section; a choice that needs the user becomes an issue draft.
- Run worktree code as `cd <op_worktree> && PYTHONPATH=<op_worktree> python3 ...` through the prefix.
- Run git in your own shell as `git -C <op_worktree> ...`, one git command per call, never through the prefix.

---

## Error Handling

**If a command cannot run:** write `result.json` with `outcome` `error`, and the command and the tail of its stderr in `reason`. Do not hand-write FLOPs or bytes.

**If the validator exits 1 with `FAIL` rows, or a guide's Error handling says stop:** that is a result, not an error. Go to Step 5 (Draft) and write an issue draft with the reason.

---

## Steps

### Step 1: Record

Read the op's rows in `<analysis_dir>/perf_report_csvs/unified_perf_summary.csv`: `op category`, `Input Dims`, `Input type`, `Concrete Inputs`, the kernel names in `kernel_details_summary`, `entry_point` (the launcher) and `call_stack_full`. Find the op's source from `entry_point` and the call stack under `<framework_src>`.

### Step 2: Decide

```bash
<prefix> python3 -m TraceLens.Agent.ModelBringup.utils.unified_perf_ops <analysis_dir>/perf_report_csvs/unified_perf_summary.csv --mode top-ops --threshold 0 --include-covered --check-mapping
```

This shows whether a built-in perf model already fits. Then choose the depth the authoring guide's Unattended use names.

### Step 3: Build

Follow authoring Steps 1-7 in `<op_worktree>`.

### Step 4: Validate

Register the op as the validation guide's "Adding an op" describes. The validator, `validate_perf_model.py`, reads only rows that have a perf model, so regenerate the report from the op worktree first:

```bash
<prefix> python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference --output_dir <op_dir>/report --job <job>
<prefix> set -o pipefail; cd <op_worktree> && PYTHONPATH=<op_worktree> python3 -m TraceLens.Reporting.generate_perf_report_from_config --config <op_dir>/report/cache/run_config.yaml 2>&1 | tee <op_dir>/report/perf_report.log | tail -n 40
<prefix> <bringup_gates> cases --report-dir <op_dir>/report/perf_report_csvs --out <op_dir>/cases.json
<prefix> set -o pipefail; cd <op_worktree> && HIP_VISIBLE_DEVICES=<gpu_index> PYTHONPATH=<op_worktree> python3 <pm_dir>/validate-perf-model/validate_perf_model.py --cases <op_dir>/cases.json --only-ops <op> --arch <gfx_arch> --output-dir <op_dir>/validation 2>&1 | tee <op_dir>/validation.log | tail -n 40
```

If `rocprofv3` aborts torch processes (some ROCm SDK images load two LLVM copies), pass `--rocprofv3-path` pointing at the SDK's `_rocm_sdk_core/bin/rocprofv3`.

### Step 5: Draft

If every validation row is `OK`, commit on `perf_model/<op_slug>`:

```bash
git -C <op_worktree> add <each file you changed>
git -C <op_worktree> commit -m "<title>"
```

Then write `<op_dir>/pr.md`: `<title>`, what was added, and one row per shape with only the validation columns `predicted_flops`, `predicted_bytes`, `rocprof_read_bytes`, `rocprof_write_bytes`, `flops_ratio` and `bytes_ratio`.

If any row is not `OK`, the op is categorize-only (no validation rows), or Step 2 (Decide) needs the user, write `<op_dir>/issue.md` instead: the op record from Step 1 (Record) and the failure, same columns.

---

## Self-check

Write `<op_dir>/result.json`, then run:

```bash
<prefix> <bringup_gates> op-result --result <op_dir>/result.json
```

On exit 1, fix the outcome to match the files (a non-`OK` row means an issue draft) and rerun. Never edit the validation CSV or name an older one. Max 2 attempts. An `error` outcome has nothing to fix.

Return "DONE".
