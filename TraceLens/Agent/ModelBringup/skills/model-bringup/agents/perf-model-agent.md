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
- `<op_worktree>`: the op's worktree, already created on its branch
- `<categorize_only>`: true when `gaps.json` marks the op categorize-only
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
- `<op_dir>/perf_model_pr.md` or `<op_dir>/perf_model_issue.md`, the draft body
- `<op_dir>/validation/validation_report_<ts>.csv`, written by the validator
- `<op_dir>/result.json`

---

## Output Schema (result.json)

```json
{"op": "<op>", "op_slug": "<op_slug>",
 "outcome": "modeled | blocked | error",
 "commit": "<sha or null>",
 "validation_csv": "<op_dir>/validation/validation_report_<ts>.csv | null",
 "reason": "<required unless outcome is modeled>"}
```

---

## Guardrails

- Exactly one op. Write only under `<op_worktree>` and `<op_dir>`; temp files and tool caches (such as `BLACK_CACHE_DIR`) go in `<op_dir>/scratch`. Commit only in `<op_worktree>`, on its checked-out branch.
- Never push, run `gh`, or ask the user. Answer every guide question per its Unattended use section; a choice that needs the user becomes outcome `blocked`.
- Run worktree code as `cd <op_worktree> && PYTHONPATH=<op_worktree> python3 ...` through the prefix.
- Run git in your own shell as `git -C <op_worktree> ...`, one git command per call, never through the prefix.

---

## Error Handling

**If a command cannot run:** write `result.json` with `outcome` `error`, and the command and the tail of its stderr in `reason`. Do not hand-write FLOPs or bytes.

**If the validator exits 1 with `FAIL` rows, or a guide's Error handling says stop:** that is a result, not an error. Go to Step 5 (Draft) with outcome `blocked` and the reason.

---

## Steps

### Step 1: Record

Read the op's rows in `<analysis_dir>/perf_report_csvs/unified_perf_summary.csv`: `op category`, `Input Dims`, `Input type`, `Concrete Inputs`, the kernel names in `kernel_details_summary`, `entry_point` (the launcher) and `call_stack_full`. Find the op's source from `entry_point` and the call stack under `<framework_src>`.

### Step 2: Decide

```bash
<prefix> python3 -m TraceLens.Agent.ModelBringup.utils.unified_perf_ops <analysis_dir>/perf_report_csvs/unified_perf_summary.csv --mode top-ops --threshold 0 --include-covered --check-mapping
```

This shows whether a built-in perf model already fits. Then choose the depth the authoring guide's Unattended use names. A categorize-only op is categorized, not modeled.

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

If every validation row is `OK`, the outcome is `modeled`. Commit in `<op_worktree>`:

```bash
git -C <op_worktree> add <each file you changed>
git -C <op_worktree> commit -m "feat(perfmodel): Add Performance Model for <op>"
```

Then write `<op_dir>/perf_model_pr.md`:

```markdown
## Operator

`<op>`: what it computes and the kernels it launches. It had no performance model.

## Changes

- Performance model: `<class>` in `<extension file>`, a subclass of `<base class>`. FLOPs are `<formula>`. Bytes are `<terms>`.
- Category mapping: `<op>` in `get_pseudo_op_mappings()` (`pseudo_ops_perf_utils.py`).
- Validator harness: `<name>` in `perf_model_harnesses.py`.
- Tests: `<test class>` in `tests/test_perfmodel_extensions.py`.

## Validation

Measured with rocprofv3 hardware counters. Ratios are measured over predicted. Every row validated OK.

| Shape | Predicted FLOPs | Predicted Bytes | Measured Read Bytes | Measured Write Bytes | FLOPs Ratio | Bytes Ratio |
|---|---|---|---|---|---|---|
| <named dims and dtype> | <predicted_flops> | <predicted_bytes> | <rocprof_read_bytes> | <rocprof_write_bytes> | <flops_ratio> | <bytes_ratio> |

## Ratio Deviations

- <ratio header> <value> (<shape>), <ratio header> <value> (<shape>): <cause>.
```

If a row is not `OK`, the op is categorize-only, or Step 2 (Decide) needs the user, the outcome is `blocked`. Write `<op_dir>/perf_model_issue.md`:

```markdown
## Operator

`<op>`: what it computes, the framework call that launches it, and its kernels.

## Observed Shapes

| Input Dims | Input Type | Concrete Inputs |
|---|---|---|
| <Input Dims> | <Input type> | <Concrete Inputs> |

## Blocking Reason

- <reason>

## Validation

Measured with rocprofv3 hardware counters. Ratios are measured over predicted.

| Shape | Predicted FLOPs | Predicted Bytes | Measured Read Bytes | Measured Write Bytes | FLOPs Ratio | Bytes Ratio |
|---|---|---|---|---|---|---|
| <named dims and dtype> | <predicted_flops> | <predicted_bytes> | <rocprof_read_bytes> | <rocprof_write_bytes> | <flops_ratio> | <bytes_ratio> |
```

Template rules:
- Keep the sections in order. Changes lists only the lines that apply. Omit Ratio Deviations when no ratio deviates, and the issue's Validation when there are no validation rows.
- Table values come from the validation CSV unchanged.
- Name no model, GPU, gfx target, host, topology, person or trace, and no local path or branch. No runtime or time-share numbers.
- Every ratio whose band is `deviation` appears under Ratio Deviations with its value as the CSV prints it, one line per cause. Take causes from the validation guide's Known ratio sources where one fits.

---

## Self-check

Write `<op_dir>/result.json`, then run:

```bash
<prefix> <bringup_gates> op-result --bringup-dir <bringup_dir> --result <op_dir>/result.json
```

On exit 1, fix the outcome or the draft (a non-`OK` row means `blocked`) and rerun. Never edit the validation CSV or name an older one. Max 2 attempts. An `error` outcome has nothing to fix.

Return "DONE".
