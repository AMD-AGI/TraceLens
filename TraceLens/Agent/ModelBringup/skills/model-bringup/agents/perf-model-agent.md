<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: perf-model-agent
description: Builds and validates the perf model for one op that has none, or only categorizes it, and drafts the PR or issue, in a worktree the model-bringup skill created. Never pushes.
---

# Perf-model sub-agent

Build and validate the perf model for one op that has none, or only categorize it, and draft the PR or issue. The model-bringup skill launches one sub-agent per op.

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
- Authoring skill: `<pm_dir>/skills/perf-model-generate/SKILL.md`; read its `reference.md` sections as each step needs them
- Validation skill: `<pm_dir>/skills/validate-perf-model/SKILL.md`; read its `reference.md` sections as each step needs them
- Draft templates: `TraceLens/Agent/ModelBringup/skills/model-bringup/templates/perf_model_pr.md` and `perf_model_issue.md`; copy the body below the header comment

`<pm_dir>` is `<op_worktree>/TraceLens/Agent/PerfModel`. Substitute it before a command goes through `<prefix>`. Follow the skills; do not restate them.

**Output files you must write:**
- `<op_dir>/perf_model_pr.md` or `<op_dir>/perf_model_issue.md`, the draft body
- `<op_dir>/validation/validation_report_<ts>.csv`, written by the validator
- `<op_dir>/result.json`

---

## Output Schema (result.json)

```json
{"op": "<op>", "op_slug": "<op_slug>",
 "outcome": "modeled | categorized | blocked | error",
 "commit": "<sha or null>",
 "validation_csv": "<op_dir>/validation/validation_report_<ts>.csv | null",
 "reason": "<required unless outcome is modeled or categorized>"}
```

---

## Guardrails

- Exactly one op. Write only under `<op_worktree>` and `<op_dir>`; temp files and tool caches (such as `BLACK_CACHE_DIR`) go in `<op_dir>/scratch`. Commit only in `<op_worktree>`, on its checked-out branch.
- Never push, run `gh`, or ask the user. Run each skill from the inputs this prompt gives: the authoring skill's Inputs defaults and Stop rules, and the validation skill's Unattended use. A choice that needs the user becomes outcome `blocked`.
- Run worktree code as `cd <op_worktree> && PYTHONPATH=<op_worktree> python3 ...` through the prefix.
- Run git in your own shell as `git -C <op_worktree> ...`, one git command per call, never through the prefix.

---

## Error Handling

**If a command cannot run:** write `result.json` with `outcome` `error`, and the command and the tail of its stderr in `reason`. Do not hand-write FLOPs or bytes.

**If the validator exits 1 with `FAIL` rows, or a skill's Stop rules or Error handling say stop:** that is a result, not an error. Go to Step 5 (Draft) with outcome `blocked` and the reason.

---

## Steps

### Step 1: Record

Read the op's rows in `<analysis_dir>/perf_report_csvs/unified_perf_summary.csv`: `op category`, `Input Dims`, `Input type`, `Concrete Inputs`, the kernel names in `kernel_details_summary`, `entry_point` (the launcher) and `call_stack_full`. Find the op's source from `entry_point` and the call stack under `<framework_src>`.

### Step 2: Decide

```bash
<prefix> python3 -m TraceLens.Agent.PerfModel.utils.unified_perf_ops <analysis_dir>/perf_report_csvs/unified_perf_summary.csv --mode top-ops --threshold 0 --include-covered --check-mapping
```

This shows whether a built-in perf model already fits. Then use the depth and output mode defaults from the authoring skill's Inputs. A categorize-only op is categorized, not modeled.

### Step 3: Build

Follow the authoring skill's workflow index, Steps 1-5, in `<op_worktree>`. Its Step 6, validation, is Step 4 here.

### Step 4: Validate

Register the op as the validation skill's "Adding an op" describes. The validator reads only rows that have a perf model, so regenerate the report from the op worktree first:

```bash
<prefix> python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference --output_dir <op_dir>/report --job <job>
<prefix> set -o pipefail; cd <op_worktree> && PYTHONPATH=<op_worktree> python3 -m TraceLens.Reporting.generate_perf_report_from_config --config <op_dir>/report/cache/run_config.yaml 2>&1 | tee <op_dir>/report/perf_report.log | tail -n 40
<prefix> <bringup_gates> cases --report-dir <op_dir>/report/perf_report_csvs --out <op_dir>/cases.json
<prefix> set -o pipefail; cd <op_worktree> && HIP_VISIBLE_DEVICES=<gpu_index> PYTHONPATH=<op_worktree> python3 -m TraceLens.Agent.PerfModel.validate_perf_model.validator --cases <op_dir>/cases.json --only-ops <op> --arch <gfx_arch> --output-dir <op_dir>/validation 2>&1 | tee <op_dir>/validation.log | tail -n 40
```

If `rocprofv3` aborts torch processes (some ROCm SDK images load two LLVM copies), pass `--rocprofv3-path` pointing at the SDK's `_rocm_sdk_core/bin/rocprofv3`.

### Step 5: Draft

First investigate every `deviation-suspect` row with the validation skill's "Investigating a deviation", which updates that row to `deviation-explained` when its cause is found. If every validation row then has `run_status` `OK` and `match_status` `match` or `deviation-explained`, the outcome is `modeled`. Commit in `<op_worktree>`:

```bash
git -C <op_worktree> add <each file you changed>
git -C <op_worktree> commit -m "feat(perfmodel): Add Performance Model for <op>"
```

Then write `<op_dir>/perf_model_pr.md` from the PR template and fill it in.

If the op is categorize-only, the outcome is `categorized`: there is no perf model to validate, so skip Step 4. Commit the category mapping in `<op_worktree>` the same way, with the message `feat(perfmodel): Categorize <op>`, set `validation_csv` to null, and write `<op_dir>/perf_model_pr.md` from the same template, keeping only Operator and Changes. Operator ends "It had no category."

If a row did not run (`run_status` not `OK`) or its `match_status` is `over-predicted`, `deviation-suspect` or `no prediction`, the op needs a pseudo op, or Step 2 (Decide) needs the user, the outcome is `blocked`. For a pseudo op, do not add it: run the authoring skill's stability and cost check ("Pseudo ops" in its reference.md), write its full table (including the measured overhead) to `<op_dir>/pseudo_op_check.md` for the user. Write `<op_dir>/perf_model_issue.md` from the issue template.

Template rules:
- Keep the sections in order. Changes lists only the lines that apply. Omit Ratio Deviations when no ratio deviates, the issue's Pseudo Op Check when no pseudo op is proposed, and the issue's Validation when there are no validation rows.
- The measured overhead of a pseudo op stays in `<op_dir>/pseudo_op_check.md`; the issue carries only its complexity and the events it adds.
- Use only the template's tables, with their columns unchanged; the issue's Validation table has the same FLOPs and bytes columns as the PR's.
- Table values come from the validation CSV unchanged.
- Name no model, GPU, gfx target, host, topology, person or trace, and no local path or branch. No runtime or time-share numbers.
- Every ratio whose band is `deviation` appears under Ratio Deviations with its value as the CSV prints it, one line per cause. Find each cause with the validation skill's "Investigating a deviation": the line names the mechanism, its source file and line (package-relative, e.g. `aiter/ops/...py:123`), and the arithmetic that reproduces the ratio. Known ratio sources and the CSV `comment` are only hypotheses. A ratio without that evidence is written as unexplained.

---

## Self-check

Write `<op_dir>/result.json`, then run:

```bash
<prefix> <bringup_gates> op-result --bringup-dir <bringup_dir> --result <op_dir>/result.json
```

On exit 1, fix the outcome or the draft (a row that did not run, or whose `match_status` is not `match` or `deviation-explained`, means `blocked`) and rerun. Edit the validation CSV only as "Investigating a deviation" allows (one row's `match_status` and `comment`, after the investigation), and never name an older one. Max 2 attempts. An `error` outcome has nothing to fix.

Return "DONE".
