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
- `<templates_dir>`: the draft templates directory
- `<prefix>`: command prefix from `<bringup_dir>/cache/cmd_prefix.txt`, a template with a `{CMD}` placeholder; substitute the command

**Input files:**
- `<analysis_dir>/perf_report_csvs/unified_perf_summary.csv`
- `<bringup_dir>/cache/model_config.json`
- `<authoring_skill>`: `<op_worktree>/TraceLens/Agent/PerfModel/skills/perf-model-generate/SKILL.md`; read its `reference.md` sections as each step needs them
- `<validation_skill>`: `<op_worktree>/TraceLens/Agent/PerfModel/skills/validate-perf-model/SKILL.md`; read its `reference.md` sections as each step needs them
- Draft templates: `<templates_dir>/perf_model_pr.md` and `perf_model_issue.md`; follow the rules in each template's header comments and copy the body below them

Follow the skills; do not restate them.

**Output files you must write:**
- `<op_dir>/perf_model_pr.md` or `<op_dir>/perf_model_issue.md`, the draft body
- `<op_dir>/validation/validation_report.csv`, written by the validator
- `<op_dir>/result.json`

---

## Output Schema (result.json)

```json
{"op": "<op>", "op_slug": "<op_slug>",
 "outcome": "modeled | categorized | blocked | error",
 "commit": "<sha or null>",
 "validation_csv": "<op_dir>/validation/validation_report.csv when the validator wrote one, else null",
 "reason": "<required unless outcome is modeled or categorized>"}
```

---

## Guardrails

- Exactly one op. Write only under `<op_worktree>` and `<op_dir>`; temp files and tool caches (such as `BLACK_CACHE_DIR`) go in `<op_dir>/scratch`. Commit only in `<op_worktree>`, on its checked-out branch.
- Never push, run `gh`, or ask the user. Run each skill from the inputs this prompt gives: each skill's Inputs defaults and Stop rules. A choice that needs the user becomes outcome `blocked`.
- Run worktree code as `cd <op_worktree> && PYTHONPATH=<op_worktree> python3 ...` through the prefix.
- Run git in your own shell as `git -C <op_worktree> ...`, one git command per call, never through the prefix.

---

## Error Handling

**If a command cannot run:** write `result.json` with `outcome` `error`, and the command and the tail of its stderr in `reason`. Do not hand-write FLOPs or bytes.

**If the validator exits 1 with `FAIL` rows, or a skill's Stop rules or Error handling say stop:** that is a result, not an error. Go to Draft with outcome `blocked` and the reason.

---

## Steps

### Step 1: Record

Read the op's rows in `<analysis_dir>/perf_report_csvs/unified_perf_summary.csv`: `op category`, `Input Dims`, `Input type`, `Concrete Inputs`, the kernel names in `kernel_details_summary`, `entry_point` (the launcher) and `call_stack_full`. Find the op's source from `entry_point` and the call stack under `<framework_src>`.

Read `model_config.json`: `launch` holds the run's TP, EP and DP settings verbatim (under AgentX the recipe overrides them); trace `Input Dims` are per rank and win; report a mismatch no setting explains in the draft, never patch it; `null` is unknown. `paper` links the model's paper or tech report (`null` when unknown); `<authoring_skill>`'s Locate step uses it.

### Step 2: Build

When `<categorize_only>` is true, the depth is categorize-only (`<authoring_skill>`'s "Per-op choices"); go to Draft after "Regenerate and check".

Follow `<authoring_skill>`'s workflow index, from "Per-op choices" through "Regenerate and check", in `<op_worktree>`; its "Hand off to validation" is the Validate step here. For "Regenerate and check", write the report to `<op_dir>/report`:

```bash
<prefix> python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference --output_dir <op_dir>/report --job <job>
<prefix> set -o pipefail; cd <op_worktree> && PYTHONPATH=<op_worktree> python3 -m TraceLens.Reporting.generate_perf_report_from_config --config <op_dir>/report/cache/run_config.yaml 2>&1 | tee <op_dir>/report/perf_report.log | tail -n 40
```

### Step 3: Validate

Register the op as `<validation_skill>`'s "Adding an op" describes. The validator reads only rows that have a perf model, so run it on the Build report:

```bash
<prefix> <bringup_gates> cases --report-dir <op_dir>/report/perf_report_csvs --out <op_dir>/cases.json
<prefix> set -o pipefail; cd <op_worktree> && HIP_VISIBLE_DEVICES=<gpu_index> PYTHONPATH=<op_worktree> python3 -m TraceLens.Agent.PerfModel.validate_perf_model.validator --cases <op_dir>/cases.json --only-ops <op> --arch <gfx_arch> --output-dir <op_dir>/validation 2>&1 | tee <op_dir>/validation.log | tail -n 40
```

If `rocprofv3` aborts, see `<validation_skill>`'s Prerequisites.

### Step 4: Draft

First investigate every `deviation-suspect` row with `<validation_skill>`'s "Investigating a deviation". Then pick the outcome:

- `modeled`: every validation row has `run_status` `OK` and `match_status` `match` or `deviation-explained`. Commit as `feat(perfmodel): Add Performance Model for <op>` and write `<op_dir>/perf_model_pr.md` from the PR template.
- `categorized`: `<categorize_only>` is true. Commit the category mapping as `feat(perfmodel): Categorize <op>` and write `<op_dir>/perf_model_pr.md` from the PR template.
- `blocked`: otherwise, including when the op needs a pseudo op or a skill's Stop rules say stop. Write `<op_dir>/perf_model_issue.md` from the issue template. For a pseudo op, do not add it: run `<authoring_skill>`'s stability and cost check ("Pseudo ops" in its reference.md), write its full table to `<op_dir>/pseudo_op_check.md`, and put the table without the measured overhead in the issue's Pseudo Op Check.

For `modeled` and `categorized`, commit in `<op_worktree>`:

```bash
git -C <op_worktree> add <each file you changed>
git -C <op_worktree> commit -m "<message>"
```

Fill each draft with table values from the validation CSV unchanged, and name no model, GPU, gfx target, host, topology, person, trace, local path or branch, and give no runtime or time-share numbers.

---

## Self-check

Write `<op_dir>/result.json`, then run:

```bash
<prefix> <bringup_gates> op-result --bringup-dir <bringup_dir> --result <op_dir>/result.json
```

On exit 1, fix the outcome or the draft and rerun. Edit the validation CSV only as `<validation_skill>`'s "Investigating a deviation" allows, and never name an older one. Max 2 attempts. An `error` outcome has nothing to fix.

Return "DONE".
