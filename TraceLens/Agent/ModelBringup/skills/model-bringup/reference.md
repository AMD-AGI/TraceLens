<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model bringup: full procedure

Run the steps in order.

## Placeholders

User inputs, the keys of `bringup_config.yaml` ([template](templates/bringup_config.yaml)):

| Placeholder | Key | Meaning |
|---|---|---|
| `<node>` | `node` | GPU node with Docker |
| `<magpie_dir>` | `magpie_dir` | Magpie checkout on `<node>`; Magpie runs from it on the host |
| `<tracelens_dir>` | `tracelens_dir` | TraceLens git checkout on `<node>` with an `origin` remote; its checked-out branch is the base for the per-op branches and PRs |
| `<tracelens_internal_dir>` | `tracelens_internal_dir` | TraceLens-internal checkout on `<node>` (arch files for GPUs not bundled in TraceLens) |
| `<bringup_dir>` | `bringup_dir` | Output root on `<node>`. Steps pass data only through files under `<bringup_dir>`. |
| `<benchmark_yaml>` | `benchmark_yaml` | Optional Magpie benchmark YAML; never edited |
| `<model_id>`, `<framework>`, `<serving_image>`, `<hf_home>`, `<workload>` | `model`, `framework`, `docker_image`, `hf_cache_path`, `workload` | From `<benchmark_yaml>` when set, else from the config. `<workload>` is `fixed_seq` or `agentx` |

Options under `options:` with their defaults: `min_share` the `gaps --threshold` default (Step 5 threshold),
`max_profile_retries` 2, `accept_triage` none (triage check names, such as `CPU_CALL_STACK_MISSING`, that Steps 2 and 4 do not block on).

Derived:

| Placeholder | Value | Set in |
|---|---|---|
| `<magpie_prefix>` | `ssh <node> "cd <magpie_dir> && PYTHONPATH=<magpie_dir> {CMD}"` | 0 |
| `<container>` | `bringup_<basename of <bringup_dir>>` | 0 |
| `<prefix>` | `ssh <node> "docker exec -i -w /tmp <container> bash -c '{CMD}'"` | 0 |
| `<bringup_gates>` | `python3 -m TraceLens.Agent.ModelBringup.utils.bringup_gates` | 0 |
| `<tl_extension>`, `<framework_src>`, `<gfx_arch>`, `<gpu_count>`, `<platform>` | printed in Step 0 | 0 |
| `<base_branch>` | `git -C <tracelens_dir> rev-parse --abbrev-ref HEAD` | 6, Publish |
| `<profile_dir>` | `<bringup_dir>/profile/attempt_<n>`, `<n>` from 1 | 1 |
| `<bench_dir>`, `<mixed_split>`, `<capture_folder>` | printed by `prepare` | 2 |
| `<job>` | `trace_path=<mixed_split>,platform=<platform>[,capture_folder=<capture_folder>]` | 2 |
| `<analysis_dir>` | `<bringup_dir>/analysis` | 3 |
| `<op>`, `<op_slug>` | `ops[]` in `<bringup_dir>/gaps.json` | 5 |
| `<gpu_index>` | a free index in 0..`<gpu_count>`-1, one per running sub-agent | 6 |
| `<op_worktree>`, `<op_dir>` | `<bringup_dir>/worktrees/pm_<op_slug>`, `<bringup_dir>/perf_models/<op_slug>` | 6 |

When `<node>` is the machine you run on (`localhost` or its hostname), drop every
`ssh <node> "..."` wrapper and run the inner command directly.

Read `<bringup_dir>/cache/cmd_prefix.txt` and substitute the command for `{CMD}`, escaping
quotes for the `ssh` and `bash -c` layers. `<prefix> <command>` below means that.

To stop at any step after Step 0: `<prefix> <bringup_gates> summary --bringup-dir <bringup_dir> --step <N> --code <CODE> --reason "<one line>"`, then end. Each stop point below names its `<CODE>`; the reason is the failing command's error or the `reason` that stopped the step.

## Step 0: Environment and Input Sanity Checks

1. **Mode.** With a named `bringup_config.yaml`, read it, apply prompt overrides, ask
   nothing. Without one, ask for `node`, `magpie_dir`, `tracelens_dir`,
   `tracelens_internal_dir`, `bringup_dir`, then `benchmark_yaml` or the model inputs, one
   question each.

2. **Inputs.**

   ```bash
   <magpie_prefix> python3 -c "import yaml, numpy, Magpie"
   ssh <node> "set -o pipefail; mkdir -p <bringup_dir>/cache && ls -d <tracelens_internal_dir>/*/Agent/Analysis/utils/arch | sed 's|.*/\([^/]*\)/Agent/Analysis/utils/arch|tl_extension=\1|'"
   ```

   Write `<magpie_prefix>` and `<prefix>` to `cache/magpie_prefix.txt` and `cache/cmd_prefix.txt`, and the effective inputs to `cache/bringup_config.yaml`.

3. **Container.**

   ```bash
   ssh <node> "docker inspect -f '{{.Config.Image}} {{.State.Running}}' <container> 2>/dev/null"
   ```

   No output: start it.
   Stopped with `<serving_image>`: `docker start` it.
   Running with `<serving_image>`: reuse it.
   Any other image: end with `[DIAG:bringup:STEP0_CONTAINER_IMAGE_MISMATCH]` and the image; Step 8 does not run.

   ```bash
   ssh <node> "docker run -d --name <container> --device=/dev/kfd --device=/dev/dri --group-add video \
     --ipc=host --network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     -v <bringup_dir>:<bringup_dir> -v <hf_home>:<hf_home> -v <tracelens_dir>:<tracelens_dir> \
     -v <tracelens_internal_dir>:<tracelens_internal_dir> -e HF_HOME=<hf_home> -e TL_EXTENSION=<tl_extension> \
     -e PYTHONDONTWRITEBYTECODE=1 --entrypoint tail <serving_image> -f /dev/null"
   ```

4. **TraceLens.**

   ```bash
   <prefix> pip install -e <tracelens_dir> && pip install --no-deps -e <tracelens_internal_dir>
   ```

   On `No module named <m>` later, `<prefix> pip install <m>` and retry.

5. **GPU.**

   ```bash
   <prefix> python3 -c "import os, torch, <framework> as f; from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms; p = torch.cuda.get_device_properties(0); print(dict(framework_src=os.path.dirname(f.__file__), gfx_arch=p.gcnArchName.split(chr(58))[0], gpu_count=torch.cuda.device_count(), platform=next((x for x in list_platforms() if x in p.name), None)))"
   ```

**Gate:** every command succeeds and `platform` is not `None`. Otherwise end with `[DIAG:bringup:STEP0_ENV_CHECK_FAILED]` and the failing command; Step 8 does not run.

## Step 1: Profile

    ssh <node> "mkdir -p <profile_dir>"

Then launch one sub-agent:

    Read and follow the FULL instructions in:
      TraceLens/Agent/Profiling/skills/magpie-benchmark-profiling/reference.md

    **Execution Context:**
    - Mode: unattended
    - Workload: <workload>; benchmark YAML: <benchmark_yaml> (or model <model_id>, framework <framework>, image <serving_image>, HF cache <hf_home>), copy to <profile_dir>/benchmark.yaml
    - Node: <node>; Magpie command: <magpie_prefix> python3 -m Magpie ...
    - Container: <container>; command prefix: Read <bringup_dir>/cache/cmd_prefix.txt and substitute {CMD}
    - TraceLens repo: <tracelens_dir>
    - Output directory: <profile_dir>
    - Output file: <profile_dir>/result.json

    Execute every step in the skill. Return "DONE" when complete.

**Gate:** `<profile_dir>/result.json` has `outcome` `profiled`. Otherwise retry (below).

**Retry:** on `retry`: if `<n>` is at most `max_profile_retries`, rerun Step 1 with `<n>` + 1; otherwise stop with `PROFILE_RETRIES_EXHAUSTED`. On `stop`, stop with `PROFILE_STOPPED`. Anything else, including a missing or unparsable file or an unknown outcome, retries.

## Step 2: Profile health

    <prefix> <bringup_gates> prepare --bringup-dir <bringup_dir> --profile-dir <profile_dir> --platform <platform>
    <prefix> python3 -m TraceLens.Agent.Analysis.triage.runner --bench-dir <bench_dir> --out-dir <bringup_dir>/profile_check
    <prefix> <bringup_gates> report-health --phase profile --bringup-dir <bringup_dir> --triage-dir <bringup_dir>/profile_check [--accept <check> ...]

Ignore the runner's exit code.

**Gate:** both gates exit 0. Exit 1 from either: retry Step 1. Exit 2: stop with `PROFILE_HEALTH_INPUT_ERROR`.

## Step 3: Baseline analysis

    <prefix> python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference \
      --output_dir <analysis_dir> --job <job>
    <prefix> cp <analysis_dir>/cache/run_config.yaml <bringup_dir>/cache/analysis_run.yaml

Then follow inline:

    Follow the analysis orchestrator installed with TraceLens and run the full agentic analysis workflow from config <bringup_dir>/cache/analysis_run.yaml, standalone, node <node>, container <container>, TraceLens repo <tracelens_dir>, output to <analysis_dir>. Ask nothing. Run the orchestrator Step 1 perf report command through the prefix with `{CMD}` set to `set -o pipefail; <command> 2>&1 | tee <analysis_dir>/perf_report.log | tail -n 40`.

**Gate:** `analysis.md`, `perf_report_csvs/unified_perf_summary.csv` and `perf_report.log`
under `<analysis_dir>` are non-empty. Otherwise stop with `ANALYSIS_OUTPUT_MISSING`.

## Step 4: Report health

    <prefix> python3 -m TraceLens.Agent.Analysis.triage.runner --run-dir <analysis_dir> --stream-file <analysis_dir>/perf_report.log
    <prefix> <bringup_gates> report-health --phase report --bringup-dir <bringup_dir> --triage-dir <analysis_dir> [--accept <check> ...]

Ignore the runner's exit code. The gate JSON carries each blocking tag with its remedy and
the graph capture merge mapping.

**Gate:** exit 0. Otherwise stop with `REPORT_HEALTH_BLOCKED`.

## Step 5: Find gaps

    <prefix> <bringup_gates> gaps --bringup-dir <bringup_dir> --unified-csv <analysis_dir>/perf_report_csvs/unified_perf_summary.csv [--threshold <min_share>]

Pass `--threshold` only when `min_share` is set. Do not edit `gaps.json`. An empty `ops` list goes to Step 8.

## Step 6: Perf-model sub-agents

1. In your own shell, one at a time:

       git -C <tracelens_dir> worktree add -b perf_model/<op_slug> <op_worktree> <base_branch>

   On a rerun, reuse `<op_worktree>`; if only the branch exists, add the worktree from it. If
   another bringup already has `perf_model/<op_slug>` checked out (same op on another
   model), use that worktree as `<op_worktree>`, so one branch and one PR cover the op.

2. Launch one sub-agent per op, at most `<gpu_count>` at once, each with its own `<gpu_index>`:

       Read and follow the FULL instructions in:
       TraceLens/Agent/ModelBringup/skills/model-bringup/agents/perf-model-agent.md

       **Execution Context:**
       - Op: <op>, op slug: <op_slug>
       - Op worktree (branch perf_model/<op_slug>, already created): <op_worktree>
       - Op output dir: <op_dir>; bringup dir: <bringup_dir>; baseline analysis dir: <analysis_dir>
       - Baseline job: <job>
       - GPU arch: <gfx_arch>; GPU index: <gpu_index>
       - Framework: <framework>, source <framework_src>
       - Gates: <bringup_gates>
       - Command prefix: Read <bringup_dir>/cache/cmd_prefix.txt and substitute {CMD}
       - Output file: <op_dir>/result.json

       Execute every step in the sub-agent file. Return "DONE" when complete.

**Gate:** `<prefix> <bringup_gates> op-result --result <op_dir>/result.json` exits 0 for each
op. Otherwise relaunch that sub-agent once; if it fails again, the op is excluded.

## Step 7: Coverage

Skip when no op has outcome `pr_draft`.

1. In your own shell, merge each `pr_draft` branch in `gaps.json` order:

       git -C <tracelens_dir> worktree add --detach <bringup_dir>/worktrees/pm_integration <base_branch>
       git -C <bringup_dir>/worktrees/pm_integration merge --no-edit perf_model/<op_slug>

   On a conflict, `merge --abort` and continue.
2. Rerun only the perf report from the merged tree:

       <prefix> python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference \
         --output_dir <bringup_dir>/coverage --job <job>
       <prefix> set -o pipefail; cd <bringup_dir>/worktrees/pm_integration && PYTHONPATH=<bringup_dir>/worktrees/pm_integration python3 -m TraceLens.Reporting.generate_perf_report_from_config --config <bringup_dir>/coverage/cache/run_config.yaml 2>&1 | tee <bringup_dir>/coverage/perf_report.log | tail -n 40

3. `<prefix> <bringup_gates> coverage --bringup-dir <bringup_dir> --before <analysis_dir>/perf_report_csvs/unified_perf_summary.csv --after <bringup_dir>/coverage/perf_report_csvs/unified_perf_summary.csv`

**Gate:** exit 0. Otherwise stop with `COVERAGE_FAILED`; the gate JSON names each failing op.

## Step 8: Summary

Always runs, including after a stop:

    <prefix> <bringup_gates> summary --bringup-dir <bringup_dir> [--step <N> --code <CODE> --reason "<one line>"]

Reply with `bringup_summary.md`, the cleanup command `ssh <node> "docker rm -f <container>"`,
and the Publish prompt it lists.

## Publish

Runs only on a prompt that says publish, names `<bringup_dir>`, and lists op slugs. Run
everything in your own shell. For each named op, read `<op_dir>/result.json`; skip it if
`<op_dir>/published_url.txt` exists.

- `pr_draft`:

      git -C <op_worktree> push -u origin perf_model/<op_slug>
      gh pr create --repo AMD-AGI/TraceLens --base <base_branch> --head perf_model/<op_slug> \
        --title "$(head -n 1 <op_dir>/pr.md)" --body "$(tail -n +2 <op_dir>/pr.md)"

- `issue_draft`:

      gh issue create --repo AMD-AGI/TraceLens --title "$(head -n 1 <op_dir>/issue.md)" --body "$(tail -n +2 <op_dir>/issue.md)"

Write the URL `gh` prints to `<op_dir>/published_url.txt`.
