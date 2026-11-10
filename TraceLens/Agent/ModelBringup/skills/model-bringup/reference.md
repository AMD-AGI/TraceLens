<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model bringup: full procedure

Run the steps in order. Steps pass data only through files under `<bringup_dir>`.

## Placeholders

User inputs, the keys of `bringup_config.yaml` ([template](templates/bringup_config.yaml)):

| Placeholder | Key | Meaning |
|---|---|---|
| `<node>` | `node` | GPU node with Docker |
| `<magpie_dir>` | `magpie_dir` | Magpie checkout on `<node>`; Magpie runs from it on the host |
| `<tracelens_dir>` | `tracelens_dir` | TraceLens git checkout on `<node>` with an `origin` remote |
| `<tracelens_internal_dir>` | `tracelens_internal_dir` | TraceLens-internal checkout on `<node>` (arch files for GPUs not bundled in TraceLens) |
| `<bringup_dir>` | `bringup_dir` | Output root on `<node>` |
| `<benchmark_yaml>` | `benchmark_yaml` | Optional Magpie benchmark YAML; never edited |
| `<model_id>`, `<framework>`, `<serving_image>`, `<hf_home>`, `<workload>` | `model`, `framework`, `docker_image`, `hf_cache_path`, `workload` | From `<benchmark_yaml>` when set, else from the config. `<workload>` is `fixed_seq` or `agentx` |

Options under `options:` with their defaults: `min_share` 0.02 (Step 5 threshold),
`max_profile_retries` 2, `accept_triage` none (sublabels Steps 2 and 4 do not block on).

Derived:

| Placeholder | Value | Set in |
|---|---|---|
| `<magpie_prefix>` | `ssh <node> "cd <magpie_dir> && PYTHONPATH=<magpie_dir> {CMD}"` | 0 |
| `<container>` | `bringup_<basename of <bringup_dir>>` | 0 |
| `<prefix>` | `ssh <node> "docker exec <container> bash -c '{CMD}'"` | 0 |
| `<tl_extension>`, `<framework_src>`, `<gfx_arch>`, `<gpu_count>`, `<platform>` | printed in Step 0 | 0 |
| `<profile_dir>` | `<bringup_dir>/profile/attempt_<n>`, `<n>` from 1 | 1 |
| `<bench_dir>`, `<mixed_split>`, `<capture_folder>` | printed by `prepare` | 2 |
| `<analysis_dir>` | `<bringup_dir>/analysis` | 3 |
| `<op>`, `<op_slug>` | `ops[]` in `<bringup_dir>/gaps.json` | 5 |
| `<op_worktree>`, `<op_dir>` | `<bringup_dir>/worktrees/pm_<op_slug>`, `<bringup_dir>/perf_models/<op_slug>` | 6 |

Read `<bringup_dir>/cache/cmd_prefix.txt` and substitute the command for `{CMD}`, escaping
quotes for the `ssh` and `bash -c` layers. `<prefix> <command>` below means that. `G` stands
for `python3 -m TraceLens.Agent.ModelBringup.utils.bringup_gates`.

To stop at any step: `<prefix> G summary --bringup-dir <bringup_dir> --stop-reason "Step <N>: <reason>"`, then end.

## Step 0: Environment and inputs

1. **Mode.** With a named `bringup_config.yaml`, read it, apply prompt overrides, ask
   nothing. Without one, ask for `node`, `magpie_dir`, `tracelens_dir`,
   `tracelens_internal_dir`, `bringup_dir`, then `benchmark_yaml` or the model inputs, one
   question each.
2. **Inputs.**

       <magpie_prefix> python3 -c "import yaml, numpy, Magpie"
       ssh <node> "mkdir -p <bringup_dir>/cache && ls -d <tracelens_internal_dir>/*/Agent/Analysis/utils/arch"

   The first checks that the host Python can run Magpie from `<magpie_dir>` (its only
   dependencies are PyYAML and numpy); on failure, stop with its output. The second prints
   the dir whose parent is `<tl_extension>`. Write
   `<magpie_prefix>` and `<prefix>` to `cache/magpie_prefix.txt` and `cache/cmd_prefix.txt`,
   and the effective inputs to `cache/bringup_config.yaml`.
3. **Container.** `ssh <node> "docker inspect -f '{{.Config.Image}} {{.State.Running}}' <container> 2>/dev/null"`.
   No output: start it. Stopped with `<serving_image>`: `docker start` it. Running with
   `<serving_image>`: reuse it. Any other image: stop.

       ssh <node> "docker run -d --name <container> --device=/dev/kfd --device=/dev/dri --group-add video \
         --ipc=host --network=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
         -v <bringup_dir>:<bringup_dir> -v <hf_home>:<hf_home> -v <tracelens_dir>:<tracelens_dir> \
         -v <tracelens_internal_dir>:<tracelens_internal_dir> -e HF_HOME=<hf_home> -e TL_EXTENSION=<tl_extension> \
         -e PYTHONDONTWRITEBYTECODE=1 --entrypoint tail <serving_image> -f /dev/null"

4. **TraceLens.** `<prefix> pip install --no-deps -e <tracelens_dir> -e <tracelens_internal_dir>`.
   On `No module named <m>` later, `<prefix> pip install <m>` and retry.
5. **GPU.**

       <prefix> python3 -c "import os, torch, <framework> as f; from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms; p = torch.cuda.get_device_properties(0); print(os.path.dirname(f.__file__), p.gcnArchName.split(':')[0], torch.cuda.device_count(), *[x for x in list_platforms() if x in p.name])"

   It prints `<framework_src> <gfx_arch> <gpu_count> <platform>`.

**Gate:** every command succeeds and `<platform>` is set. Otherwise end with the failing
command; Step 8 does not run.

## Step 1: Profile

    ssh <node> "mkdir -p <profile_dir>"

Then follow inline:

    Follow the Magpie Benchmark + Profiling skill installed with TraceLens in unattended mode: workload <workload>, benchmark YAML <benchmark_yaml> (or model <model_id>, framework <framework>, image <serving_image>, HF cache <hf_home>), copy to <profile_dir>/benchmark.yaml, node <node>, Magpie repo <magpie_dir> run as `<magpie_prefix> python3 -m Magpie ...` (no conda env), TraceLens repo <tracelens_dir>, output to <profile_dir>. Patch the image per Step 1b in unattended mode. Run the split in <container> through <prefix>.

**Gate:** Magpie exits 0. Otherwise retry (below). A failed image build is a stop, not a retry.

**Retry:** if `<n>` is at most `max_profile_retries`, rerun Step 1 with `<n>` + 1; otherwise stop.

## Step 2: Profile health

    <prefix> G prepare --bringup-dir <bringup_dir> --profile-dir <profile_dir> --platform <platform>
    <prefix> python3 -m TraceLens.Agent.Analysis.triage.runner --bench-dir <bench_dir> --out-dir <bringup_dir>/profile_check
    <prefix> G report-health --phase profile --bringup-dir <bringup_dir> --triage-dir <bringup_dir>/profile_check [--accept <sublabel> ...]

`prepare` prints `bench_dir=`, `mixed_split=`, `capture_folder=` (only the replay rank's
captures, linked) and `workload=`. Ignore the runner's exit code.

**Gate:** both gates exit 0. Exit 1 from either: retry Step 1. Exit 2: stop.

## Step 3: Baseline analysis

    <prefix> python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference \
      --output_dir <analysis_dir> --job trace_path=<mixed_split>,platform=<platform>[,capture_folder=<capture_folder>]

Then follow inline:

    Follow the analysis orchestrator installed with TraceLens and run the full agentic analysis workflow from config <analysis_dir>/cache/run_config.yaml, standalone, node <node>, container <container>, TraceLens repo <tracelens_dir>, output to <analysis_dir>. Ask nothing. Run the Step 1 perf report command as `set -o pipefail; <command> 2>&1 | tee <analysis_dir>/perf_report.log`.

**Gate:** `analysis.md`, `perf_report_csvs/unified_perf_summary.csv` and `perf_report.log`
under `<analysis_dir>` are non-empty. Otherwise stop.

## Step 4: Report health

    <prefix> python3 -m TraceLens.Agent.Analysis.triage.runner --run-dir <analysis_dir> --stream-file <analysis_dir>/perf_report.log
    <prefix> G report-health --phase report --bringup-dir <bringup_dir> --triage-dir <analysis_dir> [--accept <sublabel> ...]

Ignore the runner's exit code. The gate JSON carries each blocking tag with its remedy and
the graph capture merge mapping.

**Gate:** exit 0. Otherwise stop.

## Step 5: Find gaps

    <prefix> G gaps --bringup-dir <bringup_dir> --unified-csv <analysis_dir>/perf_report_csvs/unified_perf_summary.csv --threshold <min_share>

Do not edit `gaps.json`. An empty `ops` list goes to Step 8.

## Step 6: Perf-model workers

1. In your own shell, one at a time:

       git -C <tracelens_dir> fetch origin main
       git -C <tracelens_dir> worktree add -b perf_model/<op_slug> <op_worktree> origin/main

   On a rerun, reuse `<op_worktree>`; if only the branch exists, add the worktree from it.
2. Launch one subagent per op, at most `<gpu_count>` at once, each with its own `<k>`:

       Read and follow the FULL instructions in:
         TraceLens/Agent/ModelBringup/skills/model-bringup/agents/perf-model-worker.md

       **Execution Context:**
       - Op: <op>, op slug: <op_slug>
       - Op worktree (branch perf_model/<op_slug>, already created): <op_worktree>
       - Op output dir: <op_dir>; bringup dir: <bringup_dir>; baseline analysis dir: <analysis_dir>
       - GPU arch: <gfx_arch>; GPU: HIP_VISIBLE_DEVICES=<k>
       - Framework: <framework>, source <framework_src>
       - Command prefix: read <bringup_dir>/cache/cmd_prefix.txt and substitute {CMD}
       - Return file: <op_dir>/result.json

       Execute every step in the agent file. Return "DONE" when complete.

**Gate:** each `result.json` parses, `outcome` is `pr_draft` or `issue_draft`, and `draft`
exists. Otherwise relaunch that worker once; if it fails again, the op is excluded.

## Step 7: Coverage

Skip when no op has outcome `pr_draft`.

1. In your own shell, merge each `pr_draft` branch in `gaps.json` order:

       git -C <tracelens_dir> worktree add --detach <bringup_dir>/worktrees/pm_integration origin/main
       git -C <bringup_dir>/worktrees/pm_integration merge --no-edit perf_model/<op_slug>

   On a conflict, `merge --abort` and continue.
2. Rerun only the perf report from the merged tree:

       <prefix> python3 -m TraceLens.Agent.Analysis.utils.config_utils build --scope standalone --analysis_mode inference \
         --output_dir <bringup_dir>/coverage --job trace_path=<mixed_split>,platform=<platform>[,capture_folder=<capture_folder>]
       <prefix> cd <bringup_dir>/worktrees/pm_integration && PYTHONPATH=<bringup_dir>/worktrees/pm_integration python3 -m TraceLens.Reporting.generate_perf_report_from_config --config <bringup_dir>/coverage/cache/run_config.yaml

3. `<prefix> G coverage --bringup-dir <bringup_dir> --before <analysis_dir>/perf_report_csvs/unified_perf_summary.csv --after <bringup_dir>/coverage/perf_report_csvs/unified_perf_summary.csv`

**Gate:** exit 0. Otherwise stop; the gate JSON names each failing op.

## Step 8: Summary

Always runs, including after a stop:

    <prefix> G summary --bringup-dir <bringup_dir> [--stop-reason "Step <N>: <reason>"]

Reply with `bringup_summary.md`, the cleanup command `ssh <node> "docker rm -f <container>"`,
and the Publish prompt it lists.

## Publish

Runs only on a prompt that says publish, names `<bringup_dir>`, and lists op slugs. Run
everything in your own shell. For each named op, read `<op_dir>/result.json`; skip it if
`<op_dir>/published_url.txt` exists.

- `pr_draft`:

      git -C <op_worktree> push -u origin perf_model/<op_slug>
      gh pr create --repo AMD-AGI/TraceLens --base main --head perf_model/<op_slug> \
        --title "$(head -n 1 <op_dir>/pr.md)" --body "$(tail -n +2 <op_dir>/pr.md)"

- `issue_draft`:

      gh issue create --repo AMD-AGI/TraceLens --title "$(head -n 1 <op_dir>/issue.md)" --body "$(tail -n +2 <op_dir>/issue.md)"

Write the URL `gh` prints to `<op_dir>/published_url.txt`.
