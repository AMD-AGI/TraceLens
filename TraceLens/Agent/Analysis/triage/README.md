# TraceLens Agent Triage Toolkit

Automated triage for TraceLens analysis runs. You point it at one or more
TraceLens analysis output folders, it runs a catalog of checks, writes the
findings per run, and can roll a whole batch up into a summary report with
reproducer packages.

It works for two kinds of users:

- General users can point it at any TraceLens analysis output directory,
  whatever the surrounding layout looks like. There are no site-specific
  filesystem assumptions baked in.
- Hyperloom users can point it at a session tree and additionally get the
  session-level (GEAK) checks, plus path remapping for traces that were
  captured under a different root than where they now live.

Module path: `TraceLens.Agent.Analysis.triage`

## Install

Ships with the public `TraceLens` package:

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

Any skill or script can then import it:

```python
from TraceLens.Agent.Analysis.triage.runner import run_triage
from TraceLens.Agent.Analysis.triage.checks import ALL_CHECKS
```

## What is a "run dir"?

A run dir is a single TraceLens analysis output folder, i.e. the directory that
holds `perf_report_csvs/`, `category_data/`, `analysis.md`, and friends. The
run-level checks all operate on this folder, and they degrade gracefully when an
artifact happens to be missing.

## Modes of use

### 1. General, single run (CLI)

```bash
python -m TraceLens.Agent.Analysis.triage.runner \
    --run-dir /path/to/analysis_output --detailed
```

This writes `triage_details.csv` and `triage_diags.txt` into the run dir and
prints a `[DIAG:...]` line per finding. Adding `--detailed` also runs the
trace-loading checks, which are slower and more IO-heavy, so leave it off when
you just want a quick pass.

### 2. General, single run (library)

```python
from TraceLens.Agent.Analysis.triage.runner import run_triage
findings = run_triage("/path/to/analysis_output", detailed=True)
for f in findings:
    print(f.tag, f.failure_mode, f.evidence)
```

### 3. General, batch (any layout)

```bash
bash run_triage.sh /path/to/tracelens_outputs ./triage_report 8
```

Discovery is layout-agnostic by default: it looks for every directory that
contains a `perf_report_csvs/` subfolder under the traces root, triages each one
in parallel, and then aggregates. Everything lands in `./triage_report/`.

### 4. Hyperloom, batch (session layout)

For the Hyperloom session tree it's faster to give an explicit discovery glob
instead of walking the whole tree. You can also remap trace paths that were
captured under a different root than where they are now mounted. Replace
`<capture_root>` and `<local_mount>` with the prefixes that apply to your setup:

```bash
TRIAGE_DISCOVERY_GLOB='*/*/kernel-agent/runs/*/*/tracelens' \
TRACELENS_PATH_REMAPS='<capture_root>=<local_mount>' \
  bash run_triage.sh /path/to/sessions ./triage_report 8
```

## Check catalog

Every finding carries a DIAG tag of the form `DIAG:<category>:<sublabel>_<NAME>`:

| Section | Category | What it covers |
|---|---|---|
| 1x | `profiling` | Trace presence, size, GPU kernels, capture traces |
| 2x | `trace_quality` | Shapes, inference annotations, split traces, idle, corruption, instability |
| 3x | `perf_model` | Report correctness: synthetic and unclassified ops, missing TB/s or TFLOPs, roofline %, zero-pct ops |
| 4x | `tracelens_agent_workflow` | Orchestrator and agent pipeline outputs (perf reports, manifests, analysis.md, subagent budget) |
| 5x | `infra` | Host and environment: SSH, docker, disk, NFS, deps, context length |
| 6x | `geak_interface` | Hyperloom session GEAK checks (kernel_candidates.json) |

The trace-loading checks (`2a`, `2b`, `2c`, `2e`, `2f`) only run when you pass
`--detailed`.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `TRACELENS_PATH_REMAPS` | empty (no-op) | Comma-separated `old=new` prefix remaps applied to absolute paths read from manifests and cmd files when the original path isn't present locally. Use it when traces were captured under a different root than the current mount. |
| `TRIAGE_DISCOVERY_GLOB` | empty (auto) | Shell glob (relative to the traces root) that `run_triage.sh` uses to find run dirs. When empty it auto-discovers directories containing `perf_report_csvs/`. |
| `NUM_WORKERS` | n/a | Used by custom batch drivers (for example the roofline-bucketed runners) to bound the number of parallel triage workers. |

## Outputs

| File | Produced by | Contents |
|---|---|---|
| `<run_dir>/triage_details.csv` | `runner` | One row per finding (tag, category, failure mode, evidence, remedy). |
| `<run_dir>/triage_diags.txt` | `runner` | Human-readable DIAG lines. |
| `<report_dir>/run_dirs.txt` | `run_triage.sh` | Discovered run dirs. |
| `<report_dir>/aggregated_triage.csv` | `postprocess` | All findings across the batch. |
| `<report_dir>/summary_report.md` | `postprocess` | Funnel, top failure modes, action items, reproducers. |
| `<report_dir>/reproducers/*.tar.gz` | `postprocess` | Self-contained reproducer packages for representative runs. |
