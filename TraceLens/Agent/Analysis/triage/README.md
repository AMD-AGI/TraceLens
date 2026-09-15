<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens Agent Triage Toolkit

Automated triage for TraceLens Agent analysis runs. Given one or more TraceLens
analysis output folders, the toolkit runs a catalog of checks, records the
findings for each run, and can aggregate a batch into a summary report with
reproducer packages.

It supports two categories of user:

- General users may point it at any TraceLens agent analysis output directory,
  irrespective of the surrounding layout. No site-specific filesystem
  assumptions are built in.
- Hyperloom users may point it at a session tree to additionally obtain the
  session-level checks.

## Definition of a run directory

A run directory is a single TraceLens analysis output folder, that is, the
directory containing `perf_report_csvs/`, `category_data/`, `analysis.md`, and
the associated artifacts. The run-level checks operate on this folder and
degrade gracefully when an artifact is absent.

## Modes of use

### 1. General, single run (CLI)

```bash
python -m TraceLens.Agent.Analysis.triage.runner \
    --run-dir /path/to/analysis_output --detailed
```

This writes `triage_details.csv` and `triage_diags.txt` into the run directory
and prints one `[DIAG:...]` line per finding. The `--detailed` flag additionally
runs the trace-loading checks, which are slower and more I/O-intensive; omit it
for a faster pass.

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

Discovery is layout-agnostic by default: the script locates every directory
containing a `perf_report_csvs/` subfolder under the traces root, triages each
one in parallel, and aggregates the results. All output is written to
`./triage_report/`. The third positional argument (here `8`) is the worker
count that bounds how many runs are triaged in parallel; it defaults to `4`.

### 4. Hyperloom, batch (session layout)

For a Hyperloom session tree, providing an explicit discovery glob is faster
than walking the entire tree. Trace paths captured under a different root than
their current mount may also be remapped. Replace `<capture_root>` and
`<local_mount>` with the prefixes that apply to the environment:

```bash
TRIAGE_DISCOVERY_GLOB='*/*/kernel-agent/runs/*/*/tracelens' \
TRACELENS_PATH_REMAPS='<capture_root>=<local_mount>' \
  bash run_triage.sh /path/to/sessions ./triage_report 8
```

## Check catalog

Every finding carries a DIAG tag of the form `DIAG:<category>:<sublabel>_<NAME>`:

| Section | Category | Coverage |
|---|---|---|
| 1x | `profiling` | Trace presence, size, GPU kernels, capture traces |
| 2x | `trace_quality` | Shapes, inference annotations, split traces, idle, corruption, instability |
| 3x | `perf_model` | Report correctness: synthetic and unclassified ops, missing TB/s or TFLOPs, roofline %, zero-pct ops |
| 4x | `tracelens_agent_workflow` | Orchestrator and agent pipeline outputs (perf reports, manifests, analysis.md, subagent budget) |
| 5x | `infra` | Host and environment: SSH, docker, disk, NFS, dependencies, context length |
| 6x | `geak_interface` | Hyperloom session GEAK checks (kernel_candidates.json) |

The trace-loading checks (`2a`, `2b`, `2c`, `2e`, `2f`) run only when
`--detailed` is passed.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `TRACELENS_PATH_REMAPS` | empty (no-op) | Comma-separated `old=new` prefix remaps applied to absolute paths read from manifests and command files when the original path is not present locally. Use when traces were captured under a different root than the current mount. |
| `TRIAGE_DISCOVERY_GLOB` | empty (auto) | Shell glob, relative to the traces root, that `run_triage.sh` uses to locate run directories. When empty, it auto-discovers directories containing `perf_report_csvs/`. |

## Outputs

| File | Produced by | Contents |
|---|---|---|
| `<run_dir>/triage_details.csv` | `runner` | One row per finding (tag, category, failure mode, evidence, remedy). |
| `<run_dir>/triage_diags.txt` | `runner` | Human-readable DIAG lines. |
| `<report_dir>/run_dirs.txt` | `run_triage.sh` | Discovered run directories. |
| `<report_dir>/aggregated_triage.csv` | `postprocess` | All findings across the batch. |
| `<report_dir>/summary_report.md` | `postprocess` | Funnel, top failure modes, action items, reproducers. |
| `<report_dir>/reproducers/*.tar.gz` | `postprocess` | Self-contained reproducer packages for representative runs. |
