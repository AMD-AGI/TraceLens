<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Nightly TraceLens Processing Dashboard

**Proposal for Issue #427 — Build nightly TraceLens processing dashboard for private large-trace perf tracking**

---

## Overview

Large traces can take minutes to process through TraceLens. Runtime breaks down differently depending on the trace — some are dominated by load/parse/tree construction, others by report table generation or specific report paths. Memory usage can also be significant.

A single total runtime number is not enough. This proposal describes a nightly performance dashboard that:

- Runs TraceLens perf-report generation over a curated private trace set
- Records total wall time and stage-level timing breakdowns per trace
- Tracks peak memory (Max RSS) as a first-class metric
- Stores raw profiling artifacts privately
- Publishes summarized results to Grafana dashboards
- Supports local developer runs for testing optimization branches

The system uses **GitHub Actions** for nightly scheduling, **OpenTelemetry (OTLP)** for metrics emission, and **Grafana Cloud** for metrics storage and visualization (managed Prometheus + Grafana).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                               │
│                  (nightly cron: 0 2 * * *)                          │
│                                                                     │
│  1. Checkout TraceLens-internal                                     │
│  2. Install TraceLens + OpenTelemetry deps                          │
│  3. Download private traces (rclone / authenticated script)         │
│  4. Run profiling harness against trace manifest                    │
│  5. Upload artifacts (timing JSON, cProfile) to Actions Artifacts   │
└───────────────┬─────────────────────────────────────┬───────────────┘
                │                                     │
                │ OTLP/HTTP push                      │ GitHub Actions
                │ (stage timings, Max RSS,            │ Artifacts
                │  job metadata)                      │ (timing JSON,
                ▼                                     │  cProfile dumps)
┌───────────────────────────┐                         │
│    Grafana Cloud           │                         │
│    (managed Prometheus     │                         ▼
│     + OTLP endpoint)       │              ┌─────────────────────┐
│                            │              │  Private Artifact   │
│  Metrics:                  │              │  Storage            │
│  - stage duration (s)      │              │  (retained N days)  │
│  - total duration (s)      │              └─────────────────────┘
│  - max RSS (bytes)         │
│  - job metadata labels     │
└─────────────┬──────────────┘
              │ PromQL queries
              ▼
┌───────────────────────────┐
│    Grafana Cloud           │
│                            │
│  Dashboards:               │
│  - Runtime trends          │
│  - Stage breakdowns        │
│  - Memory trends           │
│  - Regression tables       │
│  - Workload family views   │
└────────────────────────────┘
```

---

## Repo Split

Reusable, trace-agnostic code can live in the **public TraceLens** repo. Anything that references private traces or internal storage stays in **TraceLens-internal**.

| Component | Repo | Rationale |
|---|---|---|
| Profiling harness module (`tracelens_perf_harness.py`) | Public | Trace-agnostic; instruments TraceLens stages |
| OTLP emission utilities | Public | Generic metric emission, no trace-specific data |
| Local run CLI / instructions | Public | Developer tooling, no private data |
| Private trace manifest (`trace_manifest.yaml`) | **Internal** | Contains private trace IDs, storage paths |
| Nightly workflow (`.github/workflows/nightly_perf_dashboard.yml`) | Public | Contains no sensitive data; all credentials are in `${{ secrets.* }}` placeholders |
| Prometheus / Grafana config | **Internal** | Internal infrastructure |
| Raw profiling artifacts | **Internal** | Generated from private traces |

---

## Private Trace Manifest

The trace list is defined by a YAML manifest checked into TraceLens-internal. The profiling harness reads this manifest to determine which traces to process.

### Schema

```yaml
# trace_manifest.yaml
version: 1
traces:
  - trace_id: "trace_001"
    source_location: "sharepoint:Documents/tracelens-private-traces/traces/trace_001.json.gz"
    workload_family: "vision"  # vision | nlp | recommendation | etc.
    enabled: true              # false to skip in nightly without removing
    notes: ""

  - trace_id: "trace_002"
    source_location: "sharepoint:Documents/tracelens-private-traces/traces/trace_002.json.gz"
    workload_family: "nlp"
    enabled: true
    notes: "Extremely large trace; may need extended timeout"
```

### Fields

| Field | Type | Description |
|---|---|---|
| `trace_id` | string | Stable unique identifier |
| `source_location` | string | SharePoint path in rclone remote format (e.g. `sharepoint:Documents/tracelens-private-traces/traces/trace.json.gz`) |
| `workload_family` | string | Workload category for aggregate views |
| `enabled` | bool | Whether to include in nightly runs |
| `notes` | string | Special handling notes or exclusion reasons |

---

## Python Profiling Harness

The harness is the core component. It instruments TraceLens processing stages, measures timing and memory, emits OTLP metrics, and produces structured artifacts.

### Stages Instrumented

These are the 12 stages tracked by the dashboard, each timed individually:

| # | Stage | Source Location |
|---|---|---|
| 1 | Total report generation | End-to-end wrapper |
| 2 | `TreePerfAnalyzer.from_file` | `TraceLens/TreePerf/tree_perf.py` |
| 3 | `DataLoader.load_data` | `TraceLens/util.py` |
| 4 | `TraceToTree.build_tree` | `TraceLens/Trace2Tree/trace_to_tree.py` |
| 5 | `build_host_call_stack_tree` | `TraceLens/Trace2Tree/trace_to_tree.py` |
| 6 | `label_non_gpu_paths` | `TraceLens/Trace2Tree/trace_to_tree.py` |
| 7 | `add_gpu_ops_to_tree` | `TraceLens/Trace2Tree/trace_to_tree.py` |
| 8 | `collect_unified_perf_events` | `TraceLens/TreePerf/tree_perf.py` |
| 9 | `build_df_unified_perf_table` | `TraceLens/TreePerf/tree_perf.py` |
| 10 | `get_df_kernel_launchers` | `TraceLens/TreePerf/tree_perf.py` |
| 11 | `get_df_kernels` | `TraceLens/TreePerf/tree_perf.py` |
| 12 | `get_df_gpu_timeline` | `TraceLens/TreePerf/tree_perf.py` |

All 11 stages are timed uniformly via `pstats` (`cumtime`) extracted from a `cProfile.Profile` wrapping the full run. No TraceLens source files are modified.

### Implementation Approach

```python
import pstats
import resource
import cProfile
import json
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

STAGES = {
    "total_report_generation",
    "from_file",
    "load_data",
    "build_tree",
    "build_host_call_stack_tree",
    "label_non_gpu_paths",
    "add_gpu_ops_to_tree",
    "collect_unified_perf_events",
    "build_df_unified_perf_table",
    "get_df_kernel_launchers",
    "get_df_kernels",
    "get_df_gpu_timeline",
}

def run_tracelens_with_cprofile(trace_path: str, artifact_path: str) -> cProfile.Profile:
    """Run TraceLens report generation under cProfile and save the .prof artifact."""
    pr = cProfile.Profile()
    pr.enable()
    # from_file covers stages 2-7: load_data, TraceToTree construction, build_tree,
    # and the internal build_tree sub-stages (build_host_call_stack_tree,
    # label_non_gpu_paths, add_gpu_ops_to_tree)
    analyzer = TreePerfAnalyzer.from_file(trace_path)
    # stages 8-11 are lazy methods not called by from_file; invoke explicitly
    analyzer.build_df_unified_perf_table()
    analyzer.get_df_kernel_launchers()
    analyzer.get_df_kernels()
    analyzer.get_df_gpu_timeline()
    pr.disable()
    pr.dump_stats(artifact_path)  # save .prof for artifact upload
    return pr

def extract_stage_timings(prof: cProfile.Profile) -> dict[str, float]:
    """Parse the in-memory cProfile.Profile object and return cumtime for each tracked stage."""
    stats = pstats.Stats(prof)
    timings = {}
    for (_, _, func), (_, _, _, cumtime, _) in stats.stats.items():
        if func in STAGES:
            timings[func] = cumtime
    return timings

def profile_trace(trace_path: str, trace_id: str, output_dir: str) -> dict:
    """Run TraceLens under cProfile, extract stage timings, and record peak memory."""
    artifact_path = f"{output_dir}/{trace_id}_profile.prof"

    pr = run_tracelens_with_cprofile(trace_path, artifact_path)
    timings = extract_stage_timings(pr)
    max_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # KB → bytes

    return {"stages": timings, "max_rss_bytes": max_rss_bytes, "cprofile_artifact": artifact_path}
```

**Timing:** The entire TraceLens run is wrapped in `cProfile.Profile()`. After the run, `pstats` extracts `cumtime` for each of the 11 named stage functions. `cumtime` measures the total time spent in a function including its callees, equivalent to wall-clock time for that call subtree.

**Memory:** Uses `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` to capture peak RSS after each trace completes. On Linux this returns kilobytes; the harness converts to bytes for consistency.

**cProfile artifact:** The `.prof` file is also uploaded to Actions Artifacts for manual post-hoc investigation with tools like `pstats` or `snakeviz`.

### Structured Timing JSON Output

Each run produces a JSON file with this structure:

```json
{
  "metadata": {
    "commit_sha": "abc123",
    "run_date": "2025-06-15T02:00:00Z",
    "python_version": "3.10.12",
    "tracelens_version": "0.5.0",
    "environment": "github-actions-ubuntu-latest"
  },
  "traces": [
    {
      "trace_id": "trace_001",
      "workload_family": "vision",
      "stages": {
        "total_report_generation": 145.23,
        "TreePerfAnalyzer.from_file": 142.10,
        "DataLoader.load_data": 38.50,
        "TraceToTree.build_tree": 52.30,
        "build_host_call_stack_tree": 20.10,
        "label_non_gpu_paths": 5.20,
        "add_gpu_ops_to_tree": 27.00,
        "build_df_unified_perf_table": 30.50,
        "get_df_kernel_launchers": 8.20,
        "get_df_kernels": 6.10,
        "get_df_gpu_timeline": 3.10
      },
      "max_rss_bytes": 8589934592,
      "cprofile_artifact": "artifacts/trace_001_profile.prof"
    }
  ]
}
```

### OTLP Metrics Emission

The harness uses the OpenTelemetry Python SDK to push metrics to Prometheus via the OTLP/HTTP protocol.

**Dependencies:**

```
opentelemetry-sdk
opentelemetry-exporter-otlp-proto-http
```

**Metrics emitted:**

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `tracelens.stage.duration_seconds` | Gauge | `trace_id`, `stage`, `workload_family` | `cumtime` for a single stage function, extracted from cProfile via pstats |
| `tracelens.total.duration_seconds` | Gauge | `trace_id`, `workload_family` | Total end-to-end processing time |
| `tracelens.process.max_rss_bytes` | Gauge | `trace_id`, `workload_family` | Peak resident memory after processing |

**Resource attributes** (attached to all metrics as OTLP resource labels):

| Attribute | Example | Description |
|---|---|---|
| `service.version` | `0.5.0` | TraceLens version |
| `vcs.commit.sha` | `abc123def` | Git commit under test |

**OTLP push setup:**

```python
resource = Resource.create({
    "service.version": tracelens_version,
    "vcs.commit.sha": commit_sha,
})

exporter = OTLPMetricExporter(
    endpoint=os.environ["GRAFANA_CLOUD_OTLP_ENDPOINT"],
    headers={"Authorization": f"Bearer {os.environ['GRAFANA_CLOUD_OTLP_TOKEN']}"},
)

reader = PeriodicExportingMetricReader(exporter, export_interval_millis=30000)
provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(provider)

meter = metrics.get_meter("tracelens.perf_harness")
stage_duration_gauge = meter.create_gauge(
    name="tracelens.stage.duration_seconds",
    description="Wall-clock duration of a TraceLens processing stage",
    unit="s",
)
```

After each trace completes, the harness calls `stage_duration_gauge.set(elapsed, attributes={...})` for every stage, then flushes the meter provider before exiting.

---

## GitHub Actions Workflow

The nightly workflow lives in the public TraceLens repo. All credentials are injected via GitHub Actions secrets; no sensitive data is in the workflow file itself. The private trace manifest and download scripts are checked out from TraceLens-internal at runtime.

### Workflow File: `.github/workflows/nightly_perf_dashboard.yml`

```yaml
name: Nightly Perf Dashboard

on:
  schedule:
    - cron: '0 2 * * *'   # 2:00 AM UTC daily
  workflow_dispatch:        # manual trigger for debugging

env:
  PYTHON_VERSION: '3.10'

jobs:
  nightly-perf:
    runs-on: ubuntu-latest
    timeout-minutes: 120    # large traces may take a while

    steps:
      - name: Checkout TraceLens-internal
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install .
          pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http pyyaml

      - name: Install rclone
        run: |
          curl -fsSL https://rclone.org/install.sh | sudo bash

      - name: Configure private trace access
        env:
          RCLONE_CONFIG_CONTENT: ${{ secrets.RCLONE_CONFIG }}
        run: |
          mkdir -p ~/.config/rclone
          echo "$RCLONE_CONFIG_CONTENT" > ~/.config/rclone/rclone.conf

      - name: Download private traces
        run: |
          python scripts/download_traces.py \
            --manifest config/trace_manifest.yaml \
            --output-dir /tmp/traces

      - name: Run profiling harness
        env:
          GRAFANA_CLOUD_OTLP_ENDPOINT: ${{ secrets.GRAFANA_CLOUD_OTLP_ENDPOINT }}
          GRAFANA_CLOUD_OTLP_TOKEN: ${{ secrets.GRAFANA_CLOUD_OTLP_TOKEN }}
        run: |
          python scripts/tracelens_perf_harness.py \
            --manifest config/trace_manifest.yaml \
            --trace-dir /tmp/traces \
            --output-dir ./perf_results \
            --emit-otlp

      - name: Upload timing artifacts
        uses: actions/upload-artifact@v4
        with:
          name: perf-results-${{ github.sha }}
          path: ./perf_results/
          retention-days: 90

      - name: Upload cProfile artifacts
        uses: actions/upload-artifact@v4
        with:
          name: cprofile-results-${{ github.sha }}
          path: ./perf_results/**/*.prof
          retention-days: 30
```

### Required Secrets

| Secret | Description |
|---|---|
| `RCLONE_CONFIG` | rclone configuration file containing OAuth credentials for SharePoint; written to `~/.config/rclone/rclone.conf` so that `download_traces.py` can run `rclone copy sharepoint:... /tmp/traces/` |
| `GRAFANA_CLOUD_OTLP_ENDPOINT` | OTLP/HTTP receiver URL (e.g., Prometheus with OTLP receiver, or a Grafana Cloud endpoint) |
| `GRAFANA_CLOUD_OTLP_TOKEN` | Bearer token for authenticating OTLP pushes |

### Trace Download

The `RCLONE_CONFIG` secret contains the rclone configuration file with OAuth credentials for SharePoint. The "Configure private trace access" workflow step writes this secret to `~/.config/rclone/rclone.conf`, which is the default path rclone reads for authentication when it runs.

`scripts/download_traces.py` then reads the trace manifest and, for each enabled trace, shells out to rclone using the `source_location` field as the SharePoint path:

```python
subprocess.run([
    "rclone", "copy",
    trace["source_location"],  # e.g. "sharepoint:Documents/tracelens-private-traces/traces/trace_001.json.gz"
    output_dir,
])
```

rclone picks up the SharePoint OAuth credentials from `~/.config/rclone/rclone.conf` automatically and copies each trace file to `/tmp/traces/`. This keeps the download logic separate from the profiling harness so the harness stays trace-agnostic.

---

## Prometheus Setup

Grafana Cloud provides a managed Prometheus instance with a built-in OTLP endpoint. The harness pushes directly to it:

```
https://otlp-gateway-<region>.grafana.net/otlp
```

Set `GRAFANA_CLOUD_OTLP_ENDPOINT` to this URL and `GRAFANA_CLOUD_OTLP_TOKEN` to the Grafana Cloud API key. No self-hosted Prometheus or Pushgateway is needed.

### Metric Naming

OTLP metric names are translated to Prometheus format automatically:

| OTLP Metric | Prometheus Metric |
|---|---|
| `tracelens.stage.duration_seconds` | `tracelens_stage_duration_seconds` |
| `tracelens.total.duration_seconds` | `tracelens_total_duration_seconds` |
| `tracelens.process.max_rss_bytes` | `tracelens_process_max_rss_bytes` |

Labels (`trace_id`, `stage`, `workload_family`) and resource attributes (`vcs_commit_sha`, `service_version`) are preserved as Prometheus labels.

### Retention

- **Nightly data:** 180 days minimum to enable trend analysis across optimization cycles.
- **Baseline snapshots:** Tag specific runs as baselines using a `baseline=true` label or separate recording rule.

---

## Grafana Dashboards

### Data Source

- **Type:** Prometheus
- **URL:** Points to the Prometheus instance (self-hosted or Grafana Cloud)

### Dashboard: TraceLens Nightly Performance

#### Panel 1: Runtime Trend Per Trace (Time Series)

```promql
tracelens_total_duration_seconds
```

- **Visualization:** Time series
- **Group by:** `trace_id`
- **Y-axis:** Duration (seconds)
- **Purpose:** See total processing time trends over nightly runs

#### Panel 2: Stage Breakdown Per Trace (Stacked Bar)

```promql
tracelens_stage_duration_seconds{trace_id="$trace_id"}
```

- **Visualization:** Bar chart (stacked)
- **Group by:** `stage`
- **Variable:** `$trace_id` as a dashboard dropdown
- **Purpose:** See which stages dominate for a selected trace

#### Panel 3: Max RSS Trend Per Trace (Time Series)

```promql
tracelens_process_max_rss_bytes / 1073741824
```

- **Visualization:** Time series
- **Group by:** `trace_id`
- **Y-axis:** Memory (GiB)
- **Purpose:** Track memory pressure over time

#### Panel 4: Top Regressions Since Previous Nightly (Table)

```promql
(
  tracelens_total_duration_seconds
  - tracelens_total_duration_seconds offset 1d
)
/ tracelens_total_duration_seconds offset 1d
* 100
```

- **Visualization:** Table
- **Sort:** Descending by percent change
- **Columns:** trace_id, current (s), previous (s), delta (%), commit SHA
- **Purpose:** Flag traces that got slower since yesterday

#### Panel 5: Top Regressions Since Baseline (Table)

```promql
(
  tracelens_total_duration_seconds
  - tracelens_total_duration_seconds @ <baseline_timestamp>
)
/ tracelens_total_duration_seconds @ <baseline_timestamp>
* 100
```

- **Visualization:** Table
- **Sort:** Descending by percent change
- **Purpose:** Compare current performance to a fixed baseline snapshot

#### Panel 6: Aggregate View by Workload Family (Time Series)

```promql
avg by (workload_family) (tracelens_total_duration_seconds)
```

- **Visualization:** Time series
- **Group by:** `workload_family`
- **Purpose:** High-level view of performance by workload category

#### Panel 7: Dominant Bottleneck Per Trace (Table)

```promql
topk(1,
  tracelens_stage_duration_seconds
) by (trace_id)
```

- **Visualization:** Table
- **Columns:** trace_id, dominant stage, duration (s), percent of total
- **Purpose:** Quickly identify where optimization effort should focus per trace

### Alerting (Optional)

Grafana alerts can be configured to notify on significant regressions:

```yaml
# Example alert rule
- alert: TraceLensNightlyRegression
  expr: >
    (tracelens_total_duration_seconds - tracelens_total_duration_seconds offset 1d)
    / tracelens_total_duration_seconds offset 1d > 0.15
  for: 0m
  labels:
    severity: warning
  annotations:
    summary: "TraceLens nightly regression detected for {{ $labels.trace_id }}"
    description: "Processing time increased by {{ $value | humanizePercentage }} compared to previous nightly."
```

---

## Local Developer Flow

Developers should be able to run the same profiling harness locally to test optimization branches without waiting for the nightly run.

### Running Locally

```bash
# 1. Point at a single trace file
python scripts/tracelens_perf_harness.py \
  --trace-file /path/to/local/trace.json \
  --output-dir ./my_perf_results

# 2. Point at a manifest with a subset
python scripts/tracelens_perf_harness.py \
  --manifest config/trace_manifest.yaml \
  --trace-dir /path/to/traces \
  --output-dir ./my_perf_results \
  --filter "trace_001,trace_003"

# 3. Run without OTLP emission (local-only, no Prometheus push)
python scripts/tracelens_perf_harness.py \
  --trace-file /path/to/trace.json \
  --output-dir ./my_perf_results
  # (omit --emit-otlp to skip metric push)
```

### Comparing Before/After

```bash
# Run on the base branch
git checkout main
python scripts/tracelens_perf_harness.py \
  --trace-file /path/to/trace.json \
  --output-dir ./perf_before

# Run on the optimization branch
git checkout my-optimization
python scripts/tracelens_perf_harness.py \
  --trace-file /path/to/trace.json \
  --output-dir ./perf_after

# Compare timing JSON
python scripts/compare_perf_results.py \
  --before ./perf_before/timing.json \
  --after ./perf_after/timing.json
```

The comparison script outputs a table showing per-stage deltas:

```
Stage                          Before (s)   After (s)   Delta    Change
─────────────────────────────  ──────────   ─────────   ──────   ──────
total_report_generation          145.23      128.10     -17.13   -11.8%
DataLoader.load_data              38.50       38.45      -0.05    -0.1%
TraceToTree.build_tree            52.30       40.20     -12.10   -23.1%
build_df_unified_perf_table       30.50       28.90      -1.60    -5.2%
...
Max RSS (GiB)                      8.00        7.50      -0.50    -6.3%
```

### Safeguards Against Committing Private Data

Add to `.gitignore` in TraceLens-internal:

```gitignore
# Private traces - never commit
/tmp/traces/
*.trace.json.gz
perf_results/

# cProfile artifacts
*.prof
```

The harness should also validate that output directories are not inside the git working tree, or print a warning if they are.

---

## Acceptance Criteria Mapping

Each acceptance criterion from issue #427, mapped to the component that satisfies it:

| Criterion | Satisfied By |
|---|---|
| Nightly job runs on the internal trace set and publishes a dashboard/report | GitHub Actions workflow (`nightly_perf_dashboard.yml`) + Grafana dashboards |
| Results include total runtime, stage breakdown, and memory usage | Profiling harness instruments all 11 stages + Max RSS; emits via OTLP |
| Local run instructions are documented | Local Developer Flow section above; harness supports `--trace-file` and `--filter` |
| Private traces and raw artifacts remain internal | Repo split: manifest/workflow/artifacts in TraceLens-internal only |
| The system can compare current results against previous nightly results | Grafana Panel 4 (regressions vs previous), Panel 5 (vs baseline), and `compare_perf_results.py` |
| The trace list is configurable without changing profiler code | Manifest-driven design: `trace_manifest.yaml` is separate from harness code |

---

## Implementation Sequence

Suggested order of implementation:

1. **Profiling harness** (`scripts/tracelens_perf_harness.py`) — core timing + JSON output, no OTLP yet
2. **Trace manifest** schema and loading (`config/trace_manifest.yaml`)
3. **Local run CLI** — `--trace-file`, `--manifest`, `--filter`, `--output-dir`
4. **Comparison script** (`scripts/compare_perf_results.py`)
5. **OTLP emission** — add `opentelemetry-sdk` integration to the harness
6. **Prometheus setup** — configure OTLP receiver or Grafana Cloud endpoint
7. **Grafana dashboards** — build the 7 panels described above
8. **GitHub Actions workflow** — nightly cron, trace download, harness invocation, artifact upload
9. **Trace download script** (`scripts/download_traces.py`) — rclone-based
10. **Alerting** (optional) — regression detection alerts in Grafana
