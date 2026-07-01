<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens Performance Harness

`tracelens_perf_harness.py` is the nightly performance tracking tool for TraceLens.
It runs the full TraceLens pipeline, extracting runtime metrics, optionally pushing metrics to a Grafana Cloud dashboard or hosting metrics on an endpoint.

## Installation

The harness requires `pyyaml`. For OTLP emission to Grafana Cloud it also needs the OpenTelemetry
SDK packages. For the pull-based Prometheus server it needs `prometheus_client`:

```bash
pip install pyyaml
# optional — only needed with --emit-otlp / --push-only
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
# optional — only needed with --serve-only
pip install prometheus_client
```

## Single-trace usage

Profile one trace file and write results to `./perf_results/`:

```bash
python tests/perf_tracker/tracelens_perf_harness.py \
    --trace-file /path/to/trace.json.gz \
    --output-dir ./perf_results
```

By default the trace ID is derived from the filename stem. Override it with
`--trace-id` to get a stable, human-readable key in the output:

```bash
python tests/perf_tracker/tracelens_perf_harness.py \
    --trace-file /path/to/trace.json.gz \
    --trace-id trace1 \
    --output-dir ./perf_results
```

## Manifest usage

A manifest YAML file lists all traces to profile in a single run. This is useful if the goal is to profile many traces as opposed to just one. The manifest file must follow the format below.

**Manifest format**:

```yaml
traces:
  - trace_id: "trace1"
    trace_path: "/absolute/path/to/trace1.json.gz"
    enabled: true
    notes: "optional free-text description"

  - trace_id: "trace2"
    trace_path: "/absolute/path/to/trace2.json.gz"
    enabled: false # skip this trace
    notes: ""
```

Run all enabled traces:

```bash
python tests/perf_tracker/tracelens_perf_harness.py \
    --manifest config/trace_manifest.yaml \
    --output-dir ./perf_results
```

Run only a subset of traces:

```bash
python tests/perf_tracker/tracelens_perf_harness.py \
    --manifest config/trace_manifest.yaml \
    --output-dir ./perf_results \
    --filter "trace1,trace2"
```

## Output: `timing.json`

After a successful run, `<output-dir>/timing.json` contains the runtime info of all the traces:

```json
{
  "metadata": {
    "commit_sha": "9b284f...",
    "run_date": "2026-06-08T12:00:00+00:00",
    "python_version": "3.12.0",
    "tracelens_version": "0.1.0",
    "environment": "local"
  },
  "traces": [
    {
      "trace_id": "trace1",
      "max_rss_bytes": 5800000000,
      "cprofile_artifact": "./perf_results/trace1_profile.prof",
      "stages": {
        "total_report_generation": 100.0,
        "from_file": 12.3,
        "load_data": 8.1,
        "build_tree": 5.4,
        "build_host_call_stack_tree": 3.2,
        "label_non_gpu_paths": 1.1,
        "add_gpu_ops_to_tree": 0.9,
        "collect_unified_perf_events": 2.7,
        "build_df_unified_perf_table": 60.0,
        "get_df_kernel_launchers": 4.8,
        "get_df_kernels": 1.5,
        "get_df_gpu_timeline": 0.3
      }
    }
  ],
  ...
}
```

A `<trace_id>_profile.prof` binary artifact is also written and can be further inspected.

## Prometheus metrics server

Start the server pointing at an existing `timing.json`:

```bash
python tests/perf_tracker/tracelens_perf_harness.py \
    --serve-only ./perf_results/timing.json
```

Metrics are available at `http://localhost:9100/metrics`. Use `--metrics-port` to change the port:

```bash
python tests/perf_tracker/tracelens_perf_harness.py \
    --serve-only ./perf_results/timing.json --metrics-port 9200
```

Metrics exposed:

| Metric | Labels | Description |
|--------|--------|-------------|
| `tracelens_stage_duration_seconds` | `trace_id`, `stage` | cumtime for each tracked pipeline stage |
| `tracelens_total_duration_seconds` | `trace_id` | end-to-end report generation time |
| `tracelens_process_max_rss_bytes` | `trace_id` | peak resident memory after processing |

## OTLP / Grafana Cloud

Set the following environment variables before running with `--emit-otlp`:

| Variable | Description |
|---|---|
| `GRAFANA_CLOUD_OTLP_ENDPOINT` | OTLP/HTTP base URL |
| `GRAFANA_CLOUD_OTLP_TOKEN` | Token for endpoints |

Run with OTLP emission:

```bash
export GRAFANA_CLOUD_OTLP_ENDPOINT="https://otlp-gateway-prod-us-east-0.grafana.net/otlp"
export GRAFANA_CLOUD_OTLP_TOKEN="<your-token>"

python tests/perf_tracker/tracelens_perf_harness.py \
    --manifest config/trace_manifest.yaml \
    --output-dir ./perf_results \
    --emit-otlp # note this flag
```

To push an existing `timing.json` without re-profiling (useful for retrying a
failed push):

```bash
python tests/perf_tracker/tracelens_perf_harness.py \
    --push-only ./perf_results/timing.json
```

Metrics exposed:

| Metric | Labels | Description |
|--------|--------|-------------|
| `tracelens_stage_duration_seconds` | `trace_id`, `stage` | cumtime for each tracked pipeline stage |
| `tracelens_total_duration_seconds` | `trace_id` | end-to-end report generation time |
| `tracelens_process_max_rss_bytes` | `trace_id` | peak resident memory after processing |

## Example workflow

**Push-based (OTLP to Grafana Cloud):**

```bash
# 1. Profile all enabled traces and push metrics
python tests/perf_tracker/tracelens_perf_harness.py \
    --manifest config/trace_manifest.yaml \
    --output-dir ./perf_results_$(date +%Y%m%d) \
    --emit-otlp

# 2. Inspect results locally
cat ./perf_results_$(date +%Y%m%d)/timing.json | python -m json.tool

# 3. Drill into a slow trace with snakeviz (pip install snakeviz)
snakeviz ./perf_results_$(date +%Y%m%d)/trace1_profile.prof
```

**Pull-based (Prometheus scraping):**

```bash
# 1. Profile all enabled traces (no --emit-otlp needed)
python tests/perf_tracker/tracelens_perf_harness.py \
    --manifest config/trace_manifest.yaml \
    --output-dir ./perf_results

# 2. Start the metrics server (leave this running; re-run step 1 nightly)
python tests/perf_tracker/tracelens_perf_harness.py \
    --serve-only ./perf_results/timing.json
```

For a one-off local check on a single trace without pushing metrics:

```bash
python tests/perf_tracker/tracelens_perf_harness.py \
    --trace-file traces/my_trace.json.gz \
    --trace-id my_trace \
    --output-dir /tmp/perf_check
cat /tmp/perf_check/timing.json
```
