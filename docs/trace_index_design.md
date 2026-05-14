<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Trace Index Design

This document sketches a local trace index built on top of TraceLens reports. The goal is to make a large trace corpus searchable without repeatedly opening every trace or regenerating reports for exploratory questions.

## Goals

- Catalog trace files with cheap metadata, then enrich selected traces with TraceLens performance report data.
- Use `unified_perf_summary` as the source of truth for operation-specific performance data.
- Preserve the relationship between operations and the kernels they launch.
- Support both free-text search and structured queries over performance parameters.
- Keep trace-level metadata such as device properties, profiler flags, trace name, and runtime versions available for filtering.

## Example Questions

The index should eventually answer questions like:

- Which traces have convolution ops that launch transpose or layout-conversion kernels instead of only core convolution compute kernels?
- Which traces have `C_ijk` / `Cijk` Tensile kernels?
- What range of performance was achieved for GEMMs in the indexed traces?
- Which traces contain attention, SDPA, GEMM, collectives, or specific shape patterns?
- Which traces were captured with shape recording, stack capture, FLOP capture, or memory profiling enabled?
- Which traces came from a given GPU architecture, runtime version, profiler version, or device count?

## Proposed Layers

### Catalog Layer

The catalog layer records file-level facts without running full TraceLens analysis:

- trace path or stable trace identifier
- file size and modification time
- trace format
- rank, if inferable from the path or filename
- whether the file appears to contain `traceEvents`

This layer should avoid parsing very large JSON files when possible.

### Trace Metadata Layer

Trace JSON files often contain useful top-level fields outside `traceEvents`. Examples include:

- `schemaVersion`
- `traceName`
- `trace_id`
- `displayTimeUnit`
- `baseTimeNanoseconds`
- `deviceProperties`
- profiler flags such as `record_shapes`, `with_stack`, `with_flops`, `with_modules`, and `profile_memory`
- runtime and profiler versions
- optional `metadata`

Storage overhead for these fields is small compared with event data, but extraction can be expensive for large JSON files. Prefer collecting this metadata during enrichment, when the trace is already parsed.

### Unified Performance Layer

The enrichment layer should import rows from `unified_perf_summary` as generic performance records. Each row should retain:

- trace identifier
- operation name and category
- kernel details or kernel summary fields
- raw `perf_params` or equivalent flexible fields
- selected numeric metrics for structured range queries
- enough source columns to rebuild category-specific projections

This avoids creating competing schemas for GEMM, convolution, SDPA, and other operation types.

### Category Projections

Category-specific tables or views can make common analysis easier, but they should be derived from unified performance rows:

- `gemm_perf`
- `conv_perf`
- `sdpa_perf`
- `collective_perf`

These projections should not become independent sources of truth. They are convenience views over the unified report data.

## Query Requirements

The index should support:

- keyword search over operation names, kernel names, categories, and selected metadata
- structured filters over numeric performance metrics
- queries over flexible `perf_params`
- operation-to-kernel questions, such as whether a convolution operation launched transpose/layout kernels
- trace-level filters based on capture options, device properties, and runtime versions

## Design Constraints

- Do not normalize away important report fields too early. TraceLens report structure can vary by trace type and operation type.
- Keep the raw unified row or a faithful JSON representation so future queries can be added without regenerating all reports.
- Avoid full JSON parsing in the fast catalog pass for very large traces.
- Prefer derived views/materialized tables for common categories rather than one-off parsers for each question.
