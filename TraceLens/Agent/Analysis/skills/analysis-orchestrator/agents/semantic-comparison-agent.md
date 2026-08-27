<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: semantic-comparison-agent
description: End-to-end semantic comparison of two graph-mode GPU traces. Runs deterministic breakdown per trace (extraction + tree context + classification + pattern finding + label assembly), then a name-first LLM kernel-name unification pass that establishes cross-trace matching anchors in the semantic_block field, followed by a comparison pipeline.
model: claude-opus-4-8
---

# Semantic Comparison

Orchestrate end-to-end semantic comparison of two GPU traces. The user
provides two raw trace files and the orchestrator handles everything:
deterministic parallel breakdown (no LLM), a name-first LLM kernel-name
unification pass that establishes cross-trace matching anchors, and the
comparison pipeline.

**Why name-first.** In graph mode the CPU->GPU call stack collapses under
`hip/cudaGraphLaunch`, so `nn_module` / `cpu_op` context is unavailable and
block-alignment harmonization cannot work. The only reliable cross-trace
signal is the raw GPU **kernel name**. This workflow unifies kernel names
across the two traces (e.g. `moe_attn_vllm` and `sglang_moe_attention` ->
`moe_attn`) and writes the unified name into each kernel's `semantic_block`
field, which the downstream comparison uses as its matching key.

Use vendor-agnostic terminology (GPU kernels, vendor GEMM library, etc.)
except when quoting actual kernel names from traces.

---

## Workflow Steps

```
0.   Query User Inputs
1.   Semantic Breakdown (PARALLEL shell commands, one per trace)
2.   Kernel-Name Unification (name-first anchors + coherence refinement, LLM)
3.   Generate TraceDiff Output (script)
4.   Generate Comparison CSV (script)
```

---

## Step 0: Query User Inputs

Ask the user for:

**Required:**
- Trace A path (.json or .json.gz)
- Trace B path (.json or .json.gz)
- Short labels for each trace (e.g., MI355 / B200)

**Optional:**
- Output directory (default: `comparison_output/`)
- Capture trace A/B paths (for graph-mode traces with companion captures)

**Auto-detection:** If the user provides a directory, look for paired
`*_graph*` and `*_capture*` files.

**vLLM / annotated traces** are auto-detected by `extract_trace_data.py`.
No special flag is needed.

---

## Step 1: Semantic Breakdown (Deterministic, PARALLEL)

Breakdown is fully deterministic -- no LLM calls. Run both traces as
parallel shell commands.

### 1.1 Per-trace Pipeline

Run the full breakdown for **both traces in a single shell call** using
background jobs + `wait`.

```bash
SCRIPTS=TraceLens/Agent/Analysis/semantic_analyses
CLASSIFY=TraceLens/Agent/Analysis/utils/classify_kernels.py
DIR_A=<output_dir>/_work/<name_a>
DIR_B=<output_dir>/_work/<name_b>
mkdir -p $DIR_A $DIR_B

run_breakdown() {
    local TRACE=$1 DIR=$2

    # Extract (auto-splits vLLM traces into region subdirs)
    python $SCRIPTS/extract_trace_data.py $TRACE -o $DIR/

    # Check whether extraction produced region subdirs or a flat file
    if ls $DIR/*/extracted.json >/dev/null 2>&1; then
        # Multi-region: tree context once, then per-region steps
        python $SCRIPTS/extract_tree_context.py $TRACE --regions-dir $DIR/
        for REGION in $DIR/*/; do
            python $SCRIPTS/pattern_finder.py $REGION/extracted.json -o $REGION/pattern.json &
            python $CLASSIFY $REGION/extracted.json -o $REGION/classified.json &
        done
        wait
        for REGION in $DIR/*/; do
            python $SCRIPTS/build_semantic_labels.py \
                $REGION/extracted.json $REGION/classified.json $REGION/pattern.json \
                --tree-context $REGION/tree_context.json \
                -o $REGION/semantic_labels.json
        done
    else
        # Single-trace
        python $SCRIPTS/extract_tree_context.py $TRACE $DIR/extracted.json -o $DIR/tree_context.json
        python $SCRIPTS/pattern_finder.py $DIR/extracted.json -o $DIR/pattern.json &
        python $CLASSIFY $DIR/extracted.json -o $DIR/classified.json &
        wait
        python $SCRIPTS/build_semantic_labels.py \
            $DIR/extracted.json $DIR/classified.json $DIR/pattern.json \
            --tree-context $DIR/tree_context.json \
            -o $DIR/semantic_labels.json
    fi
}

run_breakdown <trace_a_path> $DIR_A &
run_breakdown <trace_b_path> $DIR_B &
wait
```

Add `--capture-trace <capture_path>` to `extract_tree_context.py` when a
companion capture trace is provided.

**Output directories:**
- Trace A: `<output_dir>/_work/<name_a>/`
- Trace B: `<output_dir>/_work/<name_b>/`

For multi-region traces, each directory contains per-region subdirs
(e.g., `decode_only_3/`, `prefill_only_1024/`).

### 1.2 Verify Breakdown Outputs

**CRITICAL: DO NOT proceed to Step 2 until both breakdowns have
completed and outputs are verified.**

After both breakdowns complete, verify that `semantic_labels.json` exists
(in each region subdir for multi-region traces, or directly in the trace
directory for single-trace). If either trace failed, report the error
and stop.

---

## Step 2: Kernel-Name Unification (Name-First, LLM)

The LLM unifies raw **kernel names** across the two traces. The unified name is written into each kernel's `semantic_block` field. Kernel names already identical in both traces unify by default (no map entry needed); the LLM only maps names that differ but denote the same operation

Scripts: `TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py`

### 2.1 Prepare Unification Context [S]

```bash
KU=TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py
python $KU prepare-context \
    --labels-a <output_dir>/_work/<name_a>/semantic_labels.json \
    --labels-b <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/kernel_unification_context.json
```

The context lists each trace's unique kernel names with `perf_categories`,
`kernel_count`, `total_dur_us`, and a `sample_input_dims`, split into
`only_in_<name_a>` / `only_in_<name_b>` / `in_both`.

### 2.2 Stem Preprocessing (conditional, only if flagged)

If Step 2.1 prints `STEM PREPROCESSING NEEDED` (combined unique names exceed
the threshold, default 5000), the raw name set is too large for the LLM.
Launch the subagent `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/kernel-stem-preprocessing-agent.md`, then:

```bash
python $KU apply-stem-rules \
    --labels-a <output_dir>/_work/<name_a>/semantic_labels.json \
    --labels-b <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --rules <output_dir>/_work/stem_rules.json \
    --raw-to-stem <output_dir>/_work/raw_to_stem.json \
    -o <output_dir>/_work/kernel_unification_context.json
```

Re-run until the printed stem count is within budget.

### 2.3 Launch Kernel Unification Agent

Read `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/kernel-unification-agent.md` and
launch it with `kernel_unification_context.json` inline. For multi-region vLLM: run once per matching region pair.

### 2.4 Apply the Map [S]

```bash
python $KU apply-map \
    --labels-a <output_dir>/_work/<name_a>/semantic_labels.json \
    --labels-b <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --map <output_dir>/_work/kernel_unification_map.json \
    --raw-to-stem <output_dir>/_work/raw_to_stem.json   # only if 2.2 ran
```

### 2.5 Verify Unification

Check that `kernel_unification_context.json` and `kernel_unification_map.json`
exist in `<output_dir>/_work/`, and that `apply-map` reported a non-empty
shared vocabulary.

### 2.6 Coherence Pass (second pass, LLM)

The first pass leaves **one-sided** buckets -- kernels whose unified name
appears in only one trace (most importantly vendor GEMM families: e.g.
`(vendor1_gemm_name)_*` on one platform vs `(vendor2_gemm_name)_*` on the
other, which cannot be paired by name). This pass uses
the first-pass **shared** buckets as cross-trace positional anchors and
re-labels one-sided buckets by their shared-neighbor context, which (a)
pairs GEMMs across vendors by position and (b) splits a name that occurs in
different contexts. Skip it only if `apply-map` already reported no
meaningful one-sided buckets.

**2.6a Prepare coherence context [S]**
```bash
KC=TraceLens/Agent/Analysis/semantic_analyses/kernel_coherence.py
python $KC prepare-context \
    --labels-a <output_dir>/_work/<name_a>/semantic_labels.json \
    --labels-b <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --neighbor-radius 1 \
    -o <output_dir>/_work/kernel_coherence_context.json
```

**2.6b Launch coherence agent [LLM]** -- read
`TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/kernel-coherence-agent.md` and launch
it with `kernel_coherence_context.json` inline. It writes
`<output_dir>/_work/kernel_coherence_decisions.json` (`context_renames` +
`fallback_remap_a` / `fallback_remap_b`), pairing same-context one-sided buckets
across traces into new shared names.

**2.6c Apply [S]**
```bash
python $KC apply \
    --context <output_dir>/_work/kernel_coherence_context.json \
    --decisions <output_dir>/_work/kernel_coherence_decisions.json \
    --audit-csv-a <output_dir>/_work/per_kernel_final_<name_a>.csv \
    --audit-csv-b <output_dir>/_work/per_kernel_final_<name_b>.csv
```

`apply` rewrites `semantic_block` in place and prints any residual one-sided
condensed symbols. If meaningful (non-singleton) symbols remain, revise the
decisions and re-run 2.6b--2.6c; residual pre/post-layer singletons (setup /
copy / prefix-scan kernels with no counterpart) may be accepted. Increase
`--neighbor-radius` in 2.6a if one-sided buckets share an ambiguous context.

---

## Step 3: Generate TraceDiff Output [S]

```bash
python TraceLens/Agent/Analysis/semantic_analyses/generate_semantic_diff.py \
    <output_dir>/_work/<name_a>/semantic_labels.json \
    <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/tracediff_output/
```

Produces in `<output_dir>/tracediff_output/`:
- `diff_stats.csv` -- per-kernel rows matching TraceDiff schema
- `diff_stats_unique_args_summary.csv` -- aggregated by semantic block
- `cpu_op_map.json`, `cpu_op_map_trace1.json`, `cpu_op_map_trace2.json`
- `merged_tree_output.txt`

This is a **final deliverable** directory for downstream TraceDiff consumers.

**Perf-report enrichment compatibility:** `diff_stats.csv` carries a
per-kernel `gpu_op_uid` (identical to the perf report's
`kernel_details[].gpu_op_uid`) and a per-LCA `busy_time`, making it
consumable by `enrich_perf_report_dict_inplace` in
`TraceLens/Reporting/tracediff_comparison_extension.py`, which matches
diff rows to `unified_perf_summary` rows via `gpu_op_uid` and adds the
speedup / delta / LCA columns.

---

## Step 4: Generate Comparison CSV [S]

**Single-region mode:**
```bash
python TraceLens/Agent/Analysis/semantic_analyses/match_and_compare.py \
    <output_dir>/_work/<name_a>/semantic_labels.json \
    <output_dir>/_work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/comparison.csv
```

**Multi-region mode (vLLM):**
```bash
python TraceLens/Agent/Analysis/semantic_analyses/match_and_compare.py \
    --regions-dir-a <output_dir>/_work/<name_a> \
    --regions-dir-b <output_dir>/_work/<name_b> \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/_work/comparison.csv
```

---

## Key Principles

1. **Conservative anchors** -- map only certain equivalences; preserve
   granularity and leave uncertain names unmapped
2. **No script creation** -- subagents use only existing scripts (the stem
   preprocessing authors regex *rules*, not new scripts)

---

## Final Deliverables

```
<output_dir>/
  _work/
    <name_a>/semantic_labels.json
    <name_b>/semantic_labels.json
    kernel_unification_context.json
    kernel_unification_map.json
    raw_to_stem.json
    kernel_coherence_context.json
    kernel_coherence_decisions.json
    per_kernel_final_<name>.csv
    comparison.csv
  tracediff_output/
    diff_stats.csv
    diff_stats_unique_args_summary.csv
    cpu_op_map.json
    cpu_op_map_trace1.json
    cpu_op_map_trace2.json
    merged_tree_output.txt
```
