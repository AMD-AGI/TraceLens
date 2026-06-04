<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: semantic-comparison-agent
description: End-to-end semantic comparison of two GPU traces. Runs deterministic breakdown per trace (extraction + tree context + classification + pattern finding + label assembly), then a single LLM harmonization pass to assign and unify semantic_block names, followed by a comparison pipeline.
model: claude-opus-4-7-high
---

# Semantic Comparison

Orchestrate end-to-end semantic comparison of two GPU traces. The user
provides two raw trace files and the orchestrator handles everything:
deterministic parallel breakdown (no LLM), single-pass LLM harmonization
that assigns and unifies semantic_block names, and the comparison pipeline.

Use vendor-agnostic terminology (GPU kernels, vendor GEMM library, etc.)
except when quoting actual kernel names from traces.

---

## Workflow Steps

```
0.   Query User Inputs
1.   Semantic Breakdown (PARALLEL shell commands, one per trace)
2.   Semantic Harmonization (single agent, cross-trace)
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
No special flag is needed -- the script probes for annotation iterations
and splits into per-region subdirectories automatically.

---

## Step 1: Semantic Breakdown (Deterministic, PARALLEL)

Breakdown is fully deterministic -- no LLM calls. Run both traces as
**parallel shell commands** (NOT Task subagents) so the orchestrator
blocks until both finish.

### 1.1 Per-trace Pipeline

Run the full breakdown for **both traces in a single shell call** using
background jobs + `wait`. The extraction step auto-detects vLLM
annotation regions; no special flag is needed. For vLLM traces the
output directory will contain per-region subdirs; the remaining steps
(`extract_tree_context.py`, `pattern_finder.py`, `classify_kernels.py`,
`build_semantic_labels.py`) support `--regions-dir` / per-region paths
and must be run for **every** region.

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

## Step 2: Semantic Harmonization (LLM -- Label + Harmonize)

The harmonization agent is the **only LLM call** in the pipeline. It
assigns descriptive `semantic_block` names to each block AND ensures
cross-trace consistency. The enriched context includes `cpu_ops`,
`nn_module`, kernel names, and `perf_category` per block.

### 2.1 Read the Agent Definition

Read: `TraceLens/Agent/Analysis/.cursor/agents/semantic-harmonization-agent.md`

### 2.2 Launch Harmonization Agent

```
---BEGIN AGENT INSTRUCTIONS---
<full contents of semantic-harmonization-agent.md>
---END AGENT INSTRUCTIONS---

**Execution Context:**
- Trace A directory: <output_dir>/_work/<name_a>/
- Trace B directory: <output_dir>/_work/<name_b>/
- Output directory: <output_dir>/_work/
- Name A: <name_a>
- Name B: <name_b>

Follow the agent instructions above to label and harmonize the blocks.
Return the summary described in the "Return Value" section.
```

For multi-region vLLM: run once per matching region pair.

### 2.3 Verify Harmonization

Check that `alignment.json`, `harmonization_context.json`, and
`harmonization_corrections.json` exist in `<output_dir>/_work/`.
Verify that `semantic_labels.json` files now contain `semantic_block`
fields on all kernels.

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

1. **Seamless flow** -- user provides two trace paths, orchestrator handles
   everything
2. **Parallel breakdown** -- both traces processed as parallel shell jobs
3. **Cross-trace harmonization** -- separate step unifies labels for
   consistent comparison
4. **Vendor-agnostic language** -- generic terms for all recommendations
5. **Complete coverage** -- every semantic block is labeled and compared
6. **No script creation** -- subagents use only existing scripts

---

## Final Deliverables

```
<output_dir>/
  _work/
    <name_a>/semantic_labels.json     # Per-trace semantic labels
    <name_b>/semantic_labels.json
    comparison.csv                     # Cross-trace comparison
  tracediff_output/                    # TraceDiff-compatible output
    diff_stats.csv
    diff_stats_unique_args_summary.csv
    cpu_op_map.json
    cpu_op_map_trace1.json
    cpu_op_map_trace2.json
    merged_tree_output.txt
```
