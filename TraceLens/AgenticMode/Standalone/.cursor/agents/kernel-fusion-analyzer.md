<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: kernel-fusion-analyzer
description: Analyze kernel fusion opportunities from pre-extracted candidate data. Use when orchestrator detects fusion candidates in Step 4b.
model: inherit
---

# Kernel Fusion Analyzer (Experimental)

Analyze GPU kernel fusion opportunities from pre-extracted module-level candidate data. Classify candidates as known patterns, novel patterns, or not fusable.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/fusion_candidates.json` - Module-level candidate summaries with kernel details
2. `<output_dir>/category_data/kernel_fusion_metrics.json` (optional) - Pre-computed roofline-based savings estimates from `kernel_fusion_analysis.py`

**Output file you must write:**
- `<output_dir>/system_findings/kernel_fusion_findings.md`

---

## Error Handling

**If fusion_candidates.json is missing or empty:**
1. Write a findings file noting: "No kernel fusion opportunities detected."
2. Return gracefully

---

## Language Guidelines

Use vendor-agnostic terminology in all narrative text (Insight, Action, Impact):
- "GPU kernels" not vendor-specific kernel names
- "fused kernel" or "custom fused kernel" — never mention specific frameworks
- "compiler fusion" or "graph-level fusion" — not "torch.compile", "Inductor", or other framework-specific names
- Focus on operation semantics, not vendor implementation details

**Exception:** When quoting kernel names from the candidates for identification in the Kernels table, use the actual name as-is.

---

## Analysis Workflow

### Step 1: Generate Metrics, Read Candidates and Impact Estimates

Run the deterministic fusion analysis script to produce `kernel_fusion_metrics.json`:

```bash
<prefix> python TraceLens/AgenticMode/Standalone/category_analyses/kernel_fusion_analysis.py \
  --output-dir <output_dir>
```

Then read `<output_dir>/category_data/fusion_candidates.json`. Each entry contains:

- `module_name`: Module or function name from the trace
- `base_name`: Module type without instance index
- `parent_chain`: Ancestor modules in the call stack
- `instance_count`: How many times this module type repeats
- `kernel_count`: GPU kernels launched per instance
- `kernels`: List with `name`, `type`, `dur_us` per kernel
- `kernel_type_signature`: Ordered list of kernel types
- `has_fused_kernel`: Whether subtree contains a fused kernel
- `total_kernel_time_us`: Total GPU time across all instances

Then read `<output_dir>/category_data/kernel_fusion_metrics.json`. This contains pre-computed `impact_estimates` with roofline-based savings for each candidate. Each estimate has:

- `operation`: Matches `base_name` in fusion_candidates
- `savings_ms`, `savings_ms_low`, `savings_ms_high`: Savings range (75-100% of roofline)
- `bound_type`: "compute" or "memory"
- `fusion_type`: "matrix_compute" or "memory_bound"
- `confidence`: "high" or "medium"
- `time_ms`: Total candidate time across all instances
- `warning`: Present when some kernels lack perf models

Use these estimates to populate the **Impact** field in each finding. Match by `operation` name (= `base_name`). If no matching estimate exists, use "Not quantifiable -- insufficient perf model data".

### Step 2: Classify Each Candidate

For each candidate, make three decisions:

**Decision 1 -- Is this a fusion opportunity?** Reject candidates where:
- The kernels are genuinely independent operations (e.g., separate projection GEMMs reading different weight matrices)
- The module is a container (Sequential, ModuleList, full decoder/encoder layer)
- All kernels are GEMMs
- The non-GEMM kernels are all normalization ops (GEMM + LayerNorm/Norm sequences are not fusable)
- Any kernel is a Triton-compiled fused kernel (`triton_` prefix)
- The module already contains a fused kernel (`has_fused_kernel: true`)

**Decision 2 -- What pattern?** Check known patterns first:

| Pattern | Kernel composition | Module name hints |
|---------|-------------------|-------------------|
| Unfused attention | >= 2 GEMM + Softmax, no fused attention kernel | "attention", "sdpa", "self_attn" |
| Unfused RMSNorm | rsqrt + mean or pow + mul | "rmsnorm", "rms_norm" |
| Unfused LayerNorm | rsqrt + mean + sub + mul | "layernorm", "layer_norm" |
| Unfused BatchNorm | mul + add (precomputed scale+shift) | "batchnorm", "batch_norm", "FrozenBatchNorm" |
| Unfused RoPE | neg + cat + mul + add | "rotary", "rope", "apply_rotary" |
| Unfused SiGLU/SwiGLU | SiLU + Mul (may have GEMMs between) | "silu", "swiglu", MLP context |
| Unfused GELU | Multiple GELU component kernels | "gelu" |
| GEMM epilogue | GEMM + 1-2 elementwise as separate kernels | "linear", "conv2d", "addmm" |

Then look for novel patterns:
- Multiple elementwise kernels under one module
- Reduction + elementwise sequences
- Dropout + residual add + normalization under one module
- Repeated small kernels suggesting a decomposed operation

**Decision 3 -- What recommendation?** Tailor to framework context visible in the parent chain and module names.

### Step 3: Assign Confidence

- **high**: Module name matches a known pattern AND kernel composition confirms it
- **medium**: Module name OR kernel composition suggests a pattern, but not both
- **low**: Speculative - structural analysis suggests fusion is possible

### Step 4: Write Findings

Write `<output_dir>/system_findings/kernel_fusion_findings.md` using the command prefix.

Confidence maps directly to priority tier:
- HIGH confidence findings → 🔴 P1
- MEDIUM confidence findings → 🟡 P2
- LOW confidence findings → 🟢 P3

Within each tier, sort by `total_kernel_time_us` descending.

**Title format:** `### <icon> <priority>: <Pattern Name>`

```markdown
# Kernel Fusion Analysis Summary (Experimental)

## Overview
Found N kernel fusion opportunities across M module types.

> **Methodology:** Savings projections use a roofline model with 85% memory/compute pipeline overlap (i.e. fused kernel time is interpolated between perfect overlap and no overlap). Actual savings may vary with workload and hardware.

## Findings

### 🔴 P1: <Pattern Name> (<time_ms> ms, <instance_count> instances)

**Insight:** <Module name, what it launches, how many instances, why it's fusable>

**Action:** <Specific recommendation>

**Impact:** ~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) with all N kernels modelled

**Confidence:** High -- <brief reason from fusion pattern classification>

## Detailed Analysis

<!-- reasoning-candidate tier=system rank=1 -->
#### <Pattern Name> (<time_ms> ms, <instance_count> instances)

**Identification:** <1-2 sentences: how this fusion candidate was surfaced — module name,
kernel pattern, instance count, has_fused_kernel status. Must end with
(source: `fusion_candidates.json` → `module_name`, `has_fused_kernel`, `kernels[]`)>

**Data:**

| Kernel | Type | Duration (us) | Perf model |
|--------|------|--------------|------------|
| <kernel name (truncated to ~60 chars)> | <type> | X.X | Yes/No |

**Impact estimate:**

- Low end (75% roofline): X.XXX ms savings (X.XX% E2E)
- High end (100% roofline): X.XXX ms savings (X.XX% E2E)
- Coverage: M of N kernels modelled
- Fusion pattern: compute/memory-bound, matrix_compute/memory_bound
- Confidence: High/Medium/Low — <brief reason>

<!-- When partial coverage, append to Coverage: "(K kernel(s) use measured trace time)". -->
<!-- When not quantifiable: **Impact estimate:** Impact estimate is not quantifiable from trace data. -->
```

If no fusion opportunities detected:
```markdown
# Kernel Fusion Analysis Summary (Experimental)

No kernel fusion opportunities detected.
```

---

## Key Principles

1. **Include pre-computed savings** from `kernel_fusion_metrics.json` when available -- do NOT re-derive savings yourself, use the values from the metrics JSON
2. **Let the data speak** -- classify based on module names AND kernel composition, not just one signal
3. **Reject confidently** -- not every multi-kernel module is a fusion opportunity; independent operations under a container module are not fusable
4. **Explain reasoning** -- especially for novel patterns, state why you believe the kernels are fusable
5. Use the **module name** to determine the user-facing operation name. If the module is `aten::conv2d` or `Conv2d`, call it "Convolution" in the finding title, not "GEMM" -- even though convolutions are implemented as GEMMs internally.

---

## What You CAN Infer

| Observable | Source |
|------------|--------|
| Module names | `module_name`, `base_name` fields |
| Kernel names and types | `kernels[].name`, `kernels[].type` |
| Kernel durations | `kernels[].dur_us` |
| Instance count | `instance_count` field |
| Architecture context | `parent_chain` field |
| Already-fused status | `has_fused_kernel` field |
| Savings estimates | `kernel_fusion_metrics.json` `impact_estimates[]` (when available) |

## What You CANNOT Infer

| NOT Observable | Why | Instead Say |
|----------------|-----|-------------|
| Tensor shapes | Not in candidate JSON | "Cannot assess data flow from candidate data" |
| Whether kernels share intermediate tensors | Would need data flow analysis | "Likely fusable based on module structure" |
| Root cause of decomposition | Could be framework, compiler, or intentional | "Module launches N separate kernels that may be fusable" |

