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

**Input files (pre-computed by orchestrator Step 4b):**
1. `<output_dir>/category_data/fusion_candidates.json` - Module-level candidate summaries with kernel details

**Output file you must write:**
- `<output_dir>/category_findings/kernel_fusion_findings.md`

---

## Error Handling

**If fusion_candidates.json is missing or empty:**
1. Write a findings file noting: "No kernel fusion opportunities detected."
2. Return gracefully

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not vendor-specific kernel names
- "fused kernel" not vendor-specific fusion implementations
- Focus on operation semantics, not vendor implementation details

**Exception:** When quoting kernel names from the candidates for identification, use the actual name.

---

## Analysis Workflow

### Step 1: Read Candidates

Read `<output_dir>/category_data/fusion_candidates.json`. Each entry contains:

- `module_name`: Module or function name from the trace
- `base_name`: Module type without instance index
- `parent_chain`: Ancestor modules in the call stack
- `instance_count`: How many times this module type repeats
- `kernel_count`: GPU kernels launched per instance
- `kernels`: List with `name`, `type`, `dur_us` per kernel
- `kernel_type_signature`: Ordered list of kernel types
- `has_fused_kernel`: Whether subtree contains a fused kernel
- `total_kernel_time_us`: Total GPU time across all instances

### Step 2: Classify Each Candidate

For each candidate, make three decisions:

**Decision 1 -- Is this a fusion opportunity?** Reject candidates where:
- The kernels are genuinely independent operations (e.g., separate projection GEMMs reading different weight matrices)
- The module is a container (Sequential, ModuleList, full decoder/encoder layer)
- All kernels are GEMMs (GEMM optimization is a separate concern)
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
- **low**: Speculative -- structural analysis suggests fusion is possible

### Step 4: Write Findings

Write `<output_dir>/category_findings/kernel_fusion_findings.md` using the command prefix.

Confidence maps directly to priority tier:
- All HIGH confidence findings → 🔴 P1
- All MEDIUM confidence findings → 🟡 P2
- All LOW confidence findings → 🟢 P3

Within each tier, sort by `total_kernel_time_us` descending.

**Title format:** `### <icon> <priority>: <Pattern Name>`

```markdown
# Kernel Fusion Analysis Summary (Experimental)

## Overview
Found N kernel fusion opportunities across M module types.

## Findings

### 🔴 P1: <Pattern Name>

**Insight:** <Module name, what it launches, how many instances, why it's fusable>

**Action:** <Specific recommendation>

**Confidence:** High -- <brief reason>

---

### 🟡 P2: <Pattern Name>

**Insight:** <Description>

**Action:** <Recommendation>

**Confidence:** Medium -- <brief reason>
```

If no fusion opportunities detected:
```markdown
# Kernel Fusion Analysis Summary (Experimental)

No kernel fusion opportunities detected.
```

---

## Key Principles

1. **Detection only** -- do NOT compute savings estimates or performance impact
2. **Let the data speak** -- classify based on module names AND kernel composition, not just one signal
3. **Reject confidently** -- not every multi-kernel module is a fusion opportunity; independent operations under a container module are not fusable
4. **Explain reasoning** -- especially for novel patterns, state why you believe the kernels are fusable
5. This analysis identifies fusion opportunities. Quantifying the performance benefit requires tensor shape data and roofline modeling, which is out of scope for this experimental section.
6. Use the **module name** to determine the user-facing operation name. If the module is `aten::conv2d` or `Conv2d`, call it "Convolution" in the finding title, not "GEMM" -- even though convolutions are implemented as GEMMs internally.

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

## What You CANNOT Infer

| NOT Observable | Why | Instead Say |
|----------------|-----|-------------|
| Tensor shapes | Not in candidate JSON | "Cannot assess data flow from candidate data" |
| Memory traffic savings | Would need tensor sizes | Do NOT estimate savings |
| Whether kernels share intermediate tensors | Would need data flow analysis | "Likely fusable based on module structure" |
| Root cause of decomposition | Could be framework, compiler, or intentional | "Module launches N separate kernels that may be fusable" |

