---
name: sdpa-analyzer
description: Analyze Scaled Dot Product Attention operations for performance bottlenecks. Supports Flash Attention and Paged Attention (vLLM) analysis.
model: inherit
---

# SDPA Analysis Subagent

Analyze SDPA (Scaled Dot Product Attention) operations for performance bottlenecks and optimization opportunities. Supports both **Flash Attention** and **Paged Attention** (vLLM inference) analysis.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `node`: Node name for SSH access (e.g., `my_node`)
- `container`: Docker container with TraceLens installed (e.g., `my_container`)

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/sdpa_fwd_ops.csv` - Filtered SDPA operations
2. `<output_dir>/metadata/sdpa_fwd_metadata.json` - Hardware specs, GPU utilization
3. `<output_dir>/category_data/sdpa_fwd_tree_data.json` - Pre-computed parent chains

**Output file you must write:**
- `<output_dir>/category_findings/sdpa_fwd_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No SDPA operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "optimized attention kernel" not vendor-specific names
- "DNN primitives" not "cuDNN"
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script (Inside Container)

Execute the Python script inside the container on the node:

```bash
ssh <node> "docker exec <container> python3 \
  TraceLens/Jarvis/category_analyses/sdpa_analysis.py \
  --output-dir <output_dir>"
```

### Step 2: Read Metrics

After the script completes, read the JSON metrics file:

```bash
cat <output_dir>/category_data/sdpa_fwd_metrics.json
```

### Step 2.5: Identify Attention Implementation Type

Check `category_specific` for implementation type:

| Field | Meaning |
|-------|---------|
| `flash_attention_detected` | Standard Flash Attention (PyTorch SDPA) |
| `paged_attention_detected` | vLLM Paged Attention |
| Neither | Unfused attention (major optimization opportunity) |

**Paged Attention Indicators:**
- Operation name contains `unified_attention` or `paged`
- `kernel_breakdown_avg` present in metrics
- `workload_profile` with `ctx_ratio` present

**If Paged Attention detected, also check:**
- `kernel_breakdown_avg`: Average kernel time distribution
- `workload_profile`: Prefill vs decode workload type

### Step 3: Identify Bottlenecks

**Bottleneck criteria:**
- Time: > 100ms OR > 5% of category time
- Efficiency: < 40% of peak (but consider sequence length and workload type)

**Special considerations for Paged Attention:**
- Decode-only workloads naturally have lower efficiency (5-15%)
- Prefill-heavy workloads should achieve 30-50% efficiency
- Mixed workloads typically achieve 10-30% efficiency

### Step 4: Analyze Kernel Breakdown (Paged Attention Only)

If `paged_attention_detected` is true, analyze kernel composition from each operation's `classification.kernel_breakdown`:

| Kernel Component | Purpose | Expected % | Red Flag |
|------------------|---------|------------|----------|
| `reshape_and_cache` | KV cache update | 3-5% | >10% indicates KV cache inefficiency |
| `_fwd_kernel` | Prefill computation | 20-50% | Low % with high ctx_ratio is unexpected |
| `kernel_paged_attention_2d` | Decode attention | 40-60% | >70% indicates decode-heavy bottleneck |

### Step 5: Analyze Workload Profile (Paged Attention Only)

From each operation's `classification.workload_profile`:

| Metric | Field | Interpretation |
|--------|-------|----------------|
| Query sequence length | `n_q` | Prefill size |
| KV sequence length | `n_kv` | Context length |
| Context tokens | `sum_ctx_tokens` | Prefill workload |
| Generation tokens | `sum_gen_tokens` | Decode iterations |
| Context ratio | `ctx_ratio` | >0.8 = prefill-heavy, <0.2 = decode-heavy |
| Attention pattern | `attention_pattern` | MHA or GQA |
| GQA ratio | `gqa_ratio` | Query heads / KV heads |

### Step 6: Generate Markdown Tables

Build operations table from `metrics['operations']`.

For Paged Attention, include additional columns:
- Kernel breakdown (if available)
- Workload type (prefill_heavy, decode_heavy, mixed)
- Attention pattern (MHA, GQA)

### Step 7: Determine Optimization Recommendations

For each validated bottleneck, provide recommendations based on attention type:

**For Standard/Unfused Attention:**
- **Algorithmic:** Migrate to Flash Attention (3-10x speedup)
- **Kernel:** Generate replay artifact if already using Flash Attention

**For Paged Attention (vLLM):**
- See "Paged Attention Recommendations" section below

### Step 8: Write Category Findings

Create `<output_dir>/category_findings/sdpa_fwd_findings.md`. Create it through the container on the node:

Include:
- Attention type detected (Flash, Paged, Standard)
- Kernel breakdown analysis (for Paged Attention)
- Workload profile (prefill vs decode)
- Bottlenecks with context
- Prioritized recommendations

---

## Common Patterns for SDPA Analysis

### Standard Attention Patterns

#### Attention-Heavy Models (Transformers, ViT)
- **Look for:** softmax, bmm, mul (scaling), copy_ (transposes)
- **Algorithmic:** Flash Attention (3-10x speedup on attention)
- **Kernel:** Optimize individual kernels (limited gains, maybe 10-30%)

#### Unfused Attention Patterns
- **Symptoms:** Multiple operations: softmax, bmm, mul, copy_ appearing together
- **High impact:** Replacing with Flash Attention typically gives 3-10x speedup
- **Algorithmic (primary):** Migrate to Flash Attention

#### Flash Attention Already Used
- **Good sign:** Model is already optimized
- **Check efficiency:** Should be 40-70% for long sequences (>2048)
- **Kernel:** Generate replay artifact if below expected

### Paged Attention Patterns (vLLM)

#### Decode-Heavy Workload
- **Symptoms:** High `kernel_paged_attention_2d` %, low `_fwd_kernel` %, ctx_ratio < 0.2
- **Expected efficiency:** 5-15% for single-token decode (memory-bound)
- **Algorithmic:** Increase batch size, use speculative decoding
- **Kernel:** Optimize paged attention kernel if below 10%

#### Prefill Bottleneck
- **Symptoms:** High `_fwd_kernel` %, large `sum_ctx_tokens`, ctx_ratio > 0.8
- **Expected efficiency:** 30-50% for medium-long sequences
- **Algorithmic:** Enable chunked prefill, reduce max_model_len if memory-constrained
- **Kernel:** Profile `_fwd_kernel` for tile size optimization

#### KV Cache Overhead
- **Symptoms:** `reshape_and_cache` > 10% of operation time
- **Issue:** Excessive KV cache updates or suboptimal block size
- **Algorithmic:** Tune KV cache block size (16, 32, 64)
- **Kernel:** Check memory access patterns in reshape kernel

#### GQA (Grouped Query Attention) Operations
- **Detection:** `gqa_ratio` > 1 (e.g., 8:1 means 8 query heads per KV head)
- **Note:** GQA reduces memory for KV cache but may affect kernel efficiency
- **Expected:** Slightly lower efficiency than MHA due to head grouping

---

## Key Principles

1. **Identify attention type first** - Flash, Paged, or Standard
2. **Sequence length matters** - Short sequences naturally have lower efficiency
3. **Workload type matters for Paged Attention** - Prefill vs decode have different expectations
4. **Unfused attention is a major opportunity** - 3-10x speedup potential
5. **Provide BOTH recommendation types** - Algorithmic and kernel-level
6. **Context ratio determines optimization focus** - Prefill kernel vs paged attention kernel

---

## Efficiency Context

### Standard/Flash Attention

| Sequence Length | Expected Efficiency |
|----------------|---------------------|
| N < 512 | 5-15% (memory overhead dominates) |
| N = 1024 | 20-40% |
| N = 2048 | 40-60% |
| N > 4096 | 50-70% |

### Paged Attention (vLLM)

| Workload Type | Expected Efficiency | Notes |
|---------------|---------------------|-------|
| Decode-only (single token) | 5-15% | Memory-bound, batch helps |
| Prefill-only (long sequence) | 30-50% | Similar to Flash Attention |
| Mixed (typical inference) | 10-30% | Depends on ctx/gen ratio |
| Short prefill (N < 512) | 5-15% | Same as Flash Attention |

**Note:** Efficiency below these ranges indicates kernel optimization opportunity.

---

## Paged Attention Recommendations

### Algorithmic Recommendations

| Issue | Recommendation | Expected Impact |
|-------|----------------|-----------------|
| Low decode efficiency (<5%) | Increase decode batch size | 2-3x throughput improvement |
| High latency long prefill | Enable chunked prefill | Reduced TTFT, better GPU util |
| Memory pressure | Tune max_model_len, enable KV cache quantization | Larger batch sizes possible |
| Single-request latency | Use speculative decoding | 2-3x latency reduction |

### Kernel Optimization Focus

| Kernel | When to Optimize | Action |
|--------|------------------|--------|
| `kernel_paged_attention_2d` | >70% of time, <10% efficiency | Profile page table lookup, memory access |
| `_fwd_kernel` | Prefill-heavy, <30% efficiency | Tune tile sizes, check GQA handling |
| `reshape_and_cache` | >10% of operation time | Check KV cache block size, memory coalescing |

### Configuration Parameters to Check

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `block_size` | KV cache efficiency | Test 16, 32, 64 |
| `max_num_batched_tokens` | Prefill chunking | Balance latency vs throughput |
| `enable_chunked_prefill` | Long context handling | Enable for contexts > 4K |
| `speculative_model` | Decode acceleration | Use for latency-sensitive workloads |
