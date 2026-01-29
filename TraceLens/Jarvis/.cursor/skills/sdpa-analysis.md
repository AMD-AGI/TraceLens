---
name: SDPA Analysis
description: Analyze Scaled Dot Product Attention operations for performance bottlenecks
triggers:
  - sdpa analysis
  - analyze sdpa
  - attention analysis
tools:
  - terminal
  - file_read
  - file_write
---

# SDPA Analysis Skill

Analyze SDPA (Scaled Dot Product Attention) operations for performance bottlenecks and Flash Attention opportunities.

---

## Input Contract

Expects pre-filtered data:
1. **`category_data/sdpa_fwd_ops.csv`** - Filtered SDPA operations
2. **`metadata/sdpa_fwd_metadata.json`** - Hardware specs
3. **`category_data/sdpa_fwd_tree_data.json`** - Pre-computed tree data

---

## Output Contract

Produces **`category_findings/sdpa_fwd_findings.md`**

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
python3 Jarvis/category_analyses/sdpa_analysis.py --output-dir <output_dir>
```

### Step 2: Interpret Results

1. **Assess Category Significance** - Attention typically 20-40% of compute in transformers

2. **Validate Bottlenecks** - Key questions:
   - Is Flash Attention being used?
   - What are the sequence lengths? (short sequences <1024 have naturally low efficiency)
   - Are there unfused attention patterns (softmax + bmm + mul + copy_)?

3. **Contextualize Issues**:
   - Flash Attention efficiency drops significantly for short sequences (N < 1024)
   - 8-15% efficiency at N=512 is not unusual for Flash Attention
   - Unfused attention is typically 3-10x slower than Flash Attention

4. **Identify Additional Issues**:
   - Unfused attention patterns (multiple small ops instead of single fused kernel)
   - Missing Flash Attention on critical attention layers
   - Unexpected kernel selections

### Step 3: Trace Call Stacks

Review tree data to identify attention patterns:

```python
import json
with open('<output_dir>/category_data/sdpa_fwd_tree_data.json') as f:
    tree_data = json.load(f)
# Look for parent module names to identify attention layers
```

### Step 4: Determine Optimization Paths

**Path A - Fusion/Algorithmic:**
- **Unfused → Flash Attention:** 3-10x speedup potential
- Use `torch.nn.functional.scaled_dot_product_attention` with Flash Attention backend
- Check if ck_tile or composable_kernel implementations are available
- Consider sequence length when evaluating Flash Attention benefits

**Path B - Kernel Optimization:**
- If already using Flash Attention but low efficiency:
  - Generate replay artifact for kernel team
  - Check if efficiency is expected for given sequence length
- If unfused and Flash Attention not applicable:
  - Analyze specific attention pattern for custom fusion

### Step 5: Write Category Findings

Create `<output_dir>/category_findings/sdpa_fwd_findings.md`

---

## Common Patterns for SDPA Analysis

### Attention-Heavy Models (Transformers, ViT)
- **Look for:** softmax, bmm, mul (scaling), copy_ (transposes)
- **Path A:** Flash Attention (3-10x speedup on attention)
- **Path B:** Optimize individual kernels (limited gains, maybe 10-30%)

### Short Sequence Attention
- **Symptoms:** Sequence length N < 1024, Flash Attention at 8-15% efficiency
- **Expected behavior:** Memory overhead dominates when N is small
- **Not a problem:** Compare to reference platform at same N to determine if it's hardware-specific
- **Action:** Only flag as bottleneck if significantly worse than expected for that sequence length

### Unfused Attention Patterns
- **Symptoms:** Multiple operations: softmax, bmm, mul, copy_ appearing together
- **High impact:** Replacing with Flash Attention typically gives 3-10x speedup
- **Path A (primary):** Migrate to Flash Attention
- **Path B (fallback):** If Flash Attention not applicable, explore custom fusion

### Flash Attention Already Used
- **Good sign:** Model is already optimized
- **Check efficiency:** Should be 40-70% for long sequences (>2048)
- **If low efficiency:** 
  - Verify sequence length (short sequences naturally have low efficiency)
  - Generate replay artifact if unexpectedly low for given sequence length

---

## Key Principles

1. **Flash Attention is the primary optimization** - Always check if it's being used
2. **Sequence length matters** - Short sequences naturally have lower efficiency
3. **Unfused attention is a major opportunity** - 3-10x speedup potential
4. **Provide BOTH paths** - Even if Flash Attention is obvious, include kernel optimization path

---

## Efficiency Context

| Sequence Length | Expected Flash Attention Efficiency |
|----------------|-------------------------------------|
| N < 512 | 5-15% (memory overhead dominates) |
| N = 1024 | 20-40% |
| N = 2048 | 40-60% |
| N > 4096 | 50-70% |

**Unfused attention:** Typically 3-10x slower than these Flash Attention numbers
