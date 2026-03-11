<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# About Semantic Analysis

## The Problem: Graph Mode Erases Context

When PyTorch models run in **graph mode** (CUDA Graphs / HIP Graphs), the runtime captures a fixed sequence of GPU kernels and replays them without re-executing the Python-side framework code. This is excellent for performance -- it eliminates CPU overhead and enables the GPU to execute kernels back-to-back -- but it destroys the information that analysts normally rely on to understand a trace.

In an **eager-mode** trace, every GPU kernel sits beneath a rich CPU call stack:

```
torch.nn.Linear.forward
  └─ torch.matmul
       └─ aten::mm
            └─ sm80_xmma_gemm_f16f16_...   ← GPU kernel
```

This stack tells you exactly what the kernel is: a matrix multiply inside a linear layer. You can see whether it belongs to the attention block, the FFN, the embedding, etc., just by reading the stack.

In a **graph-mode** trace, that entire stack collapses to:

```
cudaGraphLaunch
  └─ sm80_xmma_gemm_f16f16_...   ← same GPU kernel, no context
```

The kernel name alone -- often a mangled C++ template like `_gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_...` -- reveals nothing about which part of the model it belongs to. Without the call stack, you cannot answer basic questions:

- Is this GEMM the QKV projection or the output projection?
- Is this elementwise kernel a residual add or a rotary embedding?
- Which transformer layer does this kernel belong to?

This matters because **the same kernel type can appear dozens of times per iteration** with completely different performance characteristics depending on its role in the model.

## The Solution: Semantic Blocks

Semantic analysis reconstructs the missing context by assigning every GPU kernel a **semantic block** label -- a human-readable name that describes the kernel's role in the model architecture:

| Kernel (opaque) | Semantic Block (meaningful) |
|---|---|
| `_gemm_a16_w16_kernel_...GRID_MN_80_...` | QKV Projection |
| `_gemm_a16_w16_kernel_...GRID_MN_180_...` | Output Projection |
| `triton_poi_fused_1` | Rotary Embedding |
| `Rmsnorm2dFwd...FusedAddEnum_1_...` | Post-Attn Norm |
| `_matmul_ogs_NNT_bf16xbf16xmxfp4_...swiglu` | MoE GateUp+SwiGLU |
| `cross_device_reduce_1stage` | Post-MoE Residual Add |

The labeling process combines three signals:

1. **Kernel classification** -- regex-based detection of kernel type from its mangled name (GEMM, norm, attention, elementwise, collective, etc.)
2. **Layer pattern detection** -- autocorrelation analysis on the kernel sequence to find the repeating period (one transformer layer) and identify preamble/epilogue boundaries
3. **Positional reasoning** -- within each detected layer, the position of a kernel relative to other classified kernels disambiguates its semantic role (e.g., the first GEMM after a norm is QKV Projection, the GEMM after attention is Output Projection)

## Why Semantic Blocks Enable Two Critical Capabilities

### 1. Cross-Trace Comparison

Without semantic labels, comparing two traces is essentially impossible. Consider comparing the same model on two different GPU platforms:

**Platform A** might have:
```
kernel_unified_attention_3d          ← custom attention kernel
_gemm_a16_w16_kernel_...            ← vendor GEMM library
Rmsnorm2dFwd...                     ← CK-Tile norm kernel
cross_device_reduce_1stage          ← custom TP allreduce
```

**Platform B** might have:
```
fmhaSm100fKernel_QkvE4m3O...       ← different attention kernel
nvjet_tst_64x8_64x16_...           ← different GEMM library
triton_red_fused__to_copy_add_...   ← Triton-generated norm
allreduce_fusion_kernel_oneshot_... ← different TP allreduce
```

These kernels have completely different names, different counts (one platform may fuse operations that another keeps separate), and different call signatures. There is no way to match them by name.

Semantic blocks provide the **common vocabulary**. Both traces get labeled with the same set of block names (`QKV Projection`, `Attention`, `Post-Attn Norm`, etc.), enabling direct apples-to-apples comparison of timing, throughput, and efficiency for each functional operation.

### 2. Selecting the Right Performance Model

GPU kernel performance analysis requires choosing the correct **roofline model** -- the theoretical framework that defines what "good" performance looks like for a given operation. Different operation types have fundamentally different performance characteristics:

| Perf Category | What it models | Key metric | Bound by |
|---|---|---|---|
| **GEMM** | Matrix multiplications (projections, expert GEMMs) | TFLOPS/s vs. peak | Compute or memory, depending on shape |
| **SDPA** | Scaled dot-product attention (Flash, Paged) | TFLOPS/s accounting for Q·K^T + softmax + P·V | Memory bandwidth at short sequences, compute at long |
| **Normalization** | RMSNorm, LayerNorm, GroupNorm | GB/s vs. memory bandwidth peak | Almost always memory bandwidth |
| **Elementwise** | Residual adds, activations, rotary, routing | GB/s vs. memory bandwidth peak | Always memory bandwidth |

Without knowing that a kernel is a "QKV Projection", you cannot know to evaluate it as a GEMM with shape M=batch, N=heads×head_dim, K=hidden. Without knowing it is "Post-Attn Norm", you cannot know to evaluate it as a normalization operation where the theoretical data movement is `num_tokens × hidden_size × bytes_per_element`.

The semantic block label determines:

- **Which roofline formula to apply** -- `2×M×N×K` FLOPS for GEMM vs. `tokens × hidden × bpe` bytes for elementwise
- **Which shape parameters to derive** from the model's `config.json` -- the QKV Projection needs `num_attention_heads`, `num_key_value_heads`, and `head_dim`; the MoE GateUp needs `num_experts`, `experts_per_token`, and `intermediate_size`
- **What "good" looks like** -- a GEMM achieving 50% of peak TFLOPS/s may be reasonable for a narrow shape, while a norm kernel achieving 10% of peak memory bandwidth indicates a clear optimization opportunity

## The Semantic Block Vocabulary

The vocabulary is a fixed set of labels defined in `category_mappings.py`. Each label maps to exactly one performance category. The vocabulary covers the standard transformer architecture and its MoE variant:

**Preamble** -- operations before the first layer:
`Embedding`, `Input Norm`

**Self-Attention** -- the attention sub-block within each layer:
`Pre-Attn Norm` → `QKV Projection` (or separate `Q`/`KV`) → `Rotary Embedding` → `Attention` → `KV Cache Store` → `Output Projection` → `Attention Output Gate` → `Post-Attn Residual Add`

**MoE FFN** -- the mixture-of-experts feed-forward sub-block:
`Post-Attn Norm` → `Router Gate` → `MoE Routing` → `MoE GateUp+SwiGLU` → `MoE Quantize` → `MoE Down Projection` → `MoE Finalize` → `Post-MoE Residual Add`

**Dense FFN** -- the standard feed-forward sub-block (non-MoE models):
`FFN Norm` → `GateUp Projection` (or separate `Gate`/`Up`) → `Activation` → `Down Projection` → `Post-FFN Residual Add`

**Epilogue** -- operations after the last layer:
`Final Norm`, `LM Head`

Every kernel in a trace must be assigned one of these labels. There is no "Other" or "Unknown" category -- the goal is complete, analyzable coverage. If a kernel cannot be confidently mapped to a specific block, it is assigned to the nearest matching label in the Elementwise category, ensuring it still participates in roofline analysis rather than being excluded.
