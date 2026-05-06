#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Classify kernels by type and perf_category using extensible regex rules.

Each kernel is classified into a kernel_type (e.g., "GEMM (cuBLAS)", "RMSNorm")
and a perf_category (GEMM, SDPA, Normalization, Elementwise, MemCpy). The perf_category
is derived from the kernel_type via KERNEL_TYPE_TO_PERF_CATEGORY.

Input: extracted trace data JSON
Output: JSON with per-kernel type + perf_category classification

Usage:
    python classify_kernels.py <extracted.json> [-o classified.json]
"""

import argparse
import json
import re
import sys

# ---------------------------------------------------------------------------
# Regex rules: (pattern, kernel_type, priority)
# Higher priority wins on conflict. Rules are evaluated in order but sorted
# by priority when multiple match.
# ---------------------------------------------------------------------------

KERNEL_TYPE_RULES = [
    # ---- Normalization (priority 20) ----
    (r"(?i)Rmsnorm|rmsnorm2d", "RMSNorm", 20),
    (r"(?i)layer_norm|layernorm", "LayerNorm", 20),
    (r"(?i)l2norm_fwd|l2norm_kernel", "L2Norm", 20),
    (r"(?i)rsqrt.*mean.*pow|mean.*mul.*pow.*rsqrt", "RMSNorm (fused triton)", 20),
    (r"(?i)_to_copy_add.*mean.*rsqrt", "RMSNorm (fused triton)", 20),
    (r"(?i)fused__to_copy_abs.*mean.*pow.*rsqrt", "RMSNorm (fused triton)", 18),
    (r"(?i)fused__to_copy_add.*mean.*pow.*rsqrt", "RMSNorm (fused triton)", 18),
    (r"(?i)fused_add_copy__mean.*pow", "RMSNorm (fused triton residual)", 18),
    (
        r"(?i)fused__to_copy_add_copy__mean.*pow",
        "RMSNorm (fused triton residual)",
        18,
    ),
    (
        r"(?i)fused__to_copy_add_mean_moe_forward_shared.*pow.*rsqrt",
        "RMSNorm (shared expert + norm)",
        20,
    ),
    (
        r"(?i)fused__to_copy_add_mean.*mul.*pow.*rsqrt",
        "RMSNorm (fused triton)",
        18,
    ),
    # ---- Attention / SDPA (priority 20) ----
    (r"(?i)attention_[23]d|unified_attention", "Attention (AMD unified)", 20),
    (r"(?i)fmha[Ss]m\d+|fmhaSm100f", "Attention (NVIDIA fmha)", 20),
    (r"(?i)paged_attention|PagedAttention", "Attention (paged)", 20),
    (r"(?i)flash_attn|flash_fwd", "Attention (flash)", 20),
    (r"(?i)reduce_segments", "Attention Reduce", 20),
    (r"(?i)reshape_and_cache_flash|reshape_and_cache", "KV Cache Store", 20),
    # ---- Linear attention kernels (priority 20) ----
    (r"(?i)chunk_gated_delta_rule_fwd", "Linear Attention (chunk delta)", 20),
    (r"(?i)chunk_fwd_kernel_o", "Linear Attention (chunk fwd)", 20),
    (r"(?i)chunk_scaled_dot_kkt", "Linear Attention (chunk kkt)", 20),
    (r"(?i)chunk_local_cumsum", "Linear Attention (chunk cumsum)", 20),
    (r"(?i)fused_recurrent_gated_delta_rule", "Linear Attention (recurrent delta)", 20),
    (r"(?i)recompute_w_u_fwd", "Linear Attention (recompute wu)", 20),
    (r"(?i)merge_\d+x\d+_to_\d+x\d+_inverse", "Linear Attention (merge inverse)", 20),
    (r"(?i)_causal_conv1d_fwd|_causal_conv1d_update", "Causal Conv1D", 20),
    (r"(?i)fused_gdn_gating", "GDN Gating", 20),
    # ---- MoE-specific (priority 20-25) ----
    (r"(?i)swiglu|swiGlu", "MoE GateUp+SwiGLU GEMM", 25),
    (r"(?i)fmoe.*g1u1.*silu|fmoe.*silu", "MoE GEMM+SiLU (aiter)", 25),
    (r"(?i)kernel_moe_gemm|MoeGemmBlockScale", "MoE GEMM (CK)", 22),
    (r"(?i)finalize.*scatter|_finalize_matmul|finalizeKernel", "MoE Finalize", 20),
    (r"(?i)topk_forward", "MoE TopK", 20),
    (r"(?i)topkGatingSoftmax", "MoE TopK Gating", 20),
    (r"(?i)bitmatrix|_sum_bitmatrix", "MoE BitMatrix", 20),
    (r"(?i)combined_routing_memset", "MoE Routing Memset", 20),
    (r"(?i)combined_routing_compute", "MoE Routing Compute", 20),
    (r"(?i)routingIndicesCluster|routingRenormalize", "MoE Routing (fused)", 20),
    (r"(?i)MoeSorting", "MoE Sorting", 20),
    (r"(?i)constant_pad.*moe|moe_forward", "MoE Pad", 20),
    (r"(?i)quantize_with_block_size", "MoE Quantize", 20),
    (
        r"(?i)per_token_group_quant|dynamic_per_group_scaled_quant",
        "Quantize (per-group)",
        18,
    ),
    # ---- Activation functions (priority 15-18) ----
    (r"(?i)act_and_mul_kernel.*silu_kernel", "Activation (SiLU+Mul)", 20),
    (r"(?i)silu_and_mul|silu_mul|SiluAndMul", "Activation (SiLU+Mul)", 18),
    (r"(?i)activationDeepSeekKernel", "Activation (DeepSeek fused)", 18),
    (r"(?i)fused_mul_sigmoid", "Activation (SiLU)", 18),
    (r"(?i)\bsilu\b|swish", "Activation (SiLU)", 15),
    (r"(?i)\bgelu\b", "Activation (GELU)", 15),
    # ---- MemCpy (priority 30 -- highest, based on trace cat=gpu_memcpy) ----
    (r"(?i)memcpy", "MemCpy", 30),
    # ---- MoE-specific (priority 20-25) ----
    # These stay distinct: used as anchors and for MoE model detection.
    (r"(?i)swiglu|swiGlu", "MoE GEMM", 25),
    (r"(?i)fmoe.*g1u1.*silu|fmoe.*silu", "MoE GEMM", 25),
    (r"(?i)kernel_moe_gemm|MoeGemmBlockScale", "MoE GEMM", 22),
    (r"(?i)finalize.*scatter|_finalize_matmul|finalizeKernel", "MoE Finalize", 20),
    (r"(?i)topk_forward", "MoE Routing", 20),
    (r"(?i)topkGatingSoftmax", "MoE Routing", 20),
    (r"(?i)bitmatrix|_sum_bitmatrix", "MoE Routing", 20),
    (r"(?i)combined_routing_memset", "MoE Routing", 20),
    (r"(?i)combined_routing_compute", "MoE Routing", 20),
    (r"(?i)routingIndicesCluster|routingRenormalize", "MoE Routing", 20),
    (r"(?i)MoeSorting", "MoE Routing", 20),
    (r"(?i)constant_pad.*moe|moe_forward", "MoE Routing", 20),
    (r"(?i)quantize_with_block_size", "MoE Quantize", 20),
    # ---- Normalization (priority 18-20) ----
    (r"(?i)Rmsnorm|rmsnorm2d", "Normalization", 20),
    (r"(?i)layer_norm|layernorm", "Normalization", 20),
    (r"(?i)l2norm_fwd|l2norm_kernel", "Normalization", 20),
    (r"(?i)rsqrt.*mean.*pow|mean.*mul.*pow.*rsqrt", "Normalization", 20),
    (r"(?i)_to_copy_add.*mean.*rsqrt", "Normalization", 20),
    (r"(?i)fused__to_copy_abs.*mean.*pow.*rsqrt", "Normalization", 18),
    (r"(?i)fused__to_copy_add.*mean.*pow.*rsqrt", "Normalization", 18),
    (r"(?i)fused_add_copy__mean.*pow", "Normalization", 18),
    (r"(?i)fused__to_copy_add_copy__mean.*pow", "Normalization", 18),
    (
        r"(?i)fused__to_copy_add_mean_moe_forward_shared.*pow.*rsqrt",
        "Normalization",
        20,
    ),
    (r"(?i)fused__to_copy_add_mean.*mul.*pow.*rsqrt", "Normalization", 18),
    # ---- Attention / SDPA (priority 20) ----
    # Distinct anchor: identifies attention blocks during alignment.
    (r"(?i)attention_[23]d|unified_attention", "Attention", 20),
    (r"(?i)fmha[Ss]m\d+|fmhaSm100f", "Attention", 20),
    (r"(?i)paged_attention|PagedAttention", "Attention", 20),
    (r"(?i)flash_attn|flash_fwd", "Attention", 20),
    (r"(?i)reduce_segments", "Attention", 20),
    # ---- Anchor: KV Cache Store (priority 20) ----
    (r"(?i)reshape_and_cache_flash|reshape_and_cache", "KV Cache Store", 20),
    # ---- Linear attention (priority 20) ----
    # Distinct anchor: identifies linear-attention blocks during alignment.
    (r"(?i)chunk_gated_delta_rule_fwd", "Linear Attention", 20),
    (r"(?i)chunk_fwd_kernel_o", "Linear Attention", 20),
    (r"(?i)chunk_scaled_dot_kkt", "Linear Attention", 20),
    (r"(?i)chunk_local_cumsum", "Linear Attention", 20),
    (r"(?i)fused_recurrent_gated_delta_rule", "Linear Attention", 20),
    (r"(?i)recompute_w_u_fwd", "Linear Attention", 20),
    (r"(?i)merge_\d+x\d+_to_\d+x\d+_inverse", "Linear Attention", 20),
    (r"(?i)_causal_conv1d_fwd|_causal_conv1d_update", "Linear Attention", 20),
    # ---- GDN Gating (priority 20) ----
    # Distinct: has perf_category SDPA-GDN, affects RLE grouping.
    (r"(?i)fused_gdn_gating", "GDN Gating", 20),
    # ---- Anchor: Rotary Embedding (priority 18) ----
    (r"(?i)rotary|rope", "Rotary Embedding", 18),
    # ---- Activation (DeepSeek fused MoE) (priority 18) ----
    # Distinct: perf_category Elementwise-MoE, affects RLE grouping.
    (r"(?i)activationDeepSeekKernel", "Activation (DeepSeek fused)", 18),
    # ---- Activation / misc elementwise (priority 5-20) ----
    (r"(?i)act_and_mul_kernel.*silu_kernel", "Elementwise", 20),
    (r"(?i)embedding", "Elementwise", 20),
    (r"(?i)gather_kernel|vectorized_gather", "Elementwise", 20),
    (r"(?i)per_token_group_quant|dynamic_per_group_scaled_quant", "Quantization", 18),
    (r"(?i)silu_and_mul|silu_mul|SiluAndMul", "Elementwise", 18),
    (r"(?i)fused_mul_sigmoid", "Elementwise", 18),
    (r"(?i)\bsilu\b|swish", "Elementwise", 15),
    (r"(?i)\bgelu\b", "Elementwise", 15),
    (r"(?i)splitKreduce|splitkreduce", "GEMM", 15),
    (r"(?i)top_k|top_p|sampling|topk_topp", "Elementwise", 15),
    (r"(?i)CUDAFunctorOnSelf_add<int>", "Elementwise", 15),
    (r"(?i)__amd_rocclr_copyBuffer", "Elementwise", 12),
    (r"(?i)memcpy32_post", "Elementwise", 12),
    (r"(?i)copy_page_indices", "Elementwise", 12),
    (r"(?i)vectorized_elementwise_kernel.*FillFunctor", "Elementwise", 10),
    (r"(?i)direct_copy_kernel|_to_copy\b", "Elementwise", 10),
    (r"(?i)CatArrayBatchedCopy", "Elementwise", 10),
    (r"(?i)elementwise.*add|FunctorOnSelf_add", "Elementwise", 8),
    (r"(?i)elementwise.*mul", "Elementwise", 8),
    (r"(?i)elementwise", "Elementwise", 5),
    # ---- GEMM rules (priority 5-12) ----
    (r"(?i)_matmul_ogs.*mxfp4", "GEMM", 12),
    (r"(?i)_gemm_a16_w16", "GEMM", 12),
    (r"(?i)nvjet_tst", "GEMM", 12),
    (r"(?i)wvSplitK", "GEMM", 12),
    (r"(?i)Cijk_", "GEMM", 12),
    (r"(?i)kernel_gemm_xdl_cshuffle", "GEMM", 12),
    (r"(?i)_gemm_a\d+w\d+", "GEMM", 10),
    (r"(?i)bmm_MxE4m3.*MxE2m1", "GEMM", 10),
    (r"(?i)bmm_Bfloat16.*MxE2m1", "GEMM", 10),
    (r"(?i)cublasLt|cublas", "GEMM", 10),
    (r"(?i)rocm_aiter_gemm_a8w8", "GEMM", 10),
    (r"(?i)triton_poi_fused.*clamp.*div.*rocm_aiter_gemm", "GEMM", 8),
    (r"(?i)triton_per_fused.*abs.*clamp.*div.*max.*rocm_aiter_gemm", "GEMM", 8),
    (r"(?i)gemm|matmul|bmm", "GEMM", 5),
    # ---- Quantization catch-all (priority 4) ----
    (r"(?i)quant", "Quantization", 4),
    # ---- Triton / generic catch-alls (lowest priority) ----
    (r"(?i)triton_poi_fused", "Elementwise", 3),
    (r"(?i)triton_red_fused", "Elementwise", 3),
    (r"(?i)triton_per_fused", "Elementwise", 3),
    (r"(?i)\bnorm\b", "Normalization", 1),
    (r"(?i)rocprim|hipcub", "Elementwise", 1),
    (r"(?i)cub::DeviceScan|DeviceRadixSort|DeviceReduce", "Elementwise", 1),
]

COMPILED_RULES = [
    (re.compile(pat), ktype, prio) for pat, ktype, prio in KERNEL_TYPE_RULES
]

# ---------------------------------------------------------------------------
# Mapping from kernel_type -> perf_category
#
# Only 15 kernel types remain. Types that serve no anchor or detection
# purpose have been collapsed into their base perf_category name.
# ---------------------------------------------------------------------------

KERNEL_TYPE_TO_PERF_CATEGORY = {
    "Normalization": "Normalization",
    "Attention": "SDPA",
    "Linear Attention": "SDPA",
    "GDN Gating": "SDPA-GDN",
    "KV Cache Store": "Elementwise",
    "Rotary Embedding": "Elementwise",
    "MoE GEMM": "GEMM-MoE",
    "MoE Routing": "Elementwise-MoE",
    "MoE Finalize": "Elementwise-MoE",
    "MoE Quantize": "Elementwise-MoE",
    "Activation (DeepSeek fused)": "Elementwise-MoE",
    "GEMM": "GEMM",
    "Quantization": "Quantization",
    "Elementwise": "Elementwise",
    "MemCpy": "MemCpy",
    "Unknown": "Others",
}


def get_perf_category(kernel_type):
    """Map kernel_type to perf_category."""
    return KERNEL_TYPE_TO_PERF_CATEGORY.get(kernel_type, "Others")


MIN_CONFIDENCE = 5


def classify_kernel(name):
    """Classify a single kernel name.

    Only accepts regex matches at priority >= MIN_CONFIDENCE. Lower-priority
    generic catch-all rules produce "Others" rather than guessing a category.

    Returns:
        (kernel_type, perf_category, confidence)
    """
    matches = []
    for regex, ktype, prio in COMPILED_RULES:
        if regex.search(name):
            matches.append((ktype, prio))
    if not matches:
        return "Unknown", "Others", 0
    matches.sort(key=lambda x: -x[1])
    ktype, prio = matches[0]
    if prio < MIN_CONFIDENCE:
        return "Unknown", "Others", 0
    return ktype, get_perf_category(ktype), prio


def classify_all(extracted_data):
    kernels = extracted_data["kernels"]
    classified = []
    for i, k in enumerate(kernels):
        ktype, perf_cat, confidence = classify_kernel(k["name"])
        classified.append(
            {
                "index": i,
                "name": k["name"],
                "dur": k["dur"],
                "kernel_type": ktype,
                "perf_category": perf_cat,
                "confidence": confidence,
            }
        )
    return classified


def run_assertions(classified):
    errors = []

    unknowns = [c for c in classified if c["kernel_type"] == "Unknown"]
    if unknowns:
        errors.append(
            f"A5.x WARNING: {len(unknowns)} kernels classified as Unknown: "
            + ", ".join(f"#{c['index']}({c['name'][:40]})" for c in unknowns[:5])
        )

    low_conf = [
        c for c in classified if c["confidence"] <= 2 and c["kernel_type"] != "Unknown"
    ]
    if len(low_conf) > len(classified) * 0.2:
        errors.append(
            f"A5.x WARNING: {len(low_conf)} kernels ({100*len(low_conf)/len(classified):.0f}%) "
            "have low classification confidence"
        )

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Classify kernels by type using regex rules"
    )
    parser.add_argument("extracted_json", help="Path to extracted trace data JSON")
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    with open(args.extracted_json) as f:
        extracted = json.load(f)

    classified = classify_all(extracted)
    errors = run_assertions(classified)
    for e in errors:
        print(e, file=sys.stderr)

    from collections import Counter

    type_counts = Counter(c["kernel_type"] for c in classified)
    cat_counts = Counter(c["perf_category"] for c in classified)

    result = {
        "source_file": extracted.get("source_file", "unknown"),
        "total_kernels": len(classified),
        "type_summary": dict(type_counts.most_common()),
        "perf_category_summary": dict(cat_counts.most_common()),
        "classified_kernels": classified,
    }

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(
            f"Classified {len(classified)} kernels into {len(type_counts)} types "
            f"({cat_counts.get('Others', 0)} others)",
            file=sys.stderr,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
