#!/usr/bin/env python3
"""
Step 5 (scriptable part): Classify kernels by type using regex rules.

This does NOT assign full semantic roles (e.g., "QKV Projection" vs "Output Projection").
It classifies each kernel into a TYPE (e.g., "GEMM", "RMSNorm", "Attention"), which the
LLM then uses to assign positional semantic roles.

Input: extracted trace data JSON
Output: JSON with per-kernel type classification

Usage:
    python classify_kernels.py <extracted.json> [-o classified.json]
"""
import argparse
import json
import re
import sys

KERNEL_TYPE_RULES = [
    # (pattern, kernel_type, priority) -- higher priority wins on conflict
    # Normalization
    (r"(?i)Rmsnorm|rmsnorm2d", "RMSNorm", 20),
    (r"(?i)layer_norm|layernorm", "LayerNorm", 20),
    (r"(?i)rsqrt.*mean.*pow|mean.*mul.*pow.*rsqrt|_to_copy_add.*mean.*rsqrt", "RMSNorm (fused triton)", 20),
    # Attention
    (r"(?i)attention_[23]d|unified_attention", "Attention (AMD unified)", 20),
    (r"(?i)fmha[Ss]m\d+|fmhaSm100f", "Attention (NVIDIA fmha)", 20),
    (r"(?i)paged_attention|PagedAttention", "Attention (paged)", 20),
    (r"(?i)flash_attn|flash_fwd", "Attention (flash)", 20),
    (r"(?i)reduce_segments", "Attention Reduce", 20),
    (r"(?i)reshape_and_cache_flash|reshape_and_cache", "KV Cache Store", 20),
    # MoE-specific
    (r"(?i)swiglu|swiGlu", "MoE GateUp+SwiGLU GEMM", 25),
    (r"(?i)finalize.*scatter|_finalize_matmul|finalizeKernel", "MoE Finalize", 20),
    (r"(?i)topk_forward", "MoE TopK", 20),
    (r"(?i)bitmatrix|_sum_bitmatrix", "MoE BitMatrix", 20),
    (r"(?i)combined_routing_memset", "MoE Routing Memset", 20),
    (r"(?i)combined_routing_compute", "MoE Routing Compute", 20),
    (r"(?i)routingIndicesCluster|routingRenormalize", "MoE Routing (fused)", 20),
    (r"(?i)constant_pad.*moe|moe_forward", "MoE Pad", 20),
    (r"(?i)quantize_with_block_size", "MoE Quantize", 20),
    # Activation functions
    (r"(?i)silu_and_mul|silu_mul|SiluAndMul", "Activation (SiLU+Mul)", 18),
    (r"(?i)\bsilu\b|swish", "Activation (SiLU)", 15),
    (r"(?i)\bgelu\b", "Activation (GELU)", 15),
    # Memory / data movement
    (r"(?i)embedding", "Embedding", 20),
    (r"(?i)gather_kernel|vectorized_gather", "Gather", 20),
    (r"(?i)rotary|rope", "Rotary Embedding", 18),
    (r"(?i)splitKreduce|splitkreduce", "SplitK Reduce", 15),
    (r"(?i)direct_copy_kernel|_to_copy\b", "Copy/Cast", 10),
    (r"(?i)CUDAFunctorOnSelf_add<int>", "Position Offset (int add)", 15),
    # Elementwise
    (r"(?i)elementwise.*add|FunctorOnSelf_add", "Elementwise Add", 8),
    (r"(?i)elementwise.*mul", "Elementwise Mul", 8),
    (r"(?i)elementwise", "Elementwise (generic)", 5),
    # Sampling / top-k / top-p
    (r"(?i)top_k|top_p|sampling|topk_topp", "Sampling", 15),
    # GEMM rules (lower priority, since specific rules above may also match)
    (r"(?i)_matmul_ogs.*mxfp4", "GEMM (aiter mxfp4)", 10),
    (r"(?i)_gemm_a16_w16", "GEMM (triton a16w16)", 10),
    (r"(?i)nvjet_tst", "GEMM (nvjet)", 10),
    (r"(?i)wvSplitK", "GEMM (rocBLAS wvSplitK)", 10),
    (r"(?i)bmm_MxE4m3.*MxE2m1", "GEMM (TRT-LLM bmm MX)", 10),
    (r"(?i)bmm_Bfloat16.*MxE2m1", "GEMM (TRT-LLM bmm bf16)", 10),
    (r"(?i)cublasLt|cublas", "GEMM (cuBLAS)", 10),
    (r"(?i)gemm|matmul|bmm_", "GEMM (generic)", 5),
    # Triton fallbacks
    (r"(?i)triton_poi_fused", "Triton Pointwise (unclassified)", 3),
    (r"(?i)triton_red_fused", "Triton Reduction (unclassified)", 3),
    (r"(?i)triton_", "Triton (unclassified)", 2),
]

COMPILED_RULES = [(re.compile(pat), ktype, prio) for pat, ktype, prio in KERNEL_TYPE_RULES]


def classify_kernel(name):
    """Classify a single kernel name. Returns (kernel_type, confidence)."""
    matches = []
    for regex, ktype, prio in COMPILED_RULES:
        if regex.search(name):
            matches.append((ktype, prio))
    if not matches:
        return "Unknown", 0
    matches.sort(key=lambda x: -x[1])
    return matches[0][0], matches[0][1]


def extract_gemm_details(name):
    """Extract tile sizes, layout, and other details from GEMM kernel names."""
    details = {}

    m = re.search(r"BLOCK_SIZE_M_(\d+)", name)
    if m:
        details["tile_M"] = int(m.group(1))
    m = re.search(r"BLOCK_SIZE_N_(\d+)", name)
    if m:
        details["tile_N"] = int(m.group(1))
    m = re.search(r"BLOCK_SIZE_K_(\d+)", name)
    if m:
        details["tile_K"] = int(m.group(1))

    if "splitK" in name or "KSPLIT" in name:
        details["is_splitK"] = True
    if "bias" in name.lower() or "ADD_BIAS_1" in name:
        details["has_bias"] = True
    if "swiglu" in name.lower() or "swiGlu" in name:
        details["has_swiglu"] = True

    for layout in ["TNT", "TNN", "NNT", "NNN"]:
        if layout in name:
            details["layout"] = layout
            break

    return details if details else None


def classify_all(extracted_data):
    kernels = extracted_data["kernels"]
    classified = []
    for i, k in enumerate(kernels):
        ktype, confidence = classify_kernel(k["name"])
        entry = {
            "index": i,
            "name": k["name"],
            "dur": k["dur"],
            "kernel_type": ktype,
            "confidence": confidence,
        }
        if "GEMM" in ktype:
            details = extract_gemm_details(k["name"])
            if details:
                entry["gemm_details"] = details
        classified.append(entry)
    return classified


def run_assertions(classified):
    errors = []

    unknowns = [c for c in classified if c["kernel_type"] == "Unknown"]
    if unknowns:
        errors.append(
            f"A5.x WARNING: {len(unknowns)} kernels classified as Unknown: "
            + ", ".join(f"#{c['index']}({c['name'][:40]})" for c in unknowns[:5])
        )

    low_conf = [c for c in classified if c["confidence"] <= 2 and c["kernel_type"] != "Unknown"]
    if len(low_conf) > len(classified) * 0.2:
        errors.append(
            f"A5.x WARNING: {len(low_conf)} kernels ({100*len(low_conf)/len(classified):.0f}%) "
            "have low classification confidence"
        )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Classify kernels by type using regex rules")
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

    result = {
        "source_file": extracted.get("source_file", "unknown"),
        "total_kernels": len(classified),
        "type_summary": dict(type_counts.most_common()),
        "classified_kernels": classified,
    }

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Classified {len(classified)} kernels into {len(type_counts)} types", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
