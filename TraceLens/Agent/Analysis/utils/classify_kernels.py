#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Classify kernels by type and perf_category using extensible regex rules.

Each kernel is classified into a kernel_type (e.g., "GEMM", "RMSNorm"), which
fixes its perf_category (GEMM, SDPA, Normalization, Elementwise, MemCpy, ...) and
the primitives it covers, via KERNEL_TYPES.

Input: extracted trace data JSON
Output: JSON with per-kernel type + perf_category classification

Usage:
    python classify_kernels.py <extracted.json> [-o classified.json]
"""

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The closed vocabulary a kernel type's `covers` is drawn from.
PRIMITIVES = frozenset(
    {
        "gemm",
        "moe_gemm",
        "softmax",
        "topk",
        "route",
        "norm",
        "batchnorm",
        "quant",
        "rope",
        "kv_store",
        "collective",
        "act",
        "mul",
        "add",
        "pointwise",
        "reduce",
        "copy",
        "library_collective",
    }
)

# Inductor kernel names start with this prefix; whether a kernel is compiled is a
# per-kernel fact, because op words in the name type many of them as op types.
COMPILED_PREFIX = r"triton_(poi|red|per)_"

# The tile-transform kernels the Case-A comparative gate skips.
LAYOUT_TRANSFORM = "LayoutTransform"

# The type of a kernel no rule matches.
UNKNOWN = "Unknown"

# The type of an Inductor-generated kernel (see COMPILED_PREFIX).
COMPILED = "Compiled"

# Regex rules: (pattern, kernel_type, priority). The highest priority wins; on a
# tie the earlier rule wins. Priority is one of the bands 1, 5, 10, 15, 20: the band
# is a rule's specificity, and within a band the more specific (fused) rules are
# listed first.
KERNEL_TYPE_RULES = [
    # ---- Band 20: MemCpy and MemSet ----
    (r"(?i)memcpy", "MemCpy", 20),
    (r"(?i)combined_routing_memset", "MoE Routing Memset", 20),
    (r"(?i)memset", "MemSet", 20),
    # ---- Band 20: fused MoE, norm and KV-store kernels ----
    (r"(?i)swiglu|swiGlu", "MoE GateUp+SwiGLU GEMM", 20),
    (
        r"(?i)fmoe.*g1u1.*silu|fmoe.*silu|mfma_moe\d+_silu_mul",
        "MoE GEMM+SiLU (aiter)",
        20,
    ),
    (
        r"(?i)add_rmsnorm_quant|Rmsnorm2dFusedAddEnumE[1-9].*Rmsnorm2dFusedQuantEnumE[1-9]",
        "Add+RMSNorm+Quant",
        20,
    ),
    (
        r"(?i)fuse_qk_rope_concat_and_cache_mla|fused_qk_rope_reshape_and_cache",
        "RoPE+KV Store",
        20,
    ),
    (r"(?i)kernel_moe_gemm|MoeGemmBlockScale|mfma_moe", "MoE GEMM (CK)", 20),
    # Templated ck_tile Rmsnorm2dFwd spells the fused-add and fused-quant enums
    # even when they are 0 (unfused), so only a non-zero enum means fused.
    (r"(?i)add_rmsnorm|Rmsnorm2dFusedAddEnumE[1-9]", "Add+RMSNorm", 20),
    (
        r"(?i)fused_rms_\w*quant|rmsnorm_quant|Rmsnorm2dFusedQuantEnumE[1-9]",
        "RMSNorm+Quant",
        20,
    ),
    (r"(?i)local_device_load_rmsnorm", "AllReduce+RMSNorm", 20),
    # ---- Band 20: Normalization ----
    (r"(?i)Rmsnorm|rmsnorm2d", "RMSNorm", 20),
    (r"(?i)layer_norm|layernorm", "LayerNorm", 20),
    (r"(?i)batch_norm|batchnorm", "BatchNorm", 20),
    (r"(?i)l2norm_fwd|l2norm_kernel", "L2Norm", 20),
    (
        r"(?i)fused__to_copy_add_mean_moe_forward_shared.*pow.*rsqrt",
        "RMSNorm (shared expert + norm)",
        20,
    ),
    (r"(?i)rsqrt.*mean.*pow|mean.*mul.*pow.*rsqrt", "RMSNorm (fused triton)", 20),
    (r"(?i)_to_copy_add.*mean.*rsqrt", "RMSNorm (fused triton)", 20),
    # ---- Band 20: Attention and KV cache ----
    (r"(?i)attention_[23]d|unified_attention", "Attention (AMD unified)", 20),
    (r"(?i)fmha[Ss]m\d+|fmhaSm100f", "Attention (NVIDIA fmha)", 20),
    (r"(?i)paged_attention|PagedAttention", "Attention (paged)", 20),
    (
        r"(?i)flash_attn|flash_fwd|attn_fwd|fmha|fort_native_sdpa",
        "Attention (flash)",
        20,
    ),
    (r"(?i)mla_a8w8", "Attention (MLA)", 20),
    (r"(?i)reduce_segments", "Attention Reduce", 20),
    (r"(?i)reshape_and_cache_flash|reshape_and_cache", "KV Cache Store", 20),
    # ---- Band 20: Linear attention ----
    (r"(?i)chunk_gated_delta_rule_fwd", "Linear Attention (chunk delta)", 20),
    (r"(?i)chunk_fwd_kernel_o", "Linear Attention (chunk fwd)", 20),
    (r"(?i)chunk_scaled_dot_kkt", "Linear Attention (chunk kkt)", 20),
    (r"(?i)chunk_local_cumsum", "Linear Attention (chunk cumsum)", 20),
    (r"(?i)fused_recurrent_gated_delta_rule", "Linear Attention (recurrent delta)", 20),
    (r"(?i)recompute_w_u_fwd", "Linear Attention (recompute wu)", 20),
    (r"(?i)merge_\d+x\d+_to_\d+x\d+_inverse", "Linear Attention (merge inverse)", 20),
    (r"(?i)_causal_conv1d_fwd|_causal_conv1d_update", "Causal Conv1D", 20),
    (r"(?i)fused_gdn_gating", "GDN Gating", 20),
    # ---- Band 20: MoE ----
    (r"(?i)finalize.*scatter|_finalize_matmul|finalizeKernel", "MoE Finalize", 20),
    (r"(?i)topk_forward", "MoE TopK", 20),
    (r"(?i)topkGatingSoftmax", "MoE TopK Gating", 20),
    (r"(?i)grouped_topk", "TopK", 20),
    (r"(?i)bitmatrix|_sum_bitmatrix", "MoE BitMatrix", 20),
    (r"(?i)combined_routing_compute", "MoE Routing Compute", 20),
    (r"(?i)routingIndicesCluster|routingRenormalize", "MoE Routing (fused)", 20),
    (r"(?i)MoeSorting", "MoE Sorting", 20),
    (r"(?i)constant_pad.*moe|moe_forward", "MoE Pad", 20),
    (r"(?i)quantize_with_block_size", "MoE Quantize", 20),
    # ---- Band 20: Activation, communication and gather ----
    (r"(?i)act_and_mul_kernel|silu_and_mul|silu_mul|SiluAndMul", "Act+Mul", 20),
    (r"(?i)cross_device|allgather_lastdim", "Collective", 20),
    (r"(?i)nccl|rccl|multimem_all_reduce", "Library Collective", 20),
    (r"(?i)gather_kernel|vectorized_gather", "Elementwise", 20),
    # ---- Band 15: fused triton Normalization, RoPE, quantization, activation ----
    (r"(?i)fused__to_copy_abs.*mean.*pow.*rsqrt", "RMSNorm (fused triton)", 15),
    (r"(?i)fused__to_copy_add.*mean.*pow.*rsqrt", "RMSNorm (fused triton)", 15),
    (r"(?i)fused_add_copy__mean.*pow", "RMSNorm (fused triton residual)", 15),
    (
        r"(?i)fused__to_copy_add_copy__mean.*pow",
        "RMSNorm (fused triton residual)",
        15,
    ),
    (
        r"(?i)fused__to_copy_add_mean.*mul.*pow.*rsqrt",
        "RMSNorm (fused triton)",
        15,
    ),
    (r"(?i)rotary|rope", "Rotary Embedding", 15),
    (
        r"(?i)per_token_group_quant|dynamic_per_group_scaled_quant",
        "Quantize (per-group)",
        15,
    ),
    (r"(?i)activationDeepSeekKernel", "Activation (DeepSeek fused)", 15),
    (r"(?i)fused_mul_sigmoid", "Activation (SiLU)", 15),
    # ---- Band 15: embedding, sampling, activation, GEMM split-K reduce ----
    (r"(?i)embedding", "Elementwise", 15),
    (r"(?i)top_k|top_p|sampling|topk_topp", "Elementwise", 15),
    (r"(?i)\bsilu(_kernel)?\b|swish", "Activation (SiLU)", 15),
    (r"(?i)\bgelu\b", "Activation (GELU)", 15),
    (r"(?i)splitKreduce|splitkreduce", "GEMM", 15),
    # ---- Band 15: copies, softmax and GEMM libraries ----
    (r"(?i)__amd_rocclr_copyBuffer", "Elementwise", 15),
    (r"(?i)memcpy32_post", "Elementwise", 15),
    (r"(?i)copy_page_indices", "Elementwise", 15),
    (r"(?i)softmax", "Softmax", 15),
    (r"(?i)_matmul_ogs.*mxfp4", "GEMM", 15),
    (r"(?i)_gemm_a16_w16", "GEMM", 15),
    (r"(?i)nvjet_tst", "GEMM", 15),
    (r"(?i)wvSplitK", "GEMM", 15),
    (r"(?i)Cijk_", "GEMM", 15),
    (r"(?i)kernel_gemm_xdl_cshuffle", "GEMM", 15),
    # ---- Band 10: elementwise and layout, then GEMM libraries ----
    (r"(?i)vectorized_elementwise_kernel.*FillFunctor", "Elementwise", 10),
    (r"(?i)direct_copy_kernel|_to_copy\b", "Elementwise", 10),
    (r"(?i)CatArrayBatchedCopy", "Elementwise", 10),
    (
        r"(?i)batched_transpose|nchw2nhwc|nhwc2nchw|transpose_nchw|transpose_cnhw",
        LAYOUT_TRANSFORM,
        10,
    ),
    (r"(?i)_gemm_a\d+w\d+", "GEMM", 10),
    (r"(?i)bmm_MxE4m3.*MxE2m1", "GEMM", 10),
    (r"(?i)bmm_Bfloat16.*MxE2m1", "GEMM", 10),
    (r"(?i)cublasLt|cublas", "GEMM", 10),
    (r"(?i)rocm_aiter_gemm_a8w8", "GEMM", 10),
    # ---- Band 10: reduce, add and mul, then fused compiled GEMM ----
    (r"(?i)\breduce_kernel", "Reduce", 10),
    (r"(?i)CUDAFunctor(OnSelf|OnOther)?_add", "Add", 10),
    (r"(?i)MulFunctor", "Mul", 10),
    (r"(?i)triton_poi_fused.*clamp.*div.*rocm_aiter_gemm", "GEMM", 10),
    (r"(?i)triton_per_fused.*abs.*clamp.*div.*max.*rocm_aiter_gemm", "GEMM", 10),
    # ---- Band 5: catch-alls, convolution first so it wins over the GEMM one ----
    (
        r"(?i)fprop|wgrad|grouped_conv|naive_conv|igemm_(fwd|bwd|wrw)_gtc",
        "Convolution",
        5,
    ),
    (r"(?i)elementwise", "Elementwise", 5),
    (r"(?i)gemm|matmul|bmm", "GEMM", 5),
    (rf"(?i){COMPILED_PREFIX}fused", COMPILED, 5),
]

# kernel_type -> (perf_category, covers). `covers` lists the primitives the kernel
# performs in unfused execution order; a type covering two or more is a fused
# kernel. Linear-attention kernels and Unknown stay outside the vocabulary.
KERNEL_TYPES = {
    "RMSNorm": ("Normalization", ("norm",)),
    "Add+RMSNorm": ("Normalization", ("add", "norm")),
    "RMSNorm+Quant": ("Normalization", ("norm", "quant")),
    "Add+RMSNorm+Quant": ("Normalization", ("add", "norm", "quant")),
    "AllReduce+RMSNorm": ("Normalization", ("collective", "norm")),
    "LayerNorm": ("Normalization", ("norm",)),
    "BatchNorm": ("Normalization", ("batchnorm",)),
    "L2Norm": ("Normalization", ("norm",)),
    "RMSNorm (fused triton)": ("Normalization", ("add", "norm")),
    "RMSNorm (fused triton residual)": ("Normalization", ("add", "norm")),
    "RMSNorm (shared expert + norm)": ("Normalization", ("add", "norm")),
    "Attention (AMD unified)": ("SDPA", ("gemm", "softmax", "gemm")),
    "Attention (NVIDIA fmha)": ("SDPA", ("gemm", "softmax", "gemm")),
    "Attention (paged)": ("SDPA", ("gemm", "softmax", "gemm")),
    "Attention (flash)": ("SDPA", ("gemm", "softmax", "gemm")),
    "Attention (MLA)": ("SDPA", ("gemm", "softmax", "gemm")),
    "Attention Reduce": ("SDPA", ("reduce",)),
    "Linear Attention (chunk delta)": ("SDPA", ()),
    "Linear Attention (chunk fwd)": ("SDPA", ()),
    "Linear Attention (chunk kkt)": ("SDPA", ()),
    "Linear Attention (chunk cumsum)": ("SDPA", ()),
    "Linear Attention (recurrent delta)": ("SDPA", ()),
    "Linear Attention (recompute wu)": ("SDPA", ()),
    "Linear Attention (merge inverse)": ("SDPA", ()),
    "Causal Conv1D": ("SDPA", ("pointwise",)),
    "GDN Gating": ("SDPA-GDN", ("pointwise",)),
    "KV Cache Store": ("Elementwise", ("kv_store",)),
    "Rotary Embedding": ("Elementwise", ("rope",)),
    "RoPE+KV Store": ("Elementwise", ("rope", "kv_store")),
    "MoE GateUp+SwiGLU GEMM": ("GEMM-MoE", ("moe_gemm", "act", "mul")),
    "MoE GEMM+SiLU (aiter)": ("GEMM-MoE", ("moe_gemm", "act", "mul")),
    "MoE GEMM (CK)": ("GEMM-MoE", ("moe_gemm",)),
    "MoE Finalize": ("Elementwise-MoE", ("route",)),
    "MoE TopK": ("Elementwise-MoE", ("topk",)),
    "MoE TopK Gating": ("Elementwise-MoE", ("softmax", "topk")),
    "TopK": ("Elementwise", ("topk",)),
    "MoE BitMatrix": ("Elementwise-MoE", ("route",)),
    "MoE Routing Memset": ("Elementwise-MoE", ("route",)),
    "MoE Routing Compute": ("Elementwise-MoE", ("route",)),
    "MoE Routing (fused)": ("Elementwise-MoE", ("topk", "softmax", "route")),
    "MoE Sorting": ("Elementwise-MoE", ("route",)),
    "MoE Pad": ("Elementwise-MoE", ("pointwise",)),
    "MoE Quantize": ("Elementwise-MoE", ("quant",)),
    "Quantize (per-group)": ("Quantization", ("quant",)),
    "Act+Mul": ("Elementwise", ("act", "mul")),
    "Activation (DeepSeek fused)": ("Elementwise-MoE", ("act", "mul")),
    "Activation (SiLU)": ("Elementwise", ("act",)),
    "Activation (GELU)": ("Elementwise", ("act",)),
    "Collective": ("Communication", ("collective",)),
    "Library Collective": ("Communication", ("library_collective",)),
    "MemCpy": ("MemCpy", ("copy",)),
    "MemSet": ("MemCpy", ("copy",)),
    "Softmax": ("Elementwise", ("softmax",)),
    "Reduce": ("Elementwise", ("reduce",)),
    "Add": ("Elementwise", ("add",)),
    "Mul": ("Elementwise", ("mul",)),
    "Elementwise": ("Elementwise", ("pointwise",)),
    LAYOUT_TRANSFORM: ("Elementwise", ("pointwise",)),
    "GEMM": ("GEMM", ("gemm",)),
    "Convolution": ("Convolution", ("gemm",)),
    # Compiler-generated kernels fuse an unknown pointwise/reduce mix.
    COMPILED: ("Elementwise", ("pointwise", "reduce")),
    UNKNOWN: ("Others", ()),
}


@dataclass(frozen=True)
class KernelClass:
    """Facts about one kernel type."""

    type: str
    perf_category: str
    covers: tuple[str, ...]


class KernelTaxonomy:
    """Kernel-name regex rules plus the facts of every kernel type."""

    _COMPILED = re.compile(COMPILED_PREFIX)

    def __init__(self, rules, types):
        self._rules = [(re.compile(pat), ktype, prio) for pat, ktype, prio in rules]
        self._types = {
            ktype: KernelClass(ktype, perf_cat, covers)
            for ktype, (perf_cat, covers) in types.items()
        }

    def _match(self, name: str) -> tuple[str, int]:
        """Return the (kernel_type, priority) of the winning rule."""
        best_type, best_prio = UNKNOWN, 0
        for regex, ktype, prio in self._rules:
            if prio > best_prio and regex.search(name):
                best_type, best_prio = ktype, prio
        return best_type, best_prio

    def classify(self, name: str) -> KernelClass:
        return self._types[self._match(name)[0]]

    def is_compiled(self, name: str) -> bool:
        return self._COMPILED.match(name) is not None

    def by_type(self, ktype: str) -> KernelClass:
        return self._types[ktype]

    def types(self) -> Iterable[KernelClass]:
        return self._types.values()


TAXONOMY = KernelTaxonomy(KERNEL_TYPE_RULES, KERNEL_TYPES)


def classify_kernel(name):
    """Classify a single kernel name.

    Returns:
        (kernel_type, perf_category, priority)
    """
    ktype, priority = TAXONOMY._match(name)
    return ktype, TAXONOMY.by_type(ktype).perf_category, priority


def classify_all(extracted_data):
    kernels = extracted_data["kernels"]
    classified = []
    for i, k in enumerate(kernels):
        ktype, perf_cat, _ = classify_kernel(k["name"])
        classified.append(
            {
                "index": i,
                "name": k["name"],
                "dur": k["dur"],
                "kernel_type": ktype,
                "perf_category": perf_cat,
            }
        )
    return classified


def run_assertions(classified):
    errors = []

    unknowns = [c for c in classified if c["kernel_type"] == UNKNOWN]
    if unknowns:
        errors.append(
            f"A5.x WARNING: {len(unknowns)} kernels classified as Unknown: "
            + ", ".join(f"#{c['index']}({c['name'][:40]})" for c in unknowns[:5])
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
