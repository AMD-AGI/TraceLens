###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Reference catalog of functional semantic labels.

Provides guidance to the LLM when assigning functional labels to aligned
semantic blocks.  The LLM should prefer labels from this catalog but is free
to invent new ones for architectures not covered here.

Structure
---------
nn_modules : list[str]
    Coarse-grained model sub-components (e.g. "Self-Attention", "MoE FFN").

cpu_ops : dict[str, list[str]]
    Fine-grained operations grouped by their parent nn_module.

Extensibility
-------------
Add new entries directly to the lists below.  The LLM will pick them up
automatically via the review context JSON.
"""

FUNCTIONAL_LABEL_CATALOG = {
    "nn_modules": [
        "Self-Attention",
        "MoE FFN",
        "Dense FFN",
        "Normalization",
        "Embedding",
        "Output Head",
        "Residual / Skip Connection",
        "Quantization / Dequantization",
        "Communication",
        "Graph Launch Overhead",
    ],
    "cpu_ops": {
        "Self-Attention": [
            "QKV Projection",
            "Q Projection",
            "K Projection",
            "V Projection",
            "Attention Score (QK^T)",
            "Softmax",
            "Attention Output (Score * V)",
            "Output Projection",
            "RoPE / Positional Encoding",
            "KV Cache Store",
            "KV Cache Load",
            "Pre-Attn RMSNorm",
            "Post-Attn RMSNorm",
            "Causal Conv1d",
            "Gated Delta Network (GDN)",
            "Linear Recurrence",
        ],
        "MoE FFN": [
            "MoE TopK Gating / Routing",
            "MoE Expert W1 GEMM (Gate)",
            "MoE Expert W2 GEMM (Down)",
            "MoE Expert W3 GEMM (Up)",
            "MoE SiLU / Activation",
            "MoE Token Permute / Scatter",
            "MoE Token Unpermute / Gather",
            "MoE Finalize / Reduce",
        ],
        "Dense FFN": [
            "FFN Gate Projection",
            "FFN Up Projection",
            "FFN Down Projection",
            "FFN Activation (SiLU/GELU/ReLU)",
        ],
        "Normalization": [
            "RMSNorm",
            "LayerNorm",
            "GroupNorm",
        ],
        "Embedding": [
            "Token Embedding Lookup",
            "Position Embedding",
        ],
        "Output Head": [
            "LM Head Projection",
            "Final LayerNorm / RMSNorm",
        ],
        "Quantization / Dequantization": [
            "Activation Quantization",
            "Weight Dequantization",
            "Per-Token Group Quantization",
        ],
        "Communication": [
            "AllReduce",
            "AllGather",
            "ReduceScatter",
        ],
        "Graph Launch Overhead": [
            "Graph Launch Preamble",
            "Graph Launch Epilogue",
        ],
    },
}
