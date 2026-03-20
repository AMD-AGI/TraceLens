#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Helper script to build semantic_labels.json from extracted, pattern, and classified data.
Used by the Semantic Breakdown Agent for the LLM labeling step.
"""

import argparse
import json
import sys

# Per-layer body mapping: offset within 16-kernel layer -> semantic_block
# Based on pattern: period=16, MoE architecture (GPT-OSS with sliding/full attention)
LAYER_BODY_MAP = {
    0: "QKV Projection",
    1: "Rotary Embedding",
    2: "KV Cache Store",
    3: "Attention",
    4: "Output Projection",
    5: "Attention Output Gate",
    6: "Router Gate",
    7: "Router Gate",
    8: "MoE Routing",
    9: "MoE Quantize",
    10: "MoE Routing",
    11: "MoE GateUp+SwiGLU",
    12: "MoE Down Projection",
    13: "MoE Finalize",
    14: "Post-MoE Residual Add",
    15: "Post-Attn Norm",
}

# Preamble labels (indices 0..preamble_size-1)
PREAMBLE_LABELS = [
    "Preamble: Embedding",
    "Preamble: Embedding",
    "Preamble: Embedding",
    "Preamble: Embedding",
    "Preamble: Embedding",
    "Preamble: Embedding",
    "Preamble: Embedding",
    "Preamble: Input Norm",
    "Preamble: Embedding",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("extracted_json", help="Path to extracted.json")
    parser.add_argument("pattern_json", help="Path to pattern.json")
    parser.add_argument("classified_json", help="Path to classified.json")
    parser.add_argument(
        "-o", "--output", required=True, help="Output semantic_labels.json path"
    )
    parser.add_argument(
        "--trace-path", default="", help="Source trace path for metadata"
    )
    args = parser.parse_args()

    with open(args.extracted_json) as f:
        extracted = json.load(f)
    with open(args.pattern_json) as f:
        pattern = json.load(f)
    with open(args.classified_json) as f:
        classified = json.load(f)

    kernels = extracted["kernels"]
    total_time = extracted["metadata"]["total_kernel_time_us"]
    is_graph = extracted["metadata"].get("is_graph_mode", False)
    preamble_size = pattern["preamble_size"]
    epilogue_size = pattern["epilogue_size"]
    period = pattern["best_period"]
    num_layers = pattern["estimated_layers"]

    classified_by_idx = {c["index"]: c for c in classified["classified_kernels"]}

    labeled = []
    for i, k in enumerate(kernels):
        name = k["name"]
        dur = k["dur"]
        ktype = classified_by_idx.get(i, {}).get("kernel_type", "Unknown")

        if i < preamble_size:
            # Preamble
            if i < len(PREAMBLE_LABELS):
                semantic = PREAMBLE_LABELS[i]
            else:
                semantic = "Preamble: Embedding"
            layer = None
        elif i >= len(kernels) - epilogue_size:
            # Epilogue
            epilogue_pos = i - (len(kernels) - epilogue_size)
            if "nvjet" in name and "192" in name:
                semantic = "Epilogue: LM Head"
            elif "rsqrt" in name.lower() or "mean" in name.lower():
                semantic = "Epilogue: Final Norm"
            elif epilogue_pos == 0:
                semantic = "Epilogue: Final Norm"
            else:
                semantic = (
                    "Epilogue: LM Head" if "nvjet" in name else "Epilogue: Final Norm"
                )
            layer = None
        else:
            # Layer body
            body_idx = (i - preamble_size) % period
            layer = (i - preamble_size) // period
            semantic = LAYER_BODY_MAP.get(body_idx, "Post-MoE Residual Add")
            # Override by kernel type for robustness
            if "Attention" in ktype and "fmha" in name:
                semantic = "Attention"
            elif "KV Cache Store" in ktype:
                semantic = "KV Cache Store"
            elif "MoE GateUp+SwiGLU" in ktype or "swiGlu" in name:
                semantic = "MoE GateUp+SwiGLU"
            elif "MoE Down" in ktype or ("bmm" in name and "GateUp" not in semantic):
                if "Down" in str(LAYER_BODY_MAP.get(body_idx, "")) or body_idx == 12:
                    semantic = "MoE Down Projection"
            elif "MoE Finalize" in ktype:
                semantic = "MoE Finalize"
            elif "MoE Routing" in ktype or "routingIndicesCluster" in name:
                semantic = "MoE Routing"
            elif "MoE Quantize" in ktype:
                semantic = "MoE Quantize"
            elif "MoE Pad" in ktype:
                semantic = "MoE Routing"
            elif "RMSNorm" in ktype and body_idx == 15:
                semantic = "Post-Attn Norm"
            elif "SplitK Reduce" in ktype:
                semantic = "Router Gate"
            elif "reshape_and_cache" in name:
                semantic = "KV Cache Store"

        labeled.append(
            {
                "index": i,
                "name": name,
                "dur": dur,
                "kernel_type": ktype,
                "semantic_block": semantic,
                "layer": layer,
            }
        )

    out = {
        "source_file": args.trace_path or extracted.get("source_file", ""),
        "total_kernel_time_us": total_time,
        "model_info": {
            "architecture": "GPT-OSS (MoE)",
            "num_layers": num_layers,
            "ffn_type": "moe",
            "graph_mode": is_graph,
        },
        "labeled_kernels": labeled,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.output} ({len(labeled)} kernels)", file=sys.stderr)


if __name__ == "__main__":
    main()
