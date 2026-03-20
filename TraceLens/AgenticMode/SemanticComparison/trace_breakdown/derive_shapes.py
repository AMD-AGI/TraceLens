#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Derive theoretical FLOPS and bytes for each semantic block using the
model's HuggingFace config.json instead of CPU op shapes.

Reads:
  - semantic_labels.json  (from LLM labeling step)
  - config.json           (HuggingFace model config)
  - num_tokens             (CLI arg, cached to run_config.json)

Writes:
  - derived_shapes.json   (per-block FLOPS, bytes, perf_params)
  - run_config.json       (cached CLI parameters for reuse)

Usage:
    python derive_shapes.py <semantic_labels.json> <config.json> \\
        --num_tokens 1 [-o derived_shapes.json] [--context_length 2048]
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

from category_mappings import (
    derive_block_shapes,
    get_group,
    get_perf_category,
    parse_model_config,
)


def load_or_prompt_run_config(output_dir, cli_num_tokens, cli_context_length):
    """Load cached run_config.json or create from CLI args."""
    cfg_path = os.path.join(output_dir, "run_config.json")
    cached = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cached = json.load(f)

    num_tokens = cli_num_tokens or cached.get("num_tokens")
    context_length = cli_context_length or cached.get("context_length")

    if num_tokens is None:
        print(
            "ERROR: --num_tokens is required on first run (no cached run_config.json found)",
            file=sys.stderr,
        )
        sys.exit(1)

    if context_length is None:
        context_length = num_tokens

    run_cfg = {
        "num_tokens": num_tokens,
        "context_length": context_length,
    }
    with open(cfg_path, "w") as f:
        json.dump(run_cfg, f, indent=2)
    print(
        f"Run config saved to {cfg_path}: T={num_tokens}, ctx={context_length}",
        file=sys.stderr,
    )
    return run_cfg


def derive_all_shapes(labeled_kernels, model_cfg, num_tokens, context_length):
    """Compute shapes for each unique semantic_block in the labeled kernels."""
    seen_blocks = OrderedDict()
    for k in labeled_kernels:
        block = k["semantic_block"]
        layer = k.get("layer")
        if block not in seen_blocks:
            seen_blocks[block] = {"layers": set(), "kernel_count": 0, "total_dur": 0.0}
        seen_blocks[block]["kernel_count"] += 1
        seen_blocks[block]["total_dur"] += k.get("dur", 0)
        if layer is not None:
            seen_blocks[block]["layers"].add(layer)

    results = []
    for block, info in seen_blocks.items():
        layer_idx = min(info["layers"]) if info["layers"] else None
        shapes = derive_block_shapes(
            block,
            model_cfg,
            num_tokens,
            context_length=context_length,
            layer_idx=layer_idx,
        )

        n_layers = len(info["layers"]) if info["layers"] else 1
        per_layer_flops = shapes["flops"] if shapes else None
        per_layer_bytes = shapes["bytes"] if shapes else None

        entry = {
            "semantic_block": block,
            "semantic_group": get_group(block),
            "perf_category": get_perf_category(block),
            "kernel_count": info["kernel_count"],
            "total_dur_us": round(info["total_dur"], 2),
            "layer_count": n_layers,
        }

        if shapes:
            total_flops = per_layer_flops * n_layers
            total_bytes = per_layer_bytes * n_layers
            entry["per_invocation_flops"] = per_layer_flops
            entry["per_invocation_bytes"] = per_layer_bytes
            entry["total_flops"] = total_flops
            entry["total_bytes"] = total_bytes
            entry["perf_params"] = shapes["perf_params"]
        else:
            entry["per_invocation_flops"] = None
            entry["per_invocation_bytes"] = None
            entry["total_flops"] = None
            entry["total_bytes"] = None
            entry["perf_params"] = None

        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Derive theoretical FLOPS/bytes from HuggingFace config"
    )
    parser.add_argument("labels_json", help="Path to semantic_labels.json")
    parser.add_argument("config_json", help="Path to HuggingFace config.json")
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=None,
        help="Number of active tokens (T). Required on first run.",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="KV context length for SDPA (defaults to num_tokens)",
    )
    parser.add_argument(
        "-o", "--output", default="derived_shapes.json", help="Output JSON path"
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    run_cfg = load_or_prompt_run_config(
        output_dir, args.num_tokens, args.context_length
    )

    with open(args.labels_json) as f:
        labels_data = json.load(f)
    with open(args.config_json) as f:
        hf_config = json.load(f)

    model_cfg = parse_model_config(hf_config)
    labeled = labels_data["labeled_kernels"]

    results = derive_all_shapes(
        labeled,
        model_cfg,
        run_cfg["num_tokens"],
        run_cfg["context_length"],
    )

    no_formula = [r for r in results if r["total_flops"] is None]
    if no_formula:
        print(
            f"INFO: {len(no_formula)} blocks have no roofline formula: "
            + ", ".join(r["semantic_block"] for r in no_formula),
            file=sys.stderr,
        )

    output_data = {
        "source_labels": args.labels_json,
        "source_config": args.config_json,
        "num_tokens": run_cfg["num_tokens"],
        "context_length": run_cfg["context_length"],
        "model_info": {
            "hidden_size": model_cfg["hidden_size"],
            "num_attention_heads": model_cfg["num_attention_heads"],
            "num_key_value_heads": model_cfg["num_key_value_heads"],
            "head_dim": model_cfg["head_dim"],
            "num_hidden_layers": model_cfg["num_hidden_layers"],
            "num_experts": model_cfg["num_experts"],
            "num_experts_per_tok": model_cfg["num_experts_per_tok"],
            "dtype": model_cfg["dtype"],
            "bpe": model_cfg["bpe"],
        },
        "blocks": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    total_flops = sum(r["total_flops"] for r in results if r["total_flops"])
    total_bytes = sum(r["total_bytes"] for r in results if r["total_bytes"])
    print(
        f"Wrote {args.output}: {len(results)} blocks, "
        f"total {total_flops/1e9:.2f} GFLOPS, "
        f"total {total_bytes/1e6:.2f} MB data moved",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
