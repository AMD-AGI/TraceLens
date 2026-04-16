#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Apply LLM-suggested category corrections and functional labels to
semantic_labels.json files.

Accepts two input formats:

1. Legacy flat array -- a JSON list of corrections:
   [{"block": "GEMM_4", "new_perf_category": "GEMM-MoE", "reason": "..."}]

2. Combined object with corrections and functional labels:
   {
     "corrections": [
       {"block": "GEMM_4", "new_perf_category": "GEMM-MoE", "reason": "..."}
     ],
     "functional_labels": {
       "GEMM_0": {"semantic_block": "QKV Projection", "perf_category": "GEMM",
                   "nn_module": "Self-Attention"},
       ...
     }
   }

When functional_labels are provided, each kernel's semantic_block is replaced
with the functional label and perf_category / nn_module are set explicitly.
Blocks not covered by functional_labels keep their original generic label.

Usage:
    python apply_category_corrections.py \
        --corrections corrections.json \
        --labels-a <dir_a>/semantic_labels.json \
        --labels-b <dir_b>/semantic_labels.json
"""
import argparse
import json
import sys


def parse_input(raw):
    """Parse the input JSON, supporting both legacy and combined formats."""
    if isinstance(raw, list):
        return raw, {}
    if isinstance(raw, dict):
        corrections = raw.get("corrections", [])
        functional_labels = raw.get("functional_labels", {})
        return corrections, functional_labels
    return [], {}


def apply_to_labels(labels_path, corrections, functional_labels):
    """Apply corrections and functional labels to a semantic_labels.json file.

    Returns (rename_count, label_count).
    """
    with open(labels_path) as f:
        data = json.load(f)

    correction_map = {c["block"]: c["new_perf_category"] for c in corrections}

    rename_count = 0
    label_count = 0

    for kernel in data["labeled_kernels"]:
        old_block = kernel["semantic_block"]

        if old_block in correction_map:
            kernel["perf_category"] = correction_map[old_block]
            rename_count += 1

        if old_block in functional_labels:
            fl = functional_labels[old_block]
            kernel["semantic_block"] = fl["semantic_block"]
            if "perf_category" in fl:
                kernel["perf_category"] = fl["perf_category"]
            if "nn_module" in fl:
                kernel["nn_module"] = fl["nn_module"]
            label_count += 1

    with open(labels_path, "w") as f:
        json.dump(data, f, indent=2)

    return rename_count, label_count


def main():
    parser = argparse.ArgumentParser(
        description="Apply LLM category corrections and functional labels to semantic_labels.json files"
    )
    parser.add_argument(
        "--corrections",
        required=True,
        help="Path to corrections JSON file (flat array or combined object)",
    )
    parser.add_argument(
        "--labels-a", required=True, help="Path to trace A semantic_labels.json"
    )
    parser.add_argument(
        "--labels-b", required=True, help="Path to trace B semantic_labels.json"
    )
    args = parser.parse_args()

    with open(args.corrections) as f:
        raw = json.load(f)

    corrections, functional_labels = parse_input(raw)

    if not corrections and not functional_labels:
        print("No corrections or functional labels to apply.", file=sys.stderr)
        return

    if corrections:
        print(f"Category corrections: {len(corrections)}", file=sys.stderr)
        for c in corrections:
            print(
                f"  {c['block']} -> perf_category={c['new_perf_category']}  "
                f"({c.get('reason', '')})",
                file=sys.stderr,
            )

    if functional_labels:
        print(f"Functional labels: {len(functional_labels)} blocks", file=sys.stderr)

    ren_a, lab_a = apply_to_labels(args.labels_a, corrections, functional_labels)
    ren_b, lab_b = apply_to_labels(args.labels_b, corrections, functional_labels)

    if corrections:
        print(
            f"Category corrections applied: {ren_a} kernels in trace A, "
            f"{ren_b} in trace B",
            file=sys.stderr,
        )
    if functional_labels:
        print(
            f"Functional labels applied: {lab_a} kernels in trace A, "
            f"{lab_b} in trace B",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
