###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared helper functions for the semantic_analyses scripts.

Merged from the former trace_breakdown/_helpers.py (build_rle, detect_period)
and trace_comparison/_helpers.py (load_labels).
"""

import json


def build_rle(kernel_indices, cls_by_idx):
    """Run-length encode kernel indices by perf_category.

    Returns list of (perf_category, count, [kernel_indices], [kernel_types]).
    """
    if not kernel_indices:
        return []

    def _cls(idx):
        c = cls_by_idx.get(idx, {})
        return c.get("perf_category", "Others"), c.get("kernel_type", "Unknown")

    first_cat, first_kt = _cls(kernel_indices[0])
    groups = []
    cur_cat = first_cat
    cur_indices = [kernel_indices[0]]
    cur_types = [first_kt]

    for idx in kernel_indices[1:]:
        cat, kt = _cls(idx)
        if cat == cur_cat:
            cur_indices.append(idx)
            cur_types.append(kt)
        else:
            groups.append((cur_cat, len(cur_indices), cur_indices[:], cur_types[:]))
            cur_cat = cat
            cur_indices = [idx]
            cur_types = [kt]

    groups.append((cur_cat, len(cur_indices), cur_indices[:], cur_types[:]))
    return groups


def detect_period(rle_groups):
    """Find the shortest repeating period in an RLE group sequence.

    Returns the period length (number of RLE groups per super-cycle).
    """
    cats = [g[0] for g in rle_groups]
    n = len(cats)
    if n < 6:
        return n

    for p in range(3, n // 2 + 1):
        prefix = cats[:p]
        matches = sum(1 for i in range(p, n) if cats[i] == prefix[i % p])
        total = n - p
        if total > 0 and matches / total > 0.85 and total >= 2 * p:
            return p

    return n


def load_labels(path):
    """Load a semantic_labels.json file."""
    with open(path) as f:
        return json.load(f)
