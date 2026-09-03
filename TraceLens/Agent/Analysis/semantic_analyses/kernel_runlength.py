#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run-length helpers over per-kernel ``semantic_block`` sequences.

Shared by the kernel-name coherence (second) pass. Operates purely on the
``semantic_block`` field of ``semantic_labels.json`` -- no ``nn_module`` /
tree_context is required, so it works on graph-mode traces.

Terminology:
    sequence   -- kernel-order list of semantic_block values (one per kernel).
    condensed  -- the sequence with consecutive duplicates collapsed
                  (``A A B D D`` -> ``A B D``); one symbol per contiguous run.
    shared     -- a symbol present in the condensed sets of BOTH traces.
    one-sided  -- a symbol present in only one trace's condensed set.
"""

import json

from _helpers import load_json


def load_sequence(labels_path):
    """Return the kernel-order list of semantic_block values for a labels file."""
    data = load_json(labels_path)
    return [k.get("semantic_block", "") for k in data.get("labeled_kernels", [])]


def collapse_consecutive(seq):
    """Collapse consecutive duplicates: ``A A B D D`` -> ``A B D``."""
    if not seq:
        return []
    out = [seq[0]]
    for s in seq[1:]:
        if s != out[-1]:
            out.append(s)
    return out


def run_index_per_kernel(seq):
    """Map each kernel position to the index of its run in ``collapse_consecutive(seq)``."""
    if not seq:
        return []
    out = []
    run_idx = 0
    prev = None
    for s in seq:
        if prev is not None and s != prev:
            run_idx += 1
        out.append(run_idx)
        prev = s
    return out


def shared_neighbor_windows_skip_non_shared(condensed, center_j, shared, radius):
    """Nearest ``radius`` *shared* symbols on each side of ``condensed[center_j]``.

    Walks outward from the center, skipping any symbol not in ``shared``
    (including other one-sided symbols), so the windows are always expressed
    in cross-trace-stable anchors. Nearest-to-center first on each side.
    Returns ``(left_window, right_window)`` as lists.
    """
    left = []
    i = center_j - 1
    while i >= 0 and len(left) < radius:
        if condensed[i] in shared:
            left.append(condensed[i])
        i -= 1

    right = []
    i = center_j + 1
    while i < len(condensed) and len(right) < radius:
        if condensed[i] in shared:
            right.append(condensed[i])
        i += 1

    return left, right
