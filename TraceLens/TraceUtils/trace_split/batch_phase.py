###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase inference for LLM-inference traces without parseable annotations.

Recovers per-iteration batch size from tensor shapes and classifies each
iteration as prefill-bearing or decode from those batch sizes -- the
shape/batch-size counterpart to the parsed-annotation phase logic in
``annotation_utils``.
"""

from bisect import bisect_left, bisect_right

from ...util import most_common_first_dim

# Smallest multiplicative gap between unique batch sizes that splits the
# distribution into decode vs prefill-bearing when max_num_seq is unknown.
_GAP_SPLIT_MIN_RATIO = 2.0


def infer_batch_sizes_from_shapes(
    roots: list[dict],
    cpu_event_index: tuple[list[dict], list[float]],
    root_tiles: dict | None = None,
) -> list[int | None]:
    """Derive batch size per iteration from the most common first dim of cpu_op Input Dims.

    Performs a lightweight scan (no full extraction) using the pre-built
    cpu_event_index for bisect-based windowing.  Returns one value per root,
    or ``None`` when no cpu_op with ``Input Dims`` falls in the window.
    """
    cpu_events, cpu_starts = cpu_event_index
    batch_sizes: list[int | None] = []

    for root in roots:
        if root_tiles is not None:
            key = (root.get("pid"), root.get("tid"), root.get("ts", 0))
            win_ts, win_end = root_tiles.get(
                key, (root["ts"], root["ts"] + root["dur"])
            )
        else:
            win_ts = root["ts"]
            win_end = win_ts + root["dur"]

        lo = bisect_left(cpu_starts, win_ts)
        hi = bisect_right(cpu_starts, win_end)
        win_dur = win_end - win_ts

        # Filter to events within the window, excluding enclosing spans
        window_events = [
            e
            for e in cpu_events[lo:hi]
            if win_ts <= e["ts"] < win_end and e["dur"] <= win_dur
        ]
        batch_sizes.append(most_common_first_dim(window_events, exclude_mem_ops=True))

    return batch_sizes


def classify_phases_from_batch_sizes(
    batch_sizes: list[int | None],
    max_num_seq: int | None = None,
) -> list[str]:
    """Classify each iteration as ``'decode'`` or ``'prefill_bearing'``.

    When *max_num_seq* is provided, any batch size above that value is
    ``'prefill_bearing'``.  Otherwise, a heuristic finds the largest
    multiplicative gap among the unique batch sizes (after dropping the
    two smallest to ignore ramp-up noise) and splits there if the gap
    exceeds 10x.
    """
    valid = [b for b in batch_sizes if b is not None]
    if not valid:
        return ["decode"] * len(batch_sizes)

    if max_num_seq is not None:
        threshold = max_num_seq
    else:
        unique = sorted(set(b for b in valid if b > 0))
        threshold = None
        if len(unique) >= 4:
            candidate = unique[2:]
            if any(v <= 64 for v in candidate):
                unique = candidate
        if len(unique) >= 2:
            max_ratio = 0.0
            split_idx = -1
            for i in range(len(unique) - 1):
                ratio = unique[i + 1] / unique[i]
                if ratio > max_ratio:
                    max_ratio = ratio
                    split_idx = i
            if max_ratio >= _GAP_SPLIT_MIN_RATIO:
                threshold = unique[split_idx]

    labels: list[str] = []
    for b in batch_sizes:
        if b is None or threshold is None or b <= threshold:
            labels.append("decode")
        else:
            labels.append("prefill_bearing")
    return labels
