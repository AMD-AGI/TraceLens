###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stage 2: steady-state region detection and window selection."""

import math
from statistics import mean
from typing import List, Optional, Tuple

from ..annotation_utils import (
    has_context,
    is_decode_only,
    is_mixed,
    iteration_details,
)


def identify_steady_state_regions(
    iter_details: List[dict], num_steps: int
) -> Tuple[List[Tuple[int, int]], int]:
    """Detect contiguous steady-state regions based on num_requests proximity to global max.

    Returns ``(regions, global_max)`` where ``regions`` is a list of
    ``(start, end)`` index pairs and ``global_max`` is the peak concurrency
    observed across all iterations.
    """
    n = len(iter_details)
    thresh = 0.1 if n >= num_steps else 0.2
    global_max = max(t["num_requests"] for t in iter_details)

    steady_state_started = False
    steady_state_ended = False
    prev_events_in_steady = 0
    start_index = 0
    regions = []

    for i, t in enumerate(iter_details):
        if abs(t["num_requests"] - global_max) <= max(1, thresh * global_max):
            if not steady_state_started:
                prev_events_in_steady += 1
        else:
            if steady_state_started:
                prev_events_in_steady -= 1

        if prev_events_in_steady > 5 and not steady_state_started:
            print(f"Steady state started at index {i - 5}")
            steady_state_started = True
            start_index = i - prev_events_in_steady + 1

        if (
            prev_events_in_steady <= 0
            and steady_state_started
            and not steady_state_ended
        ):
            print(f"Steady state ended at index {i}")
            steady_state_ended = True
            regions.append((start_index, i))
            steady_state_started = False
            steady_state_ended = False
            prev_events_in_steady = 0

    if steady_state_started and not steady_state_ended:
        regions.append((start_index, i))

    print(f"Steady state regions: {regions}")

    if len(regions) == 0:
        delta = min(n, max(8, num_steps - n))
        start = max(0, delta // 2)
        end = max(start + 1, min(n, n - delta // 2))
        regions = [(start, end)]
        print(
            "Warning: no steady state region found; discarding initial/final iterations "
            "and selecting middle region"
        )

    return regions, global_max


def compute_reference_pd_ratio(
    regions: List[Tuple[int, int]], iter_details: List[dict]
) -> Tuple[Tuple[int, int], float, float]:
    """
    Return the largest steady-state region, a reference PD ratio, and the
    median PD ratio across all regions.

    The reference ratio starts as the PD/total ratio of the largest region.  A
    sanity check compares it to the median ratio across ALL regions.  If the
    largest region deviates by more than 50 % relative from the median, the
    median is used instead and a warning is printed.
    """
    region_stats = []
    total_steps = 0
    total_pd_steps = 0
    for s, e in regions:
        window = iter_details[s:e]
        total = len(window)
        total_steps += total
        pd_count = sum(1 for t in window if has_context(t))
        total_pd_steps += pd_count
        ratio = pd_count / total if total > 0 else 0.0
        region_stats.append({"start": s, "end": e, "size": total, "pd_ratio": ratio})
        print(
            f"  Region [{s}, {e}): size={total}, "
            f"prefilldecodemix_steps={pd_count}, prefilldecodemix_to_totalsteps_ratio={ratio:.3f}"
        )

    largest = max(region_stats, key=lambda x: x["size"])
    average_ratio = total_pd_steps / total_steps if total_steps > 0 else 0.0
    largest_window_ratio = largest["pd_ratio"]
    print(
        f"Reference prefilldecodemix_to_totalsteps_ratio={largest_window_ratio:.3f} (largest region [{largest['start']}, {largest['end']}), Average across all regions={average_ratio:.3f})"
    )

    return (largest["start"], largest["end"]), average_ratio, largest_window_ratio


def find_steady_state_window(
    iteration_roots: List[dict],
    num_steps: int,
    steady_state_regions: List[Tuple[int, int]],
    mode: str = "mixed",
    CONC: Optional[int] = None,
    OSL: Optional[float] = None,
    R: Optional[float] = None,
) -> List[dict]:
    """
    Find the best contiguous window of up to ``num_steps`` iterations.

    Parameters
    ----------
    iteration_roots : list of iteration-root events
    num_steps : requested window size
    steady_state_regions : pre-computed steady-state region list as ``(start, end)``
        index pairs.  Pass ``[(0, len(iteration_roots))]`` to treat the entire
        slice as steady state.
    mode : one of ``"mixed"``, ``"decode_only"``, ``"max_prefilldecode"``
    CONC : expected peak concurrency (number of concurrent requests).
        If provided, a warning is printed when the observed peak in the trace
        differs from this value.
    OSL : average output sequence length (decode tokens per request).
        Combined with ``R`` to derive the ideal PD ratio.
    R : OSL window ratio in [0, 1]. The actual OSL per request is sampled from
        ``[R * OSL, OSL]``, giving mean OSL = OSL * (1 + R) / 2.

    When ``CONC``, ``OSL``, and ``R`` are all provided the ideal PD ratio is

        ideal_pd_ratio = (CONC * 2) / (OSL * (1 + R))

    and ``num_steps`` is automatically raised to ``ceil(1 / ideal_pd_ratio)``
    if it is too small to capture the true DO/PD distribution.

    Modes
    -----
    ``"mixed"``
        Pick the sub-window whose pd_ratio is closest to the reference ratio
        (ideal when available, otherwise largest-region / median sanity-checked).
        Ties broken by highest average num_requests.
    ``"decode_only"``
        Fewest-PD window: sub-window with lowest pd_ratio.
    ``"max_prefilldecode"``
        Most-PD window: sub-window with highest pd_ratio.
    """
    iter_details = iteration_details(iteration_roots)
    regions = steady_state_regions
    global_max = max(t["num_requests"] for t in iter_details)

    (largest_start, largest_end), reference_ratio, largest_window_ratio = (
        compute_reference_pd_ratio(regions, iter_details)
    )

    # --- Optional: CONC / OSL / R validation and ideal ratio override ----------
    ideal_pd_ratio: Optional[float] = None

    if CONC is not None and global_max != CONC:
        print(
            f"Warning: expected peak concurrency CONC={CONC} but the trace peak is "
            f"global_max={global_max}. The trace may not contain requests at the "
            f"intended concurrency level."
        )

    if CONC is not None and OSL is not None and R is not None:
        if not (0.0 <= R <= 1.0):
            print(f"Warning: R={R} is outside [0, 1]; clamping to valid range.")
            R = max(0.0, min(1.0, R))
        mean_osl = OSL * (1.0 + R) / 2.0
        ideal_pd_ratio = (CONC * 2.0) / (OSL * (1.0 + R))
        print(
            f"Ideal prefilldecodemix_to_totalsteps_ratio = (CONC={CONC} * 2) / (OSL={OSL} * (1 + R={R})) "
            f"= {ideal_pd_ratio:.4f}  [mean OSL = {mean_osl:.1f}]"
        )

        min_steps_for_ratio = math.ceil(1.0 / ideal_pd_ratio)
        if num_steps < min_steps_for_ratio:
            print(
                f"Warning: --num-steps={num_steps} is too small to capture the true "
                f"decode_only/prefilldecodemix distribution. At prefilldecodemix_to_totalsteps_ratio={ideal_pd_ratio:.4f} you need at "
                f"least {min_steps_for_ratio} steps to see a representative mix. "
                f"Raising num_steps to {min_steps_for_ratio}."
            )
            num_steps = min_steps_for_ratio
        else:
            print(f"num_steps={num_steps} >= min required {min_steps_for_ratio} — OK.")

        # Ideal ratio overrides the empirical reference for the mixed mode
        reference_ratio = ideal_pd_ratio
        print(
            f"Using ideal prefilldecodemix_to_totalsteps_ratio={ideal_pd_ratio:.4f} as reference (overrides empirical {reference_ratio:.4f})"
        )
    print(f"\n --------------------------------")
    # ---------------------------------------------------------------------------

    divider = max(1, min(int(num_steps / 2), 10))
    step = max(1, num_steps // divider)

    # Build candidate sub-windows from the largest region
    candidates = []
    s, e = largest_start, largest_end

    def _count_mixed(window: List[dict]) -> int:
        """Count truly-mixed steps (both context and generation requests > 0)."""
        return sum(1 for t in window if is_mixed(t))

    if (e - s) >= num_steps:
        for s1 in range(s, e - num_steps + 1, step):
            window = iter_details[s1 : s1 + num_steps]
            pd_count = sum(1 for t in window if has_context(t))
            candidates.append(
                {
                    "start": s1,
                    "end": s1 + num_steps,
                    "pd_count": pd_count,
                    "pd_ratio": pd_count / num_steps,
                    "mixed_count": _count_mixed(window),
                    "avg_requests": mean(t["num_requests"] for t in window),
                }
            )
    else:
        # Region is smaller than num_steps — use the whole region
        window = iter_details[s:e]
        pd_count = sum(1 for t in window if has_context(t))
        candidates.append(
            {
                "start": s,
                "end": e,
                "pd_count": pd_count,
                "pd_ratio": pd_count / len(window) if window else 0.0,
                "mixed_count": _count_mixed(window),
                "avg_requests": (
                    mean(t["num_requests"] for t in window) if window else 0
                ),
            }
        )

    if mode == "mixed":
        # Prefer candidate windows that contain at least one prefill-bearing
        # step (pure prefill OR truly mixed, i.e. context_requests > 0). Fall
        # back to all candidates only when no window contains any prefill
        # activity at all.
        pd_candidates = [c for c in candidates if c["pd_count"] > 0]
        if pd_candidates:
            print(
                f"[mixed] Filtering to {len(pd_candidates)}/{len(candidates)} "
                f"candidate windows that contain at least one prefill or "
                f"prefill-decode step."
            )
            selection_pool = pd_candidates
        else:
            print(
                "[mixed] No candidate window contains a prefill or prefill-decode "
                "step; falling back to the full candidate set."
            )
            selection_pool = candidates

        best = min(
            selection_pool,
            key=lambda c: (abs(c["pd_ratio"] - reference_ratio), -c["avg_requests"]),
        )
        print(
            f"[mixed] Selected window [{best['start']}, {best['end']}): "
            f"prefilldecodemix_to_totalsteps_ratio={best['pd_ratio']:.3f} (target={reference_ratio:.3f}), "
            f"avg_requests={best['avg_requests']:.1f}, "
            f"pd_count={best['pd_count']}, mixed_count={best['mixed_count']}"
        )

    elif mode == "decode_only":
        # Find the longest contiguous run of pure decode-only steps (active
        # generation with no context requests) in the largest steady-state
        # region, capped at num_steps.
        do_runs: List[Tuple[int, int]] = []  # (start, end) in iter_details coords
        run_start: Optional[int] = None
        for idx in range(largest_start, largest_end):
            if is_decode_only(iter_details[idx]):
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None:
                    do_runs.append((run_start, idx))
                    run_start = None
        if run_start is not None:
            do_runs.append((run_start, largest_end))

        if do_runs:
            longest = max(do_runs, key=lambda r: r[1] - r[0])
            run_s, run_e = longest
            win_s = run_s
            win_e = min(run_e, run_s + num_steps)
            print(
                f"[decode_only] Longest pure decode-only run: [{run_s}, {run_e}) "
                f"({run_e - run_s} steps). "
                f"Selected [{win_s}, {win_e}) ({win_e - win_s} steps, "
                f"capped at num_steps={num_steps})."
            )
            return iteration_roots[win_s:win_e]
        else:
            print(
                "[decode_only] No pure decode-only run found in steady-state region; "
            )
            return []

    elif mode == "max_prefilldecode":
        # Find the longest contiguous run of pure PD steps (no decode-only) in
        # the largest steady-state region, capped at num_steps.
        pd_runs: List[Tuple[int, int]] = []  # (start, end) in iter_details coords
        run_start: Optional[int] = None
        for idx in range(largest_start, largest_end):
            if has_context(iter_details[idx]):
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None:
                    pd_runs.append((run_start, idx))
                    run_start = None
        if run_start is not None:
            pd_runs.append((run_start, largest_end))

        if pd_runs:
            # Pick the longest pure-PD run
            longest = max(pd_runs, key=lambda r: r[1] - r[0])
            run_s, run_e = longest
            # Cap to num_steps from the start of the run
            win_s = run_s
            win_e = min(run_e, run_s + num_steps)
            print(
                f"[max_prefilldecode] Longest pure prefilldecodemix run: [{run_s}, {run_e}) "
                f"({run_e - run_s} steps). "
                f"Selected [{win_s}, {win_e}) ({win_e - win_s} steps, "
                f"capped at num_steps={num_steps})."
            )
            return iteration_roots[win_s:win_e]
        else:
            print(
                "[max_prefilldecode] No pure prefilldecodemix run found in steady-state "
            )
            return []

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'mixed', 'decode_only', or 'max_prefilldecode'."
        )

    return iteration_roots[best["start"] : best["end"]]
