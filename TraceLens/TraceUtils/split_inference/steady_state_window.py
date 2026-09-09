###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stage 2: steady-state region detection and window selection.

Three public entry points:

* :func:`find_steady_state_inference` — for LLM inference serving traces
  (vLLM, SGLang, ATOM) where annotations encode request concurrency.
  Uses concurrency-based region detection + prefill/decode mode selection.

* :func:`find_steady_state_inference_from_shapes` — for LLM inference traces
  without parseable serving annotations.  Uses batch sizes derived from
  cpu_op shapes as a concurrency proxy + threshold-based phase classification.

* :func:`find_steady_state_generic` — for training, diffusion, and other
  workloads where annotations lack concurrency info.  Uses duration-CV
  sliding window.

Both return ``(selected_roots, regions)`` so callers can also use the
region list (e.g. ``divide_phases_and_save``).
"""

import math
from statistics import mean, median, pstdev

from ..annotation_utils import (
    has_context,
    is_decode_only,
    is_mixed,
    iteration_details,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _identify_regions_by_peak(
    values: list[int], num_steps: int, label: str = "Steady state",
) -> tuple[list[tuple[int, int]], int]:
    """Find contiguous regions where ``values`` are near the global peak.

    The core scan shared by both the inference (num_requests) and shape-based
    (batch_sizes) steady-state finders.

    Returns ``(regions, global_max)``.
    """
    n = len(values)
    thresh = 0.1 if n >= num_steps else 0.2
    global_max = max(values)

    steady_state_started = False
    steady_state_ended = False
    prev_events_in_steady = 0
    start_index = 0
    regions: list[tuple[int, int]] = []

    for i, v in enumerate(values):
        if abs(v - global_max) <= max(1, thresh * global_max):
            if not steady_state_started:
                prev_events_in_steady += 1
        else:
            if steady_state_started:
                prev_events_in_steady -= 1

        if prev_events_in_steady > 5 and not steady_state_started:
            print(f"{label} started at index {i - 5}")
            steady_state_started = True
            start_index = i - prev_events_in_steady + 1

        if (
            prev_events_in_steady <= 0
            and steady_state_started
            and not steady_state_ended
        ):
            print(f"{label} ended at index {i}")
            steady_state_ended = True
            regions.append((start_index, i))
            steady_state_started = False
            steady_state_ended = False
            prev_events_in_steady = 0

    if steady_state_started and not steady_state_ended:
        regions.append((start_index, i))

    print(f"{label} regions: {regions}")

    if len(regions) == 0:
        delta = min(n, max(8, num_steps - n))
        start = max(0, delta // 2)
        end = max(start + 1, min(n, n - delta // 2))
        regions = [(start, end)]
        print(
            f"Warning: no {label.lower()} region found; discarding initial/final "
            "iterations and selecting middle region"
        )

    return regions, global_max


def _identify_regions_inference(
    iter_details: list[dict], num_steps: int
) -> tuple[list[tuple[int, int]], int]:
    """Detect contiguous steady-state regions based on num_requests proximity to global max."""
    return _identify_regions_by_peak(
        [t["num_requests"] for t in iter_details], num_steps,
        label="Steady state",
    )


def _compute_reference_pd_ratio(
    regions: list[tuple[int, int]], iter_details: list[dict]
) -> tuple[tuple[int, int], float, float]:
    """Return the largest steady-state region, a reference PD ratio, and the
    median PD ratio across all regions.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_steady_state_inference(
    iteration_roots: list[dict],
    num_steps: int,
    mode: str = "mixed",
    CONC: int | None = None,
    OSL: float | None = None,
    R: float | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Find the best contiguous window for an LLM inference serving trace.

    Combines concurrency-based region detection with prefill/decode-aware
    window selection.

    Returns ``(selected_roots, regions)`` — the chosen window of iteration
    roots and the steady-state region list (useful for ``divide_phases``).

    Parameters
    ----------
    iteration_roots : list of iteration-root events
    num_steps : requested window size
    mode : one of ``"mixed"``, ``"decode_only"``, ``"max_prefilldecode"``
    CONC : expected peak concurrency (number of concurrent requests).
    OSL : average output sequence length (decode tokens per request).
    R : OSL window ratio in [0, 1].

    Modes
    -----
    ``"mixed"``
        Pick the sub-window whose pd_ratio is closest to the reference ratio.
        Ties broken by highest average num_requests.
    ``"decode_only"``
        Longest contiguous run of pure decode-only steps, capped at num_steps.
    ``"max_prefilldecode"``
        Longest contiguous run of pure prefill-bearing steps, capped at num_steps.
    """
    iter_details = iteration_details(iteration_roots)
    regions, global_max = _identify_regions_inference(iter_details, num_steps)

    (largest_start, largest_end), reference_ratio, _largest_window_ratio = (
        _compute_reference_pd_ratio(regions, iter_details)
    )

    # --- Optional: CONC / OSL / R validation and ideal ratio override ----------
    ideal_pd_ratio: float | None = None

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

        reference_ratio = ideal_pd_ratio
        print(
            f"Using ideal prefilldecodemix_to_totalsteps_ratio={ideal_pd_ratio:.4f} as reference (overrides empirical {reference_ratio:.4f})"
        )
    print("\n --------------------------------")
    # ---------------------------------------------------------------------------

    divider = max(1, min(int(num_steps / 2), 10))
    step = max(1, num_steps // divider)

    candidates = []
    s, e = largest_start, largest_end

    def _count_mixed(window: list[dict]) -> int:
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
        return iteration_roots[best["start"] : best["end"]], regions

    elif mode in ("decode_only", "max_prefilldecode"):
        phase_labels = [
            "prefill_bearing" if has_context(d) else "decode"
            for d in iter_details
        ]
        target = "decode" if mode == "decode_only" else "prefill_bearing"
        return _select_run_window(
            iteration_roots, phase_labels, target,
            largest_start, largest_end, num_steps, regions, mode,
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'mixed', 'decode_only', or 'max_prefilldecode'."
        )


def _longest_contiguous_run(
    labels: list[str],
    target: str,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    """Find the longest contiguous run of ``target`` in ``labels[start:end]``.

    Returns ``(run_start, run_end)`` or ``None`` if no run exists.
    """
    best: tuple[int, int] | None = None
    run_start: int | None = None
    for idx in range(start, end):
        if labels[idx] == target:
            if run_start is None:
                run_start = idx
        else:
            if run_start is not None:
                if best is None or (idx - run_start) > (best[1] - best[0]):
                    best = (run_start, idx)
                run_start = None
    if run_start is not None:
        if best is None or (end - run_start) > (best[1] - best[0]):
            best = (run_start, end)
    return best


def _select_run_window(
    iteration_roots: list[dict],
    labels: list[str],
    target: str,
    largest_start: int,
    largest_end: int,
    num_steps: int,
    regions: list[tuple[int, int]],
    mode_tag: str,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Select the longest contiguous run of ``target``-labelled iterations,
    capped at ``num_steps``.  Shared by both annotation- and shape-based paths.
    """
    run = _longest_contiguous_run(labels, target, largest_start, largest_end)
    if run:
        run_s, run_e = run
        win_e = min(run_e, run_s + num_steps)
        print(
            f"[{mode_tag}] Longest {target} run: [{run_s}, {run_e}) "
            f"({run_e - run_s} steps). Selected [{run_s}, {win_e}) "
            f"({win_e - run_s} steps, capped at num_steps={num_steps})."
        )
        return iteration_roots[run_s:win_e], regions
    else:
        print(f"[{mode_tag}] No {target} run found in steady-state region.")
        return [], regions


PREFILL_SPIKE_FACTOR = 2.0


def classify_phases_from_batch_sizes(
    batch_sizes: list[int | None],
) -> list[str]:
    """Classify each iteration as ``'decode'`` or ``'prefill_bearing'``.

    Uses the median batch size as the decode baseline.  Iterations whose
    batch size exceeds ``PREFILL_SPIKE_FACTOR * median`` are labelled
    ``'prefill_bearing'``; the rest are ``'decode'``.
    """
    valid = [b for b in batch_sizes if b is not None]
    if not valid:
        return ["decode"] * len(batch_sizes)
    baseline = median(valid)
    threshold = PREFILL_SPIKE_FACTOR * baseline

    labels: list[str] = []
    for b in batch_sizes:
        if b is None or b <= threshold:
            labels.append("decode")
        else:
            labels.append("prefill_bearing")
    return labels


def find_steady_state_inference_from_shapes(
    iteration_roots: list[dict],
    batch_sizes: list[int],
    num_steps: int,
    mode: str = "mixed",
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Find steady state for LLM inference traces without serving annotations.

    Uses ``batch_sizes`` (derived from cpu_op shapes) as a proxy for
    concurrency and :func:`classify_phases_from_batch_sizes` for
    prefill/decode classification.

    Returns ``(selected_roots, regions)`` — same shape as
    :func:`find_steady_state_inference`.
    """
    if not batch_sizes or not iteration_roots:
        return [], []

    total = len(iteration_roots)
    if total == 0:
        return [], []

    regions, _ = _identify_regions_by_duration_cv(iteration_roots, num_steps)
    phase_labels = classify_phases_from_batch_sizes(batch_sizes)

    largest_start, largest_end = max(regions, key=lambda r: r[1] - r[0])

    divider = max(1, min(int(num_steps / 2), 10))
    step = max(1, num_steps // divider)

    if mode == "mixed":
        total_pf = sum(
            1 for i in range(largest_start, largest_end)
            if phase_labels[i] == "prefill_bearing"
        )
        region_size = largest_end - largest_start
        reference_ratio = total_pf / region_size if region_size else 0.0

        candidates = []
        if (largest_end - largest_start) >= num_steps:
            for s1 in range(largest_start, largest_end - num_steps + 1, step):
                pf_count = sum(
                    1 for i in range(s1, s1 + num_steps)
                    if phase_labels[i] == "prefill_bearing"
                )
                avg_bs = mean(
                    b for b in batch_sizes[s1 : s1 + num_steps] if b is not None
                ) if any(b is not None for b in batch_sizes[s1 : s1 + num_steps]) else 0
                candidates.append({
                    "start": s1,
                    "end": s1 + num_steps,
                    "pf_ratio": pf_count / num_steps,
                    "avg_bs": avg_bs,
                })
        else:
            pf_count = sum(
                1 for i in range(largest_start, largest_end)
                if phase_labels[i] == "prefill_bearing"
            )
            candidates.append({
                "start": largest_start,
                "end": largest_end,
                "pf_ratio": pf_count / region_size if region_size else 0.0,
                "avg_bs": mean(
                    b for b in batch_sizes[largest_start:largest_end] if b is not None
                ) if any(b is not None for b in batch_sizes[largest_start:largest_end]) else 0,
            })

        best = min(
            candidates,
            key=lambda c: (abs(c["pf_ratio"] - reference_ratio), -c["avg_bs"]),
        )
        print(
            f"[mixed/shapes] Selected window [{best['start']}, {best['end']}): "
            f"pf_ratio={best['pf_ratio']:.3f} (target={reference_ratio:.3f}), "
            f"avg_bs={best['avg_bs']:.1f}"
        )
        return iteration_roots[best["start"] : best["end"]], regions

    elif mode in ("decode_only", "max_prefilldecode"):
        target = "decode" if mode == "decode_only" else "prefill_bearing"
        return _select_run_window(
            iteration_roots, phase_labels, target,
            largest_start, largest_end, num_steps, regions, f"{mode}/shapes",
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'mixed', 'decode_only', or 'max_prefilldecode'."
        )


MIN_STEADY_WINDOW = 4
CV_THRESHOLD = 0.15


def _identify_regions_by_duration_cv(
    iteration_roots: list[dict], num_steps: int
) -> tuple[list[tuple[int, int]], float]:
    """Find the most duration-consistent region via sliding-window CV.

    Returns ``(regions, best_cv)`` where ``regions`` is a single-element
    list ``[(start, end)]`` for the best window.
    """
    total = len(iteration_roots)
    durations = [r.get("dur", 0) for r in iteration_roots]
    scan_size = min(max(num_steps, MIN_STEADY_WINDOW), total)

    windows = []
    for start in range(total - scan_size + 1):
        chunk = durations[start : start + scan_size]
        m = mean(chunk)
        cv = pstdev(chunk) / m if m else 0.0
        windows.append((start, start + scan_size, cv, m))

    if not windows:
        return [(0, min(num_steps, total))], 0.0

    passing = [(s, e, cv, m) for s, e, cv, m in windows if cv < CV_THRESHOLD]
    if passing:
        best_start, best_end, best_cv, best_mean = min(passing, key=lambda x: x[3])
    else:
        best_start, best_end, best_cv, best_mean = min(windows, key=lambda x: x[2])

    print(
        f"[duration-cv] Steady state by duration: [{best_start}, {best_end}) "
        f"cv={best_cv:.4f}, mean_dur={best_mean:.0f}us"
    )

    return [(best_start, best_end)], best_cv


def find_steady_state_generic(
    iteration_roots: list[dict],
    num_steps: int,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Find steady state by duration consistency for non-serving workloads.

    Slides an overlapping window (stride 1) across all iterations, computes
    the coefficient of variation of wall-clock duration for each position,
    and picks the fastest window whose CV is below ``CV_THRESHOLD``.  Falls
    back to the lowest-CV window when nothing passes.

    Returns ``(selected_roots, regions)`` — the chosen window and the
    region it was drawn from.
    """
    total = len(iteration_roots)
    if total == 0:
        return [], []

    region, _ = _identify_regions_by_duration_cv(iteration_roots, num_steps)
    best_start, best_end = region[0]
    scan_size = min(max(num_steps, MIN_STEADY_WINDOW), total)

    if num_steps < scan_size:
        center = (best_start + best_end) // 2
        half = num_steps // 2
        final_start = max(best_start, center - half)
        final_end = min(final_start + num_steps, total)
        final_start = max(0, final_end - num_steps)
        return iteration_roots[final_start:final_end], region

    return iteration_roots[best_start:best_end], region
