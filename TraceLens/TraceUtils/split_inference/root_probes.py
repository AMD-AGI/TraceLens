###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Step 4: escalation probes, tried in declared order when coverage falls short.

Each probe proposes a different root set. They are ordered cheapest and most
reliable first, and the runner re-measures coverage after each one, so the order
is a declared preference rather than an implicit one -- every attempt and its
effect on coverage is recorded either way.
"""

from typing import Callable, Dict, List, Optional, Sequence

from ...Trace2Tree.inference_iteration_roots import (
    PERIOD_EXACT,
    compare_periods,
    find_iteration_roots_generic,
    find_period_candidates,
)
from ..annotation_utils import inherit_identity, name_skeleton
from .detect_utils import (
    GpuAttribution,
    IntervalIndex,
    PhaseConfidence,
    Probe,
    RootSet,
    ancestors_of,
    gaps_between,
    in_gap_candidates,
)

# An ancestor level is only a plausible iteration boundary if there is roughly
# one ancestor per root. All roots sharing a single ancestor is the whole-run
# wrapper, which passes a coverage check while being useless.
ANCESTOR_COUNT_TOLERANCE = 0.8
# How far to walk upward. Unbounded walking always reaches the thread entry point.
MAX_ANCESTOR_DEPTH = 8
# Categories that can plausibly mark an iteration in an inter-root gap.
GAP_CANDIDATE_CATEGORIES = ("python_function", "cpu_op")

SYNTHETIC_KEY = "synthetic"


def _synthesize(template: dict, ts: float, dur: float, probe: str, name: str) -> dict:
    """A root that no annotation produced, tagged with where it came from."""
    return {
        **{k: v for k, v in template.items() if k in ("pid", "tid", "cat")},
        "name": name,
        "ts": ts,
        "dur": dur,
        SYNTHETIC_KEY: True,
        "probe": probe,
    }


def enclosing_ancestor_probe(events: Sequence[dict], attribution: GpuAttribution):
    """Widen each root to an enclosing span that covers more GPU work.

    Narrow scope: step 1 already handles parents that are annotations, so what
    is left here are parents that were never annotated at all.
    """
    cache: Dict[str, object] = {}

    def _tree():
        if "tree" not in cache:
            from ...Trace2Tree.trace_to_tree import TraceToTree

            try:
                tree = TraceToTree(list(events), prune_nongpu_paths=False)
                tree.build_tree(add_python_func=True)
            except Exception as exc:  # a malformed trace must not abort detection
                print(f"Probe 4a: tree build failed ({exc}); skipping.")
                tree = None
            cache["tree"] = tree
        return cache["tree"]

    def run(root_set: RootSet) -> Optional[RootSet]:
        tree = _tree()
        if tree is None:
            return None
        baseline = attribution.gpu_time_by_correlation(root_set.roots)
        chains = [ancestors_of(tree, r, MAX_ANCESTOR_DEPTH) for r in root_set.roots]
        for depth in range(MAX_ANCESTOR_DEPTH):
            level = {}
            for root, chain in zip(root_set.roots, chains):
                if depth < len(chain):
                    level.setdefault(id(chain[depth]), (chain[depth], root))
            if len(level) < ANCESTOR_COUNT_TOLERANCE * len(root_set.roots):
                continue
            ancestors = [a for a, _ in level.values()]
            if attribution.gpu_time_by_correlation(ancestors) <= baseline:
                continue
            roots = sorted(
                (inherit_identity(a, r) for a, r in level.values()),
                key=lambda e: e.get("ts", 0),
            )
            return RootSet(
                roots=roots,
                method="probe:enclosing_ancestor",
                phase_confidence=PhaseConfidence.LOW,
                diagnostics={**root_set.diagnostics, "ancestor_depth": depth + 1},
            )
        return None

    return Probe(
        name="4a_enclosing_ancestor",
        applies_to=lambda rs: bool(rs.roots),
        run=run,
    )


def generic_ratio_probe(events: Sequence[dict], attribution: GpuAttribution):
    """Cross-check against call-tree periodicity, adopting only on exact agreement.

    An integer multiple of the root count is the same loop at finer grain, which
    confirms the roots rather than replacing them. The ratio is recorded anyway.
    """

    def run(root_set: RootSet) -> Optional[RootSet]:
        diagnostics: dict = {}
        roots = find_iteration_roots_generic(list(events), diagnostics)
        if not roots:
            return None
        verdict, ratio = compare_periods(len(roots), len(root_set.roots))
        merged = {**root_set.diagnostics, "generic_ratio_k": ratio, **diagnostics}
        if verdict != PERIOD_EXACT:
            return None
        return RootSet(
            roots=roots,
            method="probe:generic_ratio",
            phase_confidence=PhaseConfidence.UNKNOWN,
            diagnostics=merged,
        )

    return Probe(
        name="4b_generic_ratio",
        applies_to=lambda rs: True,
        run=run,
    )


def gap_family_probe(events: Sequence[dict], attribution: GpuAttribution):
    """Recover work sitting in the gaps between roots.

    A family firing about once per gap is an iteration marker the annotations
    missed. Recovered events are added to the roots rather than replacing them,
    but carry no parseable identity, so phase confidence drops.
    """
    index = IntervalIndex(events)

    def run(root_set: RootSet) -> Optional[RootSet]:
        gaps = gaps_between(root_set.roots)
        if not gaps:
            return None
        candidates = in_gap_candidates(index, gaps, GAP_CANDIDATE_CATEGORIES)
        if not candidates:
            return None
        grouped: Dict[str, List[dict]] = {}
        for event in candidates:
            grouped.setdefault(name_skeleton(event.get("name", "")), []).append(event)
        best = max(
            grouped.items(),
            key=lambda kv: attribution.gpu_time_by_correlation(kv[1]),
        )
        skeleton, found = best
        if attribution.gpu_time_by_correlation(found) <= 0:
            return None
        roots = sorted([*root_set.roots, *found], key=lambda e: e.get("ts", 0))
        return RootSet(
            roots=roots,
            method="probe:gap_family",
            phase_confidence=PhaseConfidence.LOW,
            diagnostics={**root_set.diagnostics, "gap_family_skeleton": skeleton},
        )

    return Probe(
        name="4c_gap_family",
        applies_to=lambda rs: len(rs.roots) > 1,
        run=run,
    )


def kernel_series_probe(events: Sequence[dict], attribution: GpuAttribution):
    """Build roots from a repeating period in the kernels nothing accounts for.

    Last resort, and the only probe that works when the uncovered work has no
    annotation near it. Roots must be CPU-side spans since extraction windows by
    thread, so launch sites are used; a block launched entirely inside a
    captured graph has none and is skipped.
    """

    def run(root_set: RootSet) -> Optional[RootSet]:
        from .root_detection import collect_annotations

        uncovered = attribution.uncovered_kernels(collect_annotations(events))
        if len(uncovered) < 2:
            return None
        candidates = find_period_candidates([k.get("name", "") for k in uncovered])
        if not candidates:
            return None
        best = candidates[0]
        roots = []
        for index in range(best.repeats):
            lo = best.start + index * best.period
            block = uncovered[lo : lo + best.period]
            launches = attribution.cpu_launches_for(block)
            if not launches:
                continue
            start = min(e["ts"] for e in launches)
            end = max(e["ts"] + e["dur"] for e in launches)
            roots.append(
                _synthesize(
                    launches[0],
                    start,
                    end - start,
                    "4d_kernel_series",
                    f"synthetic:{name_skeleton(block[0].get('name', ''))}",
                )
            )
        if not roots:
            return None
        roots.sort(key=lambda e: e.get("ts", 0))
        return RootSet(
            roots=roots,
            method="probe:kernel_series",
            phase_confidence=PhaseConfidence.UNKNOWN,
            diagnostics={
                **root_set.diagnostics,
                "kernel_series_period": best.period,
                "n_uncovered_kernels": len(uncovered),
            },
        )

    return Probe(
        name="4d_kernel_series",
        applies_to=lambda rs: bool(attribution.kernels),
        run=run,
    )


PROBE_FACTORIES: Sequence[Callable] = (
    enclosing_ancestor_probe,
    generic_ratio_probe,
    gap_family_probe,
    kernel_series_probe,
)


def build_probes(events: Sequence[dict], attribution: GpuAttribution) -> List[Probe]:
    """Every probe, in the order they should be attempted."""
    return [factory(events, attribution) for factory in PROBE_FACTORIES]
