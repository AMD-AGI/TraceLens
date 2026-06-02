###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Match same-phase execution-root blocks between two inference traces.

Reads two large vLLM/SGLang inference traces, finds contiguous same-phase
execution-root blocks (decode_only and prefilldecode) in each, and selects
the SINGLE best (A, B) pair per phase.

Selection rule per phase
------------------------
1. Execution roots within a block are contiguous in the source trace.
2. Both blocks of a candidate pair must have the same number of execution
   roots (strict size matching).
3. decode_only blocks must have at least 2 steps; prefilldecode at least 1.
4. Score each candidate with a per-step **averaged** distance tuple
   (priority order matches the user's stated criteria):
       decode_only:    (avg |g_sq diff|, avg |g_sk diff|)
       prefilldecode:  (avg |c_req|, avg |g_req|, avg |c_sq|, avg |g_sq|)
5. Pick the pair with the lexicographically smallest distance tuple. Ties
   are broken by the **largest step count**.

Output
------
- ``match_report.json`` — one entry per phase. Each entry includes the
  selected blocks' aggregate stats AND a per-step list with name, num_context,
  num_generation, c_sq, c_sk, g_sq, g_sk, plus the path of the extracted
  trace file (deterministic, populated even with ``--no-extract``).
- ``match_report.csv`` — flat, one row per match.
- ``match_notes.json`` — diagnostics (phases that had no eligible pair).

Usage
-----
    python match_inference_trace_blocks.py <trace_a> <trace_b> -o <out_dir>
                                           [--no-extract]
                                           [--phases decode_only,prefilldecode]
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from TraceLens.util import DataLoader
from TraceLens.TraceUtils.split_inference_trace_annotation import (
    extract_and_save,
    find_phase_from_window,
    get_filename,
    preprocess_trace,
)

ANNOTATION_PATTERN = [
    re.compile(
        r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
    ),
    re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)"),
    re.compile(r"execute_new_\d+_cached_\d+"),
    re.compile(r"execute_context_\d+\(\d+_\d+\)_generation_\d+\(\d+\)"),
]

PHASE_DECODE_ONLY = "decode_only"
PHASE_PREFILLDECODE = "prefilldecode"

PER_STEP_KEYS = (
    "context_requests",
    "generation_requests",
    "c_sq",
    "c_sk",
    "g_sq",
    "g_sk",
)


def _find_events_by_pattern_quiet(events, patterns, name, cat=None):
    """Same shape as the splitter helper but without per-event spam."""
    matches = []
    for pattern in patterns:
        cur = [e for e in events if pattern.match(e.get("name", ""))]
        if cat is not None:
            cur = [e for e in cur if e.get("cat") == cat]
        matches.extend(cur)
    matches.sort(key=lambda x: x.get("ts", 0))
    print(f"Found {len(matches)} {name} events")
    if not matches:
        return None
    return matches


# ---------------------------------------------------------------------------
# Per-iteration name parser (extended to expose c_sq/c_sk/g_sq/g_sk)
# ---------------------------------------------------------------------------
def _safe_int(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def parse_iter_details(name: str) -> dict:
    """Extract per-step details including c_sq/c_sk/g_sq/g_sk where available.

    For naming patterns that don't carry sq/sk (e.g. ``execute_new_N_cached_M``
    or ``execute_context_N(M)_generation_K(L)``), the corresponding sq/sk
    fields default to 0 and ``has_sqsk`` is set to False.
    """
    raw = name
    name2 = name.replace("(", "_").replace(")", "_")
    parts = re.sub(r"[sqk]+", "_", name2).split("_")

    has_sqsk = False
    c_req = c_sq = c_sk = c_sqsq = c_sqsk = 0
    g_req = g_sq = g_sk = g_sqsq = g_sqsk = 0

    if len(parts) < 10:
        # Support for vLLM v0.13 and lower annotation format
        # execute_context_N(M)_generation_K(L)
        c_req = _safe_int(parts[2]) if len(parts) > 2 else 0
        c_sum_proxy = _safe_int(parts[3]) if len(parts) > 3 else 0
        g_req = _safe_int(parts[6]) if len(parts) > 6 else 0
        g_sum_proxy = _safe_int(parts[7]) if len(parts) > 7 else 0
        c_sq = c_sum_proxy
        g_sq = g_sum_proxy
    elif len(parts) < 12:
        # Support for vLLM v0.14 and higher annotation format
        # execute_X_context_R(sqAskBsqsqCsqskD)_generation_R'(...)
        c_req = _safe_int(parts[2])
        c_sq = _safe_int(parts[3])
        g_req = _safe_int(parts[7])
        g_sq = _safe_int(parts[8])
    else:
        # Support for TraceLens vLLM annotation format
        # Full pattern: execute_X_context_R(sqAskBsqsqCsqskD)_generation_R'(...)
        c_req = _safe_int(parts[3])
        c_sq = _safe_int(parts[5])
        c_sk = _safe_int(parts[6])
        c_sqsq = _safe_int(parts[7])
        c_sqsk = _safe_int(parts[8])
        g_req = _safe_int(parts[11])
        g_sq = _safe_int(parts[13])
        g_sk = _safe_int(parts[14])
        g_sqsq = _safe_int(parts[15])
        g_sqsk = _safe_int(parts[16])
        has_sqsk = True

    return {
        "name": raw,
        "context_requests": c_req,
        "generation_requests": g_req,
        "c_sq": c_sq,
        "c_sk": c_sk,
        "c_sqsq": c_sqsq,
        "c_sqsk": c_sqsk,
        "g_sq": g_sq,
        "g_sk": g_sk,
        "g_sqsq": g_sqsq,
        "g_sqsk": g_sqsk,
        "num_requests": c_req + g_req,
        "batch_size": c_sq + g_sq,
        "has_sqsk": has_sqsk,
    }


def classify_phase(detail: dict) -> Optional[str]:
    """Return ``decode_only``, ``prefilldecode`` or ``None`` for a step."""
    c = detail.get("context_requests", 0)
    g = detail.get("generation_requests", 0)
    if c > 0:
        return PHASE_PREFILLDECODE
    if g > 0:
        return PHASE_DECODE_ONLY
    return None


# ---------------------------------------------------------------------------
# Block discovery
# ---------------------------------------------------------------------------
@dataclass
class Block:
    phase: str
    start_idx: int  # inclusive index into iteration_roots
    end_idx: int  # exclusive
    roots: List[dict] = field(default_factory=list)
    details: List[dict] = field(default_factory=list)
    # Step count of the parent block this came from before windowing.
    # -1 means "not windowed" (this Block is the original block; equals num_steps).
    # When > num_steps, this Block represents one non-overlapping N-step window
    # carved out of a larger parent block (see ``window_blocks``).
    original_num_steps: int = -1

    @property
    def num_steps(self) -> int:
        return len(self.roots)

    @property
    def truncated(self) -> bool:
        # True iff this Block is a window of a larger parent block.
        return self.original_num_steps > 0 and self.original_num_steps != self.num_steps

    def has_full_sqsk(self) -> bool:
        return all(d.get("has_sqsk", False) for d in self.details)

    def avg(self, key: str) -> float:
        if not self.details:
            return 0.0
        return sum(d.get(key, 0) for d in self.details) / len(self.details)


def window_blocks(blocks: List["Block"], max_steps: int) -> List["Block"]:
    """Split each block into non-overlapping windows of ``max_steps`` steps.

    Behavior:
    - ``max_steps <= 0``: pass-through, no windowing.
    - block.num_steps <= max_steps: emitted as one window (unchanged).
    - block.num_steps  > max_steps: split into floor(N / max_steps)
      non-overlapping windows of exactly ``max_steps`` steps. Any tail of
      length < ``max_steps`` is dropped (so all candidates are full N-step
      windows; this keeps strict same-size matching meaningful).

    Each window remembers its parent block's original step count in
    ``original_num_steps``. ``start_idx``/``end_idx`` are absolute indices
    into the source trace's iteration_roots list.
    """
    if max_steps is None or max_steps <= 0:
        return blocks
    out: List[Block] = []
    for b in blocks:
        n = b.num_steps
        if n <= max_steps:
            out.append(
                Block(
                    phase=b.phase,
                    start_idx=b.start_idx,
                    end_idx=b.end_idx,
                    roots=b.roots,
                    details=b.details,
                    original_num_steps=n,
                )
            )
            continue
        num_windows = n // max_steps
        for w in range(num_windows):
            s = w * max_steps
            e = s + max_steps
            out.append(
                Block(
                    phase=b.phase,
                    start_idx=b.start_idx + s,
                    end_idx=b.start_idx + e,
                    roots=b.roots[s:e],
                    details=b.details[s:e],
                    original_num_steps=n,
                )
            )
    return out


def find_blocks(iteration_roots: List[dict]) -> List[Block]:
    """Return the list of contiguous same-phase blocks in iteration order.

    Blocks end at any phase transition or at a step that doesn't fit either
    phase (e.g. malformed step). Min-step filters are applied per phase:
    decode_only requires >= 2 steps, prefilldecode requires >= 1 step.
    """
    if not iteration_roots:
        return []

    details = [parse_iter_details(r.get("name", "")) for r in iteration_roots]
    phases = [classify_phase(d) for d in details]

    blocks: List[Block] = []
    cur_phase: Optional[str] = None
    cur_start = 0
    cur_roots: List[dict] = []
    cur_details: List[dict] = []

    def _flush(end_idx: int):
        if cur_phase is None or not cur_roots:
            return
        if cur_phase == PHASE_DECODE_ONLY and len(cur_roots) < 2:
            return
        if cur_phase == PHASE_PREFILLDECODE and len(cur_roots) < 1:
            return
        blocks.append(
            Block(
                phase=cur_phase,
                start_idx=cur_start,
                end_idx=end_idx,
                roots=list(cur_roots),
                details=list(cur_details),
            )
        )

    for i, (root, det, phase) in enumerate(zip(iteration_roots, details, phases)):
        if phase != cur_phase:
            _flush(i)
            cur_phase = phase
            cur_start = i
            cur_roots = []
            cur_details = []
        if phase is not None:
            cur_roots.append(root)
            cur_details.append(det)
    _flush(len(iteration_roots))

    return blocks


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _avg_block_distance(a: Block, b: Block) -> Optional[Tuple[float, ...]]:
    """Return per-step **averaged** priority distance tuple for two same-phase
    blocks of equal step count, or ``None`` if not comparable.

    Averaging makes the distance comparable across blocks of different sizes,
    so the largest-step-count tie-break is meaningful when two pairs have
    similar per-step deviation.
    """
    if a.phase != b.phase or a.num_steps != b.num_steps:
        return None
    n = a.num_steps
    if n == 0:
        return None
    da, db = a.details, b.details

    if a.phase == PHASE_DECODE_ONLY:
        d_g_sq = sum(abs(da[i]["g_sq"] - db[i]["g_sq"]) for i in range(n)) / n
        d_g_sk = sum(abs(da[i]["g_sk"] - db[i]["g_sk"]) for i in range(n)) / n
        return (d_g_sq, d_g_sk)

    if a.phase == PHASE_PREFILLDECODE:
        d_c_req = (
            sum(
                abs(da[i]["context_requests"] - db[i]["context_requests"])
                for i in range(n)
            )
            / n
        )
        d_g_req = (
            sum(
                abs(da[i]["generation_requests"] - db[i]["generation_requests"])
                for i in range(n)
            )
            / n
        )
        d_c_sq = sum(abs(da[i]["c_sq"] - db[i]["c_sq"]) for i in range(n)) / n
        d_g_sq = sum(abs(da[i]["g_sq"] - db[i]["g_sq"]) for i in range(n)) / n
        return (d_c_req, d_g_req, d_c_sq, d_g_sq)

    return None


def select_best_per_phase(
    a_blocks: List[Block], b_blocks: List[Block], phases: List[str]
) -> Tuple[List[dict], List[dict]]:
    """Pick at most one (A, B) pair per phase.

    Selection rule: smallest per-step averaged distance tuple
    (lexicographic — priority order matches the user's stated criteria), with
    ties broken by largest step count.

    Returns ``(matches, notes)`` where ``notes`` collects diagnostic info
    about phases that had no eligible candidate pair.
    """
    matches: List[dict] = []
    notes: List[dict] = []

    # Index B blocks by (phase, num_steps) for fast candidate lookup.
    # Both phases require has_full_sqsk so the distance tuple uses real
    # sq/sk values and does not produce false ties of (0, 0, ...) between
    # blocks that happened to be parsed via the simpler naming patterns.
    b_index: dict = {}
    for j, b in enumerate(b_blocks):
        if b.phase not in phases:
            continue
        if not b.has_full_sqsk():
            continue
        b_index.setdefault((b.phase, b.num_steps), []).append((j, b))

    for phase in phases:
        best = None  # tuple (sort_key, payload)
        a_eligible = 0
        a_with_candidates = 0
        for i, a in enumerate(a_blocks):
            if a.phase != phase:
                continue
            if not a.has_full_sqsk():
                continue
            a_eligible += 1
            candidates = b_index.get((phase, a.num_steps), [])
            if not candidates:
                continue
            a_with_candidates += 1
            for j, b in candidates:
                dist = _avg_block_distance(a, b)
                if dist is None:
                    continue
                # Largest step count wins ties. When blocks have been truncated
                # by --num-steps, prefer the pair whose ORIGINAL blocks were
                # longest (more representative source). Fall back to current
                # num_steps when original is unset.
                a_orig = (
                    a.original_num_steps if a.original_num_steps > 0 else a.num_steps
                )
                b_orig = (
                    b.original_num_steps if b.original_num_steps > 0 else b.num_steps
                )
                tie_key = -min(a_orig, b_orig)
                sort_key = (dist, tie_key, -a.num_steps)
                payload = {
                    "a_block_index": i,
                    "b_block_index": j,
                    "phase": phase,
                    "num_steps": a.num_steps,
                    "distance": [round(x, 6) for x in dist],
                    "a_block": a,
                    "b_block": b,
                }
                if best is None or sort_key < best[0]:
                    best = (sort_key, payload)

        if best is None:
            notes.append(
                {
                    "phase": phase,
                    "reason": (
                        f"no eligible (A, B) pair found "
                        f"(A blocks of phase: {a_eligible}, "
                        f"with same-size B candidate: {a_with_candidates})"
                    ),
                }
            )
        else:
            matches.append(best[1])

    return matches, notes


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def base_name_from_path(p: str) -> str:
    base = os.path.basename(p)
    return (
        base.replace(".pt.trace", "")
        .replace(".json.gz", "")
        .replace(".json", "")
        .replace(".zip", "")
    )


def _block_steps(b: Block) -> List[dict]:
    """Per-step list of {name, num_context, num_generation, c_sq, c_sk, g_sq, g_sk}."""
    out = []
    for step_idx, d in enumerate(b.details):
        out.append(
            {
                "step": b.start_idx + step_idx,
                "name": d.get("name", ""),
                "num_context": d.get("context_requests", 0),
                "num_generation": d.get("generation_requests", 0),
                "c_sq": d.get("c_sq", 0),
                "c_sk": d.get("c_sk", 0),
                "g_sq": d.get("g_sq", 0),
                "g_sk": d.get("g_sk", 0),
            }
        )
    return out


def _block_summary_dict(b: Block) -> dict:
    summary = {f"avg_{k}": round(b.avg(k), 3) for k in PER_STEP_KEYS}
    summary["start"] = b.start_idx
    summary["end"] = b.end_idx
    summary["num_steps"] = b.num_steps
    summary["original_num_steps"] = (
        b.original_num_steps if b.original_num_steps > 0 else b.num_steps
    )
    summary["truncated"] = b.truncated
    return summary


def compute_output_path(
    output_dir: str, phase: str, label: str, base_name: str, block: Block
) -> str:
    """Mirror the naming used by ``extract_and_save`` so the path is known
    even when ``--no-extract`` is set.
    """
    iter_details = block.details
    phase_details = find_phase_from_window(iter_details)
    if len(block.roots) == 1:
        name_append = block.roots[0]["name"].replace("(", "_").replace(")", "")
    else:
        name_append = (
            f"prefilldecode_{phase_details['num_prefilldecode']}_"
            f"decode_{phase_details['num_decode']}_"
            f"bs{phase_details['avg_bs']}_"
            f"conc{phase_details['avg_conc']}"
        )
    return os.path.join(output_dir, phase, f"{label}_{name_append}_{base_name}.json.gz")


def write_reports(
    matches: List[dict],
    notes: List[dict],
    output_dir: str,
    base_a: str,
    base_b: str,
):
    matches_out = []
    rows = []
    for idx, m in enumerate(matches):
        a, b = m["a_block"], m["b_block"]
        entry = {
            "match_id": idx,
            "phase": m["phase"],
            "num_steps": m["num_steps"],
            "distance": m["distance"],
            "TraceA": {
                "block_index": m["a_block_index"],
                "trace": base_a,
                "output_path": m.get("a_output_path"),
                **_block_summary_dict(a),
                "steps": _block_steps(a),
            },
            "TraceB": {
                "block_index": m["b_block_index"],
                "trace": base_b,
                "output_path": m.get("b_output_path"),
                **_block_summary_dict(b),
                "steps": _block_steps(b),
            },
        }
        matches_out.append(entry)

        row = {
            "match_id": idx,
            "phase": m["phase"],
            "num_steps": m["num_steps"],
            "distance": "|".join(str(x) for x in m["distance"]),
        }
        for col, internal, blk, base, out in [
            ("TraceA", "a", a, base_a, m.get("a_output_path")),
            ("TraceB", "b", b, base_b, m.get("b_output_path")),
        ]:
            row[f"{col}_trace"] = base
            row[f"{col}_block_index"] = m[f"{internal}_block_index"]
            row[f"{col}_start"] = blk.start_idx
            row[f"{col}_end"] = blk.end_idx
            row[f"{col}_original_num_steps"] = (
                blk.original_num_steps if blk.original_num_steps > 0 else blk.num_steps
            )
            row[f"{col}_truncated"] = blk.truncated
            row[f"{col}_avg_c_req"] = round(blk.avg("context_requests"), 3)
            row[f"{col}_avg_g_req"] = round(blk.avg("generation_requests"), 3)
            row[f"{col}_avg_c_sq"] = round(blk.avg("c_sq"), 3)
            row[f"{col}_avg_c_sk"] = round(blk.avg("c_sk"), 3)
            row[f"{col}_avg_g_sq"] = round(blk.avg("g_sq"), 3)
            row[f"{col}_avg_g_sk"] = round(blk.avg("g_sk"), 3)
            row[f"{col}_output_path"] = out
        rows.append(row)

    json_path = os.path.join(output_dir, "match_report.json")
    with open(json_path, "w") as f:
        json.dump(matches_out, f, indent=2)
    print(f"Wrote {json_path} ({len(matches_out)} matches)")

    if rows:
        csv_path = os.path.join(output_dir, "match_report.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

    notes_path = os.path.join(output_dir, "match_notes.json")
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"Wrote {notes_path} ({len(notes)} note(s))")


# ---------------------------------------------------------------------------
# Trace loading wrapper
# ---------------------------------------------------------------------------
def load_trace(path: str):
    print(f"\n=== Loading {path} ===")
    trace_json = DataLoader.load_data(get_filename(path))
    events = trace_json.get("traceEvents", [])
    gpu_corr_map, flow_corr_map, meta_events = preprocess_trace(events)
    print(f"Loaded {len(events)} events from {path}")
    iteration_roots = (
        _find_events_by_pattern_quiet(
            events,
            ANNOTATION_PATTERN,
            f"execution steps (iteration) [{os.path.basename(path)}]",
            cat="user_annotation",
        )
        or []
    )
    return {
        "trace_json": trace_json,
        "events": events,
        "gpu_corr_map": gpu_corr_map,
        "flow_corr_map": flow_corr_map,
        "meta_events": meta_events,
        "iteration_roots": iteration_roots,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Match decode_only / prefilldecode execution-root blocks "
        "between two inference traces."
    )
    parser.add_argument("trace_a", help="Path to trace A (.json, .json.gz, .zip)")
    parser.add_argument("trace_b", help="Path to trace B (.json, .json.gz, .zip)")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        default=False,
        help="Skip writing extracted trace files; only emit reports.",
    )
    parser.add_argument(
        "--phases",
        default="decode_only,prefilldecode",
        help="Comma-separated phases to match (default: decode_only,prefilldecode).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=16,
        help=(
            "Window size in execution steps for matching. Blocks longer than "
            "--num-steps are split into non-overlapping windows of exactly "
            "--num-steps steps (any tail shorter than --num-steps is dropped); "
            "blocks already shorter are kept as a single window. Matching and "
            "extraction operate on these windows. Use <=0 to disable. Default: 16."
        ),
    )
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    valid = {PHASE_DECODE_ONLY, PHASE_PREFILLDECODE}
    bad = [p for p in phases if p not in valid]
    if bad:
        print(f"Unknown phase(s): {bad}. Valid: {sorted(valid)}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)

    timings: dict = {}
    overall_t0 = time.perf_counter()

    t0 = time.perf_counter()
    a = load_trace(args.trace_a)
    timings["load_trace_a_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = load_trace(args.trace_b)
    timings["load_trace_b_s"] = time.perf_counter() - t0

    base_a = base_name_from_path(args.trace_a)
    base_b = base_name_from_path(args.trace_b)

    print("\n=== Discovering blocks in A ===")
    t0 = time.perf_counter()
    a_blocks_full = find_blocks(a["iteration_roots"])
    a_blocks = window_blocks(a_blocks_full, args.num_steps)
    timings["find_blocks_a_s"] = time.perf_counter() - t0
    if args.num_steps and args.num_steps > 0:
        print(
            f"  Windowing into non-overlapping chunks of --num-steps={args.num_steps}: "
            f"{len(a_blocks_full)} block(s) -> {len(a_blocks)} window(s)"
        )
    print(f"  A: {len(a_blocks)} window(s)")
    for i, blk in enumerate(a_blocks):
        win = (
            f" (window of {blk.num_steps} from parent of {blk.original_num_steps})"
            if blk.truncated
            else ""
        )
        print(
            f"    [{i}] phase={blk.phase} steps={blk.num_steps} "
            f"range=[{blk.start_idx},{blk.end_idx}){win} "
            f"avg_g_sq={blk.avg('g_sq'):.1f} avg_g_sk={blk.avg('g_sk'):.1f}"
        )

    print("\n=== Discovering blocks in B ===")
    t0 = time.perf_counter()
    b_blocks_full = find_blocks(b["iteration_roots"])
    b_blocks = window_blocks(b_blocks_full, args.num_steps)
    timings["find_blocks_b_s"] = time.perf_counter() - t0
    if args.num_steps and args.num_steps > 0:
        print(
            f"  Windowing into non-overlapping chunks of --num-steps={args.num_steps}: "
            f"{len(b_blocks_full)} block(s) -> {len(b_blocks)} window(s)"
        )
    print(f"  B: {len(b_blocks)} window(s)")
    for j, blk in enumerate(b_blocks):
        win = (
            f" (window of {blk.num_steps} from parent of {blk.original_num_steps})"
            if blk.truncated
            else ""
        )
        print(
            f"    [{j}] phase={blk.phase} steps={blk.num_steps} "
            f"range=[{blk.start_idx},{blk.end_idx}){win} "
            f"avg_g_sq={blk.avg('g_sq'):.1f} avg_g_sk={blk.avg('g_sk'):.1f}"
        )

    print("\n=== Matching (best per phase) ===")
    t0 = time.perf_counter()
    matches, notes = select_best_per_phase(a_blocks, b_blocks, phases)
    timings["match_blocks_s"] = time.perf_counter() - t0
    for m in matches:
        print(
            f"  best {m['phase']}: A[{m['a_block_index']}] <-> B[{m['b_block_index']}] "
            f"steps={m['num_steps']} avg_dist={m['distance']}"
        )
    for n in notes:
        print(f"  no match for phase={n['phase']}: {n['reason']}")

    # Pre-compute deterministic output paths so they appear in the report
    # even when --no-extract is used.
    for m in matches:
        phase = m["phase"]
        phase_dir = os.path.join(args.output_dir, phase)
        label_base = f"{phase}_best_A{m['a_block_index']}_B{m['b_block_index']}"
        m["a_output_path"] = compute_output_path(
            args.output_dir, phase, f"{label_base}_A", base_a, m["a_block"]
        )
        m["b_output_path"] = compute_output_path(
            args.output_dir, phase, f"{label_base}_B", base_b, m["b_block"]
        )

    timings["extract_traces_s"] = 0.0
    timings["extract_a_s"] = 0.0
    timings["extract_b_s"] = 0.0
    if not args.no_extract and matches:
        print("\n=== Extracting matched trace files ===")
        extract_t0 = time.perf_counter()
        for m in matches:
            phase = m["phase"]
            phase_dir = os.path.join(args.output_dir, phase)
            os.makedirs(phase_dir, exist_ok=True)
            label_base = f"{phase}_best_A{m['a_block_index']}_B{m['b_block_index']}"

            print(f"\n--- {phase} best (A) ---")
            t0 = time.perf_counter()
            a_summary = extract_and_save(
                [m["a_block"].roots],
                a["events"],
                a["trace_json"],
                phase_dir,
                base_a,
                "annotation_iteration",
                0,
                1,
                a["gpu_corr_map"],
                a["flow_corr_map"],
                a["meta_events"],
                output_label=f"{label_base}_A",
            )
            timings["extract_a_s"] += time.perf_counter() - t0
            if a_summary:
                m["a_output_path"] = a_summary[0]["output_path"]

            print(f"\n--- {phase} best (B) ---")
            t0 = time.perf_counter()
            b_summary = extract_and_save(
                [m["b_block"].roots],
                b["events"],
                b["trace_json"],
                phase_dir,
                base_b,
                "annotation_iteration",
                0,
                1,
                b["gpu_corr_map"],
                b["flow_corr_map"],
                b["meta_events"],
                output_label=f"{label_base}_B",
            )
            timings["extract_b_s"] += time.perf_counter() - t0
            if b_summary:
                m["b_output_path"] = b_summary[0]["output_path"]
        timings["extract_traces_s"] = time.perf_counter() - extract_t0

    t0 = time.perf_counter()
    write_reports(matches, notes, args.output_dir, base_a, base_b)
    timings["write_reports_s"] = time.perf_counter() - t0

    timings["total_s"] = time.perf_counter() - overall_t0

    print("\n=== Summary ===")
    print(f"  trace A: {os.path.basename(args.trace_a)}  blocks={len(a_blocks)}")
    print(f"  trace B: {os.path.basename(args.trace_b)}  blocks={len(b_blocks)}")
    print(f"  best matches: {len(matches)}  (one per phase)")
    print(f"  output_dir: {args.output_dir}")

    print("\n=== Output files ===")
    print("  reports:")
    print(f"    {os.path.join(args.output_dir, 'match_report.json')}")
    print(f"    {os.path.join(args.output_dir, 'match_report.csv')}")
    print(f"    {os.path.join(args.output_dir, 'match_notes.json')}")
    print(f"    {os.path.join(args.output_dir, 'timings.json')}")
    if matches:
        label = (
            "extracted traces"
            if not args.no_extract
            else "extracted traces (paths only; --no-extract was set)"
        )
        print(f"  {label}:")
        for m in matches:
            print(f"    [{m['phase']}] TraceA: {m.get('a_output_path', '<unknown>')}")
            print(f"    [{m['phase']}] TraceB: {m.get('b_output_path', '<unknown>')}")

    load_total = timings["load_trace_a_s"] + timings["load_trace_b_s"]
    find_total = timings["find_blocks_a_s"] + timings["find_blocks_b_s"]
    total = timings["total_s"]

    def _pct(x: float) -> str:
        return f"{(100.0 * x / total):5.1f}%" if total > 0 else "  n/a "

    print("\n=== Runtime breakdown ===")
    print(f"  load traces:         {load_total:7.2f}s  ({_pct(load_total)})")
    print(f"    - trace A:         {timings['load_trace_a_s']:7.2f}s")
    print(f"    - trace B:         {timings['load_trace_b_s']:7.2f}s")
    print(f"  find blocks:         {find_total:7.2f}s  ({_pct(find_total)})")
    print(f"    - trace A:         {timings['find_blocks_a_s']:7.2f}s")
    print(f"    - trace B:         {timings['find_blocks_b_s']:7.2f}s")
    print(
        f"  match blocks:        {timings['match_blocks_s']:7.2f}s  ({_pct(timings['match_blocks_s'])})"
    )
    print(
        f"  extract traces:      {timings['extract_traces_s']:7.2f}s  ({_pct(timings['extract_traces_s'])})"
    )
    if timings["extract_traces_s"] > 0:
        print(f"    - trace A writes:  {timings['extract_a_s']:7.2f}s")
        print(f"    - trace B writes:  {timings['extract_b_s']:7.2f}s")
    print(
        f"  write reports:       {timings['write_reports_s']:7.2f}s  ({_pct(timings['write_reports_s'])})"
    )
    print(f"  total:               {total:7.2f}s")

    timings_path = os.path.join(args.output_dir, "timings.json")
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
