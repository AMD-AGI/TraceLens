#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Score the shapes inferred for graph-replayed kernels against capture traces.

A graph replay records only device kernels. The inference report gives each
one the op and the input shapes the model says it was called with. A
graph-capture trace of the same graph records the real call: every launch sits
under the CPU ops that issued it, with ``Input Dims`` and ``Input type``.

The capture trace is the answer key here and nothing else reads it. A replay
is paired with the capture whose launch sequence it repeats kernel for
kernel, and every replayed kernel is scored:

- *matched*: the row that owns the kernel is named as one of the ops on the
  captured launch's stack;
- *exact*: that op's captured dims and dtypes also equal the row's.

A captured launch whose stack records no dims counts toward *matched* only.

Usage::

    python scripts/graph_shape_score.py REPLAY.json.gz CAPTURE_DIR \\
        --graph-model-config Qwen/Qwen3-14B-FP8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from TraceLens.util import DataLoader

_GRAPH_LAUNCHES = ("hipGraphLaunch", "cudaGraphLaunch")
_COPY_LAUNCHES = ("hipMemcpyAsync", "cudaMemcpyAsync")
_LAUNCH_APIS = _COPY_LAUNCHES + ("hipMemsetAsync", "cudaMemsetAsync")
_GPU_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
# A replayed graph runs a captured async copy as this device kernel.
_REPLAYED_COPY = "copyBuffer"


@dataclass(frozen=True)
class CapturedOp:
    """One CPU op or annotation enclosing a captured launch."""

    name: str
    dims: tuple = ()
    types: tuple = ()


@dataclass(frozen=True)
class CapturedLaunch:
    """One launch recorded between stream begin- and end-capture."""

    kernel: str
    stack: tuple[CapturedOp, ...]  # outermost first

    @property
    def has_dims(self) -> bool:
        return any(op.dims for op in self.stack)


@dataclass
class ReplayedGraph:
    """One graph launch of the replay and the kernels it ran, in order."""

    launch: dict
    kernels: list[dict]


@dataclass
class Score:
    kernels: int = 0
    matched: int = 0
    exact: int = 0
    # Captured launches that record no dims: they score on the match only.
    dimless: int = 0
    unscored_graphs: int = 0
    misses: Counter = field(default_factory=Counter)

    def add(self, other: Score) -> None:
        self.kernels += other.kernels
        self.matched += other.matched
        self.exact += other.exact
        self.dimless += other.dimless
        self.unscored_graphs += other.unscored_graphs
        self.misses.update(other.misses)

    def summary(self) -> str:
        shaped = self.kernels - self.dimless
        return (
            f"kernels={self.kernels} matched={self.matched} "
            f"({_pct(self.matched, self.kernels)}) exact={self.exact}/{shaped} "
            f"({_pct(self.exact, shaped)}) unscored_graphs={self.unscored_graphs}"
        )


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def _events(path: str) -> list[dict]:
    return DataLoader.load_data(path)["traceEvents"]


def _encloses(outer: dict, inner: dict) -> bool:
    return outer["ts"] <= inner["ts"] and inner["ts"] <= outer["ts"] + outer["dur"]


def _kernel_key(name: str) -> str:
    """The name a replayed kernel and its captured launch share."""
    return _REPLAYED_COPY if name in _COPY_LAUNCHES else name


def capture_launches(path: str) -> list[CapturedLaunch]:
    """Every launch the graph captured, with the ops that issued it."""
    events = _events(path)
    runtime = [e for e in events if e.get("cat") == "cuda_runtime"]
    begins = [e["ts"] for e in runtime if "BeginCapture" in e.get("name", "")]
    ends = [e["ts"] for e in runtime if "EndCapture" in e.get("name", "")]
    start = min(begins) if begins else float("-inf")
    stop = max(ends) if ends else float("inf")
    launches = sorted(
        (
            e
            for e in runtime
            if start <= e["ts"] <= stop
            and ("kernel" in (e.get("args") or {}) or e.get("name") in _LAUNCH_APIS)
        ),
        key=lambda e: e["ts"],
    )
    callers = defaultdict(list)
    for event in events:
        if event.get("cat") in ("cpu_op", "user_annotation") and "dur" in event:
            callers[event.get("tid")].append(event)
    captured = []
    for launch in launches:
        stack = sorted(
            (c for c in callers[launch.get("tid")] if _encloses(c, launch)),
            key=lambda c: (c["ts"], -c["dur"]),
        )
        args = launch.get("args") or {}
        captured.append(
            CapturedLaunch(
                kernel=_kernel_key(args.get("kernel") or launch["name"]),
                stack=tuple(
                    CapturedOp(
                        name=op["name"],
                        dims=_freeze((op.get("args") or {}).get("Input Dims")),
                        types=_freeze((op.get("args") or {}).get("Input type")),
                    )
                    for op in stack
                ),
            )
        )
    return captured


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value if value is not None else ()


def replayed_graphs(events: list[dict]) -> list[ReplayedGraph]:
    """Each graph launch of a replay trace with its kernels in launch order."""
    by_correlation = defaultdict(list)
    for event in events:
        if event.get("cat") in _GPU_CATEGORIES:
            by_correlation[(event.get("args") or {}).get("correlation")].append(event)
    graphs = []
    for event in events:
        if event.get("cat") == "cuda_runtime" and event.get("name") in _GRAPH_LAUNCHES:
            kernels = by_correlation.get((event.get("args") or {}).get("correlation"))
            kernels = sorted(kernels or [], key=lambda k: k["ts"])
            graphs.append(ReplayedGraph(event, kernels))
    return sorted(graphs, key=lambda g: g.launch["ts"])


def _replay_key(name: str) -> str:
    return _REPLAYED_COPY if _REPLAYED_COPY in name else name


def load_captures(capture_dir: str) -> list[tuple[str, list[CapturedLaunch]]]:
    captures = []
    for path in sorted(Path(capture_dir).glob("*.json.gz")):
        launches = capture_launches(str(path))
        # Launches before begin-capture are not graph nodes; the window
        # already excludes them.
        captures.append((path.name, launches))
    return captures


def capture_for(
    graph: ReplayedGraph, captures: list[tuple[str, list[CapturedLaunch]]]
) -> tuple[str, list[CapturedLaunch]] | None:
    """The capture whose launches the replay repeats, kernel for kernel."""
    names = [_replay_key(k.get("name", "")) for k in graph.kernels]
    for label, launches in captures:
        if [launch.kernel for launch in launches] == names:
            return label, launches
    return None


def score_graph(rows: list[dict | None], truth: list[CapturedLaunch]) -> Score:
    """Score each replayed kernel's owning row against its captured launch.

    ``rows[i]`` is the row that owns kernel ``i``: a dict with ``name`` and
    ``args`` (``Input Dims``, ``Input type``), or ``None`` when no op was
    inferred for it.
    """
    score = Score(kernels=len(truth))
    for row, launch in zip(rows, truth):
        if not launch.has_dims:
            score.dimless += 1
        if row is None:
            score.misses[
                ("unassigned", launch.stack[-1].name if launch.stack else "")
            ] += 1
            continue
        same_name = [op for op in launch.stack if op.name == row.get("name")]
        if not same_name:
            score.misses[
                (row.get("name"), launch.stack[-1].name if launch.stack else "")
            ] += 1
            continue
        score.matched += 1
        if not launch.has_dims:
            continue
        args = row.get("args") or {}
        dims = _freeze(args.get("Input Dims"))
        types = _freeze(args.get("Input type"))
        if any(op.dims == dims and op.types == types for op in same_name):
            score.exact += 1
        else:
            score.misses[(row.get("name"), "shape")] += 1
    return score


def _inferred_rows(
    replay_path: str, checkpoint: str | None, engine_version: str, settings: dict
) -> tuple[list[dict], dict[int, dict]]:
    """The replay's events and the inference report's row for each kernel UID."""
    from TraceLens.Reporting import generate_perf_report_pytorch_inference as report
    from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer

    analyzer = TreePerfAnalyzer.from_file(profile_filepath=replay_path)
    analyzer.tree.apply_annotation(
        name_filters=report._INFERENCE_ANNOTATION_OP_PREFIXES
    )
    report.attach_graph_model(
        analyzer, checkpoint, engine_version=engine_version, settings=settings
    )
    owner = {}
    for row in analyzer.collect_unified_perf_events():
        for uid in row.get("gpu_events") or ():
            owner[uid] = row
    return analyzer.tree.events, owner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replay", help="graph-replay trace (.json or .json.gz)")
    parser.add_argument("capture_dir", help="directory of graph-capture traces")
    parser.add_argument("--graph-model-config", default=None)
    parser.add_argument("--graph-engine-version", default="")
    parser.add_argument(
        "--graph-setting", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument("--json", default=None, help="write the scores here")
    args = parser.parse_args(argv)

    captures = load_captures(args.capture_dir)
    events, owner = _inferred_rows(
        args.replay,
        args.graph_model_config,
        args.graph_engine_version,
        dict(item.split("=", 1) for item in args.graph_setting if "=" in item),
    )
    per_capture: dict[str, Score] = defaultdict(Score)
    total = Score()
    for graph in replayed_graphs(events):
        found = capture_for(graph, captures)
        if found is None:
            total.unscored_graphs += 1
            continue
        label, truth = found
        rows = [owner.get(kernel.get("UID")) for kernel in graph.kernels]
        score = score_graph(rows, truth)
        per_capture[label].add(score)
        total.add(score)
    for label in sorted(per_capture):
        print(f"{label}: {per_capture[label].summary()}")
    print(f"TOTAL: {total.summary()}")
    for miss, count in total.misses.most_common(25):
        print(f"  {count:6d}  {miss}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    label: vars(s) | {"misses": [[*k, v] for k, v in s.misses.items()]}
                    for label, s in {**per_capture, "TOTAL": total}.items()
                },
                indent=1,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
