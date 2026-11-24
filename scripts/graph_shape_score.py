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

The capture tool annotates a Python function that launches kernels with no
torch op as ``<tool>::<module>_<function>_<n>``, recording the arguments its
signature types as tensors (bound with defaults, ``None`` as ``''``). A
launch the model makes inside such a function is scored against that
annotation, the same way.

Usage::

    python scripts/graph_shape_score.py REPLAY.json.gz CAPTURE_DIR \\
        --graph-model-config Qwen/Qwen3-14B-FP8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from TraceLens.TreePerf.graph_schedule import kernel_is
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
    # When the op began: the kernels one call launches share it.
    start: float = 0.0


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
    # One shape miss of each kind: (inferred dims, captured dims).
    examples: dict = field(default_factory=dict)

    def add(self, other: Score) -> None:
        self.kernels += other.kernels
        self.matched += other.matched
        self.exact += other.exact
        self.dimless += other.dimless
        self.unscored_graphs += other.unscored_graphs
        self.misses.update(other.misses)
        for key, example in other.examples.items():
            self.examples.setdefault(key, example)

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
                stack=tuple(_captured_op(op) for op in stack),
            )
        )
    return captured


_ANNOTATED_CALL = re.compile(r"^([\w.]+)\((.*)\)$")
_ANNOTATED_ARG = re.compile(r"arg(\d+):\[([^\]]*)\]")


def _captured_op(event: dict) -> CapturedOp:
    """One op on a launch's stack.

    A Python function annotated with its arguments
    (``pkg.fn(arg0:[4, 8], arg2:[16])``) is the call ``pkg.fn`` with those
    dims; arguments the annotation omits are not tensors.
    """
    args = event.get("args") or {}
    match = _ANNOTATED_CALL.match(event.get("name", ""))
    if match and not args.get("Input Dims"):
        positions = {
            int(index): tuple(int(dim) for dim in dims.split(",") if dim.strip())
            for index, dims in _ANNOTATED_ARG.findall(match.group(2))
        }
        count = max(positions, default=-1) + 1
        return CapturedOp(
            name=match.group(1),
            dims=tuple(positions.get(index, ()) for index in range(count)),
            start=event.get("ts", 0.0),
        )
    return CapturedOp(
        name=event["name"],
        dims=_freeze(args.get("Input Dims")),
        types=_freeze(args.get("Input type")),
        start=event.get("ts", 0.0),
    )


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
    graph: ReplayedGraph,
    captures: list[tuple[str, list[CapturedLaunch]]],
    batch_size: int | None = None,
) -> tuple[str, list[CapturedLaunch]] | None:
    """The capture whose launches the replay repeats, kernel for kernel.

    Captures of several batch sizes can run the same kernels; the one whose
    file names the replay's batch size (``_bs_<n>_``) is the one, and a
    batch with no capture of its own runs the next size up, padded. When the
    capture of the replay's batch launches other kernels, the replay ran for
    another batch, and which of the captures it repeats is not known.
    """
    names = [_replay_key(k.get("name", "")) for k in graph.kernels]
    found = [
        (label, launches)
        for label, launches in captures
        if [launch.kernel for launch in launches] == names
    ]
    if batch_size is None or len(found) <= 1:
        return found[0] if found else None
    sized = [item for item in found if _capture_batch(item[0]) == batch_size]
    if sized:
        return sized[0]
    runners = {label.split("_bs_")[0] for label, _ in found}
    if any(
        _capture_batch(label) == batch_size and label.split("_bs_")[0] in runners
        for label, _ in captures
    ):
        return None
    above = [item for item in found if (_capture_batch(item[0]) or 0) > batch_size]
    return min(above, key=lambda item: _capture_batch(item[0])) if above else None


def _capture_batch(label: str) -> int | None:
    found = re.search(r"_bs_(\d+)_", label)
    return int(found.group(1)) if found else None


def score_graph(
    rows: list[dict | None],
    truth: list[CapturedLaunch],
    launches: list | None = None,
    signatures=None,
) -> Score:
    """Score each replayed kernel's owning row against its captured launch.

    ``rows[i]`` is the row that owns kernel ``i``: a dict with ``name`` and
    ``args`` (``Input Dims``, ``Input type``), or ``None`` when no op was
    inferred for it. ``launches[i]`` is the model launch paired with it, whose
    Python callers are scored against the capture tool's annotations, with
    ``signatures(qualified name)`` giving a function's parameters.
    """
    score = Score(kernels=len(truth))
    launches = launches or [None] * len(truth)
    for row, launch, captured in zip(rows, launches, truth):
        if not captured.has_dims:
            score.dimless += 1
        if row is None:
            score.misses[
                ("unassigned", captured.stack[-1].name if captured.stack else "")
            ] += 1
            continue
        if not captured.has_dims and not any(
            "::" in op.name or "." in op.name for op in captured.stack[1:]
        ):
            # Captured under no op at all: the kernel itself is the match.
            if launch is not None and kernel_is(captured.kernel, launch.kernels):
                score.matched += 1
                continue
        same_name = [op for op in captured.stack if op.name == row.get("name")]
        annotated = [
            (op, caller)
            for op in captured.stack
            for caller in (getattr(launch, "callers", ()) or ())
            if tool_function(caller[0], op.name)
        ]
        if not same_name and not annotated:
            score.misses[
                (row.get("name"), captured.stack[-1].name if captured.stack else "")
            ] += 1
            continue
        score.matched += 1
        if not captured.has_dims:
            continue
        args = row.get("args") or {}
        dims = _freeze(args.get("Input Dims"))
        types = _freeze(args.get("Input type"))
        if any(_same_record(op, dims, types) for op in same_name) or any(
            (op.dims, op.types) == tool_record(caller, signatures)
            for op, caller in annotated
            if signatures is not None
        ):
            score.exact += 1
        else:
            score.misses[(row.get("name"), "shape")] += 1
            score.examples.setdefault(
                row.get("name"),
                (
                    (dims, types),
                    [(op.dims, op.types) for op in same_name]
                    or [(op.dims, op.types) for op, _ in annotated],
                ),
            )
    return score


def _trim(dims) -> tuple:
    dims = list(dims)
    while dims and dims[-1] == ():
        dims.pop()
    return tuple(dims)


def _same_record(op: CapturedOp, dims, types) -> bool:
    """A row's arguments as the captured op records them.

    An annotated call (``fn(arg0:[...], ...)``) records only its tensors'
    dims, positionally: the row's other arguments are empty there.
    """
    if op.types:
        return op.dims == dims and op.types == types
    return _trim(op.dims) == _trim(dims)


_TOOL_NAME = re.compile(r"^\w+::(.+)_\d+$")
_STRING_TENSOR = {
    "torch.Tensor",
    "Tensor",
    "Optional[torch.Tensor]",
    "Optional[Tensor]",
}


def tool_function(qualified: str, op_name: str) -> bool:
    """Whether the capture tool's ``<tool>::<module>_<function>_<n>`` is this call."""
    found = _TOOL_NAME.match(op_name)
    if found is None:
        return False
    parts = qualified.split(".")
    return any(
        found.group(1) == f"{parts[i - 1]}_{parts[-1]}"
        for i in range(len(parts) - 1, 0, -1)
    )


def _tensor_annotation(text: str, postponed: bool) -> bool:
    """Whether the capture tool maps an annotation to ``Tensor``/``Tensor?``."""
    text = " ".join(text.split())
    if postponed:
        return text in _STRING_TENSOR
    if "None" not in text and "Optional" not in text:
        return text in ("torch.Tensor", "Tensor")
    return bool(re.search(r"\b(torch\.)?Tensor\b", text))


def source_signatures(find_module):
    """``signatures(qualified)``: ``[(parameter, tensor-typed, None default)]``.

    ``find_module(module)`` gives a module's source text. ``self`` of a
    method is dropped, as the tool drops it.
    """
    import ast
    import functools

    @functools.lru_cache(maxsize=None)
    def tree(module):
        text = find_module(module)
        return ast.parse(text) if text is not None else None

    @functools.lru_cache(maxsize=None)
    def signatures(qualified):
        parts = qualified.split(".")
        for split in range(len(parts) - 1, 0, -1):
            parsed = tree(".".join(parts[:split]))
            if parsed is None:
                continue
            postponed = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
                for node in parsed.body
            )
            scope, node, in_class = parsed.body, None, False
            for name in parts[split:]:
                in_class = isinstance(node, ast.ClassDef)
                node = next(
                    (
                        item
                        for item in scope
                        if isinstance(item, ast.FunctionDef | ast.ClassDef)
                        and item.name == name
                    ),
                    None,
                )
                if node is None:
                    break
                scope = node.body
            if not isinstance(node, ast.FunctionDef):
                continue
            positional = [*node.args.posonlyargs, *node.args.args]
            defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
                node.args.defaults
            )
            found = []
            for param, default in [
                *zip(positional, defaults),
                *zip(node.args.kwonlyargs, node.args.kw_defaults),
            ]:
                text = (
                    ast.unparse(param.annotation)
                    if param.annotation is not None
                    else ""
                )
                tensor = bool(text) and _tensor_annotation(text, postponed)
                found.append(
                    (
                        param.arg,
                        tensor,
                        isinstance(default, ast.Constant) and default.value is None,
                    )
                )
            if in_class and found and found[0][0] in ("self", "cls"):
                found[0] = (found[0][0], False, False)
            return found
        return None

    return signatures


def tool_record(caller, signatures) -> tuple[tuple, tuple]:
    """``(dims, types)`` the capture tool records for a Python call."""
    name, dims, types, nones, keywords = caller
    signature = signatures(name) or []
    positional = len(dims) - len(keywords)
    by_name = {keyword: positional + index for index, keyword in enumerate(keywords)}
    kept_dims, kept_types = [], []
    for index, (param, tensor, none_default) in enumerate(signature):
        if not tensor:
            continue
        where = index if index < positional else by_name.get(param)
        if where is None or where in nones:
            if where is not None or none_default:
                kept_dims.append(())
                kept_types.append("")
            continue
        kept_dims.append(_freeze(dims[where]))
        kept_types.append(types[where])
    return tuple(kept_dims), tuple(kept_types)


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
    # Collectives too: their launches are inferred like any other.
    for row in analyzer.collect_unified_perf_events(include_nccl=True):
        for uid in row.get("gpu_events") or ():
            owner[uid] = row
    return analyzer, owner


def _paired_launches(analyzer, graph: ReplayedGraph):
    """The model launch paired with each kernel, as the report pairs them."""
    from TraceLens.TreePerf.graph_schedule import align_kernels

    names = [kernel.get("name", "") for kernel in graph.kernels]
    launches = analyzer._launches_for(graph.launch, names)
    if not launches:
        return [None] * len(names), None
    name = analyzer._annotation_for(graph.launch)
    from TraceLens.TreePerf.graph_schedule import invocation_from_annotation

    invocation = invocation_from_annotation(name) if name else None
    return align_kernels(names, launches, analyzer._eager_kernel_families()), invocation


def _module_sources(model):
    """``find(module)``: a module's source text in the engine run's checkouts."""
    from TraceLens.ModelUtils.engine_sources import engine_sources

    if model is None or not model.engine:
        return None
    sources = engine_sources(model.engine, model.gfx_arch, model.engine_version or None)

    def find(module):
        root = sources.resolve(module.partition(".")[0])
        if root is None:
            return None
        base = Path(root) / module.replace(".", "/")
        for path in (base.with_suffix(".py"), base / "__init__.py"):
            if path.is_file():
                return path.read_text(errors="ignore")
        return None

    return find


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
    analyzer, owner = _inferred_rows(
        args.replay,
        args.graph_model_config,
        args.graph_engine_version,
        dict(item.split("=", 1) for item in args.graph_setting if "=" in item),
    )
    find = _module_sources(getattr(analyzer, "graph_model", None))
    signatures = source_signatures(find) if find is not None else None
    per_capture: dict[str, Score] = defaultdict(Score)
    total = Score()
    for graph in replayed_graphs(analyzer.tree.events):
        launches, invocation = _paired_launches(analyzer, graph)
        found = capture_for(
            graph, captures, invocation.batch_size if invocation is not None else None
        )
        if found is None:
            total.unscored_graphs += 1
            rows = [owner.get(kernel.get("UID")) for kernel in graph.kernels]
            shaped = sum(
                1
                for row in rows
                if row is not None and (row.get("args") or {}).get("Input Dims")
            )
            print(
                f"unscored graph of {len(rows)} kernels ({invocation}): {shaped} given an op with shapes"
            )
            continue
        label, truth = found
        rows = [owner.get(kernel.get("UID")) for kernel in graph.kernels]
        score = score_graph(rows, truth, launches, signatures)
        per_capture[label].add(score)
        total.add(score)
    for label in sorted(per_capture):
        print(f"{label}: {per_capture[label].summary()}")
    print(f"TOTAL: {total.summary()}")
    for miss, count in total.misses.most_common(25):
        print(f"  {count:6d}  {miss}")
        if miss[1] == "shape" and miss[0] in total.examples:
            ours, captured = total.examples[miss[0]]
            print(f"            ours     {ours}\n            captured {captured}")
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
