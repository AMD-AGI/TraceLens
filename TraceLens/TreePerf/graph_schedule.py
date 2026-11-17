###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pair the kernels of a replayed graph with the launches the model issues.

A replayed graph records each device kernel and no CPU op, so its kernels
have no shapes. :mod:`TraceLens.ModelUtils.graph_launches` reads the launches
of one forward from the served model's source. This module reads what the
trace says about the run (the step annotation, the process groups, the
device) and pairs that launch list with the replayed kernels in order.
"""

from __future__ import annotations

import re

from TraceLens.ModelUtils.graph_launches import GraphModel, Invocation, Launch
from TraceLens.TraceUtils.utils.annotation_utils import parse_annotation

__all__ = [
    "GraphModel",
    "Invocation",
    "Launch",
    "align_kernels",
    "covering_annotations",
    "group_stopped_ops",
    "invocation_from_annotation",
    "launch_arguments",
    "tp_from_trace",
]


def tp_from_trace(metadata: dict | None) -> int | None:
    """Tensor-parallel degree recorded on the trace's ``distributedInfo``.

    The ``tp`` process group (``tp:device`` and its ``tp:cpu`` twin). A trace
    whose only world has one rank is not sharded. A trace with no such group,
    or whose device and cpu groups disagree, does not say.
    """
    from TraceLens.ModelUtils.parallel_plan import parallel_degree_from_trace

    degree = parallel_degree_from_trace(metadata)
    if degree is None:
        return None
    if "tp" in degree.groups:
        return degree.groups["tp"]
    if degree.groups.get("world") == 1:
        return 1
    return None


def invocation_from_annotation(
    name: str, scope: tuple[str, ...] = ()
) -> Invocation | None:
    """Batch, sequence length, and past length recorded on a step annotation.

    Decode query length is one token per request. The past length is the mean
    cached length, ``g_sk / batch``, because the annotation stores the sum.
    Prefill tokens are the context sum, and the past length is zero.
    """
    ann = parse_annotation(name)
    if not ann.matched:
        return None
    if ann.context_requests and not ann.generation_requests:
        batch = ann.context_requests or ann.batch_size or 1
        tokens = ann.context_sum or ann.c_sq or batch
        seq = tokens // batch if batch else tokens
        # c_sk sums the KV length. Whatever exceeds the query is already cached.
        kv = ann.c_sk or tokens
        past = max(kv - tokens, 0) // batch if batch else 0
        return Invocation(batch, seq, past, tokens, prefill=True, scope=scope)
    batch = ann.generation_requests or ann.batch_size or 1
    tokens = ann.g_sq or batch
    past = (ann.g_sk // batch) if batch and ann.g_sk else 0
    return Invocation(batch, 1, past, tokens, scope=scope)


def covering_annotations(events: list[dict], launch: dict) -> list[dict]:
    """The annotations whose time span contains ``launch``, outermost first."""
    start = launch.get("ts")
    if start is None:
        return []
    end = start + (launch.get("dur") or 0)
    found = []
    for event in events:
        ev_start = event.get("ts")
        if ev_start is None or event.get("tid") != launch.get("tid"):
            continue
        if ev_start <= start and end <= ev_start + (event.get("dur") or 0):
            found.append(event)
    return sorted(found, key=lambda event: (event["ts"], -(event.get("dur") or 0)))


def _identifier(symbol: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")


_SYMBOL_PATTERNS: dict[str, re.Pattern] = {}


def kernel_is(kernel_name: str, symbols: tuple[str, ...]) -> bool:
    """Whether ``kernel_name`` is one of the device functions ``symbols`` names.

    A device symbol carries its function name as one identifier
    (``void ns::name<...>(...)``, a Triton function's own name), so a symbol
    matches only as a whole identifier, never as a substring of a longer one.
    """
    for symbol in symbols:
        pattern = _SYMBOL_PATTERNS.get(symbol)
        if pattern is None:
            pattern = _SYMBOL_PATTERNS[symbol] = _identifier(symbol)
        if pattern.search(kernel_name):
            return True
    return False


# Alignment scores. A kernel the launch's source names outweighs any number of
# launches whose source names no kernel, so an anchored kernel is never given
# up to place unanchored launches around it.
_NAMED = 1000
_UNNAMED = 1
_REPEAT = 1


def align_kernels(
    kernel_names: list[str], launches: list[Launch]
) -> list[Launch | None]:
    """Pair replayed kernels with predicted launches, both in launch order.

    A launch whose source names its kernels takes only a kernel that is one
    of them, and may take several in a row (a split reduction, a second
    pass). A launch whose source names none takes exactly one kernel that no
    named launch would claim. Launches and kernels may both go unpaired; the
    pairing kept is the one that pairs the most, named kernels first.
    """
    count_k = len(kernel_names)
    count_l = len(launches)
    symbol_sets = {launch.kernels for launch in launches if launch.kernels}
    distinct = set(kernel_names)
    is_symbol = {
        (name, symbols): kernel_is(name, symbols)
        for name in distinct
        for symbols in symbol_sets
    }
    claimed = {
        name: any(is_symbol[(name, symbols)] for symbols in symbol_sets)
        for name in distinct
    }

    def pairs(launch: Launch, name: str) -> bool:
        if launch.kernels:
            return is_symbol[(name, launch.kernels)]
        return not claimed[name]

    neg = float("-inf")
    width = count_k + 1
    # Back-pointers, one byte per cell: bits 0-1 the move out of ``best``
    # (0 = pair, 1 = skip launch, 2 = skip kernel), bit 2 set when the pair
    # extends the launch's previous kernel.
    moves = bytearray((count_l + 1) * width)
    best_prev = [0.0] * width
    for j in range(1, width):
        moves[j] = 2
    for i in range(1, count_l + 1):
        launch = launches[i - 1]
        gain = _NAMED if launch.kernels else _UNNAMED
        best_row = [0.0] * width
        open_row = [neg] * width
        best_row[0] = best_prev[0]
        moves[i * width] = 1
        for j in range(1, width):
            cell = i * width + j
            flags = 0
            if pairs(launch, kernel_names[j - 1]):
                first = best_prev[j - 1] + gain
                again = open_row[j - 1] + _REPEAT if launch.kernels else neg
                if again > first:
                    open_row[j] = again
                    flags = 4
                else:
                    open_row[j] = first
            # On a tie, leave the later kernel unpaired: walking back from the
            # end, that pairs each launch with the earliest kernel it fits.
            choice, move = best_row[j - 1], 2
            if open_row[j] > choice:
                choice, move = open_row[j], 0
            if best_prev[j] > choice:
                choice, move = best_prev[j], 1
            best_row[j] = choice
            moves[cell] = move | flags
        best_prev = best_row
    assigned: list[Launch | None] = [None] * count_k
    i, j = count_l, count_k
    in_pair = False
    while i > 0 and j > 0:
        cell = moves[i * width + j]
        if not in_pair:
            move = cell & 3
            if move == 1:
                i -= 1
                continue
            if move == 2:
                j -= 1
                continue
        assigned[j - 1] = launches[i - 1]
        if cell & 4:
            in_pair = True
            j -= 1
        else:
            in_pair = False
            i, j = i - 1, j - 1
    return assigned


def group_stopped_ops(
    assigned: list[Launch | None],
) -> list[tuple[int, list[int]]]:
    """Group kernels the collector keeps on one CPU op.

    Each entry is ``(owner, members)``. The owner is the kernel whose launch
    the row carries. Consecutive kernels of one launch, and a launch marked
    ``same_call``, share the row of the launch before them.
    """
    groups: list[tuple[int, list[int]]] = []
    for index, launch in enumerate(assigned):
        previous = assigned[index - 1] if index else None
        joins = (
            groups
            and launch is not None
            and previous is not None
            and (launch is previous or launch.same_call)
        )
        if joins:
            groups[-1][1].append(index)
        else:
            groups.append((index, [index]))
    return groups


def _row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Contiguous strides of ``shape``. An empty shape is a scalar argument."""
    stride = [1] * len(shape)
    step = 1
    for index in range(len(shape) - 2, -1, -1):
        step *= shape[index + 1]
        stride[index] = step
    return tuple(stride)


def launch_arguments(launch: Launch) -> dict:
    """The argument record a captured call of ``launch`` would carry.

    The roofline reads ``Input Dims``, ``Input type`` and ``Input Strides``
    from the CPU op. A replayed graph has no CPU op, so the launch supplies
    the same fields.
    """
    count = len(launch.input_shapes)
    if launch.input_strides:
        strides = [list(stride) for stride in launch.input_strides]
    else:
        strides = [list(_row_major(shape)) for shape in launch.input_shapes]
    record = {
        "Input Dims": [list(shape) for shape in launch.input_shapes],
        "Input type": list(launch.input_types[:count]),
        "Input Strides": strides,
    }
    if launch.concrete:
        record["Concrete Inputs"] = list(launch.concrete)
    return record
