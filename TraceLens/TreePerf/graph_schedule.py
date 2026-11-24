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


from TraceLens.ModelUtils.graph_launches import GraphModel, Invocation, Launch
from TraceLens.ModelUtils.kernel_alignment import (
    align_kernels,
    kernel_family,
    kernel_is,
)
from TraceLens.TraceUtils.utils.annotation_utils import parse_annotation

__all__ = [
    "GraphModel",
    "Invocation",
    "Launch",
    "align_kernels",
    "kernel_family",
    "kernel_is",
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
