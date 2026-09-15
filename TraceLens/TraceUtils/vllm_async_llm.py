###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""vLLM AsyncLLM frontend tables for the inference performance report.

The GPU-worker report already covers kernels and ``execute_context_*``
iterations. The AsyncLLM / frontend ``torch.profiler`` file is a second,
explicit input: it records request ingress (``_send_input``) and completion
(``process_outputs`` / ``_finish_request``).

Sheets are three independent grains. They do **not** zip send-order to
finish-order: mixed output-sequence-length batches can finish out of arrival
order, so that join is not an identity.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TraceUtils.annotation_utils import (
    IterationAnnotation,
    find_iteration_roots_by_priority,
)
from TraceLens.util import DataLoader

SHEET_INGRESS = "vllm_async_ingress"
SHEET_OUTPUT_GROUPS = "vllm_async_output_groups"
SHEET_ENGINE_STEPS = "vllm_async_engine_steps"

DISPLAY_SEND_INPUT = "_send_input"
DISPLAY_ASSIGN = "assign_request_id"
DISPLAY_CREATE_COMPLETION = "create_completion"
DISPLAY_PROCESS_OUTPUTS = "process_outputs"
DISPLAY_FINISH = "_finish_request"

_INGRESS_COLUMNS = ["event_ord", "t_http_ms", "t_assign_ms", "t_send_ms"]
_OUTPUT_GROUP_COLUMNS = [
    "group",
    "finish_ord_in_group",
    "t_process_outputs_ms",
    "n_finished",
    "t_finish_ms",
]
_ENGINE_STEP_COLUMNS = [
    "engine_step",
    "annotation",
    "t_engine_step_start_ms",
    "engine_step_dur_ms",
    "n_arrivals",
    "n_finished",
    "t_process_outputs_ms",
]


def python_display_name(name: str) -> str:
    """Kineto python_function names are ``path(line): func``; return ``func``."""
    if not name:
        return ""
    if "): " in name:
        return name.rsplit("): ", 1)[-1]
    if "):" in name:
        return name.rsplit("):", 1)[-1].strip()
    return name


def _end_ts(event: dict) -> float:
    return float(event.get("ts", 0) or 0) + float(event.get("dur", 0) or 0)


def _ms(ts: float, t0: float) -> float:
    return round((float(ts) - t0) / 1000.0, 3)


def _empty_df(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def load_chrome_events(path: str) -> List[dict]:
    """Load Chrome ``traceEvents`` from ``.json`` / ``.json.gz``."""
    data = DataLoader.load_data(path)
    if isinstance(data, dict) and "traceEvents" in data:
        return list(data["traceEvents"])
    if isinstance(data, list):
        return data
    raise ValueError(f"No Chrome traceEvents in {path}")


def build_async_llm_tree(events: List[dict]) -> TraceToTree:
    """CPU/python-only tree: keep python_function events, do not prune GPU-less paths."""
    tree = TraceToTree(events, prune_nongpu_paths=False)
    tree.build_tree(add_python_func=True, link_fwd_bwd=False)
    return tree


def events_with_display(
    events: Iterable[dict],
    display: str,
    name_contains: Optional[str] = None,
) -> List[dict]:
    matched = []
    for event in events:
        if event.get("ph") == "M":
            continue
        name = event.get("name") or ""
        if python_display_name(name) != display:
            continue
        if name_contains is not None and name_contains not in name:
            continue
        matched.append(event)
    matched.sort(key=lambda e: e.get("ts", 0) or 0)
    return matched


def http_create_completion_events(events: Iterable[dict]) -> List[dict]:
    """Prefer the API-router HTTP span; fall back to serving.py, then all matches.

    Nested ``_create_completion`` helpers are excluded by exact display match.
    """
    for needle in ("api_router.py", "serving.py"):
        hits = events_with_display(events, DISPLAY_CREATE_COMPLETION, needle)
        if hits:
            return hits
    return events_with_display(events, DISPLAY_CREATE_COMPLETION)


def _descendants_named(tree: TraceToTree, event: dict, display: str) -> List[dict]:
    found: List[dict] = []
    for child in tree.get_children_events(event):
        if python_display_name(child.get("name") or "") == display:
            found.append(child)
        found.extend(_descendants_named(tree, child, display))
    return found


def finish_events_for_process_outputs(tree: TraceToTree, process_outputs: dict) -> List[dict]:
    """``_finish_request`` under one ``process_outputs`` (one engine output message).

    Prefer TraceToTree nesting, then Kineto ``Python parent id``, then the
    process_outputs time window.
    """
    nested = _descendants_named(tree, process_outputs, DISPLAY_FINISH)
    if nested:
        nested.sort(key=lambda e: e.get("ts", 0) or 0)
        return nested

    python_id = (process_outputs.get("args") or {}).get("Python id")
    if python_id is not None:
        by_parent = [
            event
            for event in tree.events
            if python_display_name(event.get("name") or "") == DISPLAY_FINISH
            and (event.get("args") or {}).get("Python parent id") == python_id
        ]
        if by_parent:
            by_parent.sort(key=lambda e: e.get("ts", 0) or 0)
            return by_parent

    start = process_outputs.get("ts", 0) or 0
    end = _end_ts(process_outputs)
    windowed = [
        event
        for event in tree.events
        if python_display_name(event.get("name") or "") == DISPLAY_FINISH
        and start <= (event.get("ts", 0) or 0) <= end
    ]
    windowed.sort(key=lambda e: e.get("ts", 0) or 0)
    return windowed


def vllm_engine_steps(worker_events: Optional[Sequence[dict]]) -> List[dict]:
    """vLLM ``execute_context_*`` roots from the GPU worker trace.

    Drops idle ``0(0)_generation_0(0)`` noise. Keeps decode-only
    ``execute_context_0(0)_generation_N(N)`` steps.
    """
    if not worker_events:
        return []
    roots = find_iteration_roots_by_priority(list(worker_events))
    steps = []
    for event in roots:
        annotation = IterationAnnotation(event.get("name") or "")
        if annotation.kind not in ("vllm_native", "vllm_detailed"):
            continue
        if annotation.context_sum == 0 and annotation.generation_sum == 0:
            continue
        steps.append(event)
    steps.sort(key=lambda e: e.get("ts", 0) or 0)
    return steps


def _pair_process_outputs_to_steps(
    steps: Sequence[dict], process_outputs: Sequence[dict]
) -> List[Optional[dict]]:
    """Greedy: first unused ``process_outputs`` at or after the engine-step start.

    If every frontend timestamp is earlier than the worker (clock skew), fall
    back to positional pairing — one output message per engine step, not a
    per-request identity.
    """
    used = set()
    paired: List[Optional[dict]] = []
    for step in steps:
        match = None
        for event in process_outputs:
            marker = id(event)
            if marker in used:
                continue
            if (event.get("ts", 0) or 0) >= (step.get("ts", 0) or 0):
                match = event
                used.add(marker)
                break
        paired.append(match)
    if steps and process_outputs and all(item is None for item in paired):
        n = min(len(steps), len(process_outputs))
        return list(process_outputs[:n]) + [None] * (len(steps) - n)
    return paired


def build_vllm_async_llm_report_dfs(
    async_llm_trace_path: str,
    worker_events: Optional[Sequence[dict]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build the three AsyncLLM sheets from an explicit frontend trace path."""
    events = load_chrome_events(async_llm_trace_path)
    tree = build_async_llm_tree(events)

    http_events = http_create_completion_events(tree.events)
    assign_events = events_with_display(tree.events, DISPLAY_ASSIGN)
    send_events = events_with_display(tree.events, DISPLAY_SEND_INPUT)
    process_outputs = events_with_display(tree.events, DISPLAY_PROCESS_OUTPUTS)
    steps = vllm_engine_steps(worker_events)

    t0_candidates = []
    for group in (http_events, send_events, steps):
        if group:
            t0_candidates.append(group[0].get("ts", 0) or 0)
    t0 = min(t0_candidates) if t0_candidates else 0.0

    # One row per send. HTTP/assign aligned by arrival order — not a
    # send↔finish identity. assign_request_id often lives on a tokenizer
    # thread, so TraceToTree cannot nest it under _send_input.
    ingress_rows = []
    for i, send in enumerate(send_events):
        http = http_events[i] if i < len(http_events) else None
        assign = assign_events[i] if i < len(assign_events) else None
        ingress_rows.append(
            {
                "event_ord": i + 1,
                "t_http_ms": _ms(http["ts"], t0) if http else None,
                "t_assign_ms": _ms(assign["ts"], t0) if assign else None,
                "t_send_ms": _ms(send.get("ts", 0) or 0, t0),
            }
        )
    df_ingress = (
        pd.DataFrame(ingress_rows, columns=_INGRESS_COLUMNS)
        if ingress_rows
        else _empty_df(_INGRESS_COLUMNS)
    )

    output_rows = []
    for group_i, po in enumerate(process_outputs, start=1):
        finishes = finish_events_for_process_outputs(tree, po)
        t_po = _ms(po.get("ts", 0) or 0, t0)
        if not finishes:
            output_rows.append(
                {
                    "group": group_i,
                    "finish_ord_in_group": None,
                    "t_process_outputs_ms": t_po,
                    "n_finished": 0,
                    "t_finish_ms": None,
                }
            )
            continue
        for finish_i, finish in enumerate(finishes, start=1):
            output_rows.append(
                {
                    "group": group_i,
                    "finish_ord_in_group": finish_i,
                    "t_process_outputs_ms": t_po,
                    "n_finished": len(finishes),
                    "t_finish_ms": _ms(finish.get("ts", 0) or 0, t0),
                }
            )
    df_output = (
        pd.DataFrame(output_rows, columns=_OUTPUT_GROUP_COLUMNS)
        if output_rows
        else _empty_df(_OUTPUT_GROUP_COLUMNS)
    )

    send_ts = [float(e.get("ts", 0) or 0) for e in send_events]
    paired_po = _pair_process_outputs_to_steps(steps, process_outputs)
    step_rows = []
    for i, step in enumerate(steps):
        prev_end = t0 if i == 0 else _end_ts(steps[i - 1])
        this_end = _end_ts(step)
        n_arrivals = sum(1 for ts in send_ts if prev_end < ts <= this_end)
        po = paired_po[i] if i < len(paired_po) else None
        n_finished: Any = None
        t_po = None
        if po is not None:
            finishes = finish_events_for_process_outputs(tree, po)
            n_finished = len(finishes)
            t_po = _ms(po.get("ts", 0) or 0, t0)
        step_rows.append(
            {
                "engine_step": i + 1,
                "annotation": step.get("name"),
                "t_engine_step_start_ms": _ms(step.get("ts", 0) or 0, t0),
                "engine_step_dur_ms": round(float(step.get("dur", 0) or 0) / 1000.0, 3),
                "n_arrivals": n_arrivals,
                "n_finished": n_finished,
                "t_process_outputs_ms": t_po,
            }
        )
    df_steps = (
        pd.DataFrame(step_rows, columns=_ENGINE_STEP_COLUMNS)
        if step_rows
        else _empty_df(_ENGINE_STEP_COLUMNS)
    )

    print(
        "vLLM AsyncLLM sheets: "
        f"{len(df_ingress)} ingress, {len(df_output)} output-group rows, "
        f"{len(df_steps)} engine steps "
        f"(sends={len(send_events)} process_outputs={len(process_outputs)})"
    )
    return {
        SHEET_INGRESS: df_ingress,
        SHEET_OUTPUT_GROUPS: df_output,
        SHEET_ENGINE_STEPS: df_steps,
    }


def maybe_add_vllm_async_llm_sheets(
    dict_name2df: Dict[str, pd.DataFrame],
    *,
    async_llm_trace: Optional[str],
    engine: Optional[str],
    worker_events: Optional[Sequence[dict]],
) -> None:
    """Merge AsyncLLM sheets into the inference report dict, or no-op."""
    if not async_llm_trace:
        return
    if engine is not None and engine != "vllm":
        warnings.warn(
            "--async_llm_trace is only supported when --engine is vllm "
            f"(got {engine!r}); skipping AsyncLLM sheets.",
            UserWarning,
            stacklevel=2,
        )
        return
    if not os.path.isfile(async_llm_trace):
        raise FileNotFoundError(f"--async_llm_trace not found: {async_llm_trace}")
    dict_name2df.update(
        build_vllm_async_llm_report_dfs(async_llm_trace, worker_events=worker_events)
    )
