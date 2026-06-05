###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Link SGLang cuda-graph replay kernels to graph-capture traces for Input Dims.

SGLang traces use ``hipGraphLaunch`` + orphan GPU kernels (no ``cpu_op`` parent).
Capture traces are stored as ``bs_<N>_rank0.json.gz`` files. Replay kernel index
*i* aligns with capture launch *i + offset* (typically offset 2).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

from .trace_capture_merge_experimental import find_closest_batch_size

logger = logging.getLogger(__name__)

GRAPH_LAUNCH_NAMES = frozenset({"hipGraphLaunch", "cudaGraphLaunch"})
CAPTURE_ARGS_KEYS = ("Input Dims", "Input type", "Input Strides", "Concrete Inputs")
DECODE_STEP_RE = re.compile(r"^step\[DECODE bs=(\d+)\]$")
CAPTURE_FILE_RE = re.compile(r"^bs_(\d+)_rank0\.json\.gz$")
DEFAULT_CAPTURE_OFFSET = 2
MAX_OFFSET_SEARCH = 6
DECODE_ANNOTATION_WINDOW_US = 100_000


def load_sglang_capture_batch_sizes(capture_folder: str) -> List[int]:
    """Return sorted batch sizes from ``bs_<N>_rank0.json.gz`` filenames."""
    batch_sizes = []
    for name in os.listdir(capture_folder):
        match = CAPTURE_FILE_RE.match(name)
        if match:
            batch_sizes.append(int(match.group(1)))
    return sorted(set(batch_sizes))


def capture_filepath(capture_folder: str, batch_size: int) -> str:
    return os.path.join(capture_folder, f"bs_{batch_size}_rank0.json.gz")


def _is_runtime_launch(event: dict) -> bool:
    return event.get("cat") in ("cuda_runtime", "cuda_driver") and "Launch" in event.get(
        "name", ""
    )


def _cpu_op_args_before_launch(events: List[dict], launch_event: dict) -> Optional[dict]:
    """Return cpu_op args with Input Dims for a capture launch event."""
    corr = launch_event.get("args", {}).get("correlation")
    if corr is not None:
        for event in events:
            if event.get("cat") != "cpu_op":
                continue
            args = event.get("args", {})
            if not args.get("Input Dims"):
                continue
            if args.get("correlation") == corr:
                return args

    launch_ts = launch_event.get("ts", 0)
    best = None
    best_ts = None
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        args = event.get("args", {})
        if not args.get("Input Dims"):
            continue
        event_ts = event.get("ts", 0)
        if event_ts <= launch_ts and (best_ts is None or event_ts > best_ts):
            best = args
            best_ts = event_ts
    return best


def _load_capture_launch_table(capture_path: str) -> List[dict]:
    """Load ordered capture launches with kernel name and cpu_op args."""
    with gzip.open(capture_path, "rt") as handle:
        events = json.load(handle)["traceEvents"]

    launches = sorted(
        [event for event in events if _is_runtime_launch(event)],
        key=lambda event: event["ts"],
    )
    table = []
    for launch in launches:
        cpu_args = _cpu_op_args_before_launch(events, launch) or {}
        table.append(
            {
                "kernel": launch.get("args", {}).get("kernel", ""),
                "args": {
                    key: cpu_args[key]
                    for key in CAPTURE_ARGS_KEYS
                    if key in cpu_args
                },
            }
        )
    return table


def detect_launch_offset(
    capture_kernels: List[str], replay_kernels: List[str], max_offset: int = MAX_OFFSET_SEARCH
) -> int:
    """Pick offset in [0, max_offset) maximizing kernel-name matches."""
    if not capture_kernels or not replay_kernels:
        return DEFAULT_CAPTURE_OFFSET

    best_offset = DEFAULT_CAPTURE_OFFSET
    best_matches = -1
    for offset in range(max_offset):
        compare_len = min(len(capture_kernels) - offset, len(replay_kernels))
        if compare_len <= 0:
            continue
        matches = sum(
            1
            for idx in range(compare_len)
            if capture_kernels[offset + idx] == replay_kernels[idx]
        )
        if matches > best_matches:
            best_matches = matches
            best_offset = offset
    return best_offset


def index_decode_annotations(events: List[dict]) -> Tuple[List[int], List[int]]:
    """Return parallel sorted lists of decode-step timestamps and batch sizes."""
    timestamps: List[int] = []
    batch_sizes: List[int] = []
    for event in events:
        match = DECODE_STEP_RE.match(event.get("name", ""))
        if not match:
            continue
        timestamps.append(event.get("ts", 0))
        batch_sizes.append(int(match.group(1)))
    paired = sorted(zip(timestamps, batch_sizes), key=lambda pair: pair[0])
    if not paired:
        return [], []
    timestamps, batch_sizes = zip(*paired)
    return list(timestamps), list(batch_sizes)


def find_decode_batch_size_for_graph_launch(
    graph_launch_ts: float,
    decode_timestamps: List[int],
    decode_batch_sizes: List[int],
    window_us: int = DECODE_ANNOTATION_WINDOW_US,
) -> Optional[int]:
    """
    Return decode batch size when *graph_launch_ts* falls within *window_us*
    after a ``step[DECODE bs=N]`` annotation.
    """
    if not decode_timestamps:
        return None

    idx = bisect_right(decode_timestamps, graph_launch_ts) - 1
    if idx < 0:
        return None

    ann_ts = decode_timestamps[idx]
    if graph_launch_ts < ann_ts or graph_launch_ts > ann_ts + window_us:
        return None
    return decode_batch_sizes[idx]


def build_graph_replay_kernel_index(
    tree,
    decode_timestamps: List[int],
    decode_batch_sizes: List[int],
) -> Dict[int, Tuple[dict, int, Optional[int]]]:
    """
    Map replay kernel UID -> (graph_launch_event, kernel_index, decode_batch_size).

    Kernel index is the position in the correlation-ordered kernel list for that
    graph launch (kernels only, sorted by timestamp).
    """
    kernel_index: Dict[int, Tuple[dict, int, Optional[int]]] = {}
    for event in tree.events:
        if event.get("name") not in GRAPH_LAUNCH_NAMES:
            continue

        graph_launch_ts = event.get("ts", 0)
        decode_bs = find_decode_batch_size_for_graph_launch(
            graph_launch_ts, decode_timestamps, decode_batch_sizes
        )
        corr = event.get("args", {}).get("correlation")
        if corr is None:
            replay_kernels = [
                child
                for child in tree.get_gpu_events(event)
                if child.get("cat") == "kernel"
            ]
        else:
            replay_kernels = [
                gpu_event
                for gpu_event in tree.linking_id_to_gpu_events.get(corr, [])
                if gpu_event.get("cat") == "kernel"
            ]
        replay_kernels.sort(key=lambda gpu_event: gpu_event.get("ts", 0))
        for idx, kernel in enumerate(replay_kernels):
            kernel_index[kernel["UID"]] = (event, idx, decode_bs)
    return kernel_index


class SGLangCaptureLinker:
    """Cache capture tables and enrich graph-replay synthetic ops."""

    def __init__(self, capture_folder: str):
        self.capture_folder = capture_folder
        self.capture_batch_sizes = load_sglang_capture_batch_sizes(capture_folder)
        self._launch_tables: Dict[int, List[dict]] = {}
        self._offsets: Dict[int, int] = {}

    def _launch_table(self, padded_bs: int) -> List[dict]:
        if padded_bs not in self._launch_tables:
            path = capture_filepath(self.capture_folder, padded_bs)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            self._launch_tables[padded_bs] = _load_capture_launch_table(path)
        return self._launch_tables[padded_bs]

    def offset_for(self, padded_bs: int, replay_kernels: List[str]) -> int:
        if padded_bs not in self._offsets:
            capture_kernels = [row["kernel"] for row in self._launch_table(padded_bs)]
            self._offsets[padded_bs] = detect_launch_offset(
                capture_kernels, replay_kernels
            )
        return self._offsets[padded_bs]

    def capture_args_for_replay_index(
        self, padded_bs: int, replay_index: int, replay_kernels: List[str]
    ) -> Optional[dict]:
        table = self._launch_table(padded_bs)
        offset = self.offset_for(padded_bs, replay_kernels)
        capture_index = replay_index + offset
        if capture_index < 0 or capture_index >= len(table):
            return None
        return table[capture_index]["args"] or None


def enrich_synthetic_ops_from_sglang_capture(
    collected: List[dict],
    tree,
    capture_folder: str,
    synthetic_op_marker: str = " (Synthetic Op)",
) -> int:
    """
    Copy capture ``Input Dims`` (and related args) onto graph-replay synthetic ops.

    Returns the number of synthetic ops enriched.
    """
    if not capture_folder or not os.path.isdir(capture_folder):
        return 0

    capture_batch_sizes = load_sglang_capture_batch_sizes(capture_folder)
    if not capture_batch_sizes:
        logger.warning(
            "SGLang capture folder %s has no bs_*_rank0.json.gz files",
            capture_folder,
        )
        return 0

    decode_timestamps, decode_batch_sizes = index_decode_annotations(tree.events)
    kernel_index = build_graph_replay_kernel_index(
        tree, decode_timestamps, decode_batch_sizes
    )
    linker = SGLangCaptureLinker(capture_folder)

    replay_sequences: Dict[int, List[str]] = {}
    for _kernel_uid, (graph_launch, _idx, decode_bs) in kernel_index.items():
        if decode_bs is None:
            continue
        padded_bs = find_closest_batch_size(decode_bs, capture_batch_sizes)
        if padded_bs is None:
            continue
        corr = graph_launch.get("args", {}).get("correlation")
        if corr is None:
            kernels = [
                child
                for child in tree.get_gpu_events(graph_launch)
                if child.get("cat") == "kernel"
            ]
        else:
            kernels = [
                gpu_event
                for gpu_event in tree.linking_id_to_gpu_events.get(corr, [])
                if gpu_event.get("cat") == "kernel"
            ]
        kernels.sort(key=lambda gpu_event: gpu_event.get("ts", 0))
        replay_sequences.setdefault(padded_bs, [kernel["name"] for kernel in kernels])

    enriched = 0
    for event in collected:
        if synthetic_op_marker not in event.get("name", ""):
            continue
        if event.get("args", {}).get("Input Dims"):
            continue

        parent_name = event.get("name", "").split("->", 1)[0]
        if parent_name not in GRAPH_LAUNCH_NAMES:
            continue

        gpu_uids = event.get("gpu_events") or []
        if len(gpu_uids) != 1:
            continue
        mapping = kernel_index.get(gpu_uids[0])
        if mapping is None:
            continue

        _graph_launch, replay_idx, decode_bs = mapping
        if decode_bs is None:
            continue

        padded_bs = find_closest_batch_size(decode_bs, capture_batch_sizes)
        if padded_bs is None:
            continue

        replay_kernels = replay_sequences.get(padded_bs, [])
        capture_args = linker.capture_args_for_replay_index(
            padded_bs, replay_idx, replay_kernels
        )
        if not capture_args:
            continue

        event.setdefault("args", {}).update(capture_args)
        enriched += 1

    if enriched:
        logger.info(
            "SGLang capture linker enriched %d synthetic ops from %s",
            enriched,
            capture_folder,
        )
    return enriched
