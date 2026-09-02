###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stage 3: preprocess traces, extract iteration windows, and write output files."""

import gzip
import json
import os
import zipfile
from bisect import bisect_left, bisect_right

from tqdm import tqdm

from ..annotation_utils import (
    ITERATION_BACKUP_PATTERNS,
    ITERATION_PATTERNS,
    find_phase_from_window,
    has_context,
    has_generation,
    is_decode_only,
    iteration_details,
)

from .detect_utils import (
    GPU_KERNEL_CATEGORIES,
    PROJECTION_CATEGORY,
    build_root_tiles,
)

# Kernels plus the annotation projections that describe them. Anything summing
# GPU *time* must use GPU_KERNEL_CATEGORIES instead, since a projection encloses
# the kernels it describes and counting both double-counts.
GPU_EVENT_CATEGORIES = [*GPU_KERNEL_CATEGORIES, PROJECTION_CATEGORY]


def get_filename(filepath: str) -> dict:
    """Load trace JSON from file (.json, .json.gz, or .zip)."""
    print(f"Loading trace: {filepath}")
    if filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zf:
            # Find the JSON file inside the zip
            json_files = [f for f in zf.namelist() if f.endswith(".json")]
            if not json_files:
                raise ValueError(f"No .json file found in {filepath}")
            json_file = json_files[0]
            print(f"  Reading {json_file} from zip...")
            return json_file
    return filepath


def preprocess_trace(events: list[dict]):
    gpu_corr_map = {}
    flow_corr_map = {}
    meta_events = []
    for e in tqdm(events):
        ts = e.get("ts")
        ph = e.get("ph")
        cat = e.get("cat")
        if ts is None:
            meta_events.append(e)
            continue
        if ph in ("s", "f"):
            corr = e.get("id")
            if corr is not None:
                flow_corr_map.setdefault(corr, []).append(e)
            continue
        if cat in GPU_EVENT_CATEGORIES:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                gpu_corr_map.setdefault(corr, []).append(e)
            continue
    return gpu_corr_map, flow_corr_map, meta_events


def build_cpu_event_index(
    events: list[dict],
) -> tuple[list[dict], list[float]]:
    """Pre-filter and sort CPU-side events for extraction.

    Returns ``(cpu_events, cpu_starts)`` -- the sorted event list and a parallel
    list of timestamps for bisect lookups.  Build once, pass to every
    :func:`extract_iteration` call on the same trace.
    """
    cpu_events = []
    for e in events:
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        if e.get("cat") in GPU_EVENT_CATEGORIES:
            continue
        cpu_events.append(e)
    cpu_events.sort(key=lambda e: e["ts"])
    cpu_starts = [e["ts"] for e in cpu_events]
    return cpu_events, cpu_starts


def extract_iteration(
    iteration_roots: list[dict],
    events: list[dict],
    trace_json: dict,
    gpu_corr_map: dict,
    flow_corr_map: dict,
    meta_events: list[dict],
    root_tiles: dict | None = None,
    gap_fill: bool = True,
    cpu_event_index: tuple[list[dict], list[float]] | None = None,
) -> dict:
    """Extract a single iteration trace.

    Events are assigned to the window containing their *start* timestamp, which
    makes the windows a partition of the timeline. Testing for full containment
    instead would drop any event straddling a boundary from both neighbours, so
    closing the gaps alone would not stop kernels going missing.

    Events longer than their window are enclosing spans -- thread roots, outer
    python frames -- which belong to no single iteration and are left out. They
    carry no correlation id, so no kernel is lost with them.

    ``root_tiles`` should come from :func:`build_root_tiles` over the *whole* root
    list: built from a selected window instead, the window's last root would lose
    the boundary of the root that follows it. Pass ``gap_fill=False`` to score
    each root by its own span.

    ``cpu_event_index`` is an optional ``(cpu_events, cpu_starts)`` tuple from
    :func:`build_cpu_event_index`.  When provided the expensive full-event scan
    and sort is skipped -- pass it when calling in a loop over many roots.
    """

    filtered_events = []
    gpu_dur = 0
    gpu_busy = 0
    num_gpu_events = 0
    batch_list = []

    if not iteration_roots:
        return trace_json.copy(), [], 0, 0, 0

    if not gap_fill:
        windows = [
            (r.get("ts", 0), r.get("ts", 0) + r.get("dur", 0)) for r in iteration_roots
        ]
    else:
        tiles = (
            root_tiles
            if root_tiles is not None
            else build_root_tiles(iteration_roots)[0]
        )
        windows = [
            tiles.get(
                (r.get("pid"), r.get("tid"), r.get("ts", 0)),
                (r.get("ts", 0), r.get("ts", 0) + r.get("dur", 0)),
            )
            for r in iteration_roots
        ]

    min_iter_ts = min(start for start, _ in windows)
    max_iter_end = max(end for _, end in windows)
    if root_tiles:
        max_iter_end = max(max_iter_end, max(end for _, end in root_tiles.values()))

    if cpu_event_index is not None:
        cpu_events, cpu_starts = cpu_event_index
    else:
        cpu_events, cpu_starts = build_cpu_event_index(events)

    # For each iteration window collect the CPU events whose start falls in it,
    # regardless of thread, then follow their correlation ids to the GPU work.
    for iteration_root, (win_ts, win_end) in zip(tqdm(iteration_roots), windows):
        start_time = []
        end_time = []
        win_dur = win_end - win_ts
        is_last = win_end == max_iter_end

        correlation_ids: set[int] = set()

        # Bisect to the window's slice so widening to all threads stays cheap.
        lo = bisect_left(cpu_starts, win_ts)
        hi = bisect_right(cpu_starts, win_end)
        for e in cpu_events[lo:hi]:
            ts = e["ts"]
            # Half-open so neighbouring windows cannot both claim an event; the
            # final window is closed so nothing at the very end is orphaned.
            within = win_ts <= ts < win_end or (is_last and ts == win_end)
            # Spans longer than the window are enclosing frames (thread roots,
            # outer python frames) that belong to no single iteration; they carry
            # no correlation id, so no kernel is lost by skipping them.
            if within and e["dur"] <= win_dur:
                filtered_events.append(e)
                corr = e.get("args", {}).get("correlation")
                if corr is not None:
                    correlation_ids.add(corr)

        # Add matching flow events
        for corr in correlation_ids:
            filtered_events.extend(flow_corr_map.get(corr, []))
        # Add matching GPU events
        for corr in correlation_ids:
            for e in gpu_corr_map.get(corr, []):
                filtered_events.append(e)
                start_time.append(e.get("ts"))
                end_time.append(e.get("ts") + e.get("dur"))
                gpu_busy += e.get("dur")
                num_gpu_events += 1
        gpu_dur += max(end_time) - min(start_time) if start_time else 0

    # Add all meta events (no timestamp)
    filtered_events.extend(meta_events)

    for e in tqdm(filtered_events):
        if "vllm::unified_attention_with_output" in e.get(
            "name", ""
        ) or "sgl_kernel::sgl_per_token_group_quant_8bit" in e.get("name", ""):
            dims = e.get("args", {}).get("Input Dims")
            if dims and len(dims) > 0 and len(dims[0]) > 0:
                batch_list.append(dims[0][0])
    # Create output trace
    output = trace_json.copy()
    output["traceEvents"] = filtered_events
    return output, list(set(batch_list)), num_gpu_events, gpu_dur, gpu_busy


def parse_range(range_str: str, max_len: int) -> tuple[int, int]:
    """Parse a range string like '10:20' or 'all'."""
    if range_str == "all":
        return 0, max_len
    parts = range_str.split(":")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start + 1
    return start, min(end, max_len)


def extract_and_save(
    roots: list[list[dict]],
    events: list[dict],
    trace_json: dict,
    output_dir: str,
    base_name: str,
    prefix: str,
    start: int,
    end: int,
    gpu_corr_map: dict,
    flow_corr_map: dict,
    meta_events: list[dict],
    output_label: str | None = None,
    root_tiles: dict | None = None,
):
    """Extract and save a range of iterations.

    If ``output_label`` is provided the output filename becomes
    ``{output_label}_{name_append}_{base_name}.json.gz`` instead of the
    default ``{base_name}_{prefix}_{idx}_{name_append}.json.gz``.

    ``root_tiles`` should be built over the whole root list so that a root at the
    edge of a selected window still knows where its successor begins.
    """
    extraction_summary = []
    if len(roots) == 0 or len(roots[0]) == 0:
        print(f"No {prefix} events found in the specified range, skipping extraction")
        return extraction_summary
    selected = roots[start:end]
    indices = range(start, end)
    if len(selected) == 0:
        print(f"No {prefix} events found in the specified range, skipping extraction")
        return extraction_summary
    cpu_idx = build_cpu_event_index(events)
    for idx, root in zip(indices, selected):
        iter_details = iteration_details(root)
        iter_trace, batch_list, num_gpu_events, gpu_dur, gpu_busy = extract_iteration(
            root,
            events,
            trace_json,
            gpu_corr_map,
            flow_corr_map,
            meta_events,
            root_tiles=root_tiles,
            cpu_event_index=cpu_idx,
        )
        is_annotation = "annotation_iteration" in prefix
        # Use the structured phase-aware name for any annotation extraction
        # produced by the steady-state code paths (output_label is set), and
        # for any multi-step annotation window. Single-step annotations from
        # --store-single-iteration keep their literal step name.
        is_structured = is_annotation and (output_label is not None or len(root) > 1)

        if (is_structured or not is_annotation) and len(batch_list) == len(
            iter_details
        ):
            for bs, iteration in zip(batch_list, iter_details):
                iteration["batch_size"] = bs

        phase_details = find_phase_from_window(iter_details)

        if is_structured:
            name_append = (
                f"prefill_{phase_details['num_prefill']}"
                f"_prefilldecode_{phase_details['num_prefilldecode']}"
                f"_decode_{phase_details['num_decode']}"
                f"_bs{phase_details['avg_bs']}_conc{phase_details['avg_conc']}"
            )
        elif is_annotation and len(root) == 1:
            root_name = root[0]["name"]
            is_known_annotation = any(
                pat.match(root_name)
                for pat in ITERATION_PATTERNS + ITERATION_BACKUP_PATTERNS
            )
            if is_known_annotation:
                name_append = (
                    root_name.replace("/", "_")
                    .replace("(", "_")
                    .replace(")", "")
                    .replace(":", "")
                    .replace(" ", "_")
                )
            else:
                name_append = ""
        else:
            if len(batch_list) == len(iter_details):
                name_append = f"batch{int(sum(batch_list)/len(batch_list))}_gpu{prefix}"
            else:
                name_append = f"batch_NA_gpu{prefix}"

        if output_label is not None:
            out_path = os.path.join(
                output_dir, f"{output_label}_{name_append}_{base_name}.json.gz"
            )
        else:
            suffix = f"_{name_append}" if name_append else ""
            out_path = os.path.join(
                output_dir, f"{base_name}_{prefix}_{idx}{suffix}.json.gz"
            )
        with gzip.open(out_path, "wb") as f:
            f.write(json.dumps(iter_trace).encode("utf-8"))

        print(
            f"  {prefix} {idx}: {len(iter_trace['traceEvents'])} events -> {out_path}"
        )
        extraction_summary.append(
            {
                "idx": idx,
                "output_path": out_path,
                "event_count": len(iter_trace["traceEvents"]),
                "num_gpu_events": num_gpu_events,
                "gpu_duration": gpu_dur,
                "gpu_busy_duration": gpu_busy,
                "steps": iter_details,
                "phase": phase_details,
            }
        )
    return extraction_summary


def extract_phases_and_save(
    roots: list[list[dict]],
    events: list[dict],
    trace_json: dict,
    output_dir: str,
    base_name: str,
    prefix: str,
    start: int,
    end: int,
    gpu_corr_map: dict,
    flow_corr_map: dict,
    meta_events: list[dict],
    root_tiles: dict | None = None,
):
    """Extract and save a range of iterations."""
    extraction_summary = []

    if "annotation_iteration" not in prefix:
        print("phase extraction only supported for annotation iterations, skipping")
        return extraction_summary
    for root in roots:
        iter_details = iteration_details(root)
        prefilldecode_steps = [r for r, i in zip(root, iter_details) if has_context(i)]
        decode_steps = [r for r, i in zip(root, iter_details) if is_decode_only(i)]

        if len(prefilldecode_steps) > 0:
            iter_details = iteration_details(prefilldecode_steps)
            phase_details = find_phase_from_window(iter_details)

            iter_trace, _batch_list, num_gpu_events, gpu_dur, gpu_busy = (
                extract_iteration(
                    prefilldecode_steps,
                    events,
                    trace_json,
                    gpu_corr_map,
                    flow_corr_map,
                    meta_events,
                    root_tiles=root_tiles,
                )
            )
            name_append = f"prefilldecode_{phase_details['num_prefilldecode']}_bs{phase_details['avg_bs']}_conc{phase_details['avg_conc']}"

            out_path = os.path.join(output_dir, f"{name_append}_{base_name}.json.gz")
            with gzip.open(out_path, "wb") as f:
                f.write(json.dumps(iter_trace).encode("utf-8"))

            print(f"  {prefix}: {len(iter_trace['traceEvents'])} events -> {out_path}")
            extraction_summary.append(
                {
                    "idx": 0,
                    "output_path": out_path,
                    "event_count": len(iter_trace["traceEvents"]),
                    "num_gpu_events": num_gpu_events,
                    "gpu_duration": gpu_dur,
                    "gpu_busy_duration": gpu_busy,
                    "steps": iter_details,
                    "phase": phase_details,
                }
            )
        if len(decode_steps) > 0:
            iter_details = iteration_details(decode_steps)
            phase_details = find_phase_from_window(iter_details)
            iter_trace, _batch_list, num_gpu_events, gpu_dur, gpu_busy = (
                extract_iteration(
                    decode_steps,
                    events,
                    trace_json,
                    gpu_corr_map,
                    flow_corr_map,
                    meta_events,
                    root_tiles=root_tiles,
                )
            )
            name_append = f"decode_{phase_details['num_decode']}_bs{phase_details['avg_bs']}_conc{phase_details['avg_conc']}"

            out_path = os.path.join(output_dir, f"{name_append}_{base_name}.json.gz")
            with gzip.open(out_path, "wb") as f:
                f.write(json.dumps(iter_trace).encode("utf-8"))

            print(f"  {prefix}: {len(iter_trace['traceEvents'])} events -> {out_path}")
            extraction_summary.append(
                {
                    "idx": 0,
                    "output_path": out_path,
                    "event_count": len(iter_trace["traceEvents"]),
                    "num_gpu_events": num_gpu_events,
                    "gpu_duration": gpu_dur,
                    "gpu_busy_duration": gpu_busy,
                    "steps": iter_details,
                    "phase": phase_details,
                }
            )
    return extraction_summary


def divide_phases_and_save(
    iteration_roots: list[dict],
    events: list[dict],
    trace_json: dict,
    output_dir: str,
    base_name: str,
    gpu_corr_map: dict,
    flow_corr_map: dict,
    meta_events: list[dict],
    steady_state_regions: list[tuple[int, int]],
    root_tiles: dict | None = None,
) -> list[dict]:
    """
    Group contiguous steps of the same phase within steady-state regions and
    save each contiguous run as a single trace file into one of two sub-folders:

    - ``{output_dir}/prefilldecodemix/`` — runs where every step has ``context_requests > 0``
    - ``{output_dir}/decode_only/``      — runs where every step has ``context_requests == 0``
                                           and ``generation_requests > 0``

    A phase transition (PD → DO or DO → PD) always starts a new file.

    Parameters
    ----------
    steady_state_regions
        Pre-computed steady-state region list as ``(start, end)`` index pairs.
        Pass ``[(0, len(iteration_roots))]`` to treat the entire slice as steady state.
    """
    iter_details = iteration_details(iteration_roots)
    regions = steady_state_regions
    print(f"[divide-phases] Steady-state regions: {regions}")

    # Build an ordered list of (phase_label, root) for all steady-state steps
    steady_steps: list[tuple[str, dict]] = []
    for s, e in regions:
        for idx in range(s, e):
            detail = iter_details[idx]
            root = iteration_roots[idx]
            if has_context(detail):
                steady_steps.append(("prefilldecodemix", root))
            elif has_generation(detail):
                steady_steps.append(("decode_only", root))
            # steps that are neither (e.g. idle) are skipped

    # Group into contiguous runs of the same phase
    runs: list[tuple[str, list[dict]]] = []  # (phase, [roots])
    for phase, root in steady_steps:
        if runs and runs[-1][0] == phase:
            runs[-1][1].append(root)
        else:
            runs.append((phase, [root]))

    pd_count = sum(1 for p, _ in runs if p == "prefilldecodemix")
    do_count = sum(1 for p, _ in runs if p == "decode_only")
    total_pd_steps = sum(len(r) for p, r in runs if p == "prefilldecodemix")
    total_do_steps = sum(len(r) for p, r in runs if p == "decode_only")
    print(
        f"\n[divide-phases] {pd_count} prefilldecodemix runs ({total_pd_steps} steps) and "
        f"{do_count} decode_only runs ({total_do_steps} steps) across all steady-state regions."
    )

    pd_dir = os.path.join(output_dir, "prefilldecodemix")
    do_dir = os.path.join(output_dir, "decode_only")
    if pd_count:
        os.makedirs(pd_dir, exist_ok=True)
    if do_count:
        os.makedirs(do_dir, exist_ok=True)

    extraction_summary = []
    pd_chunk_idx = 0
    do_chunk_idx = 0

    for phase, chunk_roots in runs:
        if phase == "prefilldecodemix":
            out_dir = pd_dir
            chunk_idx = pd_chunk_idx
            pd_chunk_idx += 1
        else:
            out_dir = do_dir
            chunk_idx = do_chunk_idx
            do_chunk_idx += 1

        phase_details = find_phase_from_window(iteration_details(chunk_roots))
        name_append = (
            f"chunk{chunk_idx}_"
            f"steps{len(chunk_roots)}_"
            f"bs{phase_details['avg_bs']}_"
            f"conc{phase_details['avg_conc']}"
        )
        extraction_summary.extend(
            extract_and_save(
                [chunk_roots],
                events,
                trace_json,
                out_dir,
                base_name,
                "annotation_iteration",
                0,
                1,
                gpu_corr_map,
                flow_corr_map,
                meta_events,
                output_label=f"{phase}_{name_append}",
                root_tiles=root_tiles,
            )
        )

    return extraction_summary
