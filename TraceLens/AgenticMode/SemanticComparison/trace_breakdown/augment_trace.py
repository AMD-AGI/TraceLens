#!/usr/bin/env python3
"""
Augment a trace JSON with gpu_user_annotation events based on semantic labels.

Adds three annotation tiers on the GPU timeline:
  1. Layer-level annotations (e.g., "Layer 0", "Preamble", "Epilogue")
  2. Block-level annotations (e.g., "QKV Projection", "Attention", "MoE Routing")
  3. Group-level annotations (e.g., "Self-Attention", "MoE / FFN")

The annotations use the same pid as the GPU kernels and use new tids to
appear as separate rows in Perfetto/Chrome trace viewers.

Also injects process/thread metadata ("M" phase events) so Perfetto
displays meaningful row names instead of raw tid numbers.

Input: original trace JSON + semantic_labels.json
Output: augmented trace JSON

Usage:
    python augment_trace.py <trace.json> <semantic_labels.json> [-o augmented_trace.json]
"""
import argparse
import json
import sys

from category_mappings import get_group


def find_gpu_pid_tid(events):
    """Auto-detect the pid/tid used by GPU kernel events."""
    for e in events:
        if isinstance(e, dict) and e.get("cat") == "kernel":
            return e["pid"], e["tid"]
    return None, None


def pick_annotation_tids(events, gpu_pid, count=3):
    """Find unused tids within the GPU pid for annotation rows."""
    used_tids = set()
    for e in events:
        if isinstance(e, dict) and e.get("pid") == gpu_pid:
            used_tids.add(e.get("tid"))

    tids = []
    for candidate in range(1000, 2000):
        if candidate not in used_tids:
            tids.append(candidate)
            if len(tids) == count:
                break
    return tids


def _clamp_spans(spans):
    """Ensure no overlaps: clamp each span's end to not exceed the next span's start.

    GPU kernel timestamps can have sub-microsecond overlaps between consecutive
    kernels, which causes annotation spans to overlap. Perfetto hides or stacks
    overlapping events on the same track, making some appear missing.

    Works with tuples of any length where the last two elements are (ts, te).
    """
    for i in range(len(spans) - 1):
        *prefix, ts, te = spans[i]
        next_ts = spans[i + 1][-2]
        if te > next_ts:
            spans[i] = (*prefix, ts, next_ts)
    return spans


def group_consecutive_blocks(labeled_kernels, kernel_events):
    """Group consecutive kernels that share the same semantic_block.

    Returns list of (semantic_block, layer, ts_start, ts_end) tuples.
    Each group spans from the first kernel's ts to the last kernel's (ts + dur).
    Consecutive groups are clamped so they never overlap.
    """
    groups = []
    current_block = None
    current_layer = None
    group_start = None
    group_end = None

    for lk in labeled_kernels:
        idx = lk["index"]
        ke = kernel_events[idx]
        ts = ke["ts"]
        dur = ke["dur"]
        end = ts + dur
        block = lk["semantic_block"]
        layer = lk.get("layer")

        if block != current_block:
            if current_block is not None:
                groups.append((current_block, current_layer, group_start, group_end))
            current_block = block
            current_layer = layer
            group_start = ts
            group_end = end
        else:
            group_end = max(group_end, end)

    if current_block is not None:
        groups.append((current_block, current_layer, group_start, group_end))

    return _clamp_spans(groups)


def build_layer_spans(groups):
    """Merge consecutive groups into layer-level spans."""
    spans = []
    current_label = None
    span_start = None
    span_end = None

    for block, layer, ts, te in groups:
        if layer is None:
            label = block.split(":")[0] if ":" in block else block
        else:
            label = f"Layer {layer}"

        if label != current_label:
            if current_label is not None:
                spans.append((current_label, span_start, span_end))
            current_label = label
            span_start = ts
            span_end = te
        else:
            span_end = max(span_end, te)

    if current_label is not None:
        spans.append((current_label, span_start, span_end))

    return _clamp_spans(spans)


def build_group_spans(groups):
    """Merge consecutive blocks into semantic_group-level spans."""
    spans = []
    current_group = None
    span_start = None
    span_end = None

    for block, layer, ts, te in groups:
        group = get_group(block)

        if group != current_group:
            if current_group is not None:
                spans.append((current_group, span_start, span_end))
            current_group = group
            span_start = ts
            span_end = te
        else:
            span_end = max(span_end, te)

    if current_group is not None:
        spans.append((current_group, span_start, span_end))

    return _clamp_spans(spans)


def create_annotation_events(groups, layer_spans, group_spans,
                              gpu_pid, group_tid, layer_tid, block_tid):
    """Create the annotation trace events for all three tiers."""
    new_events = []

    for label, ts, te in group_spans:
        dur = te - ts
        if dur <= 0:
            continue
        new_events.append({
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": label,
            "pid": gpu_pid,
            "tid": group_tid,
            "ts": ts,
            "dur": dur,
        })

    for label, ts, te in layer_spans:
        dur = te - ts
        if dur <= 0:
            continue
        new_events.append({
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": label,
            "pid": gpu_pid,
            "tid": layer_tid,
            "ts": ts,
            "dur": dur,
        })

    for block, layer, ts, te in groups:
        dur = te - ts
        if dur <= 0:
            continue
        new_events.append({
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": block,
            "pid": gpu_pid,
            "tid": block_tid,
            "ts": ts,
            "dur": dur,
        })

    return new_events


def create_metadata_events(gpu_pid, group_tid, layer_tid, block_tid):
    """Create thread name metadata events for Perfetto display."""
    return [
        {
            "ph": "M", "cat": "", "name": "thread_name",
            "pid": gpu_pid, "tid": group_tid,
            "args": {"name": "Semantic Groups"},
        },
        {
            "ph": "M", "cat": "", "name": "thread_name",
            "pid": gpu_pid, "tid": layer_tid,
            "args": {"name": "Semantic Layers"},
        },
        {
            "ph": "M", "cat": "", "name": "thread_name",
            "pid": gpu_pid, "tid": block_tid,
            "args": {"name": "Semantic Blocks"},
        },
        {
            "ph": "M", "cat": "", "name": "thread_sort_index",
            "pid": gpu_pid, "tid": group_tid,
            "args": {"sort_index": -3},
        },
        {
            "ph": "M", "cat": "", "name": "thread_sort_index",
            "pid": gpu_pid, "tid": layer_tid,
            "args": {"sort_index": -2},
        },
        {
            "ph": "M", "cat": "", "name": "thread_sort_index",
            "pid": gpu_pid, "tid": block_tid,
            "args": {"sort_index": -1},
        },
    ]


def run_assertions(labeled_kernels, kernel_events, groups, new_events):
    errors = []

    total_labeled = len(labeled_kernels)
    accounted = 0
    for block, layer, ts, te in groups:
        for lk in labeled_kernels:
            ke = kernel_events[lk["index"]]
            if ke["ts"] >= ts - 0.001 and ke["ts"] < te + 0.001:
                accounted += 1
    if accounted < total_labeled * 0.95:
        errors.append(
            f"AUG.1 WARNING: Only {accounted}/{total_labeled} kernels start within "
            "annotation time spans"
        )

    annotation_events = [e for e in new_events if e["ph"] == "X"]
    for ae in annotation_events:
        if ae["dur"] <= 0:
            errors.append(
                f"AUG.2 FAIL: Annotation '{ae['name']}' has non-positive duration {ae['dur']}"
            )

    if len(groups) == 0:
        errors.append("AUG.3 FAIL: No semantic block groups found")

    return errors


def augment(trace_path, labels_path, output_path):
    with open(trace_path) as f:
        trace_data = json.load(f)
    with open(labels_path) as f:
        labels_data = json.load(f)

    events = trace_data["traceEvents"]
    labeled = labels_data["labeled_kernels"]

    kernel_events = sorted(
        [e for e in events if isinstance(e, dict) and e.get("cat") == "kernel"],
        key=lambda e: e["ts"],
    )

    if len(kernel_events) != len(labeled):
        print(
            f"WARNING: Kernel count mismatch: trace has {len(kernel_events)}, "
            f"labels has {len(labeled)}",
            file=sys.stderr,
        )

    gpu_pid, gpu_tid = find_gpu_pid_tid(events)
    if gpu_pid is None:
        print("ERROR: No GPU kernel events found in trace", file=sys.stderr)
        sys.exit(1)

    tids = pick_annotation_tids(events, gpu_pid, count=3)
    group_tid, layer_tid, block_tid = tids[0], tids[1], tids[2]

    groups = group_consecutive_blocks(labeled, kernel_events)
    layer_spans = build_layer_spans(groups)
    group_spans = build_group_spans(groups)

    annotation_events = create_annotation_events(
        groups, layer_spans, group_spans,
        gpu_pid, group_tid, layer_tid, block_tid,
    )
    metadata_events = create_metadata_events(gpu_pid, group_tid, layer_tid, block_tid)

    all_new = metadata_events + annotation_events
    errors = run_assertions(labeled, kernel_events, groups, all_new)
    for e in errors:
        print(e, file=sys.stderr)
    if any("FAIL" in e for e in errors):
        sys.exit(1)

    trace_data["traceEvents"] = events + all_new

    with open(output_path, "w") as f:
        json.dump(trace_data, f)

    block_count = len(groups)
    layer_count = len(layer_spans)
    group_count = len(group_spans)
    print(
        f"Augmented trace written to {output_path}\n"
        f"  Added {len(all_new)} events "
        f"({group_count} group spans + {layer_count} layer spans + "
        f"{block_count} block spans + {len(metadata_events)} metadata)\n"
        f"  GPU pid={gpu_pid}, group_tid={group_tid}, "
        f"layer_tid={layer_tid}, block_tid={block_tid}",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Augment trace JSON with semantic annotation events"
    )
    parser.add_argument("trace", help="Path to original trace JSON")
    parser.add_argument("labels", help="Path to semantic_labels.json")
    parser.add_argument(
        "-o", "--output", required=True, help="Output augmented trace JSON path"
    )
    args = parser.parse_args()
    augment(args.trace, args.labels, args.output)


if __name__ == "__main__":
    main()
