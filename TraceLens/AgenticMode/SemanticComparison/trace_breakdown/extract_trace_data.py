#!/usr/bin/env python3
"""
Step 1+3: Load a Chrome trace JSON and extract structured data.

Outputs a JSON with:
  - ordered kernel list (name, duration, timestamp)
  - python call stack (nested)
  - metadata (categories, graph mode detection, total kernel time)

Usage:
    python extract_trace_data.py <trace.json> [-o output.json]
"""
import argparse
import gzip
import json
import sys


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"]
    by_cat = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        cat = e.get("cat", "unknown")
        by_cat.setdefault(cat, []).append(e)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda e: e.get("ts", 0))
    return data, by_cat


def extract_kernel_sequence(by_cat):
    kernels = sorted(by_cat.get("kernel", []), key=lambda e: e["ts"])
    return [
        {"name": k["name"], "dur": k["dur"], "ts": k["ts"],
         "args": k.get("args", {})}
        for k in kernels
    ]


def detect_graph_mode(by_cat):
    rt = by_cat.get("cuda_runtime", [])
    graph_launches = [e for e in rt if "GraphLaunch" in e.get("name", "")]
    return len(graph_launches) > 0, graph_launches


def extract_python_callstack(by_cat):
    pyfuncs = sorted(by_cat.get("python_function", []), key=lambda e: e["ts"])
    for p in pyfuncs:
        p["_end"] = p["ts"] + p["dur"]
    stack = []
    result = []
    for p in pyfuncs:
        while stack and stack[-1]["_end"] <= p["ts"]:
            stack.pop()
        result.append({"name": p["name"], "dur": p["dur"], "depth": len(stack)})
        stack.append(p)
    return result


def run_assertions(data, by_cat, kernels, is_graph_mode):
    errors = []

    if "traceEvents" not in data:
        errors.append("A1.1 FAIL: Missing traceEvents key")

    required_cats = {"kernel", "cpu_op"}
    missing = required_cats - set(by_cat.keys())
    if missing:
        errors.append(f"A1.2 FAIL: Missing categories: {missing}")

    if len(kernels) == 0:
        errors.append("A1.3 FAIL: No GPU kernels found")

    for i, k in enumerate(kernels):
        if k["dur"] <= 0:
            errors.append(f"A3.2 FAIL: Kernel {i} ({k['name'][:50]}) has non-positive duration {k['dur']}")
            break

    timestamps = [k["ts"] for k in kernels]
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            errors.append(f"A3.1 FAIL: Kernel timestamps not monotonic at index {i}")
            break

    total_time = sum(k["dur"] for k in kernels)
    if total_time <= 0:
        errors.append("A1.5 FAIL: Zero total kernel time")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Extract structured data from a Chrome trace JSON")
    parser.add_argument("trace", help="Path to trace JSON file")
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    data, by_cat = load_trace(args.trace)
    kernels = extract_kernel_sequence(by_cat)
    is_graph_mode, graph_launches = detect_graph_mode(by_cat)
    callstack = extract_python_callstack(by_cat)

    errors = run_assertions(data, by_cat, kernels, is_graph_mode)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    total_kernel_time = sum(k["dur"] for k in kernels)
    categories_found = sorted(by_cat.keys())

    result = {
        "source_file": args.trace,
        "metadata": {
            "total_kernels": len(kernels),
            "total_kernel_time_us": round(total_kernel_time, 2),
            "is_graph_mode": is_graph_mode,
            "graph_launch_count": len(graph_launches),
            "categories": categories_found,
            "has_python_stack": len(callstack) > 0,
        },
        "kernels": kernels,
        "python_callstack": callstack,
    }

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote {args.output} ({len(kernels)} kernels, {total_kernel_time:.1f}us total)", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
