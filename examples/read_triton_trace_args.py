"""Read Triton kernel metadata from a Chrome trace file.

Shows the event["args"] fields that V2 uses to compute perf metrics:
  - Concrete Inputs  -> xnumel, rnumel (exact element counts)
  - Input Dims       -> tensor shapes (for exact bytes calculation)
  - Input type       -> dtypes (for bytes-per-element)
  - kernel_file, kernel_kwargs, num_warps, num_stages

Usage:
    python read_triton_trace_args.py <trace.json or trace.json.gz>
"""

import json
import gzip
import sys


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def is_triton_cpu_event(event):
    name = event.get("name", "")
    return (
        event.get("cat") == "cpu_op"
        and (
            name.startswith("triton_poi_")
            or name.startswith("triton_red_")
            or name.startswith("triton_per_")
        )
    )


def main():
    # default_trace = "torch_trace_output_claude/triton_repro/trace.json.gz"
    default_trace = "/root/repos/TraceLens/torch_trace_output_claude/triton_repro/trace.json_mi250.gz"

    path = sys.argv[1] if len(sys.argv) > 1 else default_trace
    print(f"Reading: {path}\n")
    events = load_trace(path)
    triton_events = [e for e in events if is_triton_cpu_event(e)]

    if not triton_events:
        print("No Triton kernel CPU events found in trace.")
        sys.exit(0)

    print(f"Found {len(triton_events)} Triton kernel event(s)\n")

    for e in triton_events:
        args = e.get("args", {})
        print(f"{'=' * 70}")
        print(f"Kernel:           {e['name']}")
        print(f"Duration:         {e.get('dur', '?')} us")
        print(f"Concrete Inputs:  {args.get('Concrete Inputs')}")
        print(f"Input Dims:       {args.get('Input Dims')}")
        print(f"Input type:       {args.get('Input type')}")
        print(f"kernel_file:      {args.get('kernel_file')}")
        print(f"kernel_kwargs:    {args.get('kernel_kwargs')}")
        print(f"num_warps:        {args.get('num_warps')}")
        print(f"num_stages:       {args.get('num_stages')}")
        print()


if __name__ == "__main__":
    main()
