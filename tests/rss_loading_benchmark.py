#!/usr/bin/env python3
"""
Benchmark different trace loading strategies for RSS and parse time.

Each strategy is isolated in its own subprocess so ru_maxrss resets between runs.
Usage:
    python tests/rss_loading_benchmark.py
"""
import gzip
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRACES = [
    "traces/gptoss_vllm_example0.json.gz",
    "traces/huvideo_rank4_step3.json.gz",
    "traces/mixtral_8x22b_b200.json.gz",
]

# Categories to drop during streaming (construction-only)
DROP_CATS = {"ac2g"}
DROP_PHASES = {"M"}


# ---------------------------------------------------------------------------
# Strategy implementations (each run in a subprocess)
# ---------------------------------------------------------------------------

def strategy_orjson_baseline(path):
    """Baseline: orjson full load, no filtering."""
    import gzip, orjson
    with gzip.open(path, "rb") as f:
        raw = f.read()
    data = orjson.loads(raw)
    del raw
    events = data["traceEvents"]
    del data
    return len(events)


def strategy_orjson_postfilter(path):
    """orjson full load then immediately drop ac2g/metadata in Python."""
    import gzip, orjson
    with gzip.open(path, "rb") as f:
        raw = f.read()
    data = orjson.loads(raw)
    del raw
    events = [
        e for e in data["traceEvents"]
        if e.get("cat") not in DROP_CATS and e.get("ph") not in DROP_PHASES
    ]
    del data
    return len(events)


def strategy_ijson_filtered(path):
    """ijson streaming: parse traceEvents one at a time, drop unwanted categories."""
    import gzip, ijson
    with gzip.open(path, "rb") as f:
        events = [
            e for e in ijson.items(f, "traceEvents.item")
            if e.get("cat") not in DROP_CATS and e.get("ph") not in DROP_PHASES
        ]
    return len(events)


def strategy_ijson_full(path):
    """ijson streaming: parse all traceEvents (no filtering), for speed baseline."""
    import gzip, ijson
    with gzip.open(path, "rb") as f:
        events = list(ijson.items(f, "traceEvents.item"))
    return len(events)


def strategy_orjson_prefilter_bytes(path):
    """
    Decompress to bytes, use a fast byte-level scan to build a filtered JSON
    blob (removing ac2g objects), then parse with orjson.

    Approach: find each JSON object boundary in the traceEvents array using
    a brace-depth counter, check if it contains '"cat": "ac2g"', skip it if so.
    This avoids parsing ac2g events as Python objects at all.
    """
    import gzip, orjson

    with gzip.open(path, "rb") as f:
        raw = f.read()

    # Find the start of the traceEvents array
    te_key = b'"traceEvents"'
    te_start = raw.find(te_key)
    if te_start == -1:
        data = orjson.loads(raw)
        del raw
        return len(data.get("traceEvents", []))

    # Split: prefix (everything up to and including '[') and the events body
    bracket_pos = raw.index(b"[", te_start)
    prefix = raw[: bracket_pos + 1]          # b'...\n"traceEvents": ['
    suffix_start = bracket_pos + 1

    # Walk the events array byte-by-byte with a depth counter,
    # accumulating objects. Skip objects that contain b'"ac2g"'.
    kept_chunks = [prefix]
    i = suffix_start
    length = len(raw)
    first = True

    while i < length:
        # Skip whitespace and commas between objects
        c = raw[i]
        if c in (0x20, 0x09, 0x0A, 0x0D, 0x2C):  # space tab newline cr comma
            i += 1
            continue
        if c == 0x5D:  # ']' — end of array
            kept_chunks.append(raw[i:])  # include ']' and everything after
            break
        if c != 0x7B:  # not '{' — unexpected
            i += 1
            continue

        # Found start of an object — find its end using brace depth
        depth = 0
        j = i
        in_string = False
        escape_next = False
        while j < length:
            ch = raw[j]
            if escape_next:
                escape_next = False
            elif ch == 0x5C:  # backslash
                escape_next = True
            elif ch == 0x22:  # double-quote
                in_string = not in_string
            elif not in_string:
                if ch == 0x7B:
                    depth += 1
                elif ch == 0x7D:
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
            j += 1

        obj_bytes = raw[i:j]
        # Check category before parsing: skip if ac2g or metadata-phase
        if b'"ac2g"' not in obj_bytes and b'"ph": "M"' not in obj_bytes and b'"ph":"M"' not in obj_bytes:
            if not first:
                kept_chunks.append(b",")
            kept_chunks.append(obj_bytes)
            first = False
        i = j

    del raw
    filtered = b"".join(kept_chunks)
    data = orjson.loads(filtered)
    del filtered
    events = data["traceEvents"]
    del data
    return len(events)


def strategy_parquet_cached(path):
    """
    Load from a Parquet cache if it exists, otherwise parse with orjson and save.
    Uses pyarrow. Only loads columns needed for analysis (excludes 'args').
    """
    import gzip, orjson, pyarrow as pa, pyarrow.parquet as pq

    cache_path = path.replace(".json.gz", ".parquet")

    if os.path.exists(cache_path):
        table = pq.read_table(cache_path)
        # args stored as JSON string column — reconstruct minimal event list
        events = table.to_pylist()
        return len(events)

    # Build cache
    with gzip.open(path, "rb") as f:
        raw = f.read()
    data = orjson.loads(raw)
    del raw
    all_events = data["traceEvents"]
    del data

    # Flatten events to Arrow table (args serialized as JSON string)
    rows = []
    for e in all_events:
        rows.append({
            "ph": e.get("ph"),
            "cat": e.get("cat"),
            "name": e.get("name"),
            "pid": e.get("pid"),
            "tid": e.get("tid"),
            "ts": e.get("ts"),
            "dur": e.get("dur"),
            "id": str(e["id"]) if "id" in e else None,  # mixed int/str in traces
            "bp": e.get("bp"),
            "args": orjson.dumps(e.get("args") or {}).decode(),
        })

    schema = pa.schema([
        ("ph",   pa.string()),
        ("cat",  pa.string()),
        ("name", pa.string()),
        ("pid",  pa.string()),
        ("tid",  pa.string()),
        ("ts",   pa.float64()),
        ("dur",  pa.float64()),
        ("id",   pa.string()),
        ("bp",   pa.string()),
        ("args", pa.string()),
    ])
    # Coerce pid/tid to string (some traces use non-integer process IDs)
    for row in rows:
        row["pid"] = str(row["pid"]) if row["pid"] is not None else None
        row["tid"] = str(row["tid"]) if row["tid"] is not None else None
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, cache_path, compression="snappy")
    return len(rows)


def strategy_parquet_cached_read(path):
    """Read from existing Parquet cache (cache must already exist)."""
    import pyarrow.parquet as pq, orjson

    cache_path = path.replace(".json.gz", ".parquet")
    if not os.path.exists(cache_path):
        return -1  # cache not built

    table = pq.read_table(cache_path)
    # Reconstruct events from table rows
    events = []
    for row in table.to_pylist():
        e = {k: v for k, v in row.items() if k != "args" and v is not None}
        args = orjson.loads(row["args"])
        if args:
            e["args"] = args
        events.append(e)
    return len(events)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

STRATEGIES = {
    "orjson_baseline": strategy_orjson_baseline,
    "orjson_postfilter": strategy_orjson_postfilter,
    "orjson_prefilter_bytes": strategy_orjson_prefilter_bytes,
    "ijson_filtered": strategy_ijson_filtered,
    "ijson_full": strategy_ijson_full,
    "parquet_write": strategy_parquet_cached,
    "parquet_read": strategy_parquet_cached_read,
}

# Mapping from strategy key to the actual function name in this module
_STRATEGY_FUNC_NAMES = {
    "orjson_baseline":      "strategy_orjson_baseline",
    "orjson_postfilter":    "strategy_orjson_postfilter",
    "orjson_prefilter_bytes": "strategy_orjson_prefilter_bytes",
    "ijson_filtered":       "strategy_ijson_filtered",
    "ijson_full":           "strategy_ijson_full",
    "parquet_write":        "strategy_parquet_cached",
    "parquet_read":         "strategy_parquet_cached_read",
}


def run_strategy_in_subprocess(strategy_name, trace_path):
    """Run a single strategy in a fresh subprocess and return (rss_mb, time_s, n_events)."""
    func_name = _STRATEGY_FUNC_NAMES[strategy_name]
    script = f"""
import sys, resource, time
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from tests.rss_loading_benchmark import {func_name}, DROP_CATS, DROP_PHASES
t0 = time.perf_counter()
n = {func_name}({trace_path!r})
elapsed = time.perf_counter() - t0
rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"{{rss_kb}}|{{elapsed:.3f}}|{{n}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    if result.returncode != 0:
        return None, None, None, result.stderr.strip()
    line = result.stdout.strip()
    rss_kb, elapsed, n = line.split("|")
    rss_mb = int(rss_kb) / 1024
    return rss_mb, float(elapsed), int(n), None


def main():
    traces = [str(PROJECT_ROOT / t) for t in TRACES]

    print(f"{'Strategy':<28} {'Trace':<35} {'RSS (MB)':>10} {'Time (s)':>10} {'Events':>10}")
    print("-" * 100)

    for trace_path in traces:
        trace_name = Path(trace_path).name.replace(".json.gz", "")
        for strat_name in STRATEGIES:
            rss_mb, elapsed, n_events, err = run_strategy_in_subprocess(strat_name, trace_path)
            if err:
                print(f"{strat_name:<28} {trace_name:<35} {'ERROR':>10}  {err[:40]}")
            else:
                print(f"{strat_name:<28} {trace_name:<35} {rss_mb:>10.0f} {elapsed:>10.2f} {n_events:>10,}")
        print()


if __name__ == "__main__":
    main()
