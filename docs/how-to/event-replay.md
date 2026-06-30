---
myst:
    html_meta:
        "description": "Learn how to isolate a single GPU operation into a minimal, self-contained replay script using TraceLens EventReplay for focused debugging."
        "keywords": "TraceLens, EventReplay, GPU debugging, reproducer, operator replay, PyTorch profiler, ROCm, kernel isolation, IP-safe"
---

# Replay a single operation

This guide shows how to isolate an operation from a trace into a minimal,
self-contained replay — useful for focused debugging and for sharing IP-safe
reproducers with kernel or framework developers.

## Prerequisites

- TraceLens installed (see [Install TraceLens](../install/installation.md)).
- A PyTorch profiler trace containing the operation you want to isolate.

## How it works

`EventReplay` extracts each operation's essential attributes — name, input
shapes, strides, dtypes, and other arguments — into a lightweight, portable JSON
intermediate representation (IR), then reconstructs and re-runs the operation
from that IR. Because the IR is built from trace metadata rather than your model
code, the artifacts can be shared without exposing model IP.

```{note}
EventReplay allocates inputs with randomized data matching the recorded tensor
shapes, so replay timings approximate — but do not exactly reproduce — the
original run.
```

## Step 1: Identify the operation

Generate a PyTorch report (see
[Generate a PyTorch performance report](./generate-perf-report-pytorch.md)) and use the
`ops_unique_args` sheet to find the operation and input shape you want to
isolate, noting its UID.

## Step 2: Replay a single event (SDK)

```python
from TraceLens import TreePerfAnalyzer, EventReplayer

perf_analyzer = TreePerfAnalyzer.from_file("/path/to/profile.json")
uid = 12345  # UID of the target op (from the ops_unique_args sheet)
event = perf_analyzer.tree.get_UID2event(uid)

replayer = EventReplayer(event, device="cuda")
replayer.replay()
```

The `examples/event_replayer_example.ipynb` notebook walks through the same flow
interactively, including selecting the target event from the tree.

## Batch replay and benchmark

Extract a portable IR for many events at once, then replay and benchmark them
with the bundled `batched_replay.py` script:

```python
import json
from TraceLens import EventReplayer

repro_data = [
    EventReplayer(event, lazy=True).get_repro_info()
    for event in events_of_interest
]
with open("event_replay_ir.json", "w") as f:
    json.dump(repro_data, f, indent=4)
```

```bash
python batched_replay.py event_replay_ir.json
```

`batched_replay.py` accepts:

- `--device {cuda,cpu}` — device to run on (default `cuda`).
- `--op-filter <substring>` — only replay ops whose name contains the substring
  (e.g. `aten::convolution`).
- `--op-limit <N>` — replay at most `N` ops.
- `--stop-on-error` — abort on the first failure instead of continuing.
- `--verbose` / `-v` — print reconstructed arguments and per-op detail.

Each replayed op prints its reconstructed arguments, average time, and result
tensor, followed by a summary of attempted, successful, and failed replays.

## Package standalone artifacts

The IR plus the replay scripts can be zipped into a self-contained bundle that
runs without TraceLens or the original model. The bundle contains:

- `event_replay_ir.json` — serialized operator replay instructions.
- `utils.py` — tensor-creation and helper utilities.
- `batched_replay.py` — batch replay and benchmark script.
- `batched_replay_readme.md` — run instructions.

See the
[`event_replayer_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/event_replayer_example.ipynb)
notebook for end-to-end IR extraction and packaging.

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Install TraceLens](../install/installation.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [API reference](../reference/api-reference.md)
