<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Replay a single operation in TraceLens
```{meta}
:description: Isolate a GPU operation from a PyTorch profiler trace into a portable EventReplay IR, including custom ops, auto-import, and custom tensor initializers.
:keywords: TraceLens, EventReplay, GPU debugging, reproducer, operator replay, PyTorch profiler, ROCm, custom op, aiter, vLLM, paged attention, MoE
```

This topic shows how to isolate an operation from a trace into a minimal,
self-contained replay. That's useful for focused debugging and for sharing
IP-safe reproducers with kernel or framework developers.

EventReplay works with `aten::` ops and with custom ops from other namespaces
(for example `aiter::` and `_rocm_C::`) when the library that registers the op
is importable.

## Before you begin

Confirm you have the following before continuing.

- [TraceLens installed](../install/install.md).
- A PyTorch profiler trace containing the operation you want to isolate.
- For custom ops, the library that registers the op (for example `aiter` or
  `vllm`) installed in the same environment you replay in.

## How it works

`EventReplay` extracts each operation's essential attributes — name, input
shapes, strides, dtypes, and other arguments — into a lightweight, portable JSON
intermediate representation (IR), then reconstructs and re-runs the operation
from that IR. Because the IR is built from trace metadata rather than your model
code, the artifacts can be shared without exposing model IP.

```{note}
EventReplay allocates inputs with randomized data matching the recorded tensor
shapes, so replay timings approximate — but do not exactly reproduce — the
original run. Integer index and routing tensors default to zeros unless a
[custom initializer](#custom-initializers) fills them.
```

## Step 1: Identify the operation

Generate a PyTorch report (see
[Generate a PyTorch performance report](./generate-perf-report-pytorch.md)) and use the
`ops_unique_args` sheet to find the operation and input shape you want to
isolate, noting its UID. The UID is a unique integer identifier TraceLens
assigns to each event. In the `ops_unique_args` sheet, each row aggregates a
unique (operation, arguments) group, and a representative UID for the group
appears in the `ex_UID` column.

## Step 2: Replay a single event (SDK)

Use the `EventReplayer` class to replay the event identified by its UID:

```python
from TraceLens import TreePerfAnalyzer, EventReplayer

perf_analyzer = TreePerfAnalyzer.from_file("/path/to/profile.json")
uid = 12345  # UID of the target op (from the ops_unique_args sheet)
event = perf_analyzer.tree.get_UID2event(uid)

replayer = EventReplayer(event, device="cuda")
replayer.replay()
```

The
[`event_replayer_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/event_replayer_example.ipynb)
notebook walks through the same flow interactively, including selecting the
target event from the tree.

## Inspect the IR

The profiler stores arguments as unlabeled dimension lists. EventReplay zips
those arrays with the op's registered schema so you can read named tensors and
scalars without launching the kernel:

```python
replayer = EventReplayer(event, lazy=True)
ir = replayer.get_repro_info()
```

**Profiler event** for `aten::mm` (shapes only):

```json
{
  "cat": "cpu_op",
  "name": "aten::mm",
  "args": {
    "Input Dims": [[20, 2048], [2048, 11264]],
    "Input type": ["BFloat16", "BFloat16"]
  }
}
```

**EventReplay IR** for the same event:

```json
{
  "op_name": "aten::mm",
  "replay_ir": {
    "list_pos_args": [
      {
        "arg_name": "self",
        "arg_type": "Tensor",
        "value": {
          "shape": [20, 2048],
          "dtype": "c10::BFloat16",
          "strides": [2048, 1],
          "init": "normal"
        }
      },
      {
        "arg_name": "mat2",
        "arg_type": "Tensor",
        "value": {
          "shape": [2048, 11264],
          "dtype": "c10::BFloat16",
          "strides": [1, 2048],
          "init": "normal"
        }
      }
    ]
  }
}
```

That IR is a BF16 GEMM with `M=20`, `K=2048`, `N=11264`, and a column-major
`mat2` (stride pattern `[1, K]`). The same mapping is what makes fused custom
ops readable: a raw `aiter::ck_moe_stage1` event is a long list of unlabeled
scalars, while the IR names `hidden_states`, `w1`, `w2`, `sorted_token_ids`,
`topk`, `block_m`, and so on.

Schema lookup uses the PyTorch dispatcher (`torch._C._jit_get_all_schemas()` or
`torch.ops`). If no schema is registered, EventReplay falls back to a
schemaless IR that infers types from the profiled arrays. Ops called as plain
Python functions (for example a Triton kernel launched directly) don't appear in
the dispatcher; wrap them with `torch.library.custom_op` if you need a schema.

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

### Package standalone artifacts

Before running `batched_replay.py`, package the IR and its companion scripts
into a self-contained bundle. The bundle can be run without TraceLens or the
original model and is safe to share without exposing model IP. It contains:

- **`event_replay_ir.json`:** serialized operator replay instructions.
- **`utils.py`:** tensor-creation and helper utilities that `batched_replay.py` imports.
- **`batched_replay.py`:** batch replay and benchmark script.
- **`batched_replay_readme.md`:** run instructions.

See the
[`event_replayer_example.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/event_replayer_example.ipynb)
notebook for end-to-end IR extraction and packaging.

### Run the batch replay

From the directory containing the unpacked bundle (or `TraceLens/EventReplay/`
for a source checkout), run:

```bash
python batched_replay.py event_replay_ir.json
python batched_replay.py -v event_replay_ir.json
python batched_replay.py --op-filter aten::mm event_replay_ir.json
python batched_replay.py --op-limit 5 event_replay_ir.json
```

`batched_replay.py` imports `utils.py` from the same directory, so the command
must be run from that location.

The following table describes the CLI flags.

| Argument | Default | Description |
|---|---|---|
| `repro_file` | (required) | Path to the JSON IR file from `get_repro_info()`. |
| `--device` | `cuda` | Device to run on (`cuda` or `cpu`). Falls back to `cpu` if CUDA isn't available. |
| `--verbose` / `-v` | off | Print reconstructed arguments and per-op detail. |
| `--stop-on-error` | off | Abort on the first failure instead of continuing. |
| `--op-filter` | `None` | Only replay ops whose name contains this substring (for example `aten::add`). |
| `--op-limit` | `None` | Replay at most this many ops after filtering. |

Each replayed op prints average and median time, then a summary of attempted,
successful, and failed replays.

## Custom ops and auto-import

When EventReplayer sees a non-`aten` namespace, it tries to import the library
that registers the op schema. Built-in mappings:

| Namespace | Imported modules |
|---|---|
| `aiter` | `aiter` |
| `_rocm_C` | `vllm._rocm_C` |
| `_C` | `vllm._C` |
| `_C_cache_ops` | `vllm._C` |
| `vllm` | `vllm._C`, `vllm._rocm_C` |

Register additional namespaces:

```python
from TraceLens.EventReplay import EventReplayer

EventReplayer.register_namespace("my_lib", ["my_lib.ops"])
```

Resolution order is the JIT registry, then `torch.ops`, then a direct Python
module import, then auto-import, then any name aliases (for example
`_rocm_C::wvSplitK` → `_rocm_C::wvSpltK`).

## Custom initializers

Profiler traces record shapes and dtypes, not tensor values. Zero-filled index
and routing tensors can make a kernel short-circuit (no real work). Custom
initializers fill those tensors with plausible values before `replay()`. They
run when `auto_init=True` (the default).

### Built-in initializers

These activate when the event name matches exactly:

- **`PagedAttentionInit`:** `_rocm_C::paged_attention`. Fills `block_tables`
  (permutation of the physical block pool), `seq_lens` (`max_seq_len` for every
  sequence), and `query_start_loc` (CSR indptr of per-sequence query counts).
  Uses iteration annotations when present; otherwise heuristics.
- **`MoeRoutingInit`:** `aiter::ck_moe_stage1` and `aiter::ck_moe_stage2`.
  Builds `sorted_token_ids`, `sorted_expert_ids`, and `num_valid_ids`. Pass
  `init_kwargs={"moe_distribution": "zipf", "moe_zipf_s": 1.5}` for a skewed
  expert load; the default is uniform.

```python
replayer = EventReplayer(
    event,
    device="cuda",
    init_kwargs={"moe_distribution": "zipf", "moe_zipf_s": 1.5},
)
```

### Write your own initializer

1. Subclass `CustomInit` and set `op_patterns` to the **exact** profiler event
   name (for example `"aten::index_add_"`, not `"index_add"`).
2. Implement `initialize()` and mutate `replayer.args` / `replayer.kwargs` in
   place. Look up arguments by name from `replayer.event_replay_IR`.
3. Register with `EventReplayer.register_custom_init(YourInit())`.

```python
from TraceLens.EventReplay import EventReplayer, CustomInit

class IndexAddInit(CustomInit):
    op_patterns = ["aten::index_add_"]

    def initialize(self, replayer, **kwargs):
        import torch

        ir = replayer.event_replay_IR
        arg_names = [a["arg_name"] for a in ir["list_pos_args"]]
        self_tensor = replayer.args[arg_names.index("self")]
        dim = replayer.args[arg_names.index("dim")]
        index = replayer.args[arg_names.index("index")]
        dim_size = self_tensor.shape[dim]
        index.copy_(
            torch.randint(0, dim_size, index.shape, device=index.device)
        )
        return f"[custom init] index_add — index randint(0, {dim_size})"

EventReplayer.register_custom_init(IndexAddInit())
```

`replay()` applies the **first** matching initializer. Built-ins are registered
first; `register_custom_init` appends, so a user initializer for the same exact
op name as a built-in doesn't run. List the registry with
`EventReplayer.list_custom_inits()`.

The built-in implementations are in `TraceLens/EventReplay/custom_inits.py`.

## Iteration annotations (vLLM traces)

Paged attention's `query_start_loc` is a compressed sparse row (CSR) indptr that
encodes how many query tokens each sequence contributes. In a mixed batch some
sequences are prefill (many query tokens) and others are decode (one token
each). The profiler captures the tensor shape, not that split.

vLLM emits a `user_annotation` per `execute_model` step whose name encodes the
split, for example `execute_context_2(18)_generation_5(5)`: two prefill
sequences with 18 query tokens total, and five decode sequences with five tokens
(one each).

```python
from TraceLens import TreePerfAnalyzer
from TraceLens.EventReplay import EventReplayer, extract_batch_context

analyzer = TreePerfAnalyzer.from_file("vllm_trace.json")
num_annotated = extract_batch_context(analyzer)

event = analyzer.tree.get_UID2event(some_uid)
replayer = EventReplayer(event, device="cuda")
replayer.replay()
```

`extract_batch_context` attaches `event["batch_context"]` with `n_prefill`,
`prefill_tokens`, `n_decode`, and `decode_tokens`. `PagedAttentionInit` uses
that dict for `query_start_loc`. Without annotations it assumes pure decode
when `query_tokens == num_seqs`, and pure prefill otherwise. That approximation
is weak for mixed batches.

## Known limitations

- **Unregistered ops are invisible.** Triton kernels called directly from Python
  have no dispatcher schema. Wrap them in `torch.library.custom_op` in the
  upstream library if you need IR extraction.
- **Single-op isolation versus the real workload.** Replay runs each op with no
  surrounding memory traffic. Timings are a lower bound on in-model performance.
- **Data-dependent kernels.** Custom initializers are plausible, not bitwise
  copies of the original tensors. Timing can differ when control flow depends on
  values.

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Install TraceLens](../install/install.md)
- [Generate a PyTorch performance report](./generate-perf-report-pytorch.md)
- [API reference](../reference/api-reference.md)
- [Tensor shape metadata](../conceptual/shape-metadata.md)
