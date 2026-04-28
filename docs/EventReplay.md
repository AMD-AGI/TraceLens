<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Event Replay

Optimizing GPU performance in deep learning requires isolating and benchmarking
individual operations to identify bottlenecks. However, reproducing operations
directly from complex model code or large profiles can be cumbersome — the
profiler captures tensor dimensions and types but strips argument names, semantic
context, and the relationship between arguments.

Event Replay is a Python-based tool within TraceLens that extracts and replays
almost arbitrary PyTorch operations using minimal, portable Intermediate
Representation (IR). It translates the opaque profiler output into human-readable
JSON — named arguments, tensor shapes, dtypes, strides, and scalar values —
making profiler traces interpretable, shareable, and replayable on any machine
with the right op libraries installed.

**Contents:**
[Quick Start](#quick-start) |
[Batch Replay](#batch-replay) |
[Architecture](#architecture) |
[Custom Initializers](#custom-initializers) |
[Auto-Import](#auto-import-for-custom-ops) |
[Iteration Annotations](#iteration-annotations-vllm-traces) |
[Limitations](#known-limitations) |
[Use Cases](#use-cases)

---

## Quick Start

### Replay a Single Event

```python
from TraceLens import TreePerfAnalyzer
from TraceLens.EventReplay import EventReplayer

perf_analyzer = TreePerfAnalyzer.from_file('/path/to/profile.json')
uid = 12345
event = perf_analyzer.tree.get_UID2event(uid)

replayer = EventReplayer(event, device='cuda')
replayer.replay()
```

### Inspect the IR (without replaying)

Even without a GPU, the extracted IR is valuable for understanding what a
profiled op actually does. The profiler's native format stores arguments as
unlabeled dimension lists with no argument names or semantic context:

```json
{
  "cat": "cpu_op", "name": "aten::mm",
  "args": {
    "Input Dims": [[20, 2048], [2048, 11264]],
    "Input type": ["BFloat16", "BFloat16"]
  }
}
```

What are these two tensors? Which is the activation, which is the weight? Is
`mat2` transposed? You can't tell from `Input Dims` alone. Event Replay resolves
the op's registered schema and produces named, typed JSON:

```python
replayer = EventReplayer(event, lazy=True)
ir = replayer.get_repro_info()
```

**After — Event Replay IR for `aten::mm`:**

```json
{
  "op_name": "aten::mm",
  "replay_ir": {
    "list_pos_args": [
      {
        "arg_name": "self",
        "arg_type": "Tensor",
        "value": { "shape": [20, 2048], "dtype": "c10::BFloat16",
                   "strides": [2048, 1], "init": "normal" }
      },
      {
        "arg_name": "mat2",
        "arg_type": "Tensor",
        "value": { "shape": [2048, 11264], "dtype": "c10::BFloat16",
                   "strides": [1, 2048], "init": "normal" }
      }
    ]
  }
}
```

Now you can immediately read: BF16 GEMM, M=20, K=2048, N=11264, `mat2`
is column-major (stride pattern `[1, K]`).

The contrast is sharper for complex ops. Here's what the profiler gives you
for a MoE fused expert call:

**Raw profiler — `aiter::ck_moe_stage1`:**

```json
{
  "cat": "cpu_op", "name": "aiter::ck_moe_stage1",
  "args": {
    "Input Dims": [[2, 2048], [60, 2816, 2048], [60, 2048, 1408],
                   [1924], [61], [2], [2, 4, 1408], [], [], [], [], [], [], [], [], [], [], []],
    "Input type": ["BFloat16", "BFloat16", "BFloat16",
                   "Int", "Int", "Int", "BFloat16",
                   "Scalar", "Scalar", "Scalar", "Scalar", "Scalar", "Scalar",
                   "Scalar", "Scalar", "Scalar", "Scalar", "Scalar"]
  }
}
```

18 arguments, most labeled just "Scalar" — which one is `topk`? Which is
`block_m`? What does `[1924]` represent? Uninterpretable without reading the
source code.

**After — Event Replay IR:**

```json
{
  "op_name": "aiter::ck_moe_stage1",
  "replay_ir": {
    "list_pos_args": [
      { "arg_name": "hidden_states", "arg_type": "Tensor",
        "value": { "shape": [2, 2048], "dtype": "c10::BFloat16" } },
      { "arg_name": "w1", "arg_type": "Tensor",
        "value": { "shape": [60, 2816, 2048], "dtype": "c10::BFloat16" } },
      { "arg_name": "w2", "arg_type": "Tensor",
        "value": { "shape": [60, 2048, 1408], "dtype": "c10::BFloat16" } },
      { "arg_name": "sorted_token_ids", "arg_type": "Tensor",
        "value": { "shape": [1924], "dtype": "int", "init": "zeros" } },
      { "arg_name": "sorted_expert_ids", "arg_type": "Tensor",
        "value": { "shape": [61], "dtype": "int", "init": "zeros" } },
      { "arg_name": "num_valid_ids", "arg_type": "Tensor",
        "value": { "shape": [2], "dtype": "int", "init": "zeros" } },
      { "arg_name": "out", "arg_type": "Tensor",
        "value": { "shape": [2, 4, 1408], "dtype": "c10::BFloat16", "init": null } },
      { "arg_name": "topk", "arg_type": "SymInt", "value": 4 },
      { "arg_name": "block_m", "arg_type": "SymInt?", "value": 32 },
      { "arg_name": "use_non_temporal_load", "arg_type": "bool", "value": true }
    ]
  }
}
```

Now you can read: 2 tokens routed to top-4 of 60 experts, gate hidden dim 2048,
up-projection to 2816, down-projection through 1408, `[1924]` is
`sorted_token_ids` (the routing table), block tile size 32, NTL enabled.

---

## Batch Replay

### Extract IR for Multiple Events

```python
import json

repro_data = [EventReplayer(event, lazy=True).get_repro_info()
              for event in events_of_interest]

with open('event_replay_ir.json', 'w') as f:
    json.dump(repro_data, f, indent=4)
```

```bash
python batched_replay.py event_replay_ir.json            # default (timing only)
python batched_replay.py -v event_replay_ir.json         # verbose (shows args)
python batched_replay.py --op-filter aten::mm event_replay_ir.json  # filter by name
python batched_replay.py --op-limit 5 event_replay_ir.json          # first 5 ops
```

#### Example Output (`-v`)

```
[7/11] Replaying: aten::convolution
  Reconstructing arguments for 'aten::convolution'...
  Positional Args:
  input Tensor: {'shape': [20, 128, 28, 28], 'dtype': 'c10::BFloat16', 'strides': [100352, 784, 28, 1]}
  weight Tensor: {'shape': [256, 128, 3, 3], 'dtype': 'c10::BFloat16', 'strides': [1152, 9, 3, 1]}
  bias Tensor?: None
  stride SymInt[]: [2, 2]
  padding SymInt[]: [1, 1]
  dilation SymInt[]: [1, 1]
  transposed bool: False
  output_padding SymInt[]: [0, 0]
  groups SymInt: 1
  Keyword Args:
  Average time taken: 100.38 us  (median: 98.21 us)
  Successfully executed aten::convolution.
  Result: Tensor(shape=torch.Size([20, 256, 14, 14]), dtype=torch.bfloat16, device=cuda:0)
...
--- Replay Summary ---
Total operations in file: 11
Attempted replays: 11
Successful replays: 11
Errors encountered: 0
```

### Creating Standalone Replay Artifacts

Package the IR and scripts into a standalone zip for sharing and reproduction,
independent of the original model code or TraceLens:

```python
import zipfile, os
from TraceLens.EventReplay import utils as tl_utils
from TraceLens.EventReplay import batched_replay

files = [
    OUTPUT_REPRO_FILE,
    tl_utils.__file__,
    batched_replay.__file__,
    batched_replay.__file__.replace('batched_replay.py', 'batched_replay_readme.md')
]

with zipfile.ZipFile('/path/to/replay_code.zip', 'w') as zipf:
    for file in files:
        zipf.write(file, arcname=os.path.basename(file))
```

---

## Architecture

Event Replay operates in two distinct phases:

### Phase 1: IR Extraction (deterministic)

The profiler captures tensor dimensions and types but not argument names:

```
Input Dims:  [[2, 2048], [60, 2816, 2048], [60, 2048, 1408], [1924], [61], [2], ...]
Input type:  [BFloat16, BFloat16, BFloat16, Int, Int, Int, ...]
```

EventReplayer looks up the op's **registered schema** from the PyTorch dispatcher
(via `torch._C._jit_get_all_schemas()` or `torch.ops`). For example, querying
the registry for `aiter::ck_moe_stage1` returns:

```
aiter::ck_moe_stage1(Tensor(a0!) hidden_states, Tensor(a1!) w1, Tensor(a2!) w2,
    Tensor(a3!) sorted_token_ids, Tensor(a4!) sorted_expert_ids,
    Tensor(a5!) num_valid_ids, Tensor(a6!) out, SymInt topk,
    str? kernelName="", Tensor(a9!)? w1_scale=None,
    Tensor(a10!)? a1_scale=None, SymInt? block_m=None, ...) -> ()
```

This schema provides argument names, types, and defaults. EventReplayer zips the
schema with the profiler's `Input Dims` / `Input type` arrays to produce the
named, typed IR:

- **Op name** — the fully qualified operator name (e.g., `aten::mm`, `_rocm_C::paged_attention`)
- **Argument metadata** — for each positional and keyword argument:
  - Tensors: shape, dtype, strides, initialization hint
  - Scalars: concrete value (int, float, bool, str)
  - Lists: element values
  - Optionals: `null` when not provided

This phase is purely mechanical: the same profiler event always produces the same
IR. The output is a portable JSON dictionary.

**Prerequisite — ops must be registered with the PyTorch dispatcher.** If an op
is called as a plain Python function (e.g., a Triton kernel launched directly),
there is no schema to query and no IR can be extracted. The op must go through
`torch.ops`, `torch.library`, or the JIT registry. This is why aiter's CK-based
ops (`aiter::ck_moe_stage1`) produce full IR while direct Triton kernel calls do
not. The fix is wrapping such kernels in `torch.library.custom_op` so the
dispatcher has a schema to query.

### Phase 2: Init & Replay (requires judgment)

Given an IR, Event Replay:

1. **Allocates tensors** matching the recorded shapes, dtypes, and strides
2. **Initializes values** — `randn` for floating-point tensors, `zeros` for
   integer/bool tensors, `None` for optional arguments
3. **Resolves the op** to a callable function (via JIT registry, `torch.ops`,
   or direct module import)
4. **Calls the op** and optionally benchmarks it

The default initialization works well for compute-bound ops like GEMMs and
convolutions, where kernel performance is independent of input values. However,
**control and index tensors require realistic values** — zeroed-out metadata
produces behavior that is not representative of the true workload:

| Op family | Affected tensors | Effect of zeros |
|-----------|-----------------|-----------------|
| Paged Attention | `block_tables`, `seq_lens`, `query_start_loc` | Kernel sees 0 context length — does no real work |
| MoE Routing | `sorted_token_ids`, `sorted_expert_ids`, `num_valid_ids` | Kernel sees 0 valid tokens — skips all computation |

A true reproduction would require the exact tensor values from the original
execution, but the profiler doesn't capture tensor contents — only shapes and
dtypes. **Custom Initializers** bridge this gap by constructing plausible values
from shapes and metadata already in the IR, without additional instrumentation.

---

## Custom Initializers

Custom initializers fill metadata tensors with realistic values before replay.
They are applied automatically when `auto_init=True` (the default).

### Built-in Initializers

These ship with TraceLens and require no setup — they activate automatically
when the op name matches:

**`PagedAttentionInit`** — matches `_rocm_C::paged_attention`

Initializes the KV cache metadata so the attention kernel does real work:
- `block_tables` — random permutation of the physical block pool (simulates
  realistic scattered memory allocation)
- `seq_lens` — all sequences set to `max_seq_len`
- `query_start_loc` — CSR indptr encoding per-sequence query token counts.
  When iteration annotations are available (see [Iteration Annotations](#iteration-annotations-vllm-traces)),
  uses the exact prefill/decode split; otherwise falls back to heuristics

**`MoeRoutingInit`** — matches `aiter::ck_moe_stage1`, `aiter::ck_moe_stage2`

Constructs a complete token-to-expert routing table:
- `sorted_token_ids` — padded to `block_m` boundaries per expert
- `sorted_expert_ids` — block-level expert assignment
- `num_valid_ids` — total valid (non-padding) token slots

Supports configurable token distribution via `init_kwargs`:

```python
# Default: uniform random assignment across experts
replayer = EventReplayer(event, device='cuda')

# Zipf: skewed distribution (few experts get most tokens, closer to real routing)
replayer = EventReplayer(event, device='cuda',
                         init_kwargs={"moe_distribution": "zipf", "moe_zipf_s": 1.5})
```

### Writing Your Own Initializer

If you're replaying an op that needs realistic tensor content but isn't
covered by the built-ins, you can write your own custom initializer in
three steps. For real-world examples, see `PagedAttentionInit` and
`MoeRoutingInit` in `TraceLens/EventReplay/custom_inits.py`.

**Step 1 — Subclass `CustomInit`.** Set `op_patterns` to the exact op name(s)
you want to target. This is an exact match against the profiler event name
(e.g., `"aten::index_add_"`, not `"index_add"`):

```python
from TraceLens.EventReplay import EventReplayer, CustomInit

class IndexAddInit(CustomInit):
    op_patterns = ["aten::index_add_"]
```

**Step 2 — Implement `initialize()`.** This method receives the `replayer`
object and mutates its tensors **in-place** before the op executes. You have
access to:

- `replayer.args` — list of allocated tensors/scalars (in schema order)
- `replayer.kwargs` — dict of keyword arguments
- `replayer.event` — the raw profiler event dict
- `replayer.event_replay_IR` — the extracted IR with named argument metadata

Look up arguments **by name** from the IR rather than hardcoding positional
indices — this keeps your initializer robust to schema changes across library
versions:

```python
    def initialize(self, replayer, **kwargs):
        import torch

        ir = replayer.event_replay_IR
        arg_names = [a["arg_name"] for a in ir["list_pos_args"]]

        self_tensor = replayer.args[arg_names.index("self")]
        dim = replayer.args[arg_names.index("dim")]
        index = replayer.args[arg_names.index("index")]

        dim_size = self_tensor.shape[dim]
        index.copy_(torch.randint(0, dim_size, index.shape,
                                  device=index.device))

        return (f"[custom init] index_add — index randint(0, {dim_size}), "
                f"shape={list(index.shape)}")
```

In this example, `aten::index_add_` accumulates source rows into `self` at
positions given by `index`. The default zero-init makes every row land on
row 0 — not representative of the real scatter pattern. The initializer
fills `index` with random valid indices so the kernel exercises realistic
memory access.

**Step 3 — Register it.** Once registered, the initializer fires automatically
on every future replay of matching ops:

```python
EventReplayer.register_custom_init(IndexAddInit())
```

When `replay()` runs with `auto_init=True`, it iterates over registered
initializers and applies the **first match** (built-ins are checked first,
then user-registered ones in order).

To see what's currently registered:

```python
EventReplayer.list_custom_inits()
```

---

## Auto-Import for Custom Ops

When EventReplayer encounters an op from a non-`aten` namespace (e.g.,
`_rocm_C::paged_attention`, `aiter::ck_moe_stage1`), it automatically attempts
to import the library that registers the op's schema. The import is conditional
— it only fires if the namespace is recognized and hasn't been attempted yet.

Built-in namespace mappings:

| Namespace | Imported modules |
|-----------|-----------------|
| `aiter` | `aiter` |
| `_rocm_C` | `vllm._rocm_C` |
| `_C` | `vllm._C` |
| `vllm` | `vllm._C`, `vllm._rocm_C` |

Register additional namespaces:

```python
EventReplayer.register_namespace("my_lib", ["my_lib.ops"])
```

---

## Iteration Annotations (vLLM traces)

### The problem

Paged attention's `query_start_loc` is a CSR indptr array that encodes how
many query tokens each sequence contributes. In a mixed batch (common in
vLLM's continuous batching), some sequences are **prefill** (many query tokens)
and others are **decode** (1 token each). The profiler captures the tensor shape
but not the per-sequence breakdown, so without additional information the
custom initializer has to guess — and guessing wrong changes the compute
pattern significantly (prefill is quadratic in sequence length, decode is
linear).

### How vLLM exposes this

vLLM emits a `user_annotation` event for each `execute_model` iteration
with the exact prefill/decode composition encoded in the name:

```
execute_context_2(18)_generation_5(5)
```

This is an **iteration annotation** — it describes one forward pass. Here it
means: **2 prefill sequences** with **18 total query tokens**, and **5 decode
sequences** with **5 tokens** (1 each).

### Extracting batch context from iteration annotations

`extract_batch_context` parses these iteration annotations by timestamp and
attaches a `batch_context` dict to each paged attention event that falls
within the annotation's time range:

```python
from TraceLens import TreePerfAnalyzer
from TraceLens.EventReplay import EventReplayer, extract_batch_context

analyzer = TreePerfAnalyzer.from_file("vllm_trace.json")

# Annotate paged attention events with prefill/decode split
num_annotated = extract_batch_context(analyzer)
print(f"Annotated {num_annotated} paged_attention events")

# Now replay — PagedAttentionInit reads event["batch_context"] automatically
event = analyzer.tree.get_UID2event(some_uid)
replayer = EventReplayer(event, device='cuda')
replayer.replay()  # query_start_loc reflects the real prefill/decode split
```

After `extract_batch_context`, each annotated event carries:

```python
event["batch_context"] = {
    "n_prefill": 2,       # number of prefill sequences
    "prefill_tokens": 18,  # total query tokens across prefill sequences
    "n_decode": 5,         # number of decode sequences
    "decode_tokens": 5,    # total query tokens across decode sequences (1 each)
}
```

`PagedAttentionInit` uses this to build `query_start_loc` accurately:
prefill sequences get `prefill_tokens / n_prefill` tokens each, decode
sequences get 1 token each.

### Without iteration annotations

`PagedAttentionInit` still runs — it always initializes `block_tables`
(random permutation of the physical block pool) and `seq_lens` (set to
`max_seq_len`). The only difference is how `query_start_loc` is built.
Without annotations, it falls back to heuristics:

- `query_tokens == num_seqs` → assumes pure decode (1 token/seq)
- `query_tokens > num_seqs` → assumes pure prefill (tokens distributed
  uniformly)

This is a reasonable approximation for homogeneous batches but inaccurate
for mixed prefill+decode batches, where the iteration annotation provides
the exact split.

---

## Known Limitations

- **Unregistered ops are invisible.** Triton kernels called directly from Python
  (e.g., aiter's Triton attention path) have no schema in the profiler. The fix
  is wrapping them in `torch.library.custom_op` — a one-time registration effort
  in the upstream library.

- **Single-op isolation vs. real workload.** Replay runs each op in isolation
  with no surrounding memory traffic. Timings are a lower bound on in-model
  performance. Sequence replay (ops in trace order to reproduce natural cache
  pollution) is a planned enhancement.

- **Data-dependent kernels.** Custom initializers provide plausible but not exact
  values from the original execution (the profiler doesn't capture tensor
  contents). For most ops this doesn't matter; for ops with data-dependent
  control flow (e.g., sparse attention with variable sequence lengths), timing
  may vary.

---

## Use Cases

- **Trace Interpretation**: Translate opaque profiler arguments into named, typed JSON for understanding what each op actually computes.
- **Performance Debugging**: Isolate and reproduce performance issues from large models without running the model.
- **Regression Testing**: Automate benchmarks to detect performance regressions at the operator level.
- **Kernel Development**: Extract minimal reproducers for GPU kernel optimization and debugging.
- **Portable Sharing**: Package IR + replay scripts as standalone zip artifacts for teammates or upstream repos.
