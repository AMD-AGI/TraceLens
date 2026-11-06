<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Project a model's roofline from its checkpoint alone
```{meta}
:description: Project a kernel-by-kernel roofline for a whole model from its checkpoint config and modeling source, with no trace or GPU run.
:keywords: TraceLens, roofline, FLOPS, GEMM, SDPA, Normalization, static analysis, sizing study, no trace, ROCm, MI300X
```

`TraceLens_model_roofline_projection` builds the same merged Model Explorer
graph as `TraceLens_visualize_model_in_explorer` (statically parsing
`modeling_*.py` instead of executing any model code), maps every op with a
direct [`TraceLens.PerfModel`](./perf-model-without-trace.md) formula to its
FLOPs/bytes, and prints a kernel-by-kernel roofline table plus a model-level
rollup — no trace and no GPU run required. Reusing the same graph means a
`--tp`/`--pp`/`--ep` split draws, and costs, the exact per-GPU local shapes the
viewer would show for that split, not a second, independently-matched
approximation of it.

## Before you begin

Confirm you have the following before continuing.

- [TraceLens installed](../install/install.md).
- A Hugging Face model id or local checkpoint directory with a `config.json`
  and `modeling_*.py` (see `--github` below for checkpoints that don't ship
  their own modeling source).

## Run the tool

```bash
TraceLens_model_roofline_projection meta-llama/Llama-3.1-8B \
    --batch-size 1 --seq-len 4096 \
    --gpu-arch-name MI300X
```

This prints a table ranked by projected roofline time, one row per op in a
representative decoder layer (each annotated with that op's own repeat count —
the merged graph's own tile count for the structural variant it sits in, so a
hybrid layer mix like MoE-every-few-layers is already attributed per-variant,
not spread evenly across every layer), plus ops outside the repeated layer
stack (embedding, LM head) at their own single-instance cost. A rollup line
reports total FLOPs/bytes, the compute-bound vs. memory-bound time split, and
what fraction of the model's output elements were actually modeled.

## Projecting one GPU's share of a sharded deployment

Add `--tp`/`--pp`/`--ep` to project one GPU's own share of the work under a
tensor-/pipeline-/expert-parallel split, the same split
`TraceLens_visualize_model_in_explorer` draws:

```bash
TraceLens_model_roofline_projection MiniMaxAI/MiniMax-M3.1-Flash \
    --engine vllm --tp 2 --ep 4 \
    --batch-size 1 --seq-len 2048 \
    --gpu-arch-name MI325X
```

The split itself comes from the checkpoint's declared `base_model_tp_plan`/
`base_model_pp_plan`/`base_model_ep_plan` (Hugging Face), or, with `--engine
vllm`/`--engine sglang`, from that inference engine's own parallel-layer
classes (`ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, a
`FusedMoE`'s expert count, ...) — including a model-specific subclass of one of
these. Each GEMM is costed at its per-GPU local shape (a column-sharded
projection's output divided by the TP count, an expert stack's expert count
divided by the EP count, ...), derived by re-running the real shape-inference
engine against a locally-divided config, not by guessing which axis a shape
text happens to match. `1` on any of `--tp`/`--pp`/`--ep` leaves that axis
unsplit (the default: a single, unsharded GPU).

Without `--gpu-arch-name`/`--gpu-arch-json-path`, FLOPs and bytes are still
reported but no roofline time is computed (no hardware peaks to bound against).

## Flags

- `--batch-size`, `--seq-len`: the shape to project at (default: 1, 2048).
- `--num-predicted-tokens N`: also cost a secondary MTP / speculative draft
  stack of N tokens (default: 0, backbone only), capped at the depth the
  checkpoint declares (`mtp_config.num_nextn_predict_layers` /
  `num_mtp_layers` / `num_nextn_predict_layers`). See
  [Speculative draft stack](#speculative-draft-stack).
- `--dim KEY=VALUE` (repeatable): override a specific symbolic dimension
  (e.g. `--dim past_len=1024`); takes precedence over `--batch-size`/`--seq-len`
  for the same key.
- `--gpu-arch-name NAME`: a bundled GPU arch spec (`MI300X`, `MI325X`, ...).
- `--gpu-arch-json-path PATH`: a custom GPU arch JSON (same schema).
- `--output-csv PATH` / `--output-json PATH`: write the full per-op table (JSON
  also includes the rollup).
- `--top-n N`: rows printed to stdout (default: 20; `0` for all).

## Model source and runtime regime flags

The checkpoint-provenance flags (`--github`, `--config-path`, `--code-path`,
`--basic-op-add`/`--basic-op-remove`, `--meta-shapes`/`--no-meta-shapes`), the
runtime-regime flags (`--phase`, `--cache`, `--sequence-layout`, `--attention`,
`--mode`, `--device`, `--config-override`), and the device-partition flags
(`--engine`, `--tp`, `--pp`, `--ep`) are the same ones
`TraceLens_visualize_model_in_explorer` supports, since both tools build the
checkpoint's graph through the same backend — run
`TraceLens_model_roofline_projection --help` for the full list.

Unlike the Model Explorer export, this tool always stops AST expansion exactly
at ops with a direct `TraceLens.PerfModel` formula (`Linear`, the Normalization
family, `Softmax`, ...) rather than expanding into their internal primitive
steps — there is no `--stop-at-roofline-ops` flag here because turning it off
would only replace a correctly-costed op with several the projection has no
formula for.

## How a cost is chosen

Each row comes from the first source that applies:

1. **The drawn graph.** A leaf with a direct PerfModel formula is costed from
   that node's shape. A linear already drawn under an attention tile is that
   tile's projection, so a second analytical QKV row is not added beside it.
   The same rule skips a router or shared expert the MoE tile already draws.
2. **The parsed modeling source.** A linear's `(in, out)` comes from the class
   that assigns it. Two classes that size the same attribute differently each
   keep their own width; the shared name is used only when every class agrees.
   An engine parallel linear (`ColumnParallelLinear`, `RowParallelLinear`,
   `QKVParallelLinear`, `MergedColumnParallelLinear`, `ParallelLMHead`) is
   sharded on the axis that class shards. A plain `Linear` keeps an axis the
   formula already reduced to one GPU's share.
3. **The checkpoint config.** Widths the source did not resolve (heads, hidden
   size, expert intermediate size, latent ranks) come from `config.json`.
4. **An explicit fallback.** Used only for a speculative draft when the source
   has no draft class. Those rows are the fixed template below, not a silent
   substitute for a parsed layer.

A vLLM or SGLang run whose modeling source did not load, or whose constructors
did not resolve to widths, records that in the JSON rollup's `notes`. The
widths then come from the checkpoint.

## Op coverage and approximations

Costed op families: GEMM (`Linear`/`matmul`/`bmm`, including a vLLM/SGLang
`*ParallelLinear`/`ReplicatedLinear` class or a model-specific subclass of
one), SDPA, the Normalization family (RMSNorm, LayerNorm, GroupNorm,
BatchNorm, InstanceNorm — matched by subclass name, since real checkpoints
almost always subclass these), Softmax, generic elementwise/reduction ops,
MoE expert routing, and convolutions whose weight geometry is known.
Persistent tensors that cross forward calls (for example, recurrent state or
a cache) are modeled as separate memory-bound reads and writes from each
boundary's traced shape and dtype. This works for stateful attributes and
functional buffer mutation without model-specific cache conventions.
Embeddings and reshapes/transposes are reported as
**unmodeled** rather than silently dropped, so the rollup's coverage share
tells you how much of the model the projection actually accounts for.

### Attention

An opaque attention tile is the SDPA core plus the projections packed inside
it. When that class assigns linears and their constructor widths resolve,
those assignments are the projection rows. Otherwise the tile is one packed
QKV projection plus the output projection, or the latent projections when the
checkpoint sets `kv_lora_rank` (and `q_lora_rank` when the query is low-rank
too).

Latent attention scores against one compressed key of width `kv_lora_rank +
qk_rope_head_dim`. The query projection width is `qk_nope_head_dim +
qk_rope_head_dim` when those ranks are set, and the value width is
`v_head_dim`. `head_dim` is the width only when those ranks are absent.

`index_n_heads` and `index_head_dim` add the indexer query and key GEMMs, a
score GEMM over the context, and a top-k reduction. `index_topk` is the
length of that indexed attention. A layer that already has a window keeps the
window: indices in `local_layer_ids`, paired with `layer_types`, use
`sliding_window`, and the other layers attend the full context.

QK-norm rows are added when the checkpoint sets `qk_norm` or `use_qk_norm`.
RoPE is added when the architecture's positional encoding is RoPE. A short
depthwise convolution packed inside a fused attention leaf is costed when the
checkpoint sets `sconv_kernel_size` or `conv_kernel_size`.

SDPA follows the checkpoint's causal flag. An explicit `is_causal` or `causal`
wins. Otherwise a decoder-only model is causal, and an encoder, BERT, T5, or
vision model is not. The mask applies on cold prefill, when the query length
is the whole key/value length. A warm cache keeps the unmasked FLOP count.

### Quantized weights

`quantization_config` sets the weight traffic: fp4 / MXFP4 at 4 bits plus the
block scale, fp8 / int8 at one byte, int4 / NF4 at half a byte. A weight-only
scheme adds a weight-dequant row beside the GEMM. An fp8 matrix also quantizes
the activation. Modules named in `modules_to_not_convert`, `ignore`, or
`exclude_layers` stay at the activation dtype, which is how an LM head listed
there avoids the quantized weight width. A published fp4 or fp8 matrix peak is
used only for a GEMM or MoE row, and only when the selected GPU lists that
peak. Attention stays at the activation dtype.

### MoE and dense MLP

MoE routing is `experts_per_tok` grouped-GEMM passes per token (the gate-up and
down projections of the selected experts), never one pass per expert. The
weight traffic uses `G = min(experts on this GPU, active tokens on this GPU)`
distinct expert matrices. Tensor parallel column-shards each expert's
intermediate width. Expert parallel gives one GPU the floor of the routed rows
and adds the dispatch and combine all-to-all as data movement. The router GEMM,
its softmax, the combine, and a shared expert are separate rows when the graph
did not already draw them. A dense MLP on the same attribute is a SwiGLU over
every token at `intermediate_size`, not that routed subset.

GEMM bias is included when the node records it, or when the checkpoint sets
`attention_bias`, `mlp_bias`, or `lm_head_bias` / `head_bias`. Every
normalization is assumed affine. RMSNorm is assumed bias-free. Every other norm
is assumed to have a bias.

### Convolutions

A graph convolution is costed only when the meta-device pass threaded its
weight geometry onto the node (`conv_weight`, `conv_bias`, `conv_transposed`).
Input and output shapes alone do not recover the kernel or the groups. With
`--meta-shapes` (the default when a checkpoint is available), a meta-device
forward can replace those graph rows with one row per live convolution. A
convolution with neither threaded geometry nor a declared short-conv kernel
stays unmodeled.

## Speculative draft stack

`--num-predicted-tokens N` adds one draft depth per predicted token, capped at
the depth the checkpoint declares. The depth is the speculative class in the
modeling source (a class or assignment whose name matches MTP, NextN, EAGLE,
or the other speculative-decoding names the parser already uses):

- its norms and its own linear projections, at that class's constructor widths
- its attention, including the latent and indexer rows when the checkpoint
  declares them
- its feed-forward, dense or MoE, including a feed-forward that differs from
  the backbone
- an LM head only when the draft class assigns one

A width that differs from the backbone comes from the draft class. A width that
matches the backbone stays the checkpoint value.

When the exported graph draws that draft module, the leaves it drew are kept
and are not costed a second time. An attention core or expert GEMM the graph
inlined away is still added. When the graph does not draw the draft, the whole
parsed depth is added. When the source has no draft class, the depth is the
fixed template: two RMSNorms, a 2H→H projection, one attention, a dense MLP,
and the LM head.

## Run from a chat

A portable skill drives this CLI for agentic runners that support skill-file
discovery (Claude Code, Cursor, Codex). In a chat with a capable model,
invoke:

```text
Follow the model roofline projection skill installed with TraceLens and
project <model> at batch size <N> and sequence length <N> on <GPU arch>.
```

See [TraceLens/Agent/Visualizer/skills/model-roofline-projection/](../../TraceLens/Agent/Visualizer/skills/model-roofline-projection/SKILL.md)
for the skill's full procedure.

## Related topics

- [Plot an end-to-end roofline](./plot-e2e-roofline.md)
- [Model op performance without a trace](./perf-model-without-trace.md)
- [GEMM analysis in TraceLens](../conceptual/gemm-analysis.md)
- [API reference](../reference/api-reference.md)
