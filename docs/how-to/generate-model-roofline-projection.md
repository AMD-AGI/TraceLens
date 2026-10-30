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

## Op coverage and approximations

Costed op families: GEMM (`Linear`/`matmul`/`bmm`, including a vLLM/SGLang
`*ParallelLinear`/`ReplicatedLinear` class or a model-specific subclass of
one), SDPA, the Normalization family (RMSNorm, LayerNorm, GroupNorm,
BatchNorm, InstanceNorm — matched by subclass name, since real checkpoints
almost always subclass these), Softmax, generic elementwise/reduction ops, and
MoE expert routing. A vLLM/SGLang `ParallelLMHead`/`FusedMoE` boundary — read
from the engine's own source, where shape_inference has no rule for either —
is costed from the architecture's own dims (vocab, hidden,
experts-per-token, expert intermediate size) instead of its node shape; see
"Known approximations" below. Everything else — embeddings, convolutions,
reshapes/transposes — is reported as **unmodeled** rather than silently
dropped, so the rollup's coverage share tells you how much of the model the
projection actually accounts for.

Known approximations:

- GEMM bias is assumed absent.
- Every Normalization op is assumed affine; RMSNorm is assumed bias-free, every
  other norm is assumed to have one.
- SDPA is assumed causal exactly when the query and key/value sequence lengths
  resolve equal (a self-attention prefill shape).
- MoE routing is costed as `experts_per_tok` GEMM passes per token (the
  gate_up_proj and down_proj of only the selected experts), never
  `num_experts` passes. The router's own softmax/top-k selection and any
  shared-expert add are not separately costed.

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

- [Model op performance without a trace](./perf-model-without-trace.md)
- [GEMM analysis in TraceLens](../conceptual/gemm-analysis.md)
- [API reference](../reference/api-reference.md)
