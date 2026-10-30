<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Derive a Loom port's cfg.json architecture facts from a checkpoint

```{meta}
:description: Derive the architecture-fact fields of a Loom loom-port-llm cfg.json from a Hugging Face checkpoint's own config.json and modeling source, with no GGUF conversion.
:keywords: TraceLens, Loom, loom-port-llm, GGUF, cfg.json, RoPE, GQA, sliding window, static analysis, no trace, ROCm
```

`TraceLens_loom_port_cfg` loads a Hugging Face checkpoint through the same AST
backend as `TraceLens_visualize_model_in_explorer` and
`TraceLens_model_roofline_projection` (statically parsing `modeling_*.py`
instead of executing any model code), and derives the subset of
[`loom-port-llm`](https://github.com/loom-fun/loom-skills/tree/main/skills/loom-port-llm)'s
`cfg.json` architecture fields that follow directly from the checkpoint's own
`config.json` and modeling source — no GGUF conversion required.

## Why this exists

`loom-port-llm`'s `starter/gen_llm.py` turns a declarative `cfg.json` into
Loom IR kernels for the dense-transformer family (Llama/Qwen2/Qwen3/Mistral-
style). That `cfg.json` normally comes from the skill's own `inspect_gguf.py`,
which reverse-engineers GGUF tensor names and metadata and either hardcodes or
leaves blank several fields that its own "cfg.json review checklist" tells a
human to fix by reading `llama.cpp`'s `src/models/<arch>.cpp` — `rope_type` is
guessed from the architecture name, `qk_norm`/`attn_gate`/`post_norms` are
guessed from GGUF tensor-name presence, and `sliding_layers`/`nope_layers` are
always emitted empty.

`TraceLens.ModelUtils.extract.ArchitectureSpec` already parses the
checkpoint's real `config.json` and modeling source — ground truth, not a
GGUF round-trip — so several of these follow directly from the checkpoint
itself. This tool is a thin mapping from that spec to the `cfg.json` fields it
can state with confidence, plus a `--check` mode that diffs against an
already-authored `cfg.json` to flag mismatches.

## Before you begin

Confirm you have the following before continuing.

- [TraceLens installed](../install/install.md).
- A Hugging Face model id or local checkpoint directory with a `config.json`
  and `modeling_*.py` (see `--github` below for checkpoints that don't ship
  their own modeling source).

## Run the tool

```bash
TraceLens_loom_port_cfg meta-llama/Llama-3.1-8B
```

This prints the derived facts as JSON to stdout, and any coverage notes (MoE,
MLA, or an unrecognized hybrid layer mix the starter kit doesn't cover) to
stderr. Write it to a file with `--output`:

```bash
TraceLens_loom_port_cfg Qwen/Qwen3-8B --output cfg.json
```

## Checking an already-authored cfg.json

`--check` diffs the derived facts against an existing `cfg.json` (for example
one produced by `loom-port-llm`'s `inspect_gguf.py`) instead of writing one,
automating part of its "cfg.json review checklist":

```bash
TraceLens_loom_port_cfg Qwen/Qwen3-8B --check ./cfg.json
```

Exits `0` with "No mismatches" when every field this tool derives agrees with
the existing file, or prints each disagreement and exits `1` otherwise. A
field this tool never derives is never flagged — `--check` only cross-checks
what TraceLens can independently confirm.

## Field coverage

Always present (`null` when the checkpoint's own config never declared one):
`hidden`, `layers`, `heads`, `kv_heads`, `head_dim`, `ffn`, `vocab`,
`context_length`, `tied_embeddings`, `name`, `arch`.

Present when the config states it: `rms_eps`, `rope_theta` (defaulted to
`10000.0` when RoPE applies but no explicit theta is stated), `rope_dims`
(from `partial_rotary_factor`), `qk_norm`, `sliding_layers`/`sliding_window`
(an explicit per-layer list from `config.layer_types`, or the
`max_window_layers` convention), `recurrent_layers` (layer indices whose
`layer_types` entry names neither full nor sliding attention — a
gated-delta-net/linear-attention/Mamba-style hybrid layer).

**Deliberately not derived** — left to `inspect_gguf.py` / manual review,
and never guessed:

- `rope_type` (neox vs. norm rotary pairing) is a `llama.cpp` GGUF
  weight-**permutation** convention introduced by the GGUF conversion step,
  not a property of the HF modeling source.
- `attn_gate`, `post_norms`, `embed_scale`, `embed_norm` need
  submodule/weight-name AST inspection this tool doesn't do yet.
- `nope_layers`: when the config states `no_rope_layers`, a note flags it for
  manual translation rather than guessing its `0`/`1` polarity.
- `gdn` kernel geometry (conv kernel size, head counts) beyond which layers
  are recurrent.
- `tensors` (quantized format/shape per weight), the `gguf` path, and the
  Jinja-chat-template-derived `chat` block / `eos_id`/`bos_id` — these come
  from the GGUF conversion and tokenizer, not the HF modeling source.

## No `--tp`/`--pp`/`--ep`, no GPU-arch/batch/seq-len flags

`loom-serve` runs a model whole on one device — nothing in `loom-port-llm`
shards a model across GPUs — so `cfg.json`'s fields describe the model's full
per-layer geometry and this tool has no partition flags. It is also a static
architecture description, not a workload projection, so the roofline CLI's
`--batch-size`/`--seq-len`/`--gpu-arch-name` don't apply either.

## Model source and runtime regime flags

The checkpoint-provenance flags (`--github`, `--config-path`, `--code-path`,
`--basic-op-add`/`--basic-op-remove`, `--meta-shapes`/`--no-meta-shapes`) and
the runtime-regime flags (`--phase`, `--cache`, `--sequence-layout`,
`--attention`, `--mode`, `--device`, `--config-override`) are the same ones
`TraceLens_visualize_model_in_explorer`/`TraceLens_model_roofline_projection`
support, since all three tools build the checkpoint's architecture through the
same backend — run `TraceLens_loom_port_cfg --help` for the full list.

## Related topics

- [Project a model's roofline from its checkpoint](./generate-model-roofline-projection.md)
- [Model op performance without a trace](./perf-model-without-trace.md)
- [API reference](../reference/api-reference.md)
