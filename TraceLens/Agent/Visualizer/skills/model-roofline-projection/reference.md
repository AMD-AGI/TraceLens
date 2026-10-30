<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model roofline projection — reference

Detailed procedure for the **model-roofline-projection** skill ([SKILL.md](SKILL.md)). Read it when executing the workflow. The full flag list and op-coverage/approximation notes are in [the how-to guide](../../../../../docs/how-to/generate-model-roofline-projection.md); this file is the decision procedure.

## Contents

- [Workflow](#workflow)
- [Step 0: Resolve inputs](#step-0-resolve-inputs)
- [Step 1: Install](#step-1-install)
- [Step 2: Command](#step-2-command)
- [Step 3: Verify and report](#step-3-verify-and-report)
- [GPU architecture](#gpu-architecture)
- [Shape and dimension overrides](#shape-and-dimension-overrides)
- [Device sharding](#device-sharding)
- [Config overrides, attention, and runtime regime](#config-overrides-attention-and-runtime-regime)
- [Output formats](#output-formats)
- [Footguns](#footguns)

## Workflow

```
0. Resolve every stated option (ask only for what the prompt left out)
1. Ensure the Visualizer extra imports
2. Run one TraceLens_model_roofline_projection command
3. Read the table and the rollup line(s)
4. Report the roofline numbers and the modeled-element coverage to the user
```

The tool reuses the same `TraceLens.ModelUtils` AST backend as the model visualizer to build one merged graph (parsing `modeling_*.py`, no checkpoint weights, no GPU), then costs every op with a direct `TraceLens.PerfModel` formula against the requested shape and, when given, a GPU architecture spec. Everything else is reported as unmodeled.

## Step 0: Resolve inputs

Take every value the user already gave. Ask only for the gaps.

| Input | Required | Default when omitted |
| --- | --- | --- |
| Model source | yes | Hugging Face id, local checkpoint directory, or `--github` plus a whitelisted repo |
| Batch size / sequence length | no | `--batch-size 1 --seq-len 2048` |
| A specific dimension (e.g. a decode cache length) | no | `--dim KEY=VALUE`, repeatable; wins over `--batch-size`/`--seq-len` for the same key |
| GPU architecture | no | none — FLOPs/bytes only, no roofline time. Ask if the user wants a time projection and did not name a GPU |
| Sharding source | no | Hugging Face plans on the checkpoint. `--engine` only when the user names vLLM or SGLang |
| Partition counts | no | `--tp 1 --pp 1 --ep 1` (that axis stays unsplit) |
| Runtime regime (phase, cache, sequence layout, attention, mode, device) | no | prefill, cold, rectangular, checkpoint's own attention, inference, hip |
| Output format | no | stdout table, top 20 rows ranked by roofline time |

A local directory is a checkpoint when it contains `config.json`. A path to `modeling_*.py` is `--code-path`, not the positional source. A GitHub repo that is not already whitelisted needs `--allow-repo OWNER/REPO` or `TRACELENS_ALLOWED_GITHUB_REPOS`. Gated Hugging Face repos need `HF_TOKEN` in the environment. Do not invent a token.

Before writing the command, walk [GPU architecture](#gpu-architecture), [Device sharding](#device-sharding), and [Config overrides, attention, and runtime regime](#config-overrides-attention-and-runtime-regime). Each option the user named becomes a flag. An option they left out stays at the tool's default.

## Step 1: Install

Check the exporter before doing anything else:

```bash
python -c "import TraceLens.Visualizer.model_explorer_export.roofline_cli"
```

When that import fails, install the extra from a source checkout:

```bash
pip install -e ".[Visualizer]"
```

When the checkout is not the current directory, install the package extra instead:

```bash
pip install "TraceLens[Visualizer]"
```

Python 3.10 or newer. The extra pulls `torch`, `transformers`, and `huggingface_hub`. Do not `pip install` those yourself, and do not install a ROCm or CUDA wheel: the meta-device shape pass allocates no real tensors and the roofline projection itself makes no device calls at all.

Confirm the console script:

```bash
TraceLens_model_roofline_projection --help
```

If the script is missing but the import works, call `python -m TraceLens.Visualizer.model_explorer_export.roofline_cli` with the same arguments.

## Step 2: Command

Run one command.

FLOPs/bytes only, no GPU time (the user did not name hardware):

```bash
TraceLens_model_roofline_projection <model> --batch-size 1 --seq-len 4096
```

With a bounded roofline time against a named GPU:

```bash
TraceLens_model_roofline_projection <model> \
    --batch-size 1 --seq-len 4096 \
    --gpu-arch-name MI300X
```

One GPU's share under a sharded deployment, same split the visualizer draws:

```bash
TraceLens_model_roofline_projection MiniMaxAI/MiniMax-M3.1-Flash \
    --engine vllm --tp 2 --ep 4 \
    --batch-size 1 --seq-len 2048 \
    --gpu-arch-name MI325X
```

A decode step at a specific cache length, with a custom GPU arch file:

```bash
TraceLens_model_roofline_projection <model> \
    --phase decode --dim past_len=1024 \
    --gpu-arch-json-path ./mi300x.json
```

`<model>` is a Hugging Face id or a local checkpoint directory. `1` on any of `--tp`/`--pp`/`--ep` leaves that axis unsplit (a single, unsharded GPU). A partition count that does not divide the model evenly is still exported, with a warning on stdout — report it rather than silently rerunning with a different count.

## Step 3: Verify and report

Success is exit code 0 and a printed table. Read, in order:

1. The table itself (ranked by `total_roofline_time_us`, top 20 rows unless `--top-n` changed that), for the hottest ops.
2. The `Total:` line: total FLOPs, total bytes, modeled vs. unmodeled op counts, and the modeled-output-element share. Report this share — it is how much of the model the projection actually accounts for, not a detail to drop.
3. The `Roofline time:` line, when a GPU arch was given: total roofline time plus the compute-bound/memory-bound split. Its absence (`No --gpu-arch-name/--gpu-arch-json-path given...`) means report FLOPs/bytes only and say why there is no time estimate.

On a non-zero exit, show stderr to the user (`Error loading architecture: ...`). Retry once only when the error is a missing modeling file and a documented source switch fixes it (`--code-path`, `--github` with `--allow-repo`, or `--engine`).

## GPU architecture

| User named | Pass |
| --- | --- |
| A bundled GPU (`MI300X`, `MI325X`) | `--gpu-arch-name MI300X` |
| A custom or measured arch spec | `--gpu-arch-json-path PATH` (same JSON schema; takes precedence over `--gpu-arch-name`) |
| Nothing about hardware | omit both. FLOPs/bytes are still printed; the rollup says no roofline time was computed |

Do not guess a GPU the user did not name. Ask, unless the request is explicitly "just the FLOPs" or "just the bytes".

## Shape and dimension overrides

`--batch-size` and `--seq-len` set the common two axes. `--dim KEY=VALUE` (repeatable) overrides one named symbolic dimension directly and wins over `--batch-size`/`--seq-len` for that same key — use it for anything the two flags do not name (a cache length, an expert count override that is not already a sharding axis, etc.).

```bash
--dim past_len=1024
```

A `--dim` value must parse as an int; the tool raises a usage error otherwise.

## Device sharding

Identical semantics to the model visualizer: `--tp`, `--pp`, and `--ep` are the partition counts, `--engine` chooses whose plan those counts are applied to, and a count of 1 leaves that axis unsplit.

| User named | What to pass |
| --- | --- |
| Nothing about an engine, or "Hugging Face", "transformers", or "the checkpoint's plan" | no `--engine`. The split is `base_model_tp_plan`, `base_model_pp_plan`, and `base_model_ep_plan`, plus the causal-LM head |
| vLLM | `--engine vllm` plus the counts |
| SGLang | `--engine sglang` plus the counts |

Each GEMM is costed at its per-GPU local shape (a column-sharded projection's output divided by the TP count, an expert stack's expert count divided by the EP count, ...), derived from the same shape-inference engine the visualizer uses against a locally-divided config — not a second, independently-matched approximation.

## Config overrides, attention, and runtime regime

Same flags and precedence rules as the model visualizer: `--config-override KEY=VALUE` (repeatable, dotted keys reach nested configs, last value wins), `--attention` (leave unset to keep the checkpoint's own implementation), `--phase`/`--cache`/`--sequence-layout`/`--mode`/`--device` for the runtime regime. An option the user named is never dropped because some other default exists. See the model-visualizer skill's [reference.md](../model-visualizer/reference.md#config-overrides) for the full decision tables; they apply unchanged here.

## Output formats

| User asked for | Flag |
| --- | --- |
| Just the printed table | nothing extra; adjust `--top-n` (default 20, `0` for every row) |
| The full per-op table as a file | `--output-csv PATH` |
| The full per-op table plus the rollup as JSON | `--output-json PATH` |
| A display name for the architecture | `--title "..."` |

`--output-csv`/`--output-json` do not suppress the stdout table; both are written in addition to it.

## Footguns

- **No GPU arch means no time, not an error.** A run with neither `--gpu-arch-name` nor `--gpu-arch-json-path` exits 0 and prints FLOPs/bytes with an explicit "no roofline time" line. That is expected, not a bug to retry.
- **Unmodeled is not a failure mode.** Embeddings, convolutions, reshapes, and anything else with no direct `TraceLens.PerfModel` formula are reported, not silently skipped. A low "% of output elements modeled" is a real statement about coverage for that architecture, not a tool error — surface it rather than chasing 100%.
- **MoE routing is approximated as `experts_per_tok` GEMM passes**, never `num_experts`; the router's own softmax/top-k and any shared-expert add are not separately costed. State this when a MoE model's rollup is the headline number.
- **`--device` is not the host.** As with the visualizer, the default `hip` means the drawing follows HIP branches in the modeling source. The projection still runs on CPU regardless of this flag.
- **`--engine` and a named source do different jobs.** `--code-path` or `--github` picks the modeling file; `--engine` still picks the plan parser for the partition counts. Omit `--engine` for a Hugging Face plan; pass it for vLLM or SGLang even when the file is local.
- **This is not a trace tool.** No `.json`/`.pftrace` profiler trace is read here. A request to analyze a trace that already exists is the analysis orchestrator, not this skill.
- **GEMM bias, norm affine-ness, and SDPA causality are assumed**, not read from the checkpoint in every case (GEMM bias assumed absent; every norm assumed affine, RMSNorm bias-free; SDPA assumed causal exactly when query and key/value sequence lengths resolve equal). See the how-to guide's "Known approximations" for the full list before treating a number as exact.
