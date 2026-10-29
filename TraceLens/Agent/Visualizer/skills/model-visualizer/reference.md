<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model visualizer — reference

Detailed procedure for the **model-visualizer** skill ([SKILL.md](SKILL.md)). Read it when executing the workflow. The exporter's full flag list is in [the Visualizer README](../../../../Visualizer/README.md); this file is the decision procedure.

## Contents

- [Workflow](#workflow)
- [Step 0: Resolve inputs](#step-0-resolve-inputs)
- [Step 1: Install](#step-1-install)
- [Step 2: Command](#step-2-command)
- [Step 3: Verify](#step-3-verify)
- [Config overrides](#config-overrides)
- [Attention implementation](#attention-implementation)
- [Device sharding](#device-sharding)
- [Other flags](#other-flags)
- [Footguns](#footguns)

## Workflow

```
0. Resolve every stated option (ask only for what the prompt left out)
1. Ensure the Visualizer extra imports
2. Run one visualize_model_in_explorer command
3. Confirm the artifact or viewer URL
4. Report that path or URL to the user
```

The graph is built by parsing `modeling_*.py` with the Python `ast` module (`TraceLens.ModelUtils`) and, by default, a PyTorch meta-device pass that fills shapes static analysis cannot infer. No checkpoint weights are loaded and no GPU is used.

## Step 0: Resolve inputs

Take every value the user already gave. Ask only for the gaps.

| Input | Required | Default when omitted |
| --- | --- | --- |
| Model source | yes | Hugging Face id, local checkpoint directory, or `--github` plus a whitelisted repo |
| Output path | no | `<model-id-with-slashes-replaced>.html` in the working directory |
| Live viewer | no | HTML file only. `--serve` only if the user asked to open or serve the graph |
| Regime | no | rectangular, prefill, cold cache, inference, `--device hip` |
| Attention | no | unset, so the checkpoint's `_attn_implementation` is used (`sdpa` when that key is absent) |
| Config overrides | no | none. Pass one `--config-override` per key the user named |
| Sharding source | no | Hugging Face plans on the checkpoint. `--engine` only when the user names vLLM or SGLang |
| Partition counts | no | `--tp 1 --pp 1 --ep 1` (that axis stays unsplit) |
| Shapes | no | on (`--no-shapes` only when the user asks to skip them) |

Before writing the command, walk [Config overrides](#config-overrides), [Attention implementation](#attention-implementation), [Device sharding](#device-sharding), and [Other flags](#other-flags). Each option the user named becomes a flag. An option they left out stays at the exporter default. Do not pass a default that would overrule the checkpoint: leaving `--attention` unset is what preserves a checkpoint that asks for something other than `sdpa`.

A local directory is a checkpoint when it contains `config.json`. A path to `modeling_*.py` is `--code-path`, not the positional source. A GitHub repo that is not already whitelisted needs `--allow-repo OWNER/REPO` or `TRACELENS_ALLOWED_GITHUB_REPOS`.

Gated Hugging Face repos need `HF_TOKEN` in the environment. Do not invent a token.

## Step 1: Install

Check the exporter before doing anything else:

```bash
python -c "import TraceLens.Visualizer.model_explorer_export.cli"
```

When that import fails, install the extra from a source checkout:

```bash
pip install -e ".[Visualizer]"
```

When the checkout is not the current directory, install the package extra instead:

```bash
pip install "TraceLens[Visualizer]"
```

Python 3.10 or newer. The extra pulls `torch`, `transformers`, and `huggingface_hub`. Do not `pip install` those yourself, and do not install a ROCm or CUDA wheel. The meta-device pass allocates no real tensors.

Confirm the console script:

```bash
visualize_model_in_explorer --help
```

If the script is missing but the import works, call `python -m TraceLens.Visualizer.model_explorer_export.cli` with the same arguments.

## Step 2: Command

Run one command. Prefer the console script.

HTML file (the usual request):

```bash
visualize_model_in_explorer <model> -o <path>.html
```

`<model>` is a Hugging Face id or a local checkpoint directory. Omit `-o` to write `<model-with-slashes-as-underscores>.html` in the working directory.

Payload JSON, then HTML, when the user wants both artifacts. Stage 2 does not load the model:

```bash
visualize_model_in_explorer <model> -o <path>.json
visualize_model_in_explorer --from-payload <path>.json -o <path>.html
```

Live viewer, only when the user asked to serve or open it:

```bash
visualize_model_in_explorer <model> --serve --port 8765
```

`--open` implies `--serve` and also launches a browser. The server process blocks. Print the `Open viewer:` URL to the user before treating the command as finished. Do not start a server for an HTML-only request.

Operator list with inferred shapes, in addition to the graph:

```bash
visualize_model_in_explorer <model> -o <path>.html --operators-json <path>_operators.json
```

Examples, one per sharding source, plus overrides and attention together:

```bash
visualize_model_in_explorer moonshotai/Kimi-K3 -o kimi.html

visualize_model_in_explorer meta-llama/Llama-3.1-70B \
  --tp 8 --pp 2 -o llama-hf.html

visualize_model_in_explorer meta-llama/Llama-3.1-70B \
  --engine vllm --tp 8 --pp 2 -o llama-vllm.html

visualize_model_in_explorer MiniMaxAI/MiniMax-M3.1-flash \
  --engine sglang --tp 2 --ep 4 -o minimax.html

visualize_model_in_explorer moonshotai/Kimi-K3 \
  --sequence-layout packed --phase decode --attention eager \
  --config-override num_hidden_layers=2 \
  -o kimi-decode.html

visualize_model_in_explorer /models/qwen \
  --sequence-layout packed --phase decode -o /models/qwen/qwen-decode.html
```

The first Llama command uses the checkpoint's Hugging Face plans. The second draws vLLM's pinned modeling source. The third draws SGLang's. See [Device sharding](#device-sharding) for when `--engine` still changes the plan parser after `--code-path` or `--github` names the file. vLLM is pinned at `vllm-project/vllm@v0.32.0`. SGLang is pinned at `sgl-project/sglang@v0.5.22`.

A partition count that does not divide the model evenly is still exported, with a warning on stdout. More than one GPU group writes `<stem>_subgraph_<i>.html`.

## Step 3: Verify

Success for a file write is exit code 0 and a line `Wrote standalone viewer:` (HTML) or `Wrote Model Explorer JSON:` (JSON).

Then check the file:

- HTML: non-empty, and the text contains both `model-explorer-visualizer` and `graphCollections`.
- JSON: non-empty, and it parses as an object with a non-empty `graphCollections` array.
- Several GPU groups: every `*_subgraph_*.html` named on stdout exists.

Report the absolute path. For `--serve`, report the URL from `Open viewer:`.

On a non-zero exit, show stderr to the user. Retry once only when the error is a missing modeling file and a documented source switch fixes it (`--code-path`, `--github` with `--allow-repo`, or `--engine`). Leave `--auto-venv` and `--meta-shapes` on. `--no-meta-shapes` is a pure-static fallback only after the user accepts shapes that static analysis could not fill.

The first open of an HTML file loads `<model-explorer-visualizer>` and its font textures from unpkg. The layout worker is embedded. Tell the user that opening the file needs network once, unless the browser already cached that package.

## Config overrides

`--config-override` is the last word on a config value. Use it for every config key the user named, except the attention implementation, which is [`--attention`](#attention-implementation).

Pass one flag per key, repeated:

```bash
--config-override num_hidden_layers=2 --config-override vision_config.hidden_size=1024
```

Rules:

- The key is a field on the checkpoint config. A dotted key writes a nested block (`vision_config.hidden_size` reaches `vision_config["hidden_size"]`). When the parent is not a dict, the dotted string is stored as a single key.
- The value is JSON when it parses, and plain text otherwise. `8`, `true`, `false`, `null`, and `[1, 2]` mean what they mean in `config.json`. `torch_dtype=bfloat16` needs no extra quotes. A value that must stay text even though it is valid JSON is passed with JSON quotes in the argument, as in `label='"8"'`.
- Keys are applied left to right. A key stated twice warns and keeps the last value. Pass each key once. When the user contradicts themselves, keep their last value.
- Precedence, last writer wins: checkpoint `config.json`, then values the config class derives, then these overrides (so later default and alias resolution sees them), then config-class defaults, attribute aliases, and constructor writes from the modeling source, then these overrides again. The caller's value beats the checkpoint and beats a value the model's constructor writes back onto the config.
- Do not edit `config.json` on disk, and do not pass an override the user did not ask for. Shrinking the stack (`num_hidden_layers=2`) is valid only when they asked for a smaller drawing.
- `--attention`, when set, writes `_attn_implementation` after the overrides. Do not also pass `--config-override _attn_implementation=...`.

## Attention implementation

`--attention` is a free string, not an enum. It selects one arm of every branch on `config._attn_implementation`, including `is_flash_attention_requested`. The other arm is not drawn.

| User named | Pass |
| --- | --- |
| Nothing about attention | omit `--attention`. The checkpoint's `_attn_implementation` is used. A silent config is `sdpa` |
| eager, "matmul and softmax", the unfused path | `--attention eager` |
| sdpa, "the fused kernel", and they want it even if the checkpoint says otherwise | `--attention sdpa` |
| flash attention 2, FA2 | `--attention flash_attention_2` |
| flash attention 3, FA3 | `--attention flash_attention_3` |
| flash attention or flash attn, with no version | `--attention flash_attention` |
| `flash_attn`, `flash_attn_2`, or `kernels-community/flash-attn` | that string, unchanged |
| any other implementation the modeling file compares against | the string they gave, unchanged |

Do not pass `--attention sdpa` merely because that is the usual default. Passing it forces `sdpa` on a checkpoint that asked for something else. Combine `--attention` with `--sequence-layout`, `--phase`, `--cache`, `--mode`, and `--device` when the user named those too. Each flag is independent and each picks one arm.

## Device sharding

`--tp`, `--pp`, and `--ep` are the partition counts. `--engine` chooses whose plan those counts are applied to. The three axes combine. A count of 1 leaves that axis unsplit, so omit a flag the user left at one GPU.

Decide the source first:

| User named | What to pass |
| --- | --- |
| Nothing about an engine, or "Hugging Face", "transformers", or "the checkpoint's plan" | no `--engine`. The split is `base_model_tp_plan`, `base_model_pp_plan`, and `base_model_ep_plan`, plus the causal-LM head `_tp_plan` / `_pp_plan` |
| vLLM | `--engine vllm` plus the counts. Modeling source is the pin `vllm-project/vllm@v0.32.0` unless `--code-path` or `--github` already names a file |
| SGLang | `--engine sglang` plus the counts. Pin `sgl-project/sglang@v0.5.22`, with the same exception |

`--engine` does not replace the counts. "vLLM tensor parallel 8 and pipeline parallel 2" is `--engine vllm --tp 8 --pp 2`. "SGLang, tp 2, expert parallel 4" is `--engine sglang --tp 2 --ep 4`. "Sharded across 8 GPUs" with no engine is Hugging Face `--tp 8`.

When the user also passes `--code-path` or `--github`, that file is the modeling source and the engine pin is not fetched. Still pass `--engine` if they asked for vLLM or SGLang sharding: it selects the engine plan parser instead of the Hugging Face plan dicts.

Counts are integers of at least 1. A count that does not divide heads, intermediate size, expert count, or layer count is still exported. The warning on stdout is expected. Report every `*_subgraph_*.html` the command names.

## Other flags

Pass a flag only when the user asked for that behavior. Leave these at the exporter default, and do not put them on the command line: rectangular layout, prefill, cold cache, inference, `--device hip`, shapes on, meta-device pass on, constants dropped from HTML, auto-venv on, inline expansion on, port 8765. The positional model id is enough; do not also pass `--checkpoint`. The full flag list lives in the Visualizer README.

| User asked for | Flag |
| --- | --- |
| An HTML file | `-o <file>.html` and no `--serve` |
| A JSON payload | `-o <file>.json` |
| HTML from a payload already written | `--from-payload <file>.json -o <file>.html` (this skips loading the model) |
| A live viewer | `--serve`, and `--open` only when they asked to open a browser |
| Packed tokens `[1, nnz, ...]` | `--sequence-layout packed` |
| One new token | `--phase decode` (decode always reads a cache) |
| A cache that already holds keys and values | `--cache warm` |
| A training forward (`self.training` true) | `--mode training` |
| Branches for another accelerator | `--device` one of `cuda`, `npu`, `xpu`, `cpu` |
| No symbolic shapes | `--no-shapes` |
| Skip the meta-device pass | `--no-meta-shapes` |
| Stay on the installed transformers | `--no-auto-venv` |
| Weight and constant nodes in the HTML | `--no-drop-constants` (JSON always keeps them) |
| An explicit modeling file | `--code-path PATH` |
| Modeling source from GitHub | `--github`, plus `--allow-repo` when the repo is not already allowed |
| A flat operator list | `--operators-json PATH` |
| Opaque modules, no internal steps | `--no-inline-expansion` |

`--torch-module` debugs one plain `torch.nn.Module`. It is not the Hugging Face export. Pair it with exactly one of `--input-shape` or repeated `--input-spec`.

Allowed GitHub sources without `--allow-repo`: the checkpoint's own Hub tree, `huggingface/transformers`, `fla-org/flash-linear-attention`, and the pinned vLLM and SGLang repos. Everything else is blocked until it is allowlisted.

## Footguns

- **`--device` is not the host.** Default `hip` means the drawing follows HIP branches in the modeling source. The export still runs on CPU. Do not change it because the current machine is NVIDIA, CPU-only, or an NPU.
- **`--auto-venv` may re-exec.** A checkpoint whose `transformers` pin differs from the installed one is exported in a cached environment. That re-exec is success, not a failure. Disable it only with `--no-auto-venv`.
- **HTML drops constants by default.** Learned weights and buffers stay in the JSON payload and disappear from HTML. Pass `--no-drop-constants` when the user wants those nodes on the page.
- **unpkg on first open.** The HTML file embeds the graph and the layout worker. The visualizer element itself still loads from unpkg the first time the file is opened.
- **Do not download weights.** `huggingface_hub` fetches `config.json` and modeling source. A multi-gigabyte weight download means the wrong tool was invoked.
- **Server calls block.** `--serve` and `--open` do not return until the process is interrupted. Prefer `-o file.html` unless a live viewer was requested.
- **`--attention` outranks a config override of the same fact.** Passing both `--attention` and `--config-override _attn_implementation=...` draws the `--attention` value. Use only `--attention` for the implementation.
- **`--engine` and a named source do different jobs.** `--code-path` or `--github` picks the file. `--engine` still picks the plan parser. Omit `--engine` for a Hugging Face plan. Pass it for vLLM or SGLang even when the file is local.
