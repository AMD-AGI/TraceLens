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
- [Flag decisions](#flag-decisions)
- [Footguns](#footguns)

## Workflow

```
0. Resolve inputs (ask only for what the prompt left out)
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
| Regime | no | rectangular, prefill, cold cache, inference, attention from the checkpoint, `--device hip` |
| Partitions | no | `--tp 1 --pp 1 --ep 1`, transformers modeling source |
| Shapes | no | on (`--no-shapes` only when the user asks to skip them) |

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

Examples that match the common requests:

```bash
visualize_model_in_explorer moonshotai/Kimi-K3 -o kimi.html

visualize_model_in_explorer meta-llama/Llama-3.1-70B \
  --engine vllm --tp 8 -o llama3-70b.html

visualize_model_in_explorer /models/qwen \
  --sequence-layout packed --phase decode -o /models/qwen/qwen-decode.html
```

`--engine {vllm,sglang}` reads that engine's pinned modeling source and its sharding classes. It is ignored for source selection once `--code-path` or `--github` already names the file. vLLM is pinned at `vllm-project/vllm@v0.32.0`. SGLang is pinned at `sgl-project/sglang@v0.5.22`.

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

## Flag decisions

Set a flag only when the user asked for that behavior or a default would contradict them.

| User asked for | Flag |
| --- | --- |
| Browser file, email, or "save HTML" | `-o <file>.html` and no `--serve` |
| JSON payload for another tool | `-o <file>.json` |
| Watch it in a browser now | `--serve` (add `--open` only when they asked to open a browser) |
| No symbolic shapes | `--no-shapes` |
| Skip the meta-device pass | `--no-meta-shapes` |
| Stay on the installed transformers | `--no-auto-venv` |
| Keep weight and constant nodes in the HTML | `--no-drop-constants` |
| Padded batch `[B, S, ...]` | `--sequence-layout rectangular` (default) |
| Packed tokens `[1, nnz, ...]` | `--sequence-layout packed` |
| Prompt scoring | `--phase prefill` (default) |
| One new token | `--phase decode` (decode always reads a cache) |
| First forward | `--cache cold` (default) |
| Cache already holds keys and values | `--cache warm` |
| Eager matmul and softmax | `--attention eager` |
| One fused attention kernel | `--attention sdpa` (also the usual checkpoint default) |
| Training forward (`self.training` true) | `--mode training` |
| Draw the graph for a different accelerator | `--device` one of `hip` (default), `cuda`, `npu`, `xpu`, `cpu` |
| Override one config field | `--config-override KEY=VALUE` (repeatable; dotted keys reach nested configs; JSON values parse as JSON) |
| Tensor / pipeline / expert split | `--tp N`, `--pp N`, `--ep N` |
| vLLM or SGLang modeling source | `--engine vllm` or `--engine sglang` |
| Explicit modeling file | `--code-path PATH` |
| Modeling source from GitHub | `--github` plus `--allow-repo` when the repo is not already allowed |
| Flat op list | `--operators-json PATH` |
| Opaque modules, no internal steps | `--no-inline-expansion` |
| Debug one `nn.Module` | `--torch-module PKG.MOD:FACTORY` with exactly one of `--input-shape` or repeated `--input-spec` |

`--torch-module` is a debugging path for a plain `torch.nn.Module`. It is not the Hugging Face export. Use `--input-shape N,...` for one floating-point input traced with `torch.fx`. Use `--input-spec NAME=DIM,...[:DTYPE]` (repeatable) for several inputs, integer inputs, or shape-dependent control flow; those are traced with `torch.export`.

Allowed GitHub sources without `--allow-repo`: the checkpoint's own Hub tree, `huggingface/transformers`, `fla-org/flash-linear-attention`, and the pinned vLLM and SGLang repos. Everything else is blocked until it is allowlisted.

## Footguns

- **`--device` is not the host.** Default `hip` means the drawing follows HIP branches in the modeling source. The export still runs on CPU. Do not change it because the current machine is NVIDIA, CPU-only, or an NPU.
- **`--auto-venv` may re-exec.** A checkpoint whose `transformers` pin differs from the installed one is exported in a cached environment. That re-exec is success, not a failure. Disable it only with `--no-auto-venv`.
- **HTML drops constants by default.** Learned weights and buffers stay in the JSON payload and disappear from HTML. Pass `--no-drop-constants` when the user wants those nodes on the page.
- **unpkg on first open.** The HTML file embeds the graph and the layout worker. The visualizer element itself still loads from unpkg the first time the file is opened.
- **Do not download weights.** `huggingface_hub` fetches `config.json` and modeling source. A multi-gigabyte weight download means the wrong tool was invoked.
- **Server calls block.** `--serve` and `--open` do not return until the process is interrupted. Prefer `-o file.html` unless a live viewer was requested.
