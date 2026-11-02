<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model roofline projection evals

Routing and behavior dataset for the model roofline projection skill. The harness is [skillscope](https://github.com/amd/skillscope) at the same tag as [`.github/workflows/skill-evals.yml`](../../../../../../.github/workflows/skill-evals.yml) (`v0.1.3`). Structural checks run on every pull request. Routing and behavioral checks do not run per merge: they run only when someone dispatches the AMD Skills Checks workflow with those inputs enabled.

`evals/evals.json` holds the prompts. A prompt with only `skill_should_trigger` grades routing. Those routing prompts cover a GPU-bound time estimate (`roofline-mi300x-time`), vLLM tensor-parallel sharding (`roofline-vllm-tp-shard`), an SGLang expert-parallel decode step with a dimension override (`roofline-decode-dim`), a config override (`roofline-config-override`), and a GPU-sizing question phrased without any tool name (`roofline-gpu-sizing-question`). The negative prompts are the model visualizer's own graph-export request, the analysis orchestrator, and Magpie profiling, so a routing run that pools this skill with those stays discriminating in both directions. `roofline-tiny-llama-flops` also grades behavior: it expects a FLOPs/bytes-only run (no GPU arch named) and forbids a GPU weight load, a trace read, an `analysis.md` report, or a Model Explorer graph artifact. That case needs network access to the Hugging Face Hub and the `Visualizer` extra (`torch`, `transformers`, `huggingface_hub`). `evals/machine.yml` limits the run to Linux.

## Structural check

From the repository root, after installing the pinned harness:

```bash
uv tool install --system-certs git+https://github.com/amd/skillscope@v0.1.3
skillscope structural --skills-dir TraceLens/Agent/Visualizer/skills
```

## Routing and behavior

The CI workflow grades each skill on its own routing leg. `--routing-room all` pools this skill against the model visualizer, so the graph-export negative prompt in this dataset and the roofline-shaped negative prompts in the visualizer's own dataset both have to hold.

```bash
skillscope routing --skills-dir TraceLens/Agent/Visualizer/skills --routing-room all
skillscope behavioral --skills-dir TraceLens/Agent/Visualizer/skills --skill model-roofline-projection
```

Dispatch the same graders in CI with the `AMD Skills Checks` workflow. Enable routing or behavioral there when you want a paid run. The workflow passes `claude` through skillscope; a local `skillscope` run needs the `claude` CLI authenticated.

## Headless smoke prompt

From a checkout with the Visualizer extra installed:

```bash
agent --model claude-opus-4-8-thinking-medium --print --force --trust \
    "Follow the model roofline projection skill installed with the TraceLens pip package (look under TraceLens/Agent/Visualizer/skills/model-roofline-projection/ in the package installation directory) and project the FLOPs/bytes for hf-internal-testing/tiny-random-LlamaForCausalLM at batch size 1 and sequence length 128. No GPU time estimate needed."
```

Confirm the printed table includes a `Total:` rollup line with FLOPs and bytes.
