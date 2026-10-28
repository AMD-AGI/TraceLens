<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model visualizer evals

Routing and behavior dataset for the model visualizer skill. The harness is [skillscope](https://github.com/amd/skillscope) at the same tag as [`.github/workflows/skill-evals.yml`](../../../../../../.github/workflows/skill-evals.yml) (`v0.1.3`). Structural checks run on every pull request. Routing and behavioral checks run on workflow dispatch.

`evals/evals.json` holds the prompts. A prompt with only `skill_should_trigger` grades routing. `viz-tiny-llama-html` also grades behavior: it expects `model.html` and forbids a GPU weight load, a leftover server, and an analysis report. That case needs network access to the Hugging Face Hub and the `Visualizer` extra (`torch`, `transformers`, `huggingface_hub`). `evals/machine.yml` limits the run to Linux.

## Structural check

From the repository root, after installing the pinned harness:

```bash
uv tool install --system-certs git+https://github.com/amd/skillscope@v0.1.3
skillscope structural --skills-dir TraceLens/Agent/Visualizer/skills
```

## Routing and behavior

The CI workflow grades each skill on its own routing leg. The negative prompts are the analysis orchestrator and Magpie profiling, so a later side-by-side room still has a case that must not land here.

```bash
skillscope routing --skills-dir TraceLens/Agent/Visualizer/skills --routing-room all
skillscope behavioral --skills-dir TraceLens/Agent/Visualizer/skills --skill model-visualizer
```

Dispatch the same graders in CI with the `AMD Skills Checks` workflow. Enable routing or behavioral there when you want a paid run. The workflow passes `claude` through skillscope; a local `skillscope` run needs the `claude` CLI authenticated.

## Headless smoke prompt

From a checkout with the Visualizer extra installed:

```bash
agent --model claude-opus-4-8-thinking-medium --print --force --trust \
    "Follow the model visualizer installed with the TraceLens pip package (look under TraceLens/Agent/Visualizer/skills/model-visualizer/ in the package installation directory) and export hf-internal-testing/tiny-random-LlamaForCausalLM to model.html. Do not serve it."
```

Confirm `model.html` exists and contains `model-explorer-visualizer`.
