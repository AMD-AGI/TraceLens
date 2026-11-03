<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens Agent: Model Bringup

The [model-bringup skill](skills/model-bringup/SKILL.md) takes a model and serving image
from profiling to a validated `kernel_candidates.json`. It chains existing tools:

1. The [Magpie profiling skill](../Profiling/README.md) and its trace quality check.
2. The [analysis orchestrator](../Analysis/README.md) and the
   [triage checks](../Analysis/triage/README.md), which include the graph capture merge
   and capture mapping.
3. The perf-model authoring and validation tools on `feat/perf_model_validation_skill`,
   one parallel worker per op without a perf model.
4. A coverage rerun.
5. The Hyperloom kernel-candidates harness and `verify_kernel_candidates`.

Ask the agent to "follow the model bringup skill with `<bringup_config.yaml>`", starting
from [templates/bringup_config.yaml](skills/model-bringup/templates/bringup_config.yaml).
Perf-model PRs and issues are drafted for review and filed on `AMD-AGI/TraceLens` only
after the user approves them.
