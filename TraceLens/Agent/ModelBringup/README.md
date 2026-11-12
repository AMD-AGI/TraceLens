<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens Agent: Model Bringup

The [model-bringup skill](skills/model-bringup/SKILL.md) brings a new model up on TraceLens
in one unattended run:

1. It profiles the model with the [Magpie profiling skill](../Profiling/README.md), using
   either a fixed-sequence or an AgentX workload.
2. It gates profile health and report health with the
   [triage checks](../Analysis/triage/README.md).
3. It runs the [analysis orchestrator](../Analysis/README.md).
4. It drafts one validated perf-model PR or issue per op without a perf model.

A later Publish prompt files the drafts you approve on `AMD-AGI/TraceLens`.

Start from [templates/bringup_config.yaml](skills/model-bringup/templates/bringup_config.yaml)
and ask the agent to "follow the model-bringup skill installed with TraceLens and run it
from config `<path>/bringup_config.yaml`".

Code layout:
- `utils/bringup_gates.py` holds the step gates.
- `utils/unified_perf_ops.py` reads `unified_perf_summary.csv`: it ranks ops worth a perf
  model, measures the share of kernel time without one, and builds validator cases.
- Per-op authoring and validation stay in [`../PerfModel/`](../PerfModel/).
