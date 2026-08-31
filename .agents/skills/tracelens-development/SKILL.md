---
name: tracelens-development
description: >-
  Repo-wide development conventions for TraceLens. Use before writing, editing, or
  reviewing any code in this repository — imports, util reuse, cross-component
  consistency, avoiding redundant computation, comment/docstring style, and public-repo
  data-handling rules. Applies to all subsystems (Trace2Tree, PerfModel, TreePerf,
  NcclAnalyser, TraceFusion, TraceDiff, EventReplay, Agent).
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens development conventions

Follow these rules for any code change in this repository, regardless of subsystem.

## Imports

- Keep all imports at the top of the file (standard library, then third-party, then
  local), immediately after the copyright header.
- Do not add inline/deferred imports inside functions or methods unless required to
  avoid a genuine circular import — and prefer fixing the circular dependency instead.

## Reuse util functions

- Before writing new helper logic, check for an existing util module in the same or a
  related subsystem: `TraceLens/util.py`, `TraceLens/TraceDiff/util.py`,
  `TraceLens/EventReplay/utils.py`, `TraceLens/PerfModel/utils.py`,
  `TraceLens/NcclAnalyser/util/`, `TraceLens/AgenticMode/Standalone/utils/`,
  `TraceLens/Agent/Analysis/utils/`.
- If the same logic is needed in more than one place, extract it into the relevant
  `util(s).py` rather than duplicating it.
- Only add a new util module or function when nothing existing covers the need — don't
  create a parallel helper that shadows one that already does the job.

## Don't develop in isolation

- Before adding or changing a component, check how it is consumed elsewhere (e.g. does
  `TreePerf` depend on this `Trace2Tree` output shape? does `NcclAnalyser` or
  `TraceDiff` read this field? does an `Agent` skill parse this report format?).
- Prefer extending an existing pipeline/data structure over introducing a parallel one.
- Avoid unnecessary computation: don't recompute values already available from an
  earlier pass (e.g. an existing tree traversal, cached DataFrame, or report), don't
  reprocess a trace/file that's already been parsed, and don't add work inside hot
  loops that could be hoisted out or memoized.

## Comments and docstrings

- Match the surrounding code's existing structure and style — don't invent a new
  format for the file you're editing.
- Keep docstrings and comments short. A one-line docstring/comment is usually enough;
  avoid multi-paragraph docstrings or long comment blocks.
- Only comment on the *why* (a non-obvious constraint, workaround, or invariant), not
  the *what* — the code should already read clearly enough to convey what it does.

## Public repository — data handling

**This is a public AMD repository — never commit private, confidential, or
customer-related data.** This includes proprietary trace files, internal hostnames/IPs,
credentials, and any customer-identifying information, in code, tests, fixtures, or
example data.
