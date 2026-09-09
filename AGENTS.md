<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# AGENTS.md

The authoring contract for human or AI, changing this repo: how to make a change land
cleanly.

TraceLens is a Python library for automated performance analysis of ML training and inference
workloads from profiler traces. It parses PyTorch, JAX, and rocprofv3 traces into a hierarchical
event tree, models roofline and compute performance, analyzes collective communication, diffs
traces, and drives an agentic optimization report.

## Documentation

| Topic | Source of truth |
|---|---|
| Overview, capabilities, supported formats | [`docs/what-is-tracelens.md`](docs/what-is-tracelens.md) |
| Module architecture, per-module guides | [`README.md`](README.md) § Documentation, [`docs/`](docs/) |
| Dev setup, branch and commit style, updating references | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## Common commands

```bash
pip install -e .[dev]                                  # editable install with dev extras

python -m pytest tests/                                # full suite
python -m pytest tests/test_perf_report_regression.py  # one suite

black .                                                # required before every PR; CI pins black==26.3.1
python tests/update_copyright.py                       # fix missing copyright headers
```

The primary CLIs install as `console_scripts` (see `entry_points` in [`setup.py`](setup.py)).
Demo traces for local runs are bundled in `tests/traces`.

## Authoring rules of engagement

### Change scope and shape

- **One concern per change.** A PR fixes one issue or adds one capability; if you must bundle, say
  why in the description. Don't ride unrelated refactors in on a fix.
- **Smallest diff.** Change exactly what was asked. If you find yourself editing a file the request
  did not name, stop and confirm.
- **Small, focused PRs.** Open an issue first for a new analyser or backend integration.
- **Update the docs with the change.** When a change alters a CLI flag, API; update the affected `README.md` and `docs/` in the same PR.

### Reuse and structure

- **Build on the owning layer.** Before adding a standard loader, parser, or regex, find the
  canonical owner and use it. In `TraceLens/util.py`: `DataLoader.load_data` for trace JSON /
  `json.gz`. `Trace2Tree`/`GPUEventAnalyser` for the event tree and timeline,
  `TraceUtils/annotation_utils` for annotations.
- **Build new analysis on the existing tree, not a re-load.** Parse the trace and construct the tree once,
  then pass that tree to every downstream analysis; don't re-load the JSON or rebuild the tree per
  consumer. Reopen or reparse only when an analysis genuinely needs a different view that the existing tree cannot provide.
- **Group helpers; don't sprawl.** A pile of one-line functions across many files is a class you
  haven't named yet: put a helper in the module whose responsibility matches its purpose, not
  next to its first caller, and extend an existing helper over adding a parallel one.
- **Derive over hardcode.**  drop references to a
  concern from agents that don't own it, and template sections the downstream consumer never reads.

### Correctness and honesty

- **Catch narrowly; don't route around.** No new broad `except Exception` or bare `except`; catch
  the specific error, or let it raise. No new feature flag or env toggle to route around a design
  problem. A single computed source beats duplicated constants. Thresholds and regexes are calibrated 
  in one place; tune them at the source, never fork a second copy. Keep interfaces minimal.
- **Trust the caller.** Validate at the system boundary, then trust internal callers; redundant
  re-checks and layered fallbacks hide the failure they were added to survive.
- **Review feedback is a hypothesis.** A comment can be right about the symptom and wrong about the
  fix; converge on the design that is correct, not the one that is merely defensible.

### Hygiene

- **Delete, don't comment out.** No `# removed …` tombstones.
- **Comment why, not what,** and only where the reason is non-obvious. Never narrate the change: no
  "previously this did X," no step or plan numbering.
- **Follow the file layout.** New Python file: copyright banner → module docstring → imports
  (fully-qualified `from TraceLens…`, stdlib then third-party then local, no `sys.path.insert`, no
  mid-file imports) → a `# Constants` block with every threshold and regex → public functions,
  then private (`_name`).
- **Public repo, vendor-neutral.** Never add private, confidential, or customer data. Keep code and
  docs vendor-neutral, unless the surrounding code is already specific. Quoting an actual kernel
  name from a trace is fine. Don't commit generated output, traces, or large binaries; no destructive
  git operations without an explicit target.
- **English, nothing generated in git.** The repo is English: code, identifiers, comments, commit
  messages, and docs, regardless of the language the work was discussed in. 
- **Leave nothing behind.** Working notes and analysis write-ups are byproducts of the work, not
  deliverables; don't commit them, least of all at the repo root, unless they were asked for.
