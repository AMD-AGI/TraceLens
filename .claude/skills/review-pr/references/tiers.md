# Owning layers and backbone risk

Two jobs: it names the canonical owner of each concern (cited by `rules.md` S1/S2/S3/L2), and it
tiers the backbone for Step 3 of [`../SKILL.md`](../SKILL.md). A concern has one owner; new code
calls it rather than re-doing its work.

## Owning layers

From issue #1076 § 2, grounded on the current tree. When code moves, update the path here — the
rules cite the concern, not the path.

| Concern | Owning layer |
|---|---|
| Trace loading (JSON / `.json.gz` / xprof) | `TraceLens/util.py` — `DataLoader.load_data` |
| Event tree | `TraceLens/Trace2Tree/trace_to_tree.py` — `TraceToTree` / `JaxTraceToTree` |
| Kernel timing, overlap, timeline | `TraceLens/TreePerf/` — `GPUEventAnalyser` |
| Perf numbers, roofline, efficiency | `TraceLens/PerfModel/perf_model.py`, `run_perf_model.py`, `origami_helper.py` (+ `PerfModel/extensions/`) |
| Op mapping | `TraceLens/PerfModel/{torch,jax}_op_mapping.py` |
| Hardware specs | one typed spec module (do not scatter `gfx*`/peak-flops/mem-bw constants) |
| Model-specific and one-off scripts | `examples/`, `scripts/` — never the library package |
| Shared test fixtures | `tests/conftest.py` |

## Backbone tiers

Tiers are assigned by blast radius and by how a failure surfaces, not by file size or churn.

| Tier | File | Blast radius / failure mode |
|---|---|---|
| 1 | `util.py` (`DataLoader.load_data`) | The one trace loader. A raise blocks every analysis; a silent mis-decode feeds every model garbage |
| 1 | `Trace2Tree/trace_to_tree.py` | The one event tree every analysis stands on. A wrong parent/child or timing is silent — reports finish green with inverted numbers |
| 1 | `TreePerf/gpu_event_analyser.py` | The timeline the perf models read. A wrong span or attribution propagates with no error |
| 1 | `PerfModel/perf_model.py`, `run_perf_model.py`, `origami_helper.py` | The numbers a report is graded on. A unit or denominator error is a confident wrong figure, not a crash |
| 2 | `Agent/Analysis/post_processing/candidate_schema.py`, `analysis_json.py` | Typed `analysis.json` shape, read out of tree. Additive-only; a one-sided field change type-checks and still breaks readers |
| 2 | `TraceUtils/kernel_source/contract.py` | Persisted kernel-source audit doc; writer and readers move together |
| 2 | `setup.py` `console_scripts`, each `Reporting/*.py:main`, `TraceUtils/kernel_source/cli.py`, `TraceIndex/cli.py` | Name-routed CLI contracts (`module:main`). A renamed `main` or moved module breaks the installed command only at invocation |
| 2 | `Agent/**/skills/**/SKILL.md` | Skill-name routing resolved by string and gated by the skills-federation checks |
| 2 | `PerfModel/{torch,jax}_op_mapping.py`, `kernel_name_parser.py`, `TraceUtils/kernel_source/{resolver,library_artifact,demangle}.py` | Name/dtype/arch resolution shared across formats; a matching bug silently attributes the wrong kernel |
| 3 | everything else | one analyser, one util, one report helper, one view |

## Tiering a file not in the table

Applies to new files too.

```
Q1  — If this file raises at import time, does a report CLI (e.g.
      TraceLens_generate_perf_report_pytorch) still reach the point of
      loading a trace? The Reporting mains import the loader, the tree
      builder and the perf model at module level.      → NO  → Tier 1

Q1b — Is it reached by NAME at runtime — a console_scripts "module:main"
      in setup.py, or a skill name under Agent/**/skills/**? Name-routed
      wiring fails at invocation, not at startup.       → YES → Tier 1

Q2  — Does it decide, persist, or read a number a report is graded on —
      the event tree, the roofline/origami model, or a persisted
      analysis.json field?                              → YES → Tier 1
      (a wrong answer here is silent)

Q3  — Is it a contract with more than one owner: a schema written to
      disk and read elsewhere, or a subprocess/CLI another tool parses?
                                                        → YES → Tier 2

Q4  — Does it parse a real external artifact (a trace, a compiled
      library file, a profiler dump) whose malformed shapes it must
      survive?                                          → YES → Tier 2 min

Otherwise → Tier 3.
```

Q1b is the one that bites: the `console_scripts` in `setup.py` and the skill names under
`Agent/**/skills/**` route by string, so a renamed `main`, a moved module or a renamed skill passes
every import check, passes lint, passes collection, and fails only on invocation. Grep the string,
not the symbol.
