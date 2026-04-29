<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Contributing to TraceLens

Thanks for your interest in improving **TraceLens** — a toolkit that parses PyTorch/JAX profiler traces and generates useful insights.

---

## 📋 Before You Start

> **⚠️ NOTE FOR AMDers:**
>
> **This is a public repository. Do NOT add any private, confidential, or customer-related data, code, or information to this repo.**
>
> Please ensure all contributions are free of sensitive or proprietary content before submitting.

- Read the [README](./README.md) to understand scope and architecture.
- Search existing **issues** and **discussions** to avoid duplicates.
- For new features and enhancements (new analyser, backend integration, refactor), **open an issue** first to align on approach.
- Prefer small, modular, focused PRs.
- **Have a ready-made utility?** If your utility is already developed, you can raise a PR to add it directly to `examples/custom_workflows/`. This lets the community start using it right away while we plan a tighter integration into the core library.

---

## Table of Contents

- [Dev Setup](#dev-setup)
- [Project Structure (high level)](#project-structure-high-level)
- [Code Formatting with Black](#code-formatting-with-black)
  - [Installing Black](#installing-black)
  - [Using Black](#using-black)
- [Branch Naming Convention](#branch-naming-convention)
  - [Types (type)](#types-type)
  - [Scope (optional)](#scope-optional)
  - [Examples](#examples)
- [Commit Message Convention](#commit-message-convention)
- [Updating Reference Outputs](#updating-reference-outputs)

---

## Dev Setup

```bash
# clone
git clone https://github.com/AMD-AGI/TraceLens.git

# optional: virtual env
python3 -m venv .venv
source .venv/bin/activate

# install (editable) + dev extras
pip install -U pip
pip install -e .[dev]
```

## Project Structure (high level)

```text
TraceLens/
├── TraceLens/
│   ├── Reporting/        # CLI tools for quick start utils
│   ├── Trace2Tree/       # Trace2Tree parses trace into tree data structure
│   ├── PerfModel/        # Op meta data parsing and performance modelling code (roofline, FLOPs/Byte, etc.)
│   ├── TreePerf/         # TreePerf uses Trace Tree and PerfModel to generate perf breakdowns and perf metrics TFLOPS/s, etc. 
|   |                     # This directory also contains GPUEventAnalyzer
│   ├── NcclAnalyser/     # Analysis of collective communications
│   ├── TraceFusion/      # Merging of multi‑rank traces into a global view
│   ├── TraceDiff/        # TraceDiff uses the Trace Tree format and does morphological comparison across traces
│   └── EventReplay/      # Extracts meta data and replays almost arbitrary operations
├── docs/               # tool-specific guides
├── examples/           # example traces, notebooks, scripts, custom-workflows
├── tests/              # unit & integration tests
└── setup.py
```

## Code Formatting with Black

This project uses [Black](https://black.readthedocs.io/en/stable/) to automatically format Python code for consistency and readability.

### Installing Black

You can install Black using pip:

```sh
pip install black
```

### Using Black

To format all Python files in the project, run:

```sh
black .
```

You can also format a specific file:

```sh
black path/to/your_file.py
```

Please ensure your code is formatted with Black before submitting a pull request.

## Branch Naming Convention

Please follow this branch naming convention for all feature and bug fix branches:

```text
<type>/<scope>/<short-description>
```

### Types (type)

| Type       | Purpose                                     |
| ---------- | ------------------------------------------- |
| `feat`     | New feature or functionality                |
| `fix`      | Bug fix                                     |
| `docs`     | Documentation update                        |
| `refactor` | Code refactoring (no functionality change)  |
| `test`     | Tests and test-related changes              |
| `chore`    | Miscellaneous changes (e.g., build scripts) |
| `ci`       | Continuous integration-related changes      |

### Scope (optional)

The scope can be used to specify which part of the project is affected, for example: `trace2tree`, `perfmodel`, `tracediff`, `docs`, `tests`.

### Examples

```text
feat/perfmodel/aiter-fav3
fix/tracediff/diff-reporting-bug
docs/update-jax-docs
refactor/trace2tree/remove-dead-code
ci/add-linting-automation
```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages. Example:

```text
feat(perfmodel): add perf model for aiter fav3
fix(tracediff): resolve diff reporting error
docs(readme-tracediff): add docs for jax tracediff
```

This format helps us to automatically generate changelogs and provide more clarity in versioning.

## Updating Reference Outputs

Several regression tests compare freshly generated CSV outputs against checked-in reference CSVs. When an intentional code change legitimately alters those outputs, you can refresh the references in-place by passing the `--update-references` flag to pytest:

```sh
python -m pytest tests/test_perf_report_regression.py --update-references
```

> **⚠️ Important:** When you use `--update-references` in a PR, **explicitly note in your PR description which reference files were updated and why**.
>
> **Do not use this flag to paper over unexpected differences.** If a test starts failing, first investigate *why* the output changed. Only refresh references once you have confirmed the new output is intentionally correct (e.g. a deliberate metric formula change, a new column, or a bug fix in the reporting pipeline).
