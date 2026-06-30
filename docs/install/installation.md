---
myst:
    html_meta:
        "description": "Step-by-step instructions to install TraceLens from GitHub or source, verify the installation, and set up traceconv for Perfetto pftrace input."
        "keywords": "TraceLens, install, pip, Python, ROCm, GPU trace analysis, traceconv, pftrace, virtual environment, open source"
---

# Install TraceLens

This topic provides step-by-step instructions to install TraceLens and verify
the installation. TraceLens is distributed as a Python package; install it with
`pip` either directly from the public GitHub repository (recommended) or from a
local source checkout for development.

## Prerequisites

- Python 3.6 or later (3.10 recommended). Confirm your version:

  ```bash
  python3 --version
  ```

- A recent `pip`. Upgrade it inside your environment:

  ```bash
  python3 -m pip install --upgrade pip
  ```

- (Recommended) An isolated virtual environment:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

## Method 1: Install from GitHub (recommended)

Install the latest TraceLens directly from the public repository:

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

This installs TraceLens and its dependencies and places the `TraceLens_*`
command-line tools on your `PATH`.

## Method 2: Install from source (development)

Use an editable install when you want to modify TraceLens or run its test
suite. The `[dev]` extra installs the development and test dependencies:

```bash
git clone https://github.com/AMD-AGI/TraceLens.git && cd TraceLens
pip install -e .[dev]
python -m pytest tests/ -v
```

## Optional: `traceconv` for `.pftrace` input

`traceconv` is needed only when you pass a Perfetto-style `.pftrace` trace
(produced by `rocprofv3 --output-format pftrace`) to one of the pftrace report
tools (`TraceLens_generate_perf_report_pftrace_hip_activity`, `..._hip_api`, or
`..._memory_copy`); it converts the `.pftrace` to JSON. The other report tools
(PyTorch, JAX, rocprofv3 JSON) don't use it. You don't need to install it
manually:

- If `traceconv` is on your `PATH`, TraceLens uses it.
- Otherwise, TraceLens downloads it automatically (from
  `https://get.perfetto.dev`) into the trace file's directory, so this step
  needs network access.
- You can also point a pftrace tool at a specific binary with
  `--traceconv /path/to/traceconv`.

## Verify the installation

Run the following checks to confirm TraceLens is installed correctly:

1. Confirm the package imports:

   ```bash
   python3 -c "import TraceLens; print('TraceLens import OK')"
   ```

2. Confirm the command-line tools are available:

   ```bash
   TraceLens_generate_perf_report_pytorch --help
   ```

   Expected output: the tool prints its usage and the list of accepted
   arguments (for example, `--profile_json_path`) and exits without error.

3. (Optional) Generate a report from a bundled demo trace to confirm end-to-end
   operation:

   ```bash
   TraceLens_generate_perf_report_pytorch --profile_json_path tests/traces/<demo_trace>.json
   ```

   Expected output: an Excel report (`.xlsx`) is written next to the trace,
   containing the GPU-timeline and operator-summary sheets.

## Next steps

- Follow the [How-to guides](../how-to/index.md) for step-by-step examples.
- See the [API reference](../reference/api-reference.md) for the full set of
  command-line tools and their arguments.

## Related topics

- [What is TraceLens?](../what-is-tracelens.md)
- [Release notes](../about/release-notes.md)
- [Compatibility matrix](../reference/compatibility.md)
