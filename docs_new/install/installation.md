<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Installation instructions

This page provides step-by-step instructions to install TraceLens and verify
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

To run only the pftrace report tests:

```bash
python -m pytest \
    tests/test_pftrace_hip_api_perf_report.py \
    tests/test_pftrace_hip_activity_report.py -v
```

## Optional: traceconv for .pftrace input

Analyzing `.pftrace` files requires `traceconv` to convert them to JSON. You do
not need to install it manually:

- If `traceconv` is on your `PATH`, TraceLens uses it.
- Otherwise, TraceLens downloads `traceconv` automatically into the trace
  file's directory.
- You can also point to a specific binary with `--traceconv /path/to/traceconv`.

## Verify the installation

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
