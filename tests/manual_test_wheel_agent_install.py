###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Manual-only Docker experiment: TraceLens wheel + Cursor CLI + analysis agent.

This module is **not** collected by pytest (filename does not match ``test_*.py``).
Run from the TraceLens-internal repo root::

    python tests/manual_test_wheel_agent_install.py \\
        --input-dir /path/to/torch_trace \\
        --gpu-arch MI300X \\
        --output-dir /path/to/torch_trace_output \\
        --agent-log-dir /path/to/agent_logs

With interactive Cursor login (no API key on host)::

    python tests/manual_test_wheel_agent_install.py \\
        --interactive-login \\
        --input-dir ~/torch_trace \\
        --gpu-arch MI300X \\
        --output-dir ~/torch_trace_output \\
        --agent-log-dir ~/agent_logs

Or with a custom architecture JSON (must be named ``<Platform>.json``; it is
bind-mounted into the image ``utils/arch/`` directory so ``load_arch`` can find it)::

    python tests/manual_test_wheel_agent_install.py \\
        --input-dir ~/torch_trace \\
        --gpu-arch ~/myconfigs/MI300X.json \\
        --output-dir ~/torch_trace_output \\
        --agent-log-dir ~/agent_logs

Prerequisites:

* Docker
* Cursor CLI auth:
  - if ``CURSOR_API_KEY`` is set on the host, it is passed into the container, or
  - if not set, the script defaults to interactive login (``docker run -it`` +
    ``agent login`` before analysis). You can also force interactive mode with
    ``--interactive-login``.

The script builds ``tracelens-manual-agentic-experiment:latest`` from an
embedded image recipe unless ``--no-build`` is passed.
The wheel is resolved from ``--wheel``, else a single ``dist/TraceLens-*.whl``
when present. If ``--wheel`` is omitted and no such file exists under
``dist/``, the script runs ``python -m pip wheel`` from the repo root into
``dist/`` and then uses the built TraceLens wheel.
Only the four paths you provide are mounted from the host (plus the optional
arch file mount); your home directory is not mounted wholesale.

If ``--capture-folder`` is omitted and ``<input-dir>/capture_traces`` exists as
a directory, it is used like ``--capture-folder capture_traces`` (graph replay
+ capture). Pass ``--no-auto-capture-traces`` to ignore that directory.

All container stdout/stderr (including ``agent login`` and the main ``agent``
run) is duplicated to ``<--agent-log-dir>/<--agent-log-file>`` via ``tee``.
By default the agent uses
``--output-format stream-json --stream-partial-output`` so tool-level activity
is recorded in that log; use ``--agent-output-format text``
for plain prose only.
"""

import argparse
import glob
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

DEFAULT_IMAGE = "tracelens-manual-agentic-experiment:latest"


def _embedded_agentic_smoke_dockerfile(wheel_basename: str) -> str:
    """Dockerfile text; ``wheel_basename`` must be a valid ``.whl`` name (PEP 427)."""
    wb = os.path.basename(wheel_basename)
    if not wb.endswith(".whl") or "/" in wb or "\\" in wb:
        sys.exit(f"Invalid wheel basename for Docker COPY: {wheel_basename!r}")
    if any(c in wb for c in "\n\r\t"):
        sys.exit("Wheel basename contains illegal characters.")
    return f"""\
FROM python:3.12-slim

RUN apt-get update \\
    && apt-get install -y --no-install-recommends curl ca-certificates git \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY {wb} /tmp/{wb}

RUN pip install --no-cache-dir /tmp/{wb}

# Cursor CLI installer provides the `agent` command for headless runs.
RUN curl https://cursor.com/install -fsS | bash

ENV PATH="/root/.local/bin:/root/.cursor/bin:${{PATH}}"

CMD ["bash"]
"""


# Matches python:3.12-slim + pip install TraceLens in the embedded Dockerfile
DEFAULT_ARCH_SITE = (
    "/usr/local/lib/python3.12/site-packages/TraceLens/Agent/Analysis/utils/arch"
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _resolve_trace_basename(input_dir: str, trace: Optional[str]) -> str:
    ind = _abs(input_dir)
    if trace:
        tp = _abs(trace) if os.path.isabs(trace) else os.path.join(ind, trace)
        if not os.path.isfile(tp):
            sys.exit(f"Trace file not found: {tp}")
        if os.path.dirname(tp) != ind:
            sys.exit(
                "When using --trace, the file must live directly under --input-dir "
                f"(expected under {ind})."
            )
        return os.path.basename(tp)
    patterns = (
        "*.pt.trace.json.gz",
        "*.pt.trace.json",
        "*.json.gz",
        "*.json",
    )
    found: list[str] = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(ind, pat)))
    found = sorted(set(found))
    if len(found) == 1:
        return os.path.basename(found[0])
    if not found:
        sys.exit(
            f"No trace file found under {ind}; pass --trace PATH relative to input-dir."
        )
    sys.exit(
        "Multiple trace candidates under input-dir; pass --trace explicitly:\n  "
        + "\n  ".join(found)
    )


def _parse_gpu_arch(gpu_arch: str) -> Tuple[str, Optional[str]]:
    """Return (platform_name, host_path_to_json_or_none)."""
    expanded = os.path.expanduser(gpu_arch)
    if os.path.isfile(expanded):
        ap = _abs(expanded)
        base = os.path.basename(ap)
        if not base.endswith(".json"):
            sys.exit("--gpu-arch file must end with .json")
        platform = base[: -len(".json")]
        if not platform:
            sys.exit("Invalid arch JSON filename (empty stem).")
        return platform, ap
    # Treat as platform token (e.g. MI300X)
    token = gpu_arch.strip()
    if not token:
        sys.exit("Empty --gpu-arch")
    if any(c in token for c in "/\\"):
        sys.exit(f"--gpu-arch is not a file and looks like a path: {gpu_arch!r}")
    return token, None


def _wheel_path_for_docker_build(repo_root: str, wheel: Optional[str]) -> str:
    """Return absolute path to a TraceLens ``.whl`` for the Docker build context."""
    if wheel is not None:
        w = _abs(wheel)
        if not os.path.isfile(w):
            sys.exit(f"--wheel not found: {w}")
        if not w.endswith(".whl"):
            sys.exit("--wheel must be a .whl file")
        return w

    dist_dir = os.path.join(repo_root, "dist")
    candidates = sorted(glob.glob(os.path.join(dist_dir, "TraceLens-*.whl")))
    if len(candidates) > 1:
        sys.exit(
            "Multiple TraceLens-*.whl files in dist/; pass --wheel PATH or keep one wheel."
        )
    if len(candidates) == 1:
        return candidates[0]

    os.makedirs(dist_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        ".",
        "--no-cache-dir",
        "-w",
        dist_dir,
    ]
    print(
        f"No dist/TraceLens-*.whl found; building wheel into {dist_dir!r}.",
        flush=True,
    )
    print("+", " ".join(cmd), f"(cwd={repo_root})", flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)
    candidates = sorted(glob.glob(os.path.join(dist_dir, "TraceLens-*.whl")))
    if len(candidates) != 1:
        sys.exit(
            "Expected exactly one TraceLens-*.whl in dist/ after `pip wheel`; found "
            f"{len(candidates)}: {[os.path.basename(x) for x in candidates]}"
        )
    return candidates[0]


def _build_image(repo_root: str, image: str, wheel: Optional[str]) -> None:
    whl_path = _wheel_path_for_docker_build(repo_root, wheel)
    whl_bn = os.path.basename(whl_path)
    with tempfile.TemporaryDirectory(prefix="tracelens-agentic-docker-") as ctx:
        shutil.copy2(whl_path, os.path.join(ctx, whl_bn))

        dockerfile_path = os.path.join(ctx, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(_embedded_agentic_smoke_dockerfile(whl_bn))

        # ``-f`` must be absolute (or cwd-relative); it is not resolved relative to the
        # build context directory, so a bare ``Dockerfile`` breaks when cwd is repo root.
        cmd = ["docker", "build", "-f", dockerfile_path, "-t", image, ctx]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def _agent_cli_suffix(output_format: str) -> str:
    """Extra `agent` flags after --print --force --trust (shell-safe, no user text)."""
    if output_format == "text":
        return ""
    if output_format == "json":
        return "--output-format json"
    if output_format == "stream-json":
        return "--output-format stream-json --stream-partial-output"
    sys.exit(f"Unknown --agent-output-format: {output_format!r}")


def _resolve_capture_path(
    input_dir: str, capture_folder: str
) -> Tuple[str, Optional[Tuple[str, str]]]:
    """Resolve capture path to a container path.

    Returns:
      (container_capture_path, optional_mount)
      optional_mount is (host_abs_path, container_mount_path) when an extra mount is needed.
    """
    capture_arg = capture_folder.strip()
    if not capture_arg:
        sys.exit("--capture-folder cannot be empty")

    # Absolute capture path on host.
    if os.path.isabs(os.path.expanduser(capture_arg)):
        host_abs = _abs(capture_arg)
        if not os.path.isdir(host_abs):
            sys.exit(f"Capture folder does not exist on host: {host_abs}")
        # If capture directory is inside input_dir, reuse /input mount.
        if os.path.commonpath([input_dir, host_abs]) == input_dir:
            rel = os.path.relpath(host_abs, input_dir).strip("/\\")
            return f"/input/{rel}", None
        # Otherwise add a dedicated read-only mount.
        return "/capture_external", (host_abs, "/capture_external")

    # Relative folder under input_dir.
    rel = capture_arg.strip("/\\")
    host_abs = os.path.join(input_dir, rel)
    if not os.path.isdir(host_abs):
        sys.exit(f"Capture folder does not exist on host: {host_abs}")
    return f"/input/{rel}", None


def _effective_capture_folder(
    input_dir: str,
    capture_folder: Optional[str],
    no_auto_capture_traces: bool,
) -> Optional[str]:
    """Return the capture path argument for :func:`_resolve_capture_path`, or ``None``."""
    if capture_folder is not None:
        return capture_folder
    if no_auto_capture_traces:
        return None
    auto = os.path.join(input_dir, "capture_traces")
    if os.path.isdir(auto):
        return "capture_traces"
    return None


def _run_experiment(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    input_dir = _abs(args.input_dir)
    output_dir = _abs(args.output_dir)
    log_dir = _abs(args.agent_log_dir)
    if not os.path.isdir(input_dir):
        sys.exit(f"--input-dir is not a directory: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    trace_base = _resolve_trace_basename(input_dir, args.trace)
    platform, arch_file = _parse_gpu_arch(args.gpu_arch)

    effective_capture = _effective_capture_folder(
        input_dir,
        args.capture_folder,
        args.no_auto_capture_traces,
    )
    if effective_capture == "capture_traces" and args.capture_folder is None:
        print(
            f"Using {os.path.join(input_dir, 'capture_traces')!r} for graph replay + "
            "capture (auto-detected; use --no-auto-capture-traces to skip).",
            flush=True,
        )

    if not shutil.which("docker"):
        sys.exit("docker not found on PATH")

    if not args.no_build:
        _build_image(repo_root, args.docker_image, args.wheel)

    arch_site = os.environ.get("DOCKER_AGENTIC_ARCH_SITE", DEFAULT_ARCH_SITE)

    mounts: list[str] = [
        "-v",
        f"{input_dir}:/input:ro",
        "-v",
        f"{output_dir}:/output:rw",
        "-v",
        f"{log_dir}:/agent_log:rw",
    ]
    if arch_file:
        dest = f"{arch_site}/{os.path.basename(arch_file)}"
        mounts.extend(["-v", f"{arch_file}:{dest}:ro"])

    capture_clause = ""
    if effective_capture:
        cap_container, cap_mount = _resolve_capture_path(input_dir, effective_capture)
        if cap_mount:
            mounts.extend(["-v", f"{cap_mount[0]}:{cap_mount[1]}:ro"])
        capture_clause = (
            f", analysis mode inference, execution mode graph replay + capture, "
            f"capture folder {cap_container}"
        )
    else:
        mode = args.analysis_mode
        if mode == "inference":
            capture_clause = ", analysis mode inference, execution mode eager"
        else:
            capture_clause = ", analysis mode default"

    inner_trace = f"/input/{trace_base}"
    trace_and_platform = (
        f"for trace {inner_trace}, platform {platform}"
        f"{capture_clause}, with all artifacts under /output."
    )
    prompt = (
        "Follow the Analysis Orchestrator installed with TraceLens "
        "and run the full workflow "
        f"{trace_and_platform}\n\n"
    )

    api_key = os.environ.get("CURSOR_API_KEY")
    use_interactive_login = args.interactive_login or not api_key

    prompt_q = shlex.quote(prompt)
    log_bn = os.path.basename(args.agent_log_file)
    if (
        log_bn != args.agent_log_file
        or ".." in log_bn
        or "/" in log_bn
        or "\\" in log_bn
    ):
        sys.exit("--agent-log-file must be a basename only (no path components).")
    agent_extra = _agent_cli_suffix(args.agent_output_format)
    shell_xtrace = "set -x" if args.shell_trace else ":"
    login_block = ""
    if use_interactive_login:
        login_block = """echo "=== interactive Cursor agent login ==="
echo "Complete authentication in the browser when prompted (TTY required)."
unset NO_OPEN_BROWSER
/root/.local/bin/agent login

"""
    inner_script = f"""set -euo pipefail
LOG=/agent_log/{log_bn}
exec > >(tee "$LOG") 2>&1
{shell_xtrace}
echo "=== manual_test_wheel_agent_install begin ==="
date -u
ANALYSIS_DIR=$(python3 -c "import os, TraceLens; print(os.path.join(os.path.dirname(TraceLens.__file__), 'Agent', 'Analysis'))")
echo "ANALYSIS_DIR=$ANALYSIS_DIR"
cd "$ANALYSIS_DIR"
echo "PWD=$(pwd)"
{login_block}echo "TRACE={inner_trace}"
echo "PLATFORM={platform}"
echo "AGENT_OUTPUT_FORMAT={args.agent_output_format}"
export PYTHONUNBUFFERED=1
/root/.local/bin/agent --version || true
/root/.local/bin/agent --print --force --trust {agent_extra} {prompt_q}
echo "=== manual_test_wheel_agent_install end ==="
date -u
"""

    subprocess.run(
        ["docker", "rm", "-f", args.container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    docker_cmd: list[str] = ["docker", "run"]
    if use_interactive_login:
        docker_cmd.extend(["-i", "-t"])
    docker_cmd.extend(
        [
            "--rm",
            "--name",
            args.container_name,
            *mounts,
        ]
    )
    if api_key:
        docker_cmd.extend(["-e", f"CURSOR_API_KEY={api_key}"])
    docker_cmd.extend([args.docker_image, "bash", "-lc", inner_script])

    print("+ docker run ...", flush=True)
    print("(full command)", " ".join(docker_cmd[:8]), "... bash -lc '...'", flush=True)
    if use_interactive_login and not sys.stdin.isatty():
        print(
            "Warning: stdin is not a TTY; interactive login may not work. "
            "Run this script from a real terminal, or use CURSOR_API_KEY.",
            file=sys.stderr,
        )
    subprocess.run(docker_cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Manual Docker experiment: agentic analysis with bounded mounts."
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Host directory mounted read-only at /input (must contain the trace).",
    )
    p.add_argument(
        "--gpu-arch",
        required=True,
        help="Platform name (e.g. MI300X) or path to a <Platform>.json arch file.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Host directory mounted read-write at /output for analysis artifacts.",
    )
    p.add_argument(
        "--agent-log-dir",
        required=True,
        help=(
            "Host directory mounted read-write at /agent_log. Full container stdout/stderr "
            "(login + shell + agent) is tee'd to <this-dir>/<--agent-log-file>."
        ),
    )
    p.add_argument(
        "--agent-log-file",
        default="agent.full.log",
        help=(
            "Basename only: log file written under --agent-log-dir (default: agent.full.log)."
        ),
    )
    p.add_argument(
        "--agent-output-format",
        choices=("text", "json", "stream-json"),
        default="stream-json",
        help=(
            "Cursor `agent --print` output format. stream-json (default) records granular "
            "events; text is human-readable only."
        ),
    )
    p.set_defaults(shell_trace=True)
    p.add_argument(
        "--no-shell-trace",
        action="store_false",
        dest="shell_trace",
        help="Disable bash xtrace (set -x) in the container log.",
    )
    p.add_argument(
        "--trace",
        default=None,
        help="Trace filename inside --input-dir (required if multiple traces match).",
    )
    p.add_argument(
        "--analysis-mode",
        choices=("default", "inference"),
        default="default",
        help=(
            "Used when no capture folder applies: neither --capture-folder nor "
            "auto-detected input-dir/capture_traces (default: training-style report)."
        ),
    )
    p.add_argument(
        "--capture-folder",
        default=None,
        help=(
            "Capture trace folder for graph replay + capture mode. Accepts either "
            "an absolute host path or a path relative to --input-dir. If omitted and "
            "input-dir/capture_traces is a directory, that path is used automatically."
        ),
    )
    p.add_argument(
        "--no-auto-capture-traces",
        action="store_true",
        help=(
            "Do not use input-dir/capture_traces when --capture-folder is omitted, even "
            "if that directory exists."
        ),
    )
    p.add_argument(
        "--docker-image",
        default=DEFAULT_IMAGE,
        help=f"Docker image tag to build/run (default: {DEFAULT_IMAGE}).",
    )
    p.add_argument(
        "--container-name",
        default="tracelens-manual-agentic-run",
        help="Docker --name for the run container.",
    )
    p.add_argument(
        "--wheel",
        default=None,
        metavar="PATH",
        help=(
            "Path to a TraceLens .whl for the image (optional). If omitted, a single "
            "dist/TraceLens-*.whl is used when present; otherwise the script runs "
            "`python -m pip wheel` into dist/ to create one."
        ),
    )
    p.add_argument(
        "--no-build",
        action="store_true",
        help="Skip docker build (image must already exist).",
    )
    p.add_argument(
        "--interactive-login",
        action="store_true",
        help=(
            "Allocate a TTY (docker -it) and run `agent login` inside the container before "
            "analysis. Use from a terminal when not using CURSOR_API_KEY."
        ),
    )
    args = p.parse_args()
    _run_experiment(args)


if __name__ == "__main__":
    main()
