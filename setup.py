###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import subprocess
import urllib.request
from datetime import datetime
from setuptools import setup, find_packages

_BASE_VERSION = "0.1.0"
_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu/torch/"
# Wheel files are siblings of the index directory, not children of it.
_TORCH_CPU_WHEEL_BASE = "https://download.pytorch.org/whl/cpu/"
_TORCH_CPU_WHEEL = re.compile(
    r"^torch-(\d+\.\d+\.\d+)((?:\+|%2B)cpu)?-(cp\d+)-(cp\d+)-(.+)\.whl$"
)


def _torch_cpu_platform(tag):
    if tag == "win_amd64":
        return ("Windows", "AMD64")
    if re.fullmatch(r"manylinux.+_x86_64", tag):
        return ("Linux", "x86_64")
    if re.fullmatch(r"manylinux.+_aarch64", tag):
        return ("Linux", "aarch64")
    if re.fullmatch(r"macosx_.+_arm64", tag):
        return ("Darwin", "arm64")
    return None


def _torch_cpu_requirements():
    """Newest CPU wheel for each Python version the extra can mark.

    ``pip install torch`` installs the CUDA build from PyPI. The visualizer
    only meta-traces, and package metadata cannot point at the CPU index, so
    the extra lists the CPU wheel by URL. The index is read when this package
    is built. Each Python from 3.10 up gets the newest wheel published for it.
    """
    request = urllib.request.Request(
        _TORCH_CPU_INDEX,
        headers={"User-Agent": "tracelens-setup"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            page = response.read().decode("utf-8", "replace")
    except OSError as exc:
        raise RuntimeError(
            "Could not read the PyTorch CPU wheel index at "
            f"{_TORCH_CPU_INDEX} to select the Visualizer torch wheel."
        ) from exc

    newest = {}
    for name in set(re.findall(r"torch-[^\"'#<>\s]+\.whl", page)):
        match = _TORCH_CPU_WHEEL.match(name)
        if not match:
            continue
        version, _local, abi, abi2, platform_tag = match.groups()
        if abi != abi2:
            continue
        platform = _torch_cpu_platform(platform_tag)
        if platform is None:
            continue
        python = (int(abi[2]), int(abi[3:]))
        if python < (3, 10):
            continue
        version_key = tuple(int(part) for part in version.split("."))
        # Prefer manylinux_2_28 when one version publishes more than one tag.
        rank = 1 if "manylinux_2_28" in platform_tag else 0
        candidate = (version_key, rank, name.replace("+", "%2B"))
        key = (python, platform)
        current = newest.get(key)
        if current is None or candidate[:2] > current[:2]:
            newest[key] = candidate

    if not newest:
        raise RuntimeError(f"No CPU torch wheels found at {_TORCH_CPU_INDEX}.")

    requirements = []
    for (python, (system, machine)), (_version, _rank, wheel) in sorted(newest.items()):
        requirements.append(
            f'torch @ {_TORCH_CPU_WHEEL_BASE}{wheel} ; python_version == "{python[0]}.{python[1]}" '
            f'and platform_system == "{system}" '
            f'and platform_machine == "{machine}"'
        )
    return requirements


def _wheel_version():
    """Produce TraceLens-<date>+<commithash> wheel names (PEP 440)."""
    try:
        short_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        date_stamp = datetime.now().strftime("%Y%m%d")
        return f"{_BASE_VERSION}.dev{date_stamp}+g{short_sha}"
    except (OSError, subprocess.CalledProcessError):
        return _BASE_VERSION


with open("README.md", encoding="utf-8") as _readme:
    _LONG_DESCRIPTION = _readme.read()


def _extras_require():
    extras = {
        "jax": [
            "xprof==2.23.2",  # Current release; still writes HLO sidecars.
            "protobuf>=6.33.5",  # Minimum pulled in with xprof; protobuf 7 works.
        ],
        "comparative": [
            "slodels[openai,anthropic,google-genai]",
        ],
        "Visualizer": [
            *_torch_cpu_requirements(),
            "transformers",
            "huggingface_hub>=0.20",
            # fla-core imports Triton at package import. The visualizer never
            # launches a kernel; it only needs that import to succeed.
            'triton; platform_system == "Linux" and platform_machine == "x86_64"',
        ],
        "kernel_source": [
            "itanium-demangler>=1.0",
        ],
        "dev": [
            "pytest",
            "setuptools",
            "black==26.3.1",
        ],
    }
    extras["all"] = [dep for group in extras.values() for dep in group]
    return extras


setup(
    name="TraceLens",
    version=_wheel_version(),
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "TraceLens": [
            "**/*.md",
            "Agent/**/skills/**/*",
            "Agent/Analysis/utils/arch/*.json",
        ],
        "TraceLens.Visualizer.model_explorer_export": [
            "viewer/*.html",
            "viewer/app.js",
            "viewer/NOTICE",
            "viewer/ai-edge-model-explorer-visualizer.LICENSE",
        ],
    },
    exclude_package_data={
        "TraceLens": [
            "Agent/Analysis/skills/analysis-orchestrator/evals/*",
            "Agent/Visualizer/skills/model-visualizer/evals/*",
            "Agent/Visualizer/skills/model-roofline-projection/evals/*",
        ],
    },
    install_requires=[
        "pandas",
        "tqdm",
        'backports.strenum;python_version<"3.11"',
        'StrEnum;python_version<"3.11"',
        "openpyxl",
        "office365-rest-python-client",
        "msal",
        "tabulate",
        "orjson",
        "PyYAML",
        "matplotlib",
        "typing_extensions>=4.0; python_version < '3.11'",
        # 'openpyxl',
        # 'tensorflow',
    ],
    extras_require=_extras_require(),
    description="A library for automating the analysis of ML model performance traces",
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/AMD-AGI/TraceLens",
    classifiers=[
        "Programming Language :: Python :: 3",
        # 'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "TraceLens_generate_perf_report_jax = TraceLens.Reporting.generate_perf_report_jax:main",
            "TraceLens_generate_perf_report_pytorch = TraceLens.Reporting.generate_perf_report_pytorch:main",
            "TraceLens_generate_perf_report_pytorch_inference = TraceLens.Reporting.generate_perf_report_pytorch_inference:main",
            "TraceLens_generate_perf_report_from_config = TraceLens.Reporting.generate_perf_report_from_config:main",
            "TraceLens_generate_perf_report_rocprof = TraceLens.Reporting.generate_perf_report_rocprof:main",
            "TraceLens_compare_perf_reports_pytorch = TraceLens.Reporting.compare_perf_reports_pytorch:main",
            "TraceLens_generate_multi_rank_collective_report_pytorch = TraceLens.Reporting.generate_multi_rank_collective_report_pytorch:main",
            "TraceLens_generate_perf_report_pftrace_hip_api = TraceLens.Reporting.generate_perf_report_pftrace_hip_api:main",
            "TraceLens_generate_perf_report_pftrace_hip_activity = TraceLens.Reporting.generate_perf_report_pftrace_hip_activity:main",
            "TraceLens_generate_perf_report_pftrace_memory_copy = TraceLens.Reporting.generate_perf_report_pftrace_memory_copy:main",
            "TraceLens_generate_perf_report_genesis = TraceLens.Reporting.generate_perf_report_genesis:main",
            "TraceLens_visualize_model_in_explorer = TraceLens.Visualizer.model_explorer_export.cli:main",
            "visualize_model_in_explorer = TraceLens.Visualizer.model_explorer_export.cli:main",
            "TraceLens_model_roofline_projection = TraceLens.Visualizer.model_explorer_export.roofline_cli:main",
            "TraceLens_split_trace = TraceLens.TraceUtils.split_trace.main:main",
            "TraceLens_resolve_kernel_source = TraceLens.TraceUtils.kernel_source.cli:main",
            "TraceLens_trace_index = TraceLens.TraceIndex.cli:main",
        ],
    },
)
