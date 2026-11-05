###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import subprocess
from datetime import datetime
from setuptools import setup, find_packages

_BASE_VERSION = "0.1.0"
# Same pin as the Visualizer CI job. PyPI's `torch` wheel is the CUDA build;
# the visualizer only meta-traces, so the extra installs these CPU wheels.
_TORCH_CPU_VERSION = "2.14.0"
_TORCH_CPU_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
# (platform_system, platform_machine, wheel platform tag, filename has +cpu)
_TORCH_CPU_PLATFORMS = (
    ("Linux", "x86_64", "manylinux_2_28_x86_64", True),
    ("Linux", "aarch64", "manylinux_2_28_aarch64", True),
    ("Windows", "AMD64", "win_amd64", True),
    ("Darwin", "arm64", "macosx_14_0_arm64", False),
)


def _torch_cpu_requirements():
    """Direct CPU wheel URLs.

    Package metadata cannot select an index, and ``pip install torch`` resolves
    the CUDA build on PyPI. A direct URL is what makes ``.[Visualizer]`` fetch
    the CPU wheel on its own.
    """
    requirements = []
    for python in _TORCH_CPU_PYTHONS:
        abi = "cp" + python.replace(".", "")
        for system, machine, platform_tag, cpu_local in _TORCH_CPU_PLATFORMS:
            local = "%2Bcpu" if cpu_local else ""
            url = (
                "https://download.pytorch.org/whl/cpu/"
                f"torch-{_TORCH_CPU_VERSION}{local}-{abi}-{abi}-{platform_tag}.whl"
            )
            requirements.append(
                f'torch @ {url} ; python_version == "{python}" '
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
