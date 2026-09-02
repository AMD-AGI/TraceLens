###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import subprocess
from datetime import datetime
from pathlib import Path
from setuptools import setup, find_packages

_BASE_VERSION = "0.1.0"
_ROOT = Path(__file__).resolve().parent


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


def _read_requirements(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


_VISUALIZER_PACKAGES = find_packages(where="Visualizer")
_VISUALIZER_PACKAGE_DIR = {
    package: f"Visualizer/{package.replace('.', '/')}"
    for package in _VISUALIZER_PACKAGES
}

with open("README.md", encoding="utf-8") as _readme:
    _LONG_DESCRIPTION = _readme.read()

setup(
    name="TraceLens",
    version=_wheel_version(),
    packages=find_packages(where=".") + _VISUALIZER_PACKAGES,
    package_dir={"": ".", **_VISUALIZER_PACKAGE_DIR},
    include_package_data=True,
    package_data={
        "TraceLens": [
            "**/*.md",
            "Agent/**/skills/**/*",
            "Agent/Analysis/utils/arch/*.json",
        ],
        "model_explorer_export": [
            "viewer/*.html",
            "viewer/*.js",
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
        "xprof==2.20.1",  # Last version with HLO sidecar generation; supports JAX 0.8+ (with benign INT_MAX warnings)
        "protobuf>=6.31.1,<7.0.0",  # Required by xprof's grpcio-status dependency
        # 'openpyxl',
        # 'tensorflow',
    ],
    extras_require={
        "comparative": [
            "slodels[openai,anthropic,google-genai]",
        ],
        "Visualizer": _read_requirements(_ROOT / "Visualizer" / "requirements.txt"),
        "dev": [
            "pytest",
            "black==26.3.1",
        ],
    },
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
            "TraceLens_generate_perf_report_rocprof = TraceLens.Reporting.generate_perf_report_rocprof:main",
            "TraceLens_compare_perf_reports_pytorch = TraceLens.Reporting.compare_perf_reports_pytorch:main",
            "TraceLens_generate_multi_rank_collective_report_pytorch = TraceLens.Reporting.generate_multi_rank_collective_report_pytorch:main",
            "TraceLens_generate_perf_report_pftrace_hip_api = TraceLens.Reporting.generate_perf_report_pftrace_hip_api:main",
            "TraceLens_generate_perf_report_pftrace_hip_activity = TraceLens.Reporting.generate_perf_report_pftrace_hip_activity:main",
            "TraceLens_generate_perf_report_pftrace_memory_copy = TraceLens.Reporting.generate_perf_report_pftrace_memory_copy:main",
            "TraceLens_generate_perf_report_genesis = TraceLens.Reporting.generate_perf_report_genesis:main",
            "TraceLens_split_inference_trace = TraceLens.TraceUtils.split_inference_trace_annotation:main",
            "TraceLens_visualize_model_in_explorer = model_explorer_export.cli:main",
            "visualize_model_in_explorer = model_explorer_export.cli:main",
        ],
    },
)
