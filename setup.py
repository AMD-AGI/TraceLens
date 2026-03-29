###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from setuptools import setup, find_packages

setup(
    name="TraceLens",
    version="0.1.0",
    packages=find_packages(where="."),  # Will pick up 'TraceLens' automatically
    package_dir={"": "."},
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
        "matplotlib",
        "xprof==2.20.1",  # Last version with HLO sidecar generation; supports JAX 0.8+ (with benign INT_MAX warnings)
        "protobuf>=6.31.1,<7.0.0",  # Required by xprof's grpcio-status dependency
        # 'openpyxl',
        # 'tensorflow',
    ],
    extras_require={
        # To install slodels, use a custom index:
        # pip install "slodels[openai,anthropic,google-genai]" --extra-index-url https://atlartifactory.amd.com:8443/artifactory/api/pypi/SW-SLAI-PROD-VIRTUAL/simple
        # pip install "slodels[openai,anthropic,google-genai]"
        "comparative": [
            "slodels[openai,anthropic,google-genai]",
        ],
    },
    description="A library for Automating analysis from PyTorch trace files",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AMD-AIG-AIMA/TraceLens",
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
            "TraceLens_split_inference_trace = TraceLens.TraceUtils.split_inference_trace_annotation:main",
        ],
    },
)