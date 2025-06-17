import sys
from setuptools import setup, find_packages


setup(
    name='TraceLens',
    version='0.1.0',
    packages=find_packages(where='.'),  # Will pick up 'TraceLens' automatically
    package_dir={"": "."},
    install_requires=[
        'pandas',
        'tqdm',
        'backports.strenum;python_version<"3.11"',
        'StrEnum;python_version<"3.11"',
        'openpyxl',
    ],
    description="A library for Automating analysis from PyTorch trace files",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AMD-AIG-AIMA/TraceLens', 
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": [
            "TraceLens_perf_reporting_multiple_ranks = TraceLens.Reporting.generate_perf_report_multiple_ranks_trace_files:main",
        ],
    },
)
