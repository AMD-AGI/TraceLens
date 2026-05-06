###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Jarvis Analysis Package
Modular GPU performance analysis framework

Modules:
- tracelens_runner: TraceLens report generation and critical path analysis
- llm_prompts: LLM prompt management for AI analysis
- plotting_complete: Complete visualization and chart generation implementation
- report_generator: Final markdown report assembly
- data_extractors: Data extraction utilities for AI analysis
- jarvis_analysis: Main orchestrator
"""

from .tracelens_runner import TraceLensRunner
from .llm_prompts import LLMPromptManager
from .plotting_complete import JarvisPlotter
from .report_generator import ReportGenerator
from . import data_extractors

__all__ = [
    "TraceLensRunner",
    "LLMPromptManager",
    "JarvisPlotter",
    "ReportGenerator",
    "data_extractors",
]

__version__ = "2.0.0"
