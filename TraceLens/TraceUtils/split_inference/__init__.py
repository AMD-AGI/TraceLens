###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inference trace splitting: execution roots, steady-state windows, extraction."""

from .execution_roots import (
    DetectStatus,
    PhaseConfidence,
    RootSet,
    find_iteration_roots,
    find_iteration_roots_ex,
)
from .steady_state_window import (
    classify_workload,
    compute_reference_pd_ratio,
    find_max_pattern_window,
    find_steady_state_window,
    identify_steady_state_regions,
    select_window,
)
from .trace_extraction import (
    build_root_tiles,
    divide_phases_and_save,
    extract_and_save,
    extract_iteration,
    extract_phases_and_save,
    get_filename,
    parse_range,
    preprocess_trace,
)

__all__ = [
    "DetectStatus",
    "PhaseConfidence",
    "RootSet",
    "build_root_tiles",
    "classify_workload",
    "compute_reference_pd_ratio",
    "divide_phases_and_save",
    "extract_and_save",
    "extract_iteration",
    "extract_phases_and_save",
    "find_iteration_roots",
    "find_iteration_roots_ex",
    "find_max_pattern_window",
    "find_steady_state_window",
    "get_filename",
    "identify_steady_state_regions",
    "parse_range",
    "preprocess_trace",
    "select_window",
]
