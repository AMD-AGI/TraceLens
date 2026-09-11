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
)
from .steady_state_window import (
    classify_phases_from_batch_sizes,
    find_steady_state_generic,
    find_steady_state_inference,
    find_steady_state_inference_from_shapes,
)
from .trace_extraction import (
    build_cpu_event_index,
    build_root_tiles,
    divide_phases_and_save,
    extract_and_save,
    extract_iteration,
    extract_phases_and_save,
    get_filename,
    infer_batch_sizes_from_shapes,
    parse_range,
    preprocess_trace,
)
from ...util import most_common_first_dim

__all__ = [
    "DetectStatus",
    "PhaseConfidence",
    "RootSet",
    "build_cpu_event_index",
    "build_root_tiles",
    "classify_phases_from_batch_sizes",
    "divide_phases_and_save",
    "extract_and_save",
    "extract_iteration",
    "extract_phases_and_save",
    "find_iteration_roots",
    "find_steady_state_generic",
    "find_steady_state_inference",
    "find_steady_state_inference_from_shapes",
    "get_filename",
    "infer_batch_sizes_from_shapes",
    "most_common_first_dim",
    "parse_range",
    "preprocess_trace",
]
