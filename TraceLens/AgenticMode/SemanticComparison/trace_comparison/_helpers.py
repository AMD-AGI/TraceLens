###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared helper functions for trace_comparison scripts.
"""

import json


def load_labels(path):
    """Load a semantic_labels.json file."""
    with open(path) as f:
        return json.load(f)


def load_shapes(path):
    """Load derived_shapes.json and index blocks by semantic_block name."""
    with open(path) as f:
        data = json.load(f)
    return {b["semantic_block"]: b for b in data.get("blocks", [])}
