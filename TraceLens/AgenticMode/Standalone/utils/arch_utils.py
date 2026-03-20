###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Loader for per-platform GPU architecture JSON specs.

Each JSON file in utils/arch/ follows the TraceLens arch dict schema
(see examples/gpu_arch_example.md) with an additional ``memory_gb`` field.
"""

import json
import os

_ARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch")


def list_platforms():
    """Return available platform names derived from arch/*.json filenames."""
    return sorted(
        f.removesuffix(".json") for f in os.listdir(_ARCH_DIR) if f.endswith(".json")
    )


def load_arch(platform: str) -> dict:
    """Read and return the arch JSON dict for the given platform."""
    path = os.path.join(_ARCH_DIR, f"{platform}.json")
    with open(path) as f:
        return json.load(f)
