###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The weight files a Hugging Face repository lists, by name and size only.

Shapes come from the model's configuration and code: a parameter has its
shape when the model is built. The engine finds its weight files where it
looks for them, and each holds no tensor, so its loading loop runs no step:
no checkpoint byte, header or tensor is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")

# The dtype names safetensors reports.
_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass
class CheckpointFile:
    """A weight file of a repository: listed and sized, holding no tensor."""

    repo: str
    revision: str
    path: str
    size: int
    tensors: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def checkpoint_files(
    repo: str, revision: str | None = None
) -> dict[str, CheckpointFile]:
    """Every weight file of ``repo``, by path within it, from the Hub's file listing."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    revision = info.sha or revision or "main"
    return {
        sibling.rfilename: CheckpointFile(
            repo, revision, sibling.rfilename, sibling.size or 0
        )
        for sibling in info.siblings or ()
        if sibling.rfilename.endswith(WEIGHT_SUFFIXES)
    }
