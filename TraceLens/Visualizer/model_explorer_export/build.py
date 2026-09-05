###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build Model Explorer payloads from PyTorch model tracing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from TraceLens.ModelUtils.torch_trace import build_graph


def build_model_explorer_payload(
    checkpoint: str | Path,
    *,
    seq_len: int = 128,
    batch_size: int = 1,
    title: str | None = None,
) -> dict[str, Any]:
    """Build a Model Explorer payload via PyTorch tracing."""
    return build_graph(
        checkpoint,
        seq_len=seq_len,
        batch_size=batch_size,
        title=title,
    )


def save_model_explorer_payload(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a Model Explorer payload to JSON."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
