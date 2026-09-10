###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Serialization helper for Model Explorer payloads.

The graph itself is built by the AST backend
(``ast_build.build_model_explorer_payload``); this module only handles
writing the resulting payload to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_model_explorer_payload(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a Model Explorer payload to JSON."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target
