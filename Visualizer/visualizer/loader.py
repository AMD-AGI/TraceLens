###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared Hugging Face / local model loading for Model Explorer export."""

from __future__ import annotations

from pathlib import Path

from visualizer.basic_ops import COMMON_LEAF_PATTERNS, BasicOpFilter
from visualizer.extract import ArchitectureSpec, load_architecture


def resolve_checkpoint_arg(
    *,
    checkpoint: str | Path | None = None,
    source: str | Path | None = None,
) -> str | Path | None:
    """Resolve the checkpoint positional/flag pair from the Model Explorer CLI."""
    return checkpoint or source


def build_detailed_basic_ops(
    *,
    add: list[str] | None = None,
    remove: list[str] | None = None,
) -> BasicOpFilter:
    """Basic-op filter used by ``--detailed`` Model Explorer export."""
    extra_add = list(add or [])
    for pattern in COMMON_LEAF_PATTERNS:
        if pattern not in extra_add:
            extra_add.append(pattern)
    return BasicOpFilter.from_cli(add=extra_add, remove=remove or [])


def load_model_spec(
    checkpoint: str | Path | None,
    *,
    github: str | None = None,
    config_path: str | None = None,
    code_path: str | Path | None = None,
    name: str | None = None,
    analyze_code: bool = True,
    detailed: bool = True,
    basic_ops: BasicOpFilter | None = None,
    require_code: bool = False,
    allow_github_repos: list[str] | None = None,
) -> ArchitectureSpec:
    """Load architecture metadata for Model Explorer export.

    When ``checkpoint`` is a Hugging Face model id, ``config.json`` and
    Python sources in the cached snapshot (or listed on the hub) are
    resolved automatically unless ``--code-path`` / ``--github`` override
    the modeling source.
    """
    if require_code and not analyze_code:
        analyze_code = True

    resolved_basic_ops = basic_ops
    if detailed and resolved_basic_ops is None:
        resolved_basic_ops = build_detailed_basic_ops()

    spec = load_architecture(
        checkpoint,
        name=name,
        github=github,
        config_path=config_path,
        code_path=code_path,
        analyze_code=analyze_code,
        detailed=detailed,
        basic_ops=resolved_basic_ops,
        allow_github_repos=allow_github_repos,
    )

    if require_code and analyze_code and not spec.class_registry:
        raise FileNotFoundError(
            "No modeling source found for AST inspection. Pass a Hugging Face model "
            "id or local checkpoint directory (config.json and nested .py files are "
            "discovered automatically), or set --code-path / --github for the modeling file."
        )

    if require_code and not spec.export_block_trees:
        raise ValueError(
            "Model loaded but no computation block trees were built. The modeling "
            "source may lack a parseable decoder stack or forward() methods."
        )

    return spec
