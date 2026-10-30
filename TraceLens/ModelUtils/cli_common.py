###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared argparse groups and ``ArchitectureSpec`` loading for ModelUtils CLIs.

Every CLI that loads a checkpoint through the AST backend (Model Explorer
export, the roofline projection tool, ...) needs the same checkpoint-provenance,
basic-op-filter, and runtime-regime flags. Defining them once here keeps the
flags and their help text from drifting between tools.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from TraceLens.ModelUtils.ast_analyze import RuntimeRegime
from TraceLens.ModelUtils.basic_ops import DEFAULT_BASIC_OP_PATTERNS, BasicOpFilter
from TraceLens.ModelUtils.extract import ArchitectureSpec, parse_config_overrides
from TraceLens.ModelUtils.loader import (
    build_detailed_basic_ops,
    build_roofline_basic_ops,
    load_model_spec,
)


def add_model_source_arguments(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:
    """Add the "model source options" group: checkpoint provenance and basic-op filters."""
    group = parser.add_argument_group("model source options")
    group.add_argument(
        "--github",
        "-g",
        help=(
            "Optional GitHub repo URL or github:owner/repo@ref:path for modeling source "
            "when the HF repo does not ship modeling_*.py (repo must be whitelisted)"
        ),
    )
    group.add_argument(
        "--allow-repo",
        action="append",
        default=[],
        metavar="OWNER/REPO",
        help=(
            "Whitelist an extra GitHub repository for remote source introspection "
            "(repeatable; huggingface/transformers is allowed by default)"
        ),
    )
    group.add_argument(
        "--config-path",
        help="Explicit config.json path inside the checkpoint",
    )
    group.add_argument(
        "--code-path",
        type=Path,
        help="Explicit path to modeling_*.py when auto-discovery is insufficient",
    )
    group.add_argument(
        "--basic-op-add",
        action="append",
        default=[],
        metavar="REGEX",
        help=(
            "Regex for block names treated as basic leaf operations "
            f"(repeatable; defaults: {', '.join(DEFAULT_BASIC_OP_PATTERNS)})"
        ),
    )
    group.add_argument(
        "--basic-op-remove",
        action="append",
        default=[],
        metavar="REGEX",
        help="Remove a default basic-op regex (repeatable; pass exact pattern text)",
    )
    group.add_argument(
        "--meta-shapes",
        dest="meta_shapes",
        action="store_true",
        default=True,
        help=(
            "Run a meta-device forward pass (requires torch + transformers) to "
            "capture ground-truth shapes for nn.Module layers AND for inline "
            "operations the static analysis can't resolve (default: on)"
        ),
    )
    group.add_argument(
        "--no-meta-shapes",
        dest="meta_shapes",
        action="store_false",
        help="Disable torch-assisted shape inference (pure static analysis)",
    )
    return group


def add_roofline_op_boundary_argument(
    parser: argparse.ArgumentParser,
    *,
    group: argparse._ArgumentGroup | None = None,
    default: bool = True,
) -> argparse._ArgumentGroup:
    """Add ``--stop-at-roofline-ops`` to *group* (a new "model source options" group by default).

    Tools that always want this behavior (the roofline projection CLI) should
    not add this flag at all and instead pass ``stop_at_roofline_ops=True``
    directly to :func:`load_architecture_spec_from_args`.
    """
    group = group or parser.add_argument_group("model source options")
    group.add_argument(
        "--stop-at-roofline-ops",
        dest="stop_at_roofline_ops",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=(
            "Stop expanding a module once it matches an op with a TraceLens.PerfModel "
            "formula (Linear, Conv, RMSNorm/LayerNorm/GroupNorm/BatchNorm/InstanceNorm, "
            "Softmax), instead of continuing into its internal primitive steps "
            f"(default: {'on' if default else 'off'})."
        ),
    )
    return group


def add_runtime_regime_arguments(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:
    """Add the "runtime regime" group: which forward-pass arm the export depicts."""
    group = parser.add_argument_group(
        "runtime regime",
        "Which RUN the export depicts. A forward branches on things no config "
        "decides and only a run does, and both arms are real: drawing both "
        "leaves every shape below the branch true of only one of them. Each "
        "option picks exactly one arm, so the arm not taken is never used.",
    )
    group.add_argument(
        "--sequence-layout",
        choices=list(RuntimeRegime.SEQUENCE_LAYOUTS),
        default=RuntimeRegime.SEQUENCE_LAYOUTS[0],
        help=(
            "rectangular keeps [B, S, ...]; packed removes the padding first "
            "and carries every real token as one row, [1, nnz, ...] "
            "(default: rectangular)."
        ),
    )
    group.add_argument(
        "--phase",
        choices=list(RuntimeRegime.PHASES),
        default=RuntimeRegime.PHASES[0],
        help=(
            "prefill scores a whole prompt; decode advances one token, which "
            "is what makes the query length 1 and can select different kernels "
            "entirely (default: prefill)."
        ),
    )
    group.add_argument(
        "--cache",
        choices=list(RuntimeRegime.CACHES),
        default=RuntimeRegime.CACHES[0],
        help=(
            "cold is the first forward, with nothing stored yet; warm "
            "continues from a cache that already holds keys and values "
            "(default: cold). decode always reads a cache."
        ),
    )
    group.add_argument(
        "--attention",
        default=None,
        help=(
            "Attention implementation to draw (eager, sdpa, "
            "flash_attention_2, ...). eager draws the real matmul/softmax "
            "chain where sdpa draws one fused kernel. Unset takes the "
            "checkpoint's own setting, which is sdpa unless it says otherwise."
        ),
    )
    group.add_argument(
        "--mode",
        choices=list(RuntimeRegime.MODES),
        default=RuntimeRegime.MODES[0],
        help="Whether `self.training` reads true (default: inference).",
    )
    group.add_argument(
        "--device",
        choices=list(RuntimeRegime.DEVICES),
        default=RuntimeRegime.DEVICES[0],
        help=(
            "The accelerator being modeled, for files that branch on "
            "is_npu()/is_cuda()/is_hip()/... at import time. Stated, never "
            "detected: this never probes the machine running the export "
            "(default: hip, this repo's own vendor)."
        ),
    )
    group.add_argument(
        "--config-override",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override one config key, repeatable. Applied left to right over "
            "everything else -- the checkpoint, the config class defaults and "
            "the model's own constructor writes -- so this is the last word. "
            "A dotted key reaches a nested config "
            "(vision_config.hidden_size=1024); the value is read as JSON when "
            "it parses as one (8, true, null, [1, 2]) and as text otherwise. "
            "Stating one key twice warns and keeps the last."
        ),
    )
    return group


def add_engine_argument(
    parser: argparse.ArgumentParser,
    *,
    group: argparse._ArgumentGroup | None = None,
) -> argparse._ArgumentGroup:
    """Add just ``--engine`` to *group* (a new "model source options" group by default).

    Pass an existing group (e.g. a caller's own "device partitions" group) to
    keep ``--engine`` next to related flags like ``--tp``/``--pp``/``--ep``
    instead of opening a second "model source options" section in ``--help``.
    """
    group = group or parser.add_argument_group("model source options")
    group.add_argument(
        "--engine",
        choices=("vllm", "sglang"),
        help=(
            "Draw this checkpoint's architecture from vLLM's or SGLang's own "
            "pinned modeling source (ColumnParallelLinear, make_layers, FusedMoE, "
            "...) instead of transformers'. Ignored once --code-path / --github "
            "already name the source."
        ),
    )
    return group


def build_runtime_regime(args: argparse.Namespace) -> RuntimeRegime:
    """Build a :class:`RuntimeRegime` from the shared runtime-regime flags."""
    return RuntimeRegime(
        sequence_layout=args.sequence_layout,
        phase=args.phase,
        cache=args.cache,
        attention=args.attention,
        mode=args.mode,
        device=args.device,
    )


def build_basic_ops_from_args(
    args: argparse.Namespace,
    *,
    stop_at_roofline_ops: bool | None = None,
) -> BasicOpFilter:
    """Pick the leaf-op granularity and build its :class:`BasicOpFilter`.

    ``stop_at_roofline_ops=True`` stops expansion at ops with a PerfModel formula
    (:func:`build_roofline_basic_ops`); ``False`` uses the usual detailed-export
    leaf set (:func:`build_detailed_basic_ops`). Defaults to
    ``args.stop_at_roofline_ops`` when the caller added
    :func:`add_roofline_op_boundary_argument`, else ``True``.
    """
    resolved = (
        stop_at_roofline_ops
        if stop_at_roofline_ops is not None
        else getattr(args, "stop_at_roofline_ops", True)
    )
    builder = build_roofline_basic_ops if resolved else build_detailed_basic_ops
    return builder(add=args.basic_op_add, remove=args.basic_op_remove)


def load_architecture_spec_from_args(
    checkpoint: str | Path | None,
    args: argparse.Namespace,
    *,
    title: str | None = None,
    stop_at_roofline_ops: bool | None = None,
) -> ArchitectureSpec:
    """Load an :class:`ArchitectureSpec` using the shared model-source / regime flags.

    Common path for every CLI built on :func:`add_model_source_arguments` and
    :func:`add_runtime_regime_arguments` (and, optionally, :func:`add_engine_argument`).
    See :func:`build_basic_ops_from_args` for ``stop_at_roofline_ops``.
    """
    basic_ops = build_basic_ops_from_args(
        args, stop_at_roofline_ops=stop_at_roofline_ops
    )
    return load_model_spec(
        checkpoint,
        github=args.github,
        config_path=args.config_path,
        code_path=args.code_path,
        name=title,
        analyze_code=True,
        detailed=True,
        basic_ops=basic_ops,
        require_code=True,
        allow_github_repos=args.allow_repo,
        regime=build_runtime_regime(args),
        config_overrides=parse_config_overrides(args.config_override),
        engine=getattr(args, "engine", None),
    )
