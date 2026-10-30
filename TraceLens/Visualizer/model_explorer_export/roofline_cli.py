#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI for a static, trace-free full-model roofline projection.

Loads a checkpoint through the same AST backend as the Model Explorer export
(``TraceLens.ModelUtils`` loader), stops expansion at ops with a direct
``TraceLens.PerfModel`` formula, and prints a kernel-by-kernel roofline table
plus a model-level rollup -- no trace or GPU run required.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms, load_arch
from TraceLens.ModelUtils.cli_common import (
    add_engine_argument,
    add_model_source_arguments,
    add_runtime_regime_arguments,
    load_architecture_spec_from_args,
)
from TraceLens.ModelUtils.loader import resolve_checkpoint_arg
from TraceLens.ModelUtils.parallel_plan import ParallelDegree, build_parallel_layout
from TraceLens.Visualizer.model_explorer_export.roofline_projection import (
    RooflineOp,
    project_roofline,
)

# Constants

_TABLE_COLUMNS = (
    "section",
    "name",
    "category",
    "instances",
    "total_flops",
    "total_bytes",
    "compute_spec",
    "roofline_bound",
    "total_roofline_time_us",
    "reason",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TraceLens_model_roofline_projection",
        description=(
            "Project a kernel-by-kernel roofline for a model checkpoint from its "
            "config and modeling source alone, with no trace or GPU run. Ops are "
            "costed with TraceLens.PerfModel's analytical FLOPs/bytes formulas; "
            "ops with no direct formula (embeddings, MoE routing, convolutions, "
            "...) are reported as unmodeled rather than silently dropped."
        ),
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Hugging Face model id or local checkpoint directory (alias for --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        help="Hugging Face model id or local checkpoint path for config.json",
    )
    parser.add_argument(
        "--title",
        help="Architecture display name override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size to project at (default: 1)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        metavar="N",
        help="Sequence length to project at (default: 2048)",
    )
    parser.add_argument(
        "--dim",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one symbolic shape dimension with a concrete int, "
            "repeatable (e.g. --dim past_len=1024). Takes precedence over "
            "--batch-size/--seq-len for the same key."
        ),
    )
    parser.add_argument(
        "--gpu-arch-name",
        metavar="NAME",
        help=(
            "Named GPU architecture spec to bound against "
            f"(available: {', '.join(list_platforms())}). Without this or "
            "--gpu-arch-json-path, FLOPs/bytes are still reported but no "
            "roofline time/bound is computed."
        ),
    )
    parser.add_argument(
        "--gpu-arch-json-path",
        type=Path,
        metavar="PATH",
        help="Path to a GPU arch JSON (same schema as --gpu-arch-name); takes precedence",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        metavar="PATH",
        help="Write the per-op table as CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        metavar="PATH",
        help="Write the per-op table and rollup as JSON",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="Rows to print to stdout, ranked by total roofline time (default: 20; 0 for all)",
    )
    add_model_source_arguments(parser)
    add_runtime_regime_arguments(parser)

    partition_group = parser.add_argument_group(
        "device partitions",
        "How many ways to split the model before projecting its roofline. The "
        "split itself comes from the plans on its Hugging Face config "
        "(base_model_tp_plan, base_model_pp_plan, base_model_ep_plan, plus the "
        "causal-LM head), or, with --engine, from that inference engine's own "
        "parallel-layer classes. 1 leaves that axis unsplit; the projection then "
        "covers one GPU's own share of the work, the same per-GPU shapes the "
        "Model Explorer viewer would draw for this split.",
    )
    add_engine_argument(parser, group=partition_group)
    partition_group.add_argument(
        "--tp",
        type=int,
        default=1,
        metavar="N",
        help="Tensor-parallel partitions (default: 1).",
    )
    partition_group.add_argument(
        "--pp",
        type=int,
        default=1,
        metavar="N",
        help="Pipeline-parallel partitions (default: 1).",
    )
    partition_group.add_argument(
        "--ep",
        type=int,
        default=1,
        metavar="N",
        help="Expert-parallel partitions (default: 1).",
    )
    return parser


def parse_dim_overrides(entries: list[str]) -> dict[str, int]:
    """Parse ``--dim KEY=VALUE`` pairs into an int-valued dict."""
    overrides: dict[str, int] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--dim must be KEY=VALUE, got {entry!r}")
        try:
            overrides[key.strip()] = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"--dim value must be an int, got {entry!r}") from exc
    return overrides


def _op_row(op: RooflineOp) -> dict[str, object]:
    return {
        "section": op.section,
        "name": op.name,
        "computation": op.computation,
        "category": op.category,
        "instances": op.instances,
        "flops": op.flops,
        "bytes": op.bytes,
        "total_flops": op.total_flops,
        "total_bytes": op.total_bytes,
        "compute_spec": op.compute_spec,
        "roofline_bound": op.roofline_bound,
        "roofline_time_us": op.roofline_time_us,
        "total_roofline_time_us": op.total_roofline_time_us,
        "reason": op.reason,
    }


def format_table(ops: list[RooflineOp], *, top_n: int) -> str:
    """Render a fixed-width table of *ops*, ranked by total roofline time."""
    ranked = sorted(ops, key=lambda op: op.total_roofline_time_us or 0, reverse=True)
    if top_n > 0:
        ranked = ranked[:top_n]
    rows = [_op_row(op) for op in ranked]
    widths = {
        col: (
            max(len(col), *(len(_format_cell(row[col])) for row in rows))
            if rows
            else len(col)
        )
        for col in _TABLE_COLUMNS
    }
    lines = [" | ".join(col.ljust(widths[col]) for col in _TABLE_COLUMNS)]
    lines.append("-+-".join("-" * widths[col] for col in _TABLE_COLUMNS))
    for row in rows:
        lines.append(
            " | ".join(
                _format_cell(row[col]).ljust(widths[col]) for col in _TABLE_COLUMNS
            )
        )
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def write_csv(ops: list[RooflineOp], path: Path) -> None:
    rows = [_op_row(op) for op in ops]
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_projection(args):
    """Resolve the model and project its roofline from parsed CLI ``args``.

    Shared by the ``TraceLens_model_roofline_projection`` CLI and the
    orchestrator roofline-augmentation utility (``roofline_augment``). Raises
    ``ValueError`` for an invalid model source or ``--dim`` override, ``KeyError``
    for an unknown ``--gpu-arch-name``, and ``RuntimeError`` when the architecture
    fails to load. Returns ``(projection, gpu_arch, layout_warnings)``.
    """
    checkpoint = resolve_checkpoint_arg(checkpoint=args.checkpoint, source=args.source)
    if checkpoint is None and args.github is None:
        raise ValueError(
            "Provide a Hugging Face checkpoint (SOURCE or --checkpoint) and/or --github"
        )
    extra_dims = parse_dim_overrides(args.dim)
    gpu_arch = None
    if args.gpu_arch_json_path is not None:
        gpu_arch = json.loads(args.gpu_arch_json_path.read_text(encoding="utf-8"))
    elif args.gpu_arch_name is not None:
        gpu_arch = load_arch(args.gpu_arch_name)
    try:
        spec = load_architecture_spec_from_args(
            checkpoint, args, title=args.title, stop_at_roofline_ops=True
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Error loading architecture: {exc}") from exc
    degree = ParallelDegree(tp=args.tp, pp=args.pp, ep=args.ep)
    layout_warnings: list[str] = []
    if degree.requested():
        spec.parallel_layout = build_parallel_layout(spec, degree)
        for line in spec.parallel_layout.fact_lines():
            text = line.strip()
            if (
                "does not divide" in text
                or text.startswith("The model declares")
                or text.startswith("No Hugging Face")
            ):
                layout_warnings.append(text)
    projection = project_roofline(
        spec,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        extra_dims=extra_dims,
        gpu_arch=gpu_arch,
    )
    return projection, gpu_arch, layout_warnings


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        projection, gpu_arch, layout_warnings = run_projection(args)
    except ValueError as exc:
        parser.error(str(exc))
    except KeyError:
        parser.error(
            f"Unknown --gpu-arch-name {args.gpu_arch_name!r}; "
            f"available: {', '.join(list_platforms())}"
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for text in layout_warnings:
        print(f"Warning: {text}", file=sys.stderr)

    print(format_table(projection.ops, top_n=args.top_n))
    print()
    totals = projection.totals()
    print(
        f"Total: {totals['total_flops']:,.0f} FLOPs, {totals['total_bytes']:,.0f} bytes, "
        f"modeled {totals['modeled_op_count']} ops / unmodeled {totals['unmodeled_op_count']} ops "
        f"({(totals['modeled_output_element_share'] or 0) * 100:.1f}% of output elements modeled)"
    )
    if totals["total_roofline_time_us"]:
        print(
            f"Roofline time: {totals['total_roofline_time_us']:,.1f} us "
            f"(compute-bound {totals['compute_bound_time_us']:,.1f} us, "
            f"memory-bound {totals['memory_bound_time_us']:,.1f} us)"
        )
    elif gpu_arch is None:
        print(
            "No --gpu-arch-name/--gpu-arch-json-path given: FLOPs/bytes only, no roofline time."
        )

    if args.output_csv is not None:
        write_csv(projection.ops, args.output_csv)
        print(f"Wrote CSV: {args.output_csv}")
    if args.output_json is not None:
        payload = {"ops": [_op_row(op) for op in projection.ops], "totals": totals}
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
