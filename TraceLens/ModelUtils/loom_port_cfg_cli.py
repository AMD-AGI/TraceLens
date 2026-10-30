#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI for a Loom ``loom-port-llm`` ``cfg.json`` architecture-facts export.

Loads a checkpoint through the same AST backend as the Model Explorer export
and the roofline projection (``TraceLens.ModelUtils`` loader), and derives the
subset of ``loom-port-llm``'s ``cfg.json`` architecture fields TraceLens can
read from the checkpoint's own ``config.json``/modeling source -- no GGUF
conversion, chat-template, or kernel-tensor-format work attempted. See
``TraceLens.ModelUtils.loom_port_cfg`` for exactly which fields and why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from TraceLens.ModelUtils.cli_common import (
    add_model_source_arguments,
    add_runtime_regime_arguments,
    load_architecture_spec_from_args,
)
from TraceLens.ModelUtils.loader import resolve_checkpoint_arg
from TraceLens.ModelUtils.loom_port_cfg import (
    build_loom_arch_facts,
    coverage_notes,
    diff_against_existing,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TraceLens_loom_port_cfg",
        description=(
            "Derive the architecture-fact fields of a Loom loom-port-llm cfg.json "
            "(hidden/layers/heads/kv_heads/head_dim/ffn/vocab/context_length, "
            "rms_eps, rope_theta/rope_dims, qk_norm, sliding_layers/sliding_window, "
            "recurrent_layers) from a Hugging Face checkpoint's own config.json and "
            "modeling source, with no GGUF conversion required. loom-serve runs a "
            "model on one device, so there are no --tp/--pp/--ep flags here; "
            "quantized tensor formats, the GGUF path, and the chat-template-derived "
            "block stay owned by loom-port-llm's own inspect_gguf.py/chat_probe.py."
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
        "--output",
        "-o",
        type=Path,
        metavar="PATH",
        help="Write the derived facts as JSON (default: stdout)",
    )
    parser.add_argument(
        "--check",
        type=Path,
        metavar="EXISTING_CFG.json",
        help=(
            "Diff the derived facts against an already-authored cfg.json (e.g. "
            "one produced by loom-port-llm's inspect_gguf.py) and print mismatches "
            "instead of writing --output."
        ),
    )
    add_model_source_arguments(parser)
    add_runtime_regime_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint = resolve_checkpoint_arg(checkpoint=args.checkpoint, source=args.source)
    if checkpoint is None and args.github is None:
        parser.error(
            "Provide a Hugging Face checkpoint (SOURCE or --checkpoint) and/or --github"
        )

    try:
        spec = load_architecture_spec_from_args(
            checkpoint, args, title=args.title, stop_at_roofline_ops=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading architecture: {exc}", file=sys.stderr)
        return 1

    facts = build_loom_arch_facts(
        spec, name=args.title or (str(checkpoint) if checkpoint else None)
    )
    for note in coverage_notes(spec):
        print(f"Note: {note}", file=sys.stderr)

    if args.check is not None:
        try:
            existing = json.loads(args.check.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Error reading {args.check}: {exc}", file=sys.stderr)
            return 1
        mismatches = diff_against_existing(facts, existing)
        if not mismatches:
            print(f"No mismatches against {args.check}.")
            return 0
        print(f"Mismatches against {args.check}:")
        for line in mismatches:
            print(f"  {line}")
        return 1

    text = json.dumps(facts, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote Loom architecture facts: {args.output}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
