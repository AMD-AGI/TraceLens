#!/usr/bin/env python3
"""Load a Hugging Face model like the SVG visualizer and export operator graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from visualizer.basic_ops import DEFAULT_BASIC_OP_PATTERNS
from visualizer.extract import dump_model_ast
from visualizer.loader import build_detailed_basic_ops, load_model_spec, resolve_checkpoint_arg
from visualizer.shape_inference import build_operator_export, save_operator_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export-operator-graph",
        description=(
            "TraceLens operator export — load a Hugging Face model id or local checkpoint "
            "for config.json, resolve modeling_*.py from the hub (same as the SVG "
            "visualizer), infer symbolic tensor shapes/dtypes, and write a flat operator "
            "list. Composite GPU kernels remain opaque leaf ops; all other modules expand "
            "via parsed forward() (with __init__ fallback when forward is missing)."
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
        "--github",
        "-g",
        help=(
            "Optional GitHub repo URL or github:owner/repo@ref:path for modeling source "
            "when the HF repo does not ship modeling_*.py"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path (default: <model>_operators.json in cwd)",
    )
    parser.add_argument(
        "--title",
        help="Architecture display name override",
    )
    parser.add_argument(
        "--config-path",
        help="Explicit config.json path inside the checkpoint (e.g. FL2VA/text_encoder/config.json)",
    )
    parser.add_argument(
        "--code-path",
        type=Path,
        help="Explicit path to modeling_*.py when auto-discovery is insufficient",
    )
    parser.add_argument(
        "--dump-ast",
        type=Path,
        help="Write the parsed Python AST dump for the modeling file",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Ignored — operator export always requires modeling source AST inspection",
    )
    parser.add_argument(
        "--basic-op-add",
        action="append",
        default=[],
        metavar="REGEX",
        help=(
            "Regex for block names treated as basic leaf operations "
            f"(repeatable; defaults include {', '.join(DEFAULT_BASIC_OP_PATTERNS)})"
        ),
    )
    parser.add_argument(
        "--basic-op-remove",
        action="append",
        default=[],
        metavar="REGEX",
        help="Remove a default basic-op regex (repeatable; pass exact pattern text)",
    )
    parser.add_argument(
        "--no-model-output",
        action="store_true",
        help="Do not append the terminal output/logits operator",
    )
    return parser


def default_output_path(checkpoint: str | Path | None, github: str | None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.exists():
            stem = path.name if path.is_dir() else path.stem
        else:
            stem = str(checkpoint).split("/")[-1]
    elif github:
        stem = github.rstrip("/").split("/")[-1].replace(".git", "")
    else:
        stem = "architecture"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return Path.cwd() / f"{safe}_operators.json"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint = resolve_checkpoint_arg(checkpoint=args.checkpoint, source=args.source)
    if checkpoint is None:
        parser.error(
            "Provide a Hugging Face model id or local checkpoint directory "
            "(SOURCE or --checkpoint). Config and modeling source are resolved from the hub "
            "the same way as tracelens-visualizer --detailed."
        )

    if args.config_only:
        print(
            "Warning: --config-only is ignored; operator export requires modeling AST inspection",
            file=sys.stderr,
        )

    basic_ops = build_detailed_basic_ops(add=args.basic_op_add, remove=args.basic_op_remove)

    try:
        if args.dump_ast:
            ast_dump = dump_model_ast(
                checkpoint,
                github=args.github,
                config_path=args.config_path,
                code_path=args.code_path,
            )
            args.dump_ast.parent.mkdir(parents=True, exist_ok=True)
            args.dump_ast.write_text(ast_dump + "\n", encoding="utf-8")
            print(f"Wrote AST dump: {args.dump_ast}")

        spec = load_model_spec(
            checkpoint,
            github=args.github,
            config_path=args.config_path,
            code_path=args.code_path,
            name=args.title,
            analyze_code=True,
            detailed=True,
            basic_ops=basic_ops,
            require_code=True,
        )
    except Exception as exc:  # noqa: BLE001 - clear CLI errors
        print(f"Error loading model: {exc}", file=sys.stderr)
        return 1

    payload = build_operator_export(spec, include_model_output=not args.no_model_output)
    output = args.output or default_output_path(checkpoint, args.github)
    save_operator_export(payload, output)
    operator_count = sum(len(section["operators"]) for section in payload["sections"])
    sources = ", ".join(payload.get("code_sources") or spec.code_sources[:3])
    print(
        f"Wrote {operator_count} operators across {len(payload['sections'])} sections "
        f"from {sources}: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
