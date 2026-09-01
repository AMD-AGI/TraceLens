#!/usr/bin/env python3
"""CLI for exporting TraceLens computation graphs to Model Explorer."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from visualizer.basic_ops import DEFAULT_BASIC_OP_PATTERNS
from visualizer.extract import dump_model_ast
from visualizer.loader import build_detailed_basic_ops, load_model_spec, resolve_checkpoint_arg

from model_explorer_export.build import (
    build_model_explorer_payload,
    build_operator_export_payload,
    save_model_explorer_payload,
)
from model_explorer_export.serve import open_viewer, serve_viewer, viewer_url
from model_explorer_export.viewer_page import is_html_output, save_viewer_html
from visualizer.shape_inference import save_operator_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualize-model-in-explorer",
        description=(
            "TraceLens Model Explorer export — load a Hugging Face model or local "
            "checkpoint, build computation graphs from parsed modeling code, and "
            "serve or write a standalone viewer page."
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
            "when the HF repo does not ship modeling_*.py (repo must be whitelisted)"
        ),
    )
    parser.add_argument(
        "--allow-repo",
        action="append",
        default=[],
        metavar="OWNER/REPO",
        help=(
            "Whitelist an extra GitHub repository for remote source introspection "
            "(repeatable; huggingface/transformers is allowed by default)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="__default__",
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Write a standalone .html viewer page (default: <model_with_slashes_as_underscores>.html) "
            "or an explicit .html / .json path."
        ),
    )
    parser.add_argument(
        "--title",
        help="Architecture display name override",
    )
    parser.add_argument(
        "--config-path",
        help="Explicit config.json path inside the checkpoint",
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
        "--basic-op-add",
        action="append",
        default=[],
        metavar="REGEX",
        help=(
            "Regex for block names treated as basic leaf operations "
            f"(repeatable; defaults: {', '.join(DEFAULT_BASIC_OP_PATTERNS)})"
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
        "--all-tensor-ops",
        action="store_true",
        help="Include tensor housekeeping operations in detailed computation graphs",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local HTTP server with the graph embedded in the viewer page",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --serve (default: 8765)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the viewer in a browser after export (implies --serve)",
    )
    parser.add_argument(
        "--no-shapes",
        dest="shapes",
        action="store_false",
        help="Skip symbolic output_shape/output_dtype annotations on graph nodes",
    )
    parser.set_defaults(shapes=True)
    parser.add_argument(
        "--operators-json",
        type=Path,
        metavar="PATH",
        help="Also write flat operator export JSON with inferred tensor shapes",
    )
    return parser


def model_output_stem(checkpoint: str | Path | None, github: str | None) -> str:
    if checkpoint is not None:
        return str(checkpoint)
    if github:
        return github.rstrip("/").replace(".git", "")
    return "architecture"


def default_html_output_path(checkpoint: str | Path | None, github: str | None) -> Path:
    stem = model_output_stem(checkpoint, github)
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.exists():
            stem = path.name if path.is_dir() else path.stem
    elif github:
        stem = stem.rstrip("/").split("/")[-1]
    filename = stem.replace("/", "_") + ".html"
    return Path.cwd() / filename


def write_optional_output(payload: dict, output: Path) -> Path:
    if is_html_output(output):
        saved = save_viewer_html(payload, output)
        print(f"Wrote standalone viewer: {saved}")
        return saved
    saved = save_model_explorer_payload(payload, output)
    print(f"Wrote Model Explorer JSON: {saved}")
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint = resolve_checkpoint_arg(checkpoint=args.checkpoint, source=args.source)
    if checkpoint is None and args.github is None:
        parser.error("Provide a Hugging Face checkpoint (SOURCE or --checkpoint) and/or --github")

    basic_ops = build_detailed_basic_ops(add=args.basic_op_add, remove=args.basic_op_remove)

    if args.dump_ast:
        ast_dump = dump_model_ast(
            checkpoint,
            github=args.github,
            config_path=args.config_path,
            code_path=args.code_path,
            allow_github_repos=args.allow_repo,
        )
        args.dump_ast.parent.mkdir(parents=True, exist_ok=True)
        args.dump_ast.write_text(ast_dump + "\n", encoding="utf-8")
        print(f"Wrote AST dump: {args.dump_ast}")

    try:
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
            allow_github_repos=args.allow_repo,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading architecture: {exc}", file=sys.stderr)
        return 1

    try:
        payload = build_model_explorer_payload(
            spec,
            basic_ops=basic_ops,
            include_shapes=args.shapes,
            include_operator_export=args.operators_json is not None,
        )
        if not payload["graphCollections"][0]["graphs"]:
            raise ValueError("No computation graphs were built from the modeling source.")
    except Exception as exc:  # noqa: BLE001
        print(f"Error exporting Model Explorer payload: {exc}", file=sys.stderr)
        return 1

    if args.operators_json is not None:
        try:
            operator_payload = payload["tracelensViewer"].get("operatorExport")
            if operator_payload is None:
                operator_payload = build_operator_export_payload(spec)
            saved = save_operator_export(operator_payload, args.operators_json)
            print(f"Wrote operator export JSON: {saved}")
        except Exception as exc:  # noqa: BLE001
            print(f"Error writing operator export: {exc}", file=sys.stderr)
            return 1

    serve_requested = args.serve or args.open

    if args.output is not None:
        output = (
            default_html_output_path(checkpoint, args.github)
            if args.output == Path("__default__")
            else args.output
        )
        try:
            write_optional_output(payload, output)
        except Exception as exc:  # noqa: BLE001
            print(f"Error writing output: {exc}", file=sys.stderr)
            return 1
    elif not serve_requested:
        try:
            output = default_html_output_path(checkpoint, args.github)
            write_optional_output(payload, output)
        except Exception as exc:  # noqa: BLE001
            print(f"Error writing output: {exc}", file=sys.stderr)
            return 1

    if serve_requested:
        url = viewer_url(args.port)
        print(f"Open viewer: {url}")
        try:
            if args.open:
                open_viewer(url)
            serve_viewer(payload=payload, port=args.port, block=args.serve)
            if not args.serve:
                print("Viewer started in the background. Press Ctrl+C to exit.")
                try:
                    threading.Event().wait()
                except KeyboardInterrupt:
                    pass
        except Exception as exc:  # noqa: BLE001
            print(f"Error serving viewer: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
