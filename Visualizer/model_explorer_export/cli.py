#!/usr/bin/env python3
"""CLI for exporting TraceLens computation graphs to Model Explorer."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from urllib.parse import quote

from visualizer.basic_ops import DEFAULT_BASIC_OP_PATTERNS
from visualizer.extract import dump_model_ast
from visualizer.loader import build_detailed_basic_ops, load_model_spec, resolve_checkpoint_arg

from model_explorer_export.build import build_model_explorer_payload, save_model_explorer_payload
from model_explorer_export.serve import open_viewer, serve_viewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export-model-explorer",
        description=(
            "TraceLens Model Explorer export — load a Hugging Face model or local "
            "checkpoint, build computation graphs from parsed modeling code, and write "
            "JSON for the ai-edge-model-explorer-visualizer web component."
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
        help="Output JSON path (default: <model>_model_explorer.json in cwd)",
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
        help="Start a local HTTP server for the bundled Model Explorer viewer",
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
        help="Open the viewer in a browser after export (implies --serve unless JSON already exists)",
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
    return Path.cwd() / f"{safe}_model_explorer.json"


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
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading architecture: {exc}", file=sys.stderr)
        return 1

    output = args.output or default_output_path(checkpoint, args.github)
    try:
        payload = build_model_explorer_payload(spec, basic_ops=basic_ops)
        if not payload["graphCollections"][0]["graphs"]:
            raise ValueError("No computation graphs were built from the modeling source.")
        saved = save_model_explorer_payload(payload, output)
    except Exception as exc:  # noqa: BLE001
        print(f"Error exporting Model Explorer JSON: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote Model Explorer JSON: {saved}")

    if args.serve or args.open:
        try:
            if args.serve:
                url = f"http://127.0.0.1:{args.port}/index.html?graph={quote(saved.name)}"
                print(f"Serving viewer at {url}")
                if args.open:
                    open_viewer(url)
                serve_viewer(json_path=saved, port=args.port, block=True)
            else:
                url = serve_viewer(json_path=saved, port=args.port, block=False)
                print(f"Serving viewer at {url}")
                if args.open:
                    open_viewer(url)
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
