#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI for exporting TraceLens model graphs to Model Explorer."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from TraceLens.Visualizer.model_explorer_export.build import (
    build_model_explorer_payload,
    save_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.serve import open_viewer, serve_viewer, viewer_url
from TraceLens.Visualizer.model_explorer_export.viewer_page import is_html_output, save_viewer_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualize-model-in-explorer",
        description=(
            "TraceLens Model Explorer export — load a Hugging Face model checkpoint, "
            "trace it with PyTorch on the meta device, and serve or write a standalone "
            "viewer page."
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
        "-o",
        "--output",
        nargs="?",
        const="__default__",
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Write a standalone .html viewer page (default: <model>.html) "
            "or an explicit .html / .json path."
        ),
    )
    parser.add_argument(
        "--title",
        help="Architecture display name override",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for meta-device tracing (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for meta-device tracing (default: 1)",
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
    return parser


def _resolve_checkpoint(checkpoint: str | None, source: str | None) -> str | None:
    """Resolve --checkpoint / positional source to a single value."""
    if checkpoint and source:
        return checkpoint
    return checkpoint or source


def default_html_output_path(checkpoint: str | Path | None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.exists():
            stem = path.name if path.is_dir() else path.stem
        else:
            stem = str(checkpoint).replace("/", "_")
    else:
        stem = "architecture"
    return Path.cwd() / (stem + ".html")


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

    checkpoint = _resolve_checkpoint(
        checkpoint=args.checkpoint, source=args.source
    )
    if checkpoint is None:
        parser.error("Provide a Hugging Face checkpoint (SOURCE or --checkpoint)")

    try:
        payload = build_model_explorer_payload(
            checkpoint,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            title=args.title,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error building model graph: {exc}", file=sys.stderr)
        return 1

    serve_requested = args.serve or args.open

    if args.output is not None:
        output = (
            default_html_output_path(checkpoint)
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
            output = default_html_output_path(checkpoint)
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
