#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI for exporting TraceLens model graphs to Model Explorer.

Two backends are available via ``--backend``:

- ``torch`` (default): loads the checkpoint and traces it with PyTorch on
  the meta device (``TraceLens.ModelUtils.torch_trace``).
- ``ast``: parses the model's ``modeling_*.py`` source with Python's
  ``ast`` module instead of executing any model code
  (``TraceLens.ModelUtils.extract`` / ``computation_graph`` / ...). Kept as
  a switchable fallback, e.g. for checkpoints that cannot be instantiated
  or traced.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from TraceLens.ModelUtils.basic_ops import DEFAULT_BASIC_OP_PATTERNS
from TraceLens.ModelUtils.extract import ArchitectureSpec, dump_model_ast
from TraceLens.ModelUtils.loader import (
    build_detailed_basic_ops,
    load_model_spec,
    resolve_checkpoint_arg,
)
from TraceLens.ModelUtils.shape_inference import save_operator_export

from TraceLens.Visualizer.model_explorer_export.ast_build import (
    build_model_explorer_payload as build_ast_model_explorer_payload,
    build_operator_export_payload,
)
from TraceLens.Visualizer.model_explorer_export.build import (
    build_model_explorer_payload as build_torch_model_explorer_payload,
    save_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.serve import (
    open_viewer,
    serve_viewer,
    viewer_url,
)
from TraceLens.Visualizer.model_explorer_export.viewer_page import (
    is_html_output,
    save_viewer_html,
)

BACKENDS = ("torch", "ast")
DEFAULT_BACKEND = "torch"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualize-model-in-explorer",
        description=(
            "TraceLens Model Explorer export — load a Hugging Face model checkpoint "
            "and build a Model Explorer graph, then serve or write a standalone "
            "viewer page. Use --backend to pick how the graph is built: 'torch' "
            "(default) traces the model with PyTorch on the meta device; 'ast' "
            "statically parses the modeling source instead of running any code."
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
        "--backend",
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help=(
            "Graph-building backend: 'torch' traces the model with PyTorch on the "
            f"meta device (default), 'ast' statically parses the modeling source "
            "without executing any code (backup path)"
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
            "Write a standalone .html viewer page (default: <model>.html) "
            "or an explicit .html / .json path."
        ),
    )
    parser.add_argument(
        "--title",
        help="Architecture display name override",
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

    torch_group = parser.add_argument_group(
        "torch backend", "Options used when --backend torch (the default)"
    )
    torch_group.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for meta-device tracing (default: 128)",
    )
    torch_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for meta-device tracing (default: 1)",
    )

    ast_group = parser.add_argument_group(
        "ast backend", "Options used when --backend ast"
    )
    ast_group.add_argument(
        "--github",
        "-g",
        help=(
            "Optional GitHub repo URL or github:owner/repo@ref:path for modeling source "
            "when the HF repo does not ship modeling_*.py (repo must be whitelisted)"
        ),
    )
    ast_group.add_argument(
        "--allow-repo",
        action="append",
        default=[],
        metavar="OWNER/REPO",
        help=(
            "Whitelist an extra GitHub repository for remote source introspection "
            "(repeatable; huggingface/transformers is allowed by default)"
        ),
    )
    ast_group.add_argument(
        "--config-path",
        help="Explicit config.json path inside the checkpoint",
    )
    ast_group.add_argument(
        "--code-path",
        type=Path,
        help="Explicit path to modeling_*.py when auto-discovery is insufficient",
    )
    ast_group.add_argument(
        "--dump-ast",
        type=Path,
        help="Write the parsed Python AST dump for the modeling file",
    )
    ast_group.add_argument(
        "--basic-op-add",
        action="append",
        default=[],
        metavar="REGEX",
        help=(
            "Regex for block names treated as basic leaf operations "
            f"(repeatable; defaults: {', '.join(DEFAULT_BASIC_OP_PATTERNS)})"
        ),
    )
    ast_group.add_argument(
        "--basic-op-remove",
        action="append",
        default=[],
        metavar="REGEX",
        help="Remove a default basic-op regex (repeatable; pass exact pattern text)",
    )
    ast_group.add_argument(
        "--all-tensor-ops",
        action="store_true",
        help="Include tensor housekeeping operations in detailed computation graphs",
    )
    ast_group.add_argument(
        "--no-inline-expansion",
        dest="inline_expansion",
        action="store_false",
        help="Keep composite modules as opaque tiles instead of expanding into internal steps",
    )
    parser.set_defaults(inline_expansion=True)
    ast_group.add_argument(
        "--no-shapes",
        dest="shapes",
        action="store_false",
        help="Skip symbolic output_shape/output_dtype annotations on graph nodes",
    )
    parser.set_defaults(shapes=True)
    ast_group.add_argument(
        "--operators-json",
        type=Path,
        metavar="PATH",
        help="Also write flat operator export JSON with inferred tensor shapes",
    )
    ast_group.add_argument(
        "--meta-shapes",
        action="store_true",
        help=(
            "Run a meta-device forward pass (requires torch + transformers) "
            "to capture ground-truth output shapes for nn.Module layers"
        ),
    )
    return parser


def model_output_stem(checkpoint: str | Path | None, github: str | None) -> str:
    if checkpoint is not None:
        return str(checkpoint)
    if github:
        return github.rstrip("/").replace(".git", "")
    return "architecture"


def default_html_output_path(
    checkpoint: str | Path | None, github: str | None = None
) -> Path:
    stem = model_output_stem(checkpoint, github)
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.exists():
            stem = path.name if path.is_dir() else path.stem
    elif github:
        stem = stem.rstrip("/").split("/")[-1]
    return Path.cwd() / (stem.replace("/", "_") + ".html")


def write_optional_output(payload: dict, output: Path) -> Path:
    if is_html_output(output):
        saved = save_viewer_html(payload, output)
        print(f"Wrote standalone viewer: {saved}")
        return saved
    saved = save_model_explorer_payload(payload, output)
    print(f"Wrote Model Explorer JSON: {saved}")
    return saved


def _build_torch_payload(checkpoint: str, args: argparse.Namespace) -> dict:
    return build_torch_model_explorer_payload(
        checkpoint,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        title=args.title,
    )


def _load_ast_spec(
    checkpoint: str | None, args: argparse.Namespace
) -> ArchitectureSpec:
    basic_ops = build_detailed_basic_ops(
        add=args.basic_op_add, remove=args.basic_op_remove
    )

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

    return load_model_spec(
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


def _build_ast_payload(
    checkpoint: str | None, spec: ArchitectureSpec, args: argparse.Namespace
) -> dict:
    basic_ops = build_detailed_basic_ops(
        add=args.basic_op_add, remove=args.basic_op_remove
    )
    payload = build_ast_model_explorer_payload(
        spec,
        basic_ops=basic_ops,
        include_shapes=args.shapes,
        include_operator_export=args.operators_json is not None,
        inline_expansion=args.inline_expansion,
        meta_shapes_checkpoint=(str(checkpoint) if args.meta_shapes else None),
    )
    if not payload["graphCollections"][0]["graphs"]:
        raise ValueError("No computation graphs were built from the modeling source.")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint = resolve_checkpoint_arg(checkpoint=args.checkpoint, source=args.source)

    if args.backend == "torch":
        if checkpoint is None:
            parser.error("Provide a Hugging Face checkpoint (SOURCE or --checkpoint)")
        try:
            payload = _build_torch_payload(checkpoint, args)
        except Exception as exc:  # noqa: BLE001
            print(f"Error building model graph: {exc}", file=sys.stderr)
            return 1
        spec = None
    else:
        if checkpoint is None and args.github is None:
            parser.error(
                "Provide a Hugging Face checkpoint (SOURCE or --checkpoint) and/or --github"
            )
        try:
            spec = _load_ast_spec(checkpoint, args)
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading architecture: {exc}", file=sys.stderr)
            return 1

        try:
            payload = _build_ast_payload(checkpoint, spec, args)
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
            default_html_output_path(checkpoint, getattr(args, "github", None))
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
            output = default_html_output_path(checkpoint, getattr(args, "github", None))
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
