#!/usr/bin/env python3
"""Generate LLM architecture diagrams from Hugging Face repos or local checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from visualizer.basic_ops import DEFAULT_BASIC_OP_PATTERNS, BasicOpFilter
from visualizer.extract import dump_model_ast, load_architecture, architecture_section_trees
from visualizer.loader import build_detailed_basic_ops, resolve_checkpoint_arg
from visualizer.model_graph import build_architecture_model_graphs, save_architecture_model_graphs
from visualizer.render import render_diagram, _fact_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracelens-visualizer",
        description=(
            "TraceLens Visualizer — generate Raschka-style LLM architecture diagrams. "
            "Load config from a Hugging Face checkpoint and inspect modeling code from "
            "GitHub, a local path, or the checkpoint repo (CPU-only, no weights)."
        ),
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Hugging Face checkpoint id or local checkpoint directory (alias for --checkpoint)",
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
            "GitHub repo URL or github:owner/repo@ref:path for modeling source "
            "(e.g. https://github.com/org/repo/tree/main/src)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (.svg, .png, .pdf). Defaults to <model>_architecture.svg, or <model>_architecture_detailed.svg with --detailed",
    )
    parser.add_argument(
        "--title",
        help="Diagram title (defaults to architecture class name from config)",
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
        "--config-only",
        action="store_true",
        help="Skip AST inspection and use config.json heuristics only",
    )
    parser.add_argument(
        "--dump-ast",
        type=Path,
        help="Write the parsed Python AST dump for the modeling file",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for raster formats (default: 150)",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        help="Also write parsed architecture metadata to JSON",
    )
    parser.add_argument(
        "--graph-json",
        dest="graph_json_out",
        type=Path,
        help=(
            "Write serializable model graph IR (nodes, edges, inline frames, subgraphs) "
            "to JSON. Requires --detailed and modeling source AST inspection."
        ),
    )
    parser.add_argument(
        "--facts",
        action="store_true",
        help="Print fact sheet to stdout",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help=(
            "Include recursive internal block diagrams below the main model "
            "(requires modeling source AST inspection)"
        ),
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
        "--no-inline-linear-frames",
        action="store_false",
        dest="inline_linear_frames",
        help="Disable dotted frames around straight-line sub-blocks inlined in parent diagrams",
    )
    parser.set_defaults(inline_linear_frames=True)
    return parser


def _resolve_checkpoint_arg(args: argparse.Namespace) -> str | Path | None:
    return resolve_checkpoint_arg(checkpoint=args.checkpoint, source=args.source)


def default_output_path(
    checkpoint: str | Path | None,
    github: str | None,
    *,
    detailed: bool = False,
) -> Path:
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
    suffix = "_architecture_detailed.svg" if detailed else "_architecture.svg"
    return Path.cwd() / f"{safe}{suffix}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint = _resolve_checkpoint_arg(args)
    if checkpoint is None and args.github is None:
        parser.error("Provide a Hugging Face checkpoint (SOURCE or --checkpoint) and/or --github")

    try:
        analyze_code = not args.config_only
        if args.detailed and args.config_only:
            print(
                "Warning: --detailed requires modeling source; ignoring --config-only",
                file=sys.stderr,
            )
            analyze_code = True

        extra_add = list(args.basic_op_add or [])
        if args.detailed:
            basic_ops = build_detailed_basic_ops(add=extra_add, remove=args.basic_op_remove)
        else:
            basic_ops = BasicOpFilter.from_cli(add=extra_add, remove=args.basic_op_remove)

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

        spec = load_architecture(
            checkpoint,
            name=args.title,
            github=args.github,
            config_path=args.config_path,
            code_path=args.code_path,
            analyze_code=analyze_code,
            detailed=args.detailed,
            basic_ops=basic_ops,
        )
    except Exception as exc:  # noqa: BLE001 - surface clear CLI errors
        print(f"Error loading architecture: {exc}", file=sys.stderr)
        return 1

    output = args.output or default_output_path(checkpoint, args.github, detailed=args.detailed)
    try:
        saved = render_diagram(
            spec,
            output,
            dpi=args.dpi,
            title=args.title,
            detailed=args.detailed,
            inline_linear_frames=args.inline_linear_frames,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error rendering diagram: {exc}", file=sys.stderr)
        return 1

    if args.graph_json_out:
        if not args.detailed:
            print(
                "Warning: --graph-json requires detailed block trees; re-run with --detailed",
                file=sys.stderr,
            )
        elif not architecture_section_trees(spec):
            print(
                "Warning: no block trees available; graph JSON not written",
                file=sys.stderr,
            )
        else:
            graph_payload = build_architecture_model_graphs(spec, basic_ops=basic_ops)
            save_architecture_model_graphs(graph_payload, args.graph_json_out)
            print(f"Wrote model graph: {args.graph_json_out}")

    if args.json_out:
        payload = {
            "name": spec.name,
            "model_type": spec.model_type,
            "architectures": spec.architectures,
            "decoder_type": spec.decoder_type,
            "attention_type": spec.attention_type,
            "positional_encoding": spec.positional_encoding,
            "norm_type": spec.norm_type,
            "norm_placement": spec.norm_placement,
            "ffn_type": spec.ffn_type,
            "hidden_size": spec.hidden_size,
            "num_hidden_layers": spec.num_hidden_layers,
            "num_attention_heads": spec.num_attention_heads,
            "num_key_value_heads": spec.num_key_value_heads,
            "num_experts": spec.num_experts,
            "num_experts_per_tok": spec.num_experts_per_tok,
            "layer_mix": spec.layer_mix,
            "kv_cache_per_token_bf16": spec.kv_cache_per_token_bf16,
            "total_params_hint": spec.total_params_hint,
            "active_params_hint": spec.active_params_hint,
            "highlights": spec.highlights,
            "source_path": spec.source_path,
            "checkpoint_source": spec.checkpoint_source,
            "github_source": spec.github_source,
            "decoder_class": spec.decoder_class,
            "forward_sequence": spec.forward_sequence,
            "code_sources": spec.code_sources,
            "analysis_notes": spec.analysis_notes,
            "custom_blocks": spec.custom_blocks,
            "block_components": [
                {
                    "attr_name": comp.attr_name,
                    "class_name": comp.class_name,
                    "role": comp.role,
                    "label": comp.label,
                    "forward_order": comp.forward_order,
                    "details": comp.details,
                }
                for comp in spec.block_components
            ],
            "detailed_block_trees": [
                {
                    "title": title,
                    "class_name": tree.class_name,
                    "attr_name": tree.attr_name,
                    "is_basic": tree.is_basic,
                    "children": len(tree.children),
                }
                for title, tree in architecture_section_trees(spec)
            ],
            "basic_op_patterns": basic_ops.pattern_strings(),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metadata: {args.json_out}")

    if args.facts:
        print(f"\n{spec.name}\n" + "=" * len(spec.name))
        for line in _fact_lines(spec):
            print(line)

    print(f"Wrote diagram: {saved}")
    return 0
