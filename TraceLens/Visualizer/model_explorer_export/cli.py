#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CLI for exporting TraceLens model graphs to Model Explorer.

The graph is built by statically parsing the model's ``modeling_*.py`` source
with Python's ``ast`` module instead of executing any model code
(``TraceLens.ModelUtils.extract`` / ``computation_graph`` / ...). A PyTorch
meta-device pass (``--meta-shapes``, on by default) is used only to fill in
tensor shapes the static analysis can't infer.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from TraceLens.ModelUtils.basic_ops import DEFAULT_BASIC_OP_PATTERNS
from TraceLens.ModelUtils.parallel_plan import (
    ParallelDegree,
    build_parallel_layout,
    gpu_group_report,
    scope_payload_to_gpu_group,
    unique_gpu_groups,
)
from TraceLens.ModelUtils.ast_analyze import RuntimeRegime
from TraceLens.ModelUtils.extract import (
    ArchitectureSpec,
    dump_model_ast,
    parse_config_overrides,
)
from TraceLens.ModelUtils.loader import (
    build_detailed_basic_ops,
    load_model_spec,
    resolve_checkpoint_arg,
)
from TraceLens.ModelUtils.shape_inference import save_operator_export
from TraceLens.ModelUtils.torch_introspect import InputSpec

from TraceLens.Visualizer.model_explorer_export.ast_build import (
    build_model_explorer_payload as build_ast_model_explorer_payload,
    build_operator_export_payload,
)
from TraceLens.Visualizer.model_explorer_export.build import (
    save_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.shapes import (
    canonicalize_payload_shapes,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualize-model-in-explorer",
        description=(
            "TraceLens Model Explorer export — load a Hugging Face model checkpoint "
            "and build a Model Explorer graph, then serve or write a standalone "
            "viewer page. The graph is built by statically parsing the modeling "
            "source; a PyTorch meta-device pass (--meta-shapes, on by default) "
            "fills in shapes the static analysis can't infer."
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
        "--from-payload",
        type=Path,
        metavar="PATH",
        help=(
            "Render a standalone .html viewer from an existing Model Explorer "
            "payload JSON (stage 2 of model -> JSON -> HTML); skips loading the "
            "model. The viewer 'Generated' timestamp is taken from the JSON "
            "file's creation date."
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
    parser.add_argument(
        "--drop-constants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop constant/learned-weight/buffer nodes when rendering HTML "
            "(default: on). Use --no-drop-constants to render them. The exported "
            "JSON always keeps constants regardless of this flag."
        ),
    )

    parser.add_argument(
        "--auto-venv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Re-run against the transformers release named in the checkpoint "
            "when that release is newer than the one installed here, and also "
            "when it is older and this interpreter cannot import the model "
            "(default: on). An unpublished newer pin uses the newest PyPI "
            "release after it, or the transformers git HEAD when none is "
            "newer. The environment is cached and reused. "
            "Use --no-auto-venv to always stay in the current interpreter."
        ),
    )

    partition_group = parser.add_argument_group(
        "device partitions",
        "How many ways to split the model. The split itself comes from the "
        "plans on its Hugging Face config (base_model_tp_plan, "
        "base_model_pp_plan, base_model_ep_plan, plus the causal-LM head), or, "
        "with --engine, from that inference engine's own parallel-layer "
        "classes. 1 leaves that axis unsplit.",
    )
    partition_group.add_argument(
        "--engine",
        choices=("vllm", "sglang"),
        help=(
            "Draw this checkpoint's architecture from vLLM's or SGLang's own "
            "pinned modeling source (ColumnParallelLinear, make_layers, FusedMoE, "
            "...) instead of transformers', and read --tp/--pp/--ep from that "
            "engine's sharding idioms instead of a declared base_model_tp_plan. "
            "Ignored once --code-path / --github already name the source."
        ),
    )
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

    regime_group = parser.add_argument_group(
        "runtime regime",
        "Which RUN the drawing depicts. A forward branches on things no config "
        "decides and only a run does, and both arms are real: drawing both "
        "leaves every shape below the branch true of only one of them. Each "
        "option picks exactly one arm, so the arm not taken is never drawn.",
    )
    regime_group.add_argument(
        "--sequence-layout",
        choices=list(RuntimeRegime.SEQUENCE_LAYOUTS),
        default=RuntimeRegime.SEQUENCE_LAYOUTS[0],
        help=(
            "rectangular keeps [B, S, ...]; packed removes the padding first "
            "and carries every real token as one row, [1, nnz, ...] "
            "(default: rectangular)."
        ),
    )
    regime_group.add_argument(
        "--phase",
        choices=list(RuntimeRegime.PHASES),
        default=RuntimeRegime.PHASES[0],
        help=(
            "prefill scores a whole prompt; decode advances one token, which "
            "is what makes the query length 1 and can select different kernels "
            "entirely (default: prefill)."
        ),
    )
    regime_group.add_argument(
        "--cache",
        choices=list(RuntimeRegime.CACHES),
        default=RuntimeRegime.CACHES[0],
        help=(
            "cold is the first forward, with nothing stored yet; warm "
            "continues from a cache that already holds keys and values "
            "(default: cold). decode always reads a cache."
        ),
    )
    regime_group.add_argument(
        "--attention",
        default=None,
        help=(
            "Attention implementation to draw (eager, sdpa, "
            "flash_attention_2, ...). eager draws the real matmul/softmax "
            "chain where sdpa draws one fused kernel. Unset takes the "
            "checkpoint's own setting, which is sdpa unless it says otherwise."
        ),
    )
    regime_group.add_argument(
        "--mode",
        choices=list(RuntimeRegime.MODES),
        default=RuntimeRegime.MODES[0],
        help="Whether `self.training` reads true (default: inference).",
    )
    regime_group.add_argument(
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
    regime_group.add_argument(
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

    ast_group = parser.add_argument_group("model source options")
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
        dest="meta_shapes",
        action="store_true",
        default=True,
        help=(
            "Run a meta-device forward pass (requires torch + transformers) to "
            "capture ground-truth shapes for nn.Module layers AND for inline "
            "operations the static analysis can't resolve (default: on)"
        ),
    )
    ast_group.add_argument(
        "--no-meta-shapes",
        dest="meta_shapes",
        action="store_false",
        help="Disable torch-assisted shape inference (pure static analysis)",
    )

    torch_group = parser.add_argument_group("pure-PyTorch module options")
    torch_group.add_argument(
        "--torch-module",
        metavar="PKG.MOD:FACTORY",
        help=(
            "Introspect a plain torch.nn.Module instead of an HF checkpoint. "
            "Give an importable 'package.module:callable' that returns an "
            "nn.Module (a class or zero-arg factory). Requires torch and "
            "exactly one of --input-shape (torch.fx) or --input-spec (torch.export)."
        ),
    )
    torch_group.add_argument(
        "--input-shape",
        metavar="N,...",
        help="Comma-separated primary input tensor shape for --torch-module (e.g. 2,128)",
    )
    torch_group.add_argument(
        "--input-spec",
        action="append",
        default=[],
        metavar="NAME=DIM,...[:DTYPE]",
        help=(
            "A named forward input for --torch-module, traced with torch.export "
            "(repeatable). Each DIM is an int or a symbolic name kept symbolic "
            "in every shape; a name shared across inputs is one dimension. "
            "DTYPE defaults to the module's float dtype "
            "(e.g. --input-spec input_ids=B,S:int64)."
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


def file_created_at(path: Path) -> datetime:
    """Best-effort file creation time (birth time when available, else mtime)."""
    stat = Path(path).stat()
    timestamp = getattr(stat, "st_birthtime", None) or stat.st_mtime
    return datetime.fromtimestamp(timestamp)


def subgraph_output_path(output: Path, index: int, count: int) -> Path:
    """``model_subgraph_0.html`` when more than one group is written."""
    if count <= 1:
        return output
    return output.with_name(f"{output.stem}_subgraph_{index}{output.suffix}")


def prepare_gpu_group_payloads(
    spec: Any, payload: dict
) -> list[tuple[Any, dict]] | None:
    """One payload per unique GPU group. ``None`` when the model is not split.

    Prints the GPU-to-subgraph map. A single group keeps the caller's payload
    object; several groups are copies so each file can name its own devices.
    """
    layout = getattr(spec, "parallel_layout", None)
    if layout is None or not layout.degree.requested():
        return None
    groups = unique_gpu_groups(layout, spec)
    if not groups:
        return None
    for line in gpu_group_report(groups, layout.degree):
        print(line)
    prepared: list[tuple[Any, dict]] = []
    for group in groups:
        layout.gpu_group = group
        stamped = payload if len(groups) == 1 else copy.deepcopy(payload)
        scope_payload_to_gpu_group(stamped, layout)
        _refresh_displayed_shapes(stamped)
        _refresh_fact_sheet(spec, stamped)
        prepared.append((group, stamped))
    return prepared


def _refresh_displayed_shapes(payload: dict) -> None:
    """Drawn group shapes follow the per-GPU node shapes just written."""
    from TraceLens.Visualizer.model_explorer_export.shapes import (
        refresh_displayed_group_shapes,
    )

    for collection in payload.get("graphCollections") or []:
        if not isinstance(collection, dict):
            continue
        for graph in collection.get("graphs") or []:
            if isinstance(graph, dict):
                refresh_displayed_group_shapes(graph)


def _refresh_fact_sheet(spec: Any, payload: dict) -> None:
    from TraceLens.Visualizer.model_explorer_export.fact_sheet import (
        build_fact_sheet_viewer,
    )

    viewer = build_fact_sheet_viewer(spec)
    viewer_meta = payload.setdefault("tracelensViewer", {})
    if isinstance(viewer_meta, dict):
        viewer_meta["factSheet"] = viewer
    for collection in payload.get("graphCollections") or []:
        if not isinstance(collection, dict):
            continue
        for graph in collection.get("graphs") or []:
            if not isinstance(graph, dict):
                continue
            attrs = graph.get("groupNodeAttributes")
            if not isinstance(attrs, dict):
                continue
            root = attrs.get("")
            if isinstance(root, dict):
                root["architecture_fact_sheet"] = viewer["body"]


def write_optional_output(
    payload: dict,
    output: Path,
    *,
    generated_at: datetime | None = None,
    drop_constants: bool = True,
) -> Path:
    if is_html_output(output):
        saved = save_viewer_html(
            payload, output, generated_at=generated_at, drop_constants=drop_constants
        )
        print(f"Wrote standalone viewer: {saved}")
        return saved
    # The JSON artifact always keeps constants -- the flag only affects HTML.
    saved = save_model_explorer_payload(payload, output)
    print(f"Wrote Model Explorer JSON: {saved}")
    return saved


def _run_from_payload(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Render a viewer HTML from an existing Model Explorer payload JSON."""
    source = args.from_payload
    if not source.exists():
        print(f"Error: payload not found: {source}", file=sys.stderr)
        return 1
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        canonicalize_payload_shapes(payload)
    except (OSError, ValueError) as exc:
        print(f"Error reading payload {source}: {exc}", file=sys.stderr)
        return 1

    generated_at = file_created_at(source)

    if args.output is not None and args.output != Path("__default__"):
        output = args.output
    else:
        output = source.with_suffix(".html")

    serve_requested = args.serve or args.open
    if not serve_requested or args.output is not None:
        try:
            write_optional_output(
                payload,
                output,
                generated_at=generated_at,
                drop_constants=args.drop_constants,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error writing output: {exc}", file=sys.stderr)
            return 1

    if serve_requested:
        url = viewer_url(args.port)
        print(f"Open viewer: {url}")
        try:
            if args.open:
                open_viewer(url)
            serve_viewer(
                payload=payload,
                port=args.port,
                block=args.serve,
                drop_constants=args.drop_constants,
            )
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
        regime=RuntimeRegime(
            sequence_layout=args.sequence_layout,
            phase=args.phase,
            cache=args.cache,
            attention=args.attention,
            mode=args.mode,
            device=args.device,
        ),
        config_overrides=parse_config_overrides(args.config_override),
        engine=args.engine,
    )


def _build_ast_payload(
    checkpoint: str | None, spec: ArchitectureSpec, args: argparse.Namespace
) -> dict:
    basic_ops = build_detailed_basic_ops(
        add=args.basic_op_add, remove=args.basic_op_remove
    )
    degree = ParallelDegree(tp=args.tp, pp=args.pp, ep=args.ep)
    if degree.requested():
        spec.parallel_layout = build_parallel_layout(spec, degree)
        for line in spec.parallel_layout.fact_lines():
            text = line.strip()
            if (
                "does not divide" in text
                or text.startswith("The model declares")
                or text.startswith("No Hugging Face")
            ):
                print(f"Warning: {text}", file=sys.stderr)

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


def parse_input_shape(text: str) -> tuple[int, ...]:
    """Parse a ``"2,128"`` style ``--input-shape`` into a tuple of ints."""
    dims = [part.strip() for part in str(text).split(",") if part.strip()]
    if not dims:
        raise ValueError("--input-shape must list at least one dimension, e.g. 2,128")
    try:
        return tuple(int(dim) for dim in dims)
    except ValueError as exc:  # noqa: TRY003
        raise ValueError(
            f"--input-shape must be comma-separated integers: {text!r}"
        ) from exc


def parse_input_spec(text: str) -> InputSpec:
    """Parse a ``"W=B,S:int64"`` style ``--input-spec`` into an :class:`InputSpec`."""
    name, sep, rest = str(text).partition("=")
    dims_text, _, dtype = rest.partition(":")
    parts = [part.strip() for part in dims_text.split(",")]
    if not sep or not name.strip().isidentifier() or not all(parts):
        raise ValueError(
            f"--input-spec must be NAME=DIM,...[:DTYPE], e.g. input_ids=B,S:int64; "
            f"got {text!r}"
        )
    dims: list[int | str] = []
    for part in parts:
        if part.isdigit():
            dims.append(int(part))
        elif part.isidentifier():
            dims.append(part)
        else:
            raise ValueError(
                f"--input-spec dim {part!r} is neither an int nor a name: {text!r}"
            )
    return InputSpec(name.strip(), tuple(dims), dtype.strip() or None)


def _load_torch_module(target: str):  # pragma: no cover
    """Import ``package.module:callable`` and instantiate an ``nn.Module``."""
    import importlib

    module_path, _, attr = target.partition(":")
    if not module_path or not attr:
        raise ValueError(
            f"--torch-module must be 'package.module:callable', got {target!r}"
        )
    module = importlib.import_module(module_path)
    factory = getattr(module, attr)
    instance = factory() if callable(factory) else factory
    return instance


def _build_torch_module_payload(args: argparse.Namespace) -> dict:  # pragma: no cover
    from TraceLens.Visualizer.model_explorer_export.torch_payload import (
        build_exported_module_payload,
        build_torch_module_payload,
    )

    module = _load_torch_module(args.torch_module)
    name = args.title or type(module).__name__
    if args.input_spec:
        input_specs = [parse_input_spec(text) for text in args.input_spec]
        return build_exported_module_payload(
            module, input_specs, include_shapes=args.shapes, name=name
        )
    input_shape = parse_input_shape(args.input_shape)
    return build_torch_module_payload(
        module, input_shape, include_shapes=args.shapes, name=name
    )


def _run_torch_module(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:  # pragma: no cover
    try:
        payload = _build_torch_module_payload(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error introspecting torch module: {exc}", file=sys.stderr)
        return 1

    serve_requested = args.serve or args.open
    module_stem = args.torch_module.replace(":", "_").rsplit(".", 1)[-1]

    if args.output is not None or not serve_requested:
        output = args.output
        if output is None or output == Path("__default__"):
            output = Path.cwd() / (module_stem + ".html")
        try:
            write_optional_output(
                payload,
                output,
                generated_at=datetime.now(),
                drop_constants=args.drop_constants,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error writing output: {exc}", file=sys.stderr)
            return 1

    if serve_requested:
        url = viewer_url(args.port)
        print(f"Open viewer: {url}")
        try:
            if args.open:
                open_viewer(url)
            serve_viewer(
                payload=payload,
                port=args.port,
                block=args.serve,
                drop_constants=args.drop_constants,
            )
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.from_payload is not None:
        return _run_from_payload(args, parser)

    if args.torch_module is not None:
        if (args.input_shape is None) == (not args.input_spec):
            parser.error(
                "--torch-module requires exactly one of --input-shape or --input-spec"
            )
        return _run_torch_module(args, parser)

    for name in ("tp", "pp", "ep"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be a positive number of partitions")

    checkpoint = resolve_checkpoint_arg(checkpoint=args.checkpoint, source=args.source)

    if checkpoint is None and args.github is None:
        parser.error(
            "Provide a Hugging Face checkpoint (SOURCE or --checkpoint) and/or --github"
        )

    if args.auto_venv and checkpoint is not None:
        # Re-executes when the checkpoint names a newer transformers release,
        # or an older one that this interpreter cannot import. Returns None
        # (and we carry on in-process) when the installed release should stay.
        from TraceLens.ModelUtils.model_env import reexec_in_model_env

        status = reexec_in_model_env(
            checkpoint, list(sys.argv[1:] if argv is None else argv)
        )
        if status is not None:
            return status

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

    prepared = prepare_gpu_group_payloads(spec, payload)
    serve_payload = prepared[0][1] if prepared else payload
    serve_requested = args.serve or args.open
    write_files = (
        args.output is not None
        or not serve_requested
        or (prepared is not None and len(prepared) > 1)
    )
    if write_files:
        output = (
            default_html_output_path(checkpoint, getattr(args, "github", None))
            if args.output is None or args.output == Path("__default__")
            else args.output
        )
        outputs = prepared if prepared else [(None, payload)]
        count = len(prepared) if prepared else 1
        for group, stamped in outputs:
            path = (
                output
                if group is None
                else subgraph_output_path(output, group.index, count)
            )
            try:
                write_optional_output(
                    stamped,
                    path,
                    generated_at=datetime.now(),
                    drop_constants=args.drop_constants,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Error writing output: {exc}", file=sys.stderr)
                return 1

    if serve_requested:
        url = viewer_url(args.port)
        print(f"Open viewer: {url}")
        try:
            if args.open:
                open_viewer(url)
            serve_viewer(
                payload=serve_payload,
                port=args.port,
                block=args.serve,
                drop_constants=args.drop_constants,
            )
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
