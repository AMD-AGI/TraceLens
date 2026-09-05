###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Focused branch coverage for Model Explorer export and serving helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from TraceLens.Visualizer.model_explorer_export import adapter, build, cli, fact_sheet, serve, viewer_page
from TraceLens.ModelUtils.basic_ops import BasicOpFilter
from TraceLens.ModelUtils.block_tree import BlockNode
from TraceLens.ModelUtils.computation_graph import (
    ComputationGraph,
    GraphNodeSpec,
    InlineFrameSpec,
)
from TraceLens.ModelUtils.extract import ArchitectureSpec


def _spec(**overrides) -> ArchitectureSpec:
    values = {
        "name": "Coverage Model",
        "model_type": "coverage",
        "basic_ops": BasicOpFilter([r"Linear"]),
    }
    values.update(overrides)
    return ArchitectureSpec(**values)


def _payload(graphs=None) -> dict:
    return {
        "tracelensViewer": {},
        "graphCollections": [
            {"graphs": [{"id": "model"}] if graphs is None else graphs}
        ],
    }


def _patch_cli_success(monkeypatch: pytest.MonkeyPatch, payload=None) -> None:
    monkeypatch.setattr(cli, "resolve_checkpoint_arg", lambda **kwargs: "org/model")
    monkeypatch.setattr(
        cli,
        "build_detailed_basic_ops",
        lambda **kwargs: BasicOpFilter([r"Linear"]),
    )
    monkeypatch.setattr(cli, "load_model_spec", lambda *args, **kwargs: _spec())
    monkeypatch.setattr(
        cli,
        "build_model_explorer_payload",
        lambda *args, **kwargs: payload or _payload(),
    )


def test_cli_parser_and_output_name_helpers(tmp_path: Path, monkeypatch):
    args = cli.build_parser().parse_args(
        [
            "org/model",
            "--basic-op-add",
            "Add",
            "--basic-op-remove",
            "Remove",
            "--no-shapes",
            "--output",
        ]
    )
    assert args.output == Path("__default__")
    assert not args.shapes
    assert cli.model_output_stem(None, None) == "architecture"
    assert cli.model_output_stem(None, "https://github.com/o/r.git/") == (
        "https://github.com/o/r"
    )

    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_text("x", encoding="utf-8")
    assert cli.default_html_output_path(checkpoint, None).name == "weights.html"
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    assert cli.default_html_output_path(directory, None).name == "checkpoint.html"
    assert cli.default_html_output_path(None, "https://github.com/o/repo").name == (
        "repo.html"
    )


def test_write_optional_output_selects_html_or_json(monkeypatch, tmp_path, capsys):
    html = tmp_path / "viewer.HTML"
    json_path = tmp_path / "viewer.json"
    save_html = Mock(return_value=html.resolve())
    save_json = Mock(return_value=json_path.resolve())
    monkeypatch.setattr(cli, "save_viewer_html", save_html)
    monkeypatch.setattr(cli, "save_model_explorer_payload", save_json)

    assert cli.write_optional_output({"a": 1}, html) == html.resolve()
    assert cli.write_optional_output({"a": 1}, json_path) == json_path.resolve()
    assert "Wrote standalone viewer" in capsys.readouterr().out
    save_html.assert_called_once()
    save_json.assert_called_once()


def test_cli_requires_a_source():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_cli_dump_ast_operator_fallback_and_explicit_output(
    monkeypatch, tmp_path, capsys
):
    _patch_cli_success(monkeypatch)
    ast_path = tmp_path / "nested" / "ast.txt"
    output = tmp_path / "graph.json"
    operators = tmp_path / "operators.json"
    monkeypatch.setattr(cli, "dump_model_ast", lambda *args, **kwargs: "Module()")
    fallback = Mock(return_value={"operators": [1]})
    save_operators = Mock(return_value=operators.resolve())
    write_output = Mock(return_value=output.resolve())
    monkeypatch.setattr(cli, "build_operator_export_payload", fallback)
    monkeypatch.setattr(cli, "save_operator_export", save_operators)
    monkeypatch.setattr(cli, "write_optional_output", write_output)

    result = cli.main(
        [
            "org/model",
            "--dump-ast",
            str(ast_path),
            "--operators-json",
            str(operators),
            "-o",
            str(output),
        ]
    )

    assert result == 0
    assert ast_path.read_text(encoding="utf-8") == "Module()\n"
    fallback.assert_called_once()
    save_operators.assert_called_once_with({"operators": [1]}, operators)
    write_output.assert_called_once_with(_payload(), output)
    assert "Wrote AST dump" in capsys.readouterr().out


def test_cli_uses_embedded_operator_export_and_default_output(monkeypatch, tmp_path):
    payload = _payload()
    payload["tracelensViewer"]["operatorExport"] = {"embedded": True}
    _patch_cli_success(monkeypatch, payload)
    monkeypatch.chdir(tmp_path)
    save_operators = Mock(return_value=(tmp_path / "ops.json"))
    write_output = Mock()
    fallback = Mock()
    monkeypatch.setattr(cli, "save_operator_export", save_operators)
    monkeypatch.setattr(cli, "write_optional_output", write_output)
    monkeypatch.setattr(cli, "build_operator_export_payload", fallback)

    assert cli.main(["org/model", "--operators-json", "ops.json"]) == 0
    save_operators.assert_called_once_with({"embedded": True}, Path("ops.json"))
    fallback.assert_not_called()
    assert write_output.call_args.args[1] == tmp_path / "org_model.html"


@pytest.mark.parametrize(
    ("failing_name", "expected"),
    [
        ("load_model_spec", "Error loading architecture"),
        ("build_model_explorer_payload", "Error exporting Model Explorer payload"),
        ("save_operator_export", "Error writing operator export"),
        ("write_optional_output", "Error writing output"),
    ],
)
def test_cli_reports_pipeline_failures(monkeypatch, capsys, failing_name, expected):
    _patch_cli_success(monkeypatch)
    monkeypatch.setattr(
        cli,
        failing_name,
        Mock(side_effect=RuntimeError("deliberate failure")),
    )
    argv = ["org/model"]
    if failing_name == "save_operator_export":
        argv.extend(["--operators-json", "ops.json"])

    assert cli.main(argv) == 1
    assert expected in capsys.readouterr().err


def test_cli_reports_explicit_output_failure(monkeypatch, capsys):
    _patch_cli_success(monkeypatch)
    monkeypatch.setattr(
        cli, "write_optional_output", Mock(side_effect=OSError("read only"))
    )
    assert cli.main(["org/model", "--output", "graph.json"]) == 1
    assert "Error writing output: read only" in capsys.readouterr().err


def test_cli_rejects_empty_graphs(monkeypatch, capsys):
    _patch_cli_success(monkeypatch, _payload(graphs=[]))
    assert cli.main(["org/model"]) == 1
    assert "No computation graphs" in capsys.readouterr().err


def test_cli_serves_opens_and_handles_server_error(monkeypatch, capsys):
    _patch_cli_success(monkeypatch)
    monkeypatch.setattr(cli, "viewer_url", lambda port: f"url:{port}")
    opened = Mock()
    served = Mock()
    monkeypatch.setattr(cli, "open_viewer", opened)
    monkeypatch.setattr(cli, "serve_viewer", served)

    assert cli.main(["org/model", "--open", "--serve", "--port", "42"]) == 0
    opened.assert_called_once_with("url:42")
    served.assert_called_once_with(payload=_payload(), port=42, block=True)
    assert "Open viewer: url:42" in capsys.readouterr().out

    served.side_effect = RuntimeError("busy")
    assert cli.main(["org/model", "--serve"]) == 1
    assert "Error serving viewer: busy" in capsys.readouterr().err


def test_cli_open_only_background_wait_handles_interrupt(monkeypatch):
    _patch_cli_success(monkeypatch)
    monkeypatch.setattr(cli, "open_viewer", Mock())
    monkeypatch.setattr(cli, "serve_viewer", Mock())
    wait = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(cli.threading, "Event", lambda: SimpleNamespace(wait=wait))

    assert cli.main(["org/model", "--open"]) == 0
    wait.assert_called_once()


def test_load_payload_and_asset_copy(monkeypatch, tmp_path):
    payload = {"name": "direct"}
    assert serve._load_payload(payload=payload, json_path=None) is payload
    with pytest.raises(ValueError, match="payload or json_path"):
        serve._load_payload(payload=None, json_path=None)

    json_path = tmp_path / "payload.json"
    json_path.write_text('{"name": "disk"}', encoding="utf-8")
    assert serve._load_payload(payload=None, json_path=json_path) == {"name": "disk"}

    dist = tmp_path / "dist"
    viewer = tmp_path / "viewer"
    dist.mkdir()
    (dist / "worker.js").write_text("worker", encoding="utf-8")
    monkeypatch.setattr(serve, "VISUALIZER_DIST", dist)
    monkeypatch.setattr(serve, "VIEWER_DIR", viewer)
    serve.ensure_viewer_assets()
    assert (viewer / "worker.js").read_text(encoding="utf-8") == "worker"
    (dist / "worker.js").unlink()
    serve.ensure_viewer_assets()


def test_serve_viewer_handler_and_thread_lifecycle(monkeypatch, tmp_path):
    captured = {}

    class FakeServer:
        def __init__(self, address, handler):
            captured["address"] = address
            captured["handler"] = handler
            self.closed = False

        def serve_forever(self):
            captured["served"] = True

        def server_close(self):
            self.closed = True
            captured["closed"] = True

    class FakeThread:
        def __init__(self, *, target, daemon):
            captured["target"] = target
            captured["daemon"] = daemon

        def start(self):
            captured["target"]()

        def join(self):
            captured["joined"] = True

    monkeypatch.setattr(serve, "ensure_viewer_assets", Mock())
    monkeypatch.setattr(
        serve, "compose_viewer_html", lambda payload, **kwargs: "<page>"
    )
    monkeypatch.setattr(serve, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(serve.threading, "Thread", FakeThread)

    assert serve.serve_viewer(payload={"x": 1}, port=99, block=True) == (
        "http://127.0.0.1:99/"
    )
    assert captured["address"] == ("127.0.0.1", 99)
    assert captured["daemon"] is False
    assert captured["joined"] and captured["closed"]

    handler_type = captured["handler"]
    parent_init = Mock()
    monkeypatch.setattr(serve.SimpleHTTPRequestHandler, "__init__", parent_init)
    handler_type("request", ("client", 1), "server")
    parent_init.assert_called_once_with(
        "request", ("client", 1), "server", directory=str(serve.VIEWER_DIR)
    )

    handler = handler_type.__new__(handler_type)
    handler.path = "/index.html?fresh=1"
    handler.wfile = io.BytesIO()
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler.do_GET()
    handler.send_response.assert_called_once_with(200)
    assert handler.wfile.getvalue() == b"<page>"

    parent_get = Mock()
    monkeypatch.setattr(serve.SimpleHTTPRequestHandler, "do_GET", parent_get)
    handler.path = "/asset.js"
    handler.do_GET()
    parent_get.assert_called_once_with()

    parent_headers = Mock()
    monkeypatch.setattr(serve.SimpleHTTPRequestHandler, "end_headers", parent_headers)
    del handler.end_headers
    handler.send_header = Mock()
    handler.path = "/asset.json?x=1"
    handler.end_headers()
    handler.send_header.assert_called_once_with("Cache-Control", "no-cache")
    parent_headers.assert_called_once_with()
    handler.path = "/image.png"
    handler.send_header.reset_mock()
    handler.end_headers()
    handler.send_header.assert_not_called()
    assert handler.log_message("ignored") is None


def test_open_viewer_delegates_to_browser(monkeypatch):
    opened = Mock()
    monkeypatch.setattr(serve.webbrowser, "open", opened)
    serve.open_viewer("http://local/")
    opened.assert_called_once_with("http://local/")


def test_serve_main_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["serve", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit, match="File not found"):
        serve.main()


def test_serve_main_happy_path(monkeypatch, tmp_path):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["serve", str(payload_path), "--port", "81", "--open"]
    )
    opened = Mock()
    served = Mock()
    monkeypatch.setattr(serve, "open_viewer", opened)
    monkeypatch.setattr(serve, "serve_viewer", served)

    assert serve.main() == 0
    opened.assert_called_once_with("http://127.0.0.1:81/")
    served.assert_called_once_with(json_path=payload_path, port=81, block=True)


def test_build_payload_with_and_without_inference(monkeypatch):
    spec = _spec()
    exported_filter = {}
    graph = {"id": "merged", "nodes": []}

    class FakeInferencer:
        def __init__(self, received_spec):
            assert received_spec is spec
            self.context = SimpleNamespace(dims={"H": object()}, dtype="bf16")

        def export_architecture(self):
            return {"operators": ["linear"]}

    def fake_merge(received_spec, *, basic_ops, shape_inferencer, inline_expansion=True):
        assert received_spec is spec
        exported_filter["value"] = basic_ops
        exported_filter["inferencer"] = shape_inferencer
        return graph

    monkeypatch.setattr(build, "ShapeInferencer", FakeInferencer)
    monkeypatch.setattr(build, "serialize_dim", lambda value: "serialized")
    monkeypatch.setattr(build, "build_merged_model_graph", fake_merge)
    monkeypatch.setattr(
        build, "build_fact_sheet_viewer", lambda value: {"title": value.name}
    )

    payload = build.build_model_explorer_payload(
        spec,
        collection_label="Custom",
        include_shapes=False,
        include_operator_export=True,
    )
    assert payload["graphCollections"] == [{"label": "Custom", "graphs": [graph]}]
    assert payload["tracelensViewer"]["dimensions"] == {"H": "serialized"}
    assert payload["tracelensViewer"]["operatorExport"] == {"operators": ["linear"]}
    assert exported_filter["value"].patterns == spec.basic_ops.patterns
    assert exported_filter["value"].basic_only is False

    plain = build.build_model_explorer_payload(
        spec, include_shapes=False, include_operator_export=False
    )
    assert "dimensions" not in plain["tracelensViewer"]
    assert exported_filter["inferencer"] is None


def test_operator_export_and_json_save(monkeypatch, tmp_path):
    spec = _spec()
    operator_export = Mock(return_value={"flat": True})
    monkeypatch.setattr(build, "build_operator_export", operator_export)
    assert build.build_operator_export_payload(spec) == {"flat": True}

    target = build.save_model_explorer_payload(
        {"unicode": "→"}, tmp_path / "nested" / "payload.json"
    )
    assert target == (tmp_path / "nested" / "payload.json").resolve()
    assert json.loads(target.read_text(encoding="utf-8")) == {"unicode": "→"}
    assert target.read_text(encoding="utf-8").endswith("\n")


def _viewer_files(tmp_path: Path) -> tuple[Path, Path]:
    viewer = tmp_path / "viewer"
    dist = tmp_path / "dist"
    viewer.mkdir()
    dist.mkdir()
    (viewer / "index.html").write_text(
        '<html>\n    <script src="./app.js?v=7"></script>\n</html>',
        encoding="utf-8",
    )
    (viewer / "app.js").write_text("const close = '</script>';", encoding="utf-8")
    return viewer, dist


def test_viewer_page_rendering_and_worker_fallback(monkeypatch, tmp_path):
    viewer, dist = _viewer_files(tmp_path)
    monkeypatch.setattr(viewer_page, "VIEWER_DIR", viewer)
    monkeypatch.setattr(viewer_page, "VISUALIZER_DIST", dist)
    (viewer / "worker.js").write_text("worker </script>", encoding="utf-8")

    assert viewer_page.is_html_output("X.HTML")
    assert not viewer_page.is_html_output("x.json")
    payload_script = viewer_page.render_payload_script({"x": "</script>", "u": "→"})
    assert "<\\/script>" in payload_script and "→" in payload_script
    assert "<\\/script>" in viewer_page.render_worker_script("</script>")

    external = viewer_page.compose_viewer_html({"x": 1})
    assert 'id="tracelens-payload"' in external
    assert 'src="./app.js?v=10"' in external
    inline = viewer_page.compose_viewer_html({"x": 1}, inline_app=True)
    assert 'id="tracelens-worker-source"' in inline
    assert "const close" in inline

    (dist / "worker.js").write_text("preferred", encoding="utf-8")
    assert viewer_page._worker_js_source() == dist / "worker.js"
    (dist / "worker.js").unlink()
    (viewer / "worker.js").unlink()
    with pytest.raises(FileNotFoundError, match="worker.js not found"):
        viewer_page._worker_js_source()


def test_viewer_page_missing_script_and_save(monkeypatch, tmp_path):
    viewer, dist = _viewer_files(tmp_path)
    monkeypatch.setattr(viewer_page, "VIEWER_DIR", viewer)
    monkeypatch.setattr(viewer_page, "VISUALIZER_DIST", dist)
    (dist / "worker.js").write_text("worker", encoding="utf-8")

    target = viewer_page.save_viewer_html({"x": 1}, tmp_path / "out" / "page.html")
    assert target.exists()
    assert "tracelens-payload" in target.read_text(encoding="utf-8")

    (viewer / "index.html").write_text("<html></html>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing the app.js"):
        viewer_page.compose_viewer_html()


def test_adapter_graph_conversion_covers_metadata_namespaces_and_bad_edges():
    block = BlockNode(
        attr_name="proj",
        class_name="Linear",
        role="other",
        label="Linear",
        details=["bias: false"],
    )
    graph = ComputationGraph(
        nodes=[
            GraphNodeSpec(key="", label="", synthetic="@input"),
            GraphNodeSpec(
                key="proj",
                block=block,
                label="Linear",
                sublabel="dense",
                port_label="x",
                port_style="inline",
            ),
        ],
        links=[(0, 1), (99, 1)],
        link_port_labels={(0, 1): "hidden"},
        inline_frames=[
            InlineFrameSpec(frame_id="a", label=" bad / frame ", node_indices=[1, 9]),
            InlineFrameSpec(frame_id="b", label="", node_indices=[1]),
        ],
    )
    result = adapter.computation_graph_to_explorer_graph(
        graph, graph_id="detail", label="Detail"
    )

    assert result["nodes"][0]["id"] == "node:0"
    projected = result["nodes"][1]
    assert projected["namespace"] == "bad_frame/group"
    assert projected["incomingEdges"][0]["metadata"] == {"port_label": "hidden"}
    attrs = {item["key"]: item["value"] for item in projected["attrs"]}
    assert attrs["attr_name"] == "proj"
    assert attrs["class_name"] == "Linear"
    assert attrs["role"] == "other"
    assert attrs["sublabel"] == "dense"
    assert attrs["port_label"] == "x"
    assert attrs["port_style"] == "inline"
    assert attrs["details"] == "bias: false"
    assert result["groupNodeAttributes"][""]["title"] == "Detail"

    unlabeled = adapter.computation_graph_to_explorer_graph(
        ComputationGraph(nodes=[GraphNodeSpec(key="only", label="Only")]),
        graph_id="plain",
    )
    assert "groupNodeAttributes" not in unlabeled


def test_attach_subgraph_links_deduplicates_and_avoids_self_links():
    graphs = [
        {
            "id": "root",
            "nodes": [
                {
                    "id": "child",
                    "attrs": [
                        {"key": "attr_name", "value": "child"},
                        {"key": "attr_name", "value": 7},
                        {"key": "other", "value": "ignored"},
                    ],
                },
                {"id": "root", "attrs": [{"key": "attr_name", "value": "root"}]},
            ],
        }
    ]
    adapter.attach_subgraph_links(
        graphs, attr_name_to_graph_id={"child": "nested", "root": "root"}
    )
    assert graphs[0]["nodes"][0]["subgraphIds"] == ["nested"]
    assert "subgraphIds" not in graphs[0]["nodes"][1]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (" https://example.test/model ", "https://example.test/model"),
        ("hf://org/model", "https://huggingface.co/org/model"),
        (
            "hf://org/model/config.json",
            "https://huggingface.co/org/model/blob/main/config.json",
        ),
        ("local/path", None),
    ],
)
def test_checkpoint_source_url_branches(label, expected):
    assert fact_sheet.checkpoint_source_url(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("http://example.test/repo", "http://example.test/repo"),
        (
            "github://org/repo@main/path/model.py",
            "https://github.com/org/repo/blob/main/path/model.py",
        ),
        (
            "github://org/repo@main",
            "https://github.com/org/repo/tree/main",
        ),
        ("not-github", None),
    ],
)
def test_github_source_url_display_branches(label, expected):
    assert fact_sheet.github_source_url(label) == expected


def test_github_source_url_parsed_reference(monkeypatch):
    monkeypatch.setattr(fact_sheet, "is_github_url", lambda value: True)
    monkeypatch.setattr(
        fact_sheet,
        "parse_github_url",
        lambda value: SimpleNamespace(owner="o", repo="r", ref="v1", subpath="m.py"),
    )
    assert fact_sheet.github_source_url("github:o/r") == (
        "https://github.com/o/r/blob/v1/m.py"
    )
    monkeypatch.setattr(
        fact_sheet,
        "parse_github_url",
        lambda value: SimpleNamespace(owner="o", repo="r", ref="v1", subpath=""),
    )
    assert fact_sheet.github_source_url("github:o/r") == (
        "https://github.com/o/r/tree/v1"
    )


def test_fact_sheet_rich_metadata_and_html_escaping(monkeypatch):
    monkeypatch.setattr(fact_sheet, "_classify_role", lambda attr, cls: "attention")
    monkeypatch.setattr(fact_sheet, "_label_for", lambda role, cls, attr: "Attention")
    monkeypatch.setattr(
        fact_sheet,
        "format_forward_sequence",
        lambda spec, arrow=" → ": arrow.join(["Embed", "Decode"]),
    )
    spec = _spec(
        decoder_type="MoE",
        attention_type="GQA",
        positional_encoding="RoPE",
        norm_type="RMSNorm",
        norm_placement="Pre-Norm",
        decoder_class="Decoder<&>",
        checkpoint_source="hf://org/model/config.json",
        github_source="local<&>",
        num_hidden_layers=2,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=1000,
        max_position_embeddings=8192,
        total_params_hint="7B",
        active_params_hint="2B",
        kv_cache_per_token_bf16="1 KiB",
        layer_repeat_lines=["2 layers", "attn → RawClass (conditional)", "unmatched"],
        forward_sequence=["embed", "decode"],
        moe_notes=["first", "second", "ignored"],
        layer_notes=["one", "ignored"],
        analysis_notes=["Forward order: raw", "@op internal"],
        highlights=["Fast <unsafe>"],
    )

    viewer = fact_sheet.build_fact_sheet_viewer(spec)
    assert "Decoder class: Decoder<&>" in viewer["body"]
    assert "Heads: 32 Q / 8 KV" in viewer["body"]
    assert "Params (est.): 7B (2B active)" in viewer["body"]
    assert "attn → Attention (conditional)" in viewer["body"]
    assert "Forward: Embed -> Decode" in viewer["body"]
    assert "AST: Forward order" not in viewer["body"]
    assert "ignored" not in viewer["body"]
    assert "Fast &lt;unsafe&gt;" in viewer["bodyHtml"]
    assert "local&lt;&amp;&gt;" in viewer["bodyHtml"]
    assert fact_sheet.build_fact_sheet_group_attributes(spec) == {
        "architecture_fact_sheet": viewer["body"]
    }


def test_fact_sheet_layers_mix_analysis_and_empty_highlights():
    spec = _spec(
        num_hidden_layers=3,
        layer_mix="alternating",
        analysis_notes=["Useful note"],
    )
    viewer = fact_sheet.build_fact_sheet_viewer(spec)
    assert "- Layers: 3" in viewer["body"]
    assert "- Layer mix: alternating" in viewer["body"]
    assert "- AST: Useful note" in viewer["body"]
    assert "Highlights:" not in viewer["body"]


def test_fact_sheet_analysis_filters_operation_notes():
    spec = _spec(analysis_notes=["@op generated"])
    assert "AST:" not in fact_sheet.build_fact_sheet_viewer(spec)["body"]
