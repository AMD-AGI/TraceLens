###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Focused unit coverage for source loading, kernel analysis, and graph export."""

from __future__ import annotations

import ast
import io
import json
import sys
import tarfile
import types
import urllib.error
from pathlib import Path

import pytest

import visualizer.github as github
import visualizer.kernel_pipeline as kernel
import visualizer.loader as loader
import visualizer.model_graph as model_graph
import visualizer.source as source
import visualizer.source_policy as source_policy
from visualizer.block_tree import BlockNode
from visualizer.computation_graph import GraphNodeSpec, InlineFrameSpec
from visualizer.kernel_pipeline import KernelPipelineStep
from visualizer.model_graph import (
    GraphEdge,
    InlineFrame,
    ModelGraph,
    ModelGraphNode,
    NodeKind,
    OperationKind,
)


def _tarball(*names: str) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in names:
            data = b"x = 1\n"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def _function(text: str, name: str = "forward") -> ast.FunctionDef:
    tree = ast.parse(text)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _block(**changes) -> BlockNode:
    values = {
        "attr_name": "op",
        "class_name": "Linear",
        "role": "other",
        "label": "Linear",
    }
    values.update(changes)
    return BlockNode(**values)


@pytest.fixture(autouse=True)
def _reset_source_policy():
    source_policy.set_source_policy(None)
    yield
    source_policy.set_source_policy(None)


def test_github_refs_validation_and_priority(tmp_path: Path):
    assert github.is_github_url(" github:AMD/Repo@dev:src ")
    assert not github.is_github_url("https://example.com/a/b")

    short = github.parse_github_url("github:AMD/Repo@feature:pkg/model.py")
    assert short.slug == "AMD_Repo_feature"
    assert short.display == "github://AMD/Repo@feature/pkg/model.py"

    web = github.parse_github_url(
        "https://www.github.com/AMD/Repo/blob/main/pkg/modeling_demo.py"
    )
    assert web.subpath == "pkg/modeling_demo.py"
    with pytest.raises(ValueError, match="must point to a .py"):
        github.parse_github_url("https://github.com/AMD/Repo/blob/main/README.md")
    with pytest.raises(ValueError, match="Unsupported"):
        github.parse_github_url("gitlab.com/AMD/Repo")

    paths = [
        tmp_path / "deep" / "models.py",
        tmp_path / "model.py",
        tmp_path / "modeling_x.py",
        tmp_path / "helper.py",
    ]
    assert sorted(paths, key=github.python_source_priority)[0].name == "modeling_x.py"


def test_download_extract_and_archive_cache(tmp_path: Path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"downloaded"

    response = Response()
    monkeypatch.setattr(
        github.urllib.request, "urlopen", lambda request, timeout: response
    )
    assert github._download_bytes("https://example.invalid") == b"downloaded"

    extracted = github._extract_tarball(
        _tarball("repo-main/modeling_demo.py"), tmp_path / "one"
    )
    assert extracted.name == "repo-main"
    assert (extracted / "modeling_demo.py").is_file()

    multi = github._extract_tarball(_tarball("a/a.py", "b/b.py"), tmp_path / "many")
    assert multi == tmp_path / "many"

    cached = tmp_path / "cached"
    cached.mkdir()
    (cached / "keep.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        github, "_download_bytes", lambda url: pytest.fail("cache should be reused")
    )
    assert github._fetch_archive(github.GitHubRef("a", "b"), cached) == cached


def test_archive_fallback_and_failure(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def download(url: str) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.URLError("first failed")
        return _tarball("repo-main/model.py")

    monkeypatch.setattr(github, "_download_bytes", download)
    result = github._fetch_archive(
        github.GitHubRef("amd", "repo", ref="dev"), tmp_path / "cache"
    )
    assert result.name == "repo-main"
    assert len(calls) == 2

    monkeypatch.setattr(
        github,
        "_download_bytes",
        lambda url: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    with pytest.raises(FileNotFoundError, match="Could not download"):
        github._fetch_archive(github.GitHubRef("amd", "missing"), tmp_path / "missing")


def test_single_file_cache_and_fetch_subpaths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(github, "CACHE_ROOT", tmp_path / "single")
    ref = github.GitHubRef(
        "huggingface",
        "transformers",
        subpath="src/modeling_demo.py",
    )
    with pytest.raises(ValueError, match="Expected a Python"):
        github._fetch_single_file(github.GitHubRef("a", "b", subpath="src"))

    downloads: list[str] = []
    monkeypatch.setattr(
        github,
        "_download_bytes",
        lambda url: downloads.append(url) or b"VALUE = 3\n",
    )
    first = github._fetch_single_file(ref)
    assert first.read_bytes() == b"VALUE = 3\n"
    assert github._fetch_single_file(ref) == first
    assert len(downloads) == 1

    policy = source_policy.SourcePolicy()
    monkeypatch.setattr(github, "_fetch_single_file", lambda item: tmp_path / "one.py")
    assert github.fetch_github_source(ref, source_policy=policy) == tmp_path / "one.py"

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "model.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(github, "_fetch_archive", lambda item, cache: repo)
    directory_ref = github.GitHubRef("huggingface", "transformers", subpath="pkg")
    assert (
        github.fetch_github_source(directory_ref, source_policy=policy) == repo / "pkg"
    )
    file_ref = github.GitHubRef("huggingface", "transformers", subpath="pkg/model.py")
    assert github.fetch_github_source(file_ref, source_policy=policy).is_file()
    with pytest.raises(FileNotFoundError, match="not found"):
        github.fetch_github_source(
            github.GitHubRef("huggingface", "transformers", subpath="does/not/exist"),
            source_policy=policy,
        )


def test_modeling_file_discovery_and_config_path(tmp_path: Path):
    single = tmp_path / "model.py"
    single.write_text("", encoding="utf-8")
    assert github.find_modeling_files(single) == [single.absolute()]
    assert github.find_modeling_files(tmp_path / "missing") == []

    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    assert github.github_config_path(tmp_path) == config
    assert github.github_config_path(single) == config
    config.unlink()
    assert github.github_config_path(single) is None


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("github://AMD/Repo.git@main", ("AMD", "Repo")),
        ("github:AMD/Repo", ("AMD", "Repo")),
        (" AMD/Repo ", ("AMD", "Repo")),
    ],
)
def test_source_policy_repo_specs(spec: str, expected: tuple[str, str]):
    assert source_policy._parse_repo_spec(spec) == expected


def test_source_policy_env_cli_global_and_errors(monkeypatch):
    monkeypatch.setenv(
        "TRACELENS_ALLOWED_GITHUB_REPOS", "AMD/One, github://AMD/Two.git@dev"
    )
    policy = source_policy.SourcePolicy.from_env_and_cli(["Extra/Repo"])
    assert policy.is_github_repo_allowed("amd", "one")
    assert policy.is_github_repo_allowed("AMD", "TWO")
    assert ("extra", "repo") in policy.allowed_repo_keys()
    with pytest.raises(PermissionError, match="Allowed repositories"):
        policy.require_github_repo_allowed("bad", "repo")
    policy.require_github_repo_allowed("extra", "repo")

    source_policy.set_source_policy(policy)
    assert source_policy.get_source_policy() is policy
    source_policy.set_source_policy(None)
    assert isinstance(source_policy.get_source_policy(), source_policy.SourcePolicy)

    for bad in ("owner", "/repo", "owner/"):
        with pytest.raises(ValueError, match="Invalid GitHub repo spec"):
            source_policy._parse_repo_spec(bad)


def test_source_helpers_and_model_types():
    assert source._module_file("pkg.modeling.Model") == "pkg.py"
    assert source._collect_auto_map_files(
        {
            "auto_map": {
                "AutoModel": "modeling_x.Model",
                "duplicate": "modeling_x.Other",
                "ignored": ["not", "text"],
            }
        }
    ) == ["modeling_x.py"]
    assert source._collect_auto_map_files({"auto_map": "bad"}) == []
    assert source._config_model_types(
        {
            "_wrapper_model_type": "multi-modal",
            "model_type": "multi-modal",
            "text_config": {"model_type": "text-backbone"},
            "llm_config": "invalid",
        }
    ) == ["multi_modal", "text_backbone"]
    files = [
        Path("configuration_x.py"),
        Path("model.py"),
        Path("model.py"),
        Path("models.py"),
    ]
    assert source._has_modeling_implementation(files)
    assert source._dedupe_paths(files) == files[:2] + [files[3]]


def test_hub_snapshot_selection(tmp_path: Path, monkeypatch):
    cache = tmp_path / "hub"
    base = cache / "models--org--model"
    old = base / "snapshots" / "old"
    new = base / "snapshots" / "new"
    old.mkdir(parents=True)
    new.mkdir()
    (base / "refs").mkdir()
    (base / "refs" / "main").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(source, "_hub_cache_root", lambda: cache)
    assert source._hub_snapshot_root("org/model") == old

    (base / "refs" / "main").unlink()
    old.touch()
    new.touch()
    assert source._hub_snapshot_root("org/model") in {old, new}
    assert source._hub_snapshot_root("absent/model") is None


def test_hub_list_and_download_are_failure_tolerant(tmp_path: Path, monkeypatch):
    module = types.ModuleType("huggingface_hub")
    module.list_repo_files = lambda model_id: [
        "model.py",
        "__pycache__/bad.py",
        "README.md",
    ]

    def download(model_id: str, name: str):
        if name == "bad.py":
            raise RuntimeError("missing")
        return str(tmp_path / name)

    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    assert source._list_repo_python_files("org/model") == ["model.py"]
    assert source._download_repo_files("org/model", []) == []
    assert source._download_repo_files("org/model", ["good.py", "bad.py"]) == [
        tmp_path / "good.py"
    ]
    module.list_repo_files = lambda model_id: (_ for _ in ()).throw(RuntimeError())
    assert source._list_repo_python_files("org/model") == []


def test_transformers_modeling_path_variants(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    origin = tmp_path / "modeling_demo.py"
    origin.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        source.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(origin=str(origin)),
    )
    assert source._transformers_modeling_path("my-model") == origin
    monkeypatch.setattr(source.importlib.util, "find_spec", lambda name: None)
    assert source._transformers_modeling_path("none") is None
    monkeypatch.setattr(
        source.importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError()),
    )
    assert source._transformers_modeling_path("missing") is None


def test_resolve_sources_explicit_local_github_and_reads(tmp_path: Path, monkeypatch):
    model = tmp_path / "modeling_local.py"
    model.write_text("VALUE = 1\n", encoding="utf-8")
    config = {"model_type": ""}
    assert source.resolve_source_files(None, config, code_path=model) == (
        [model.resolve()],
        [str(model.resolve())],
    )
    with pytest.raises(FileNotFoundError, match="Code path"):
        source.resolve_source_files(None, config, code_path=tmp_path / "missing.py")
    assert source.resolve_source_files(model, config)[0] == [model.resolve()]

    monkeypatch.setattr(
        source,
        "resolve_github_files",
        lambda value, source_policy=None: ([model], "github://amd/repo@main"),
    )
    files, labels = source.resolve_source_files("github:amd/repo", config)
    assert files == [model]
    assert labels == ["github://amd/repo@main"]
    assert source.read_sources([model, tmp_path / "missing"]) == {model: "VALUE = 1\n"}


def test_resolve_github_empty_and_transformers_fallback(tmp_path: Path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(source, "fetch_github_source", lambda ref, **kwargs: empty)
    policy = source_policy.SourcePolicy.from_env_and_cli(["amd/repo"])
    with pytest.raises(FileNotFoundError, match="No Python source"):
        source.resolve_github_files("github:amd/repo", source_policy=policy)

    upstream = tmp_path / "modeling_native.py"
    upstream.write_text("", encoding="utf-8")
    attempts: list[str] = []

    def fetch(ref, **kwargs):
        attempts.append(ref.subpath)
        if "/native/" not in ref.subpath:
            raise FileNotFoundError
        return upstream

    monkeypatch.setattr(source, "fetch_github_source", fetch)
    result = source._transformers_github_modeling_file(
        ["wrapper", "native"], source_policy=source_policy.SourcePolicy()
    )
    assert result == (
        upstream,
        "github://huggingface/transformers@main/src/transformers/models/native/"
        "modeling_native.py",
    )
    assert len(attempts) == 2


def test_resolve_hf_download_and_candidate_fallback(tmp_path: Path, monkeypatch):
    downloaded = tmp_path / "configuration.py"
    fallback = tmp_path / "modeling_demo.py"
    downloaded.write_text("", encoding="utf-8")
    fallback.write_text("", encoding="utf-8")
    monkeypatch.setattr(source, "_hub_snapshot_root", lambda model_id: None)
    monkeypatch.setattr(source, "_transformers_modeling_path", lambda model_type: None)
    monkeypatch.setattr(source, "_list_repo_python_files", lambda model_id: [])
    monkeypatch.setattr(
        source, "_transformers_github_modeling_file", lambda *args, **kwargs: None
    )
    calls: list[list[str]] = []

    def download(model_id: str, names: list[str]) -> list[Path]:
        calls.append(names)
        if names == ["modeling_demo.py", "modeling.py"]:
            return [fallback]
        return []

    monkeypatch.setattr(source, "_download_repo_files", download)
    files, labels = source.resolve_source_files("org/demo", {"model_type": "demo"})
    assert files == [fallback]
    assert labels == ["hf://org/demo"]
    assert ["modeling_demo.py", "modeling.py"] in calls


def test_kernel_parsers_labels_conditions_and_targets():
    details = [
        "kernel: chunk_kda",
        "kwarg: use_gate=True",
        "kwarg: mode=fast",
        "kwarg: malformed",
        "import: pkg.ops#run",
    ]
    assert kernel.parse_kernel_call_flags(details) == {
        "_kernel": "chunk_kda",
        "use_gate": True,
        "mode": "fast",
    }
    assert kernel.parse_kernel_import(details) == ("pkg.ops", "run")
    assert kernel.parse_kernel_import(["import: pkg.ops.run"]) == ("pkg.ops", "run")
    assert kernel.parse_kernel_import(["import: standalone"]) == (
        "standalone",
        "standalone",
    )
    assert kernel.parse_kernel_import([]) is None

    assert (
        kernel.tensor_port_kernel_frame_label("forward_l2norm_fwd_q") == "l2norm_fwd_q"
    )
    assert kernel.tensor_port_kernel_frame_label("other") is None
    assert kernel.kernel_op_display_label("l2norm_fwd") == "L2Norm"
    assert kernel.kernel_op_display_label("foo_bar_fwd") == "Foo bar"
    assert kernel.kernel_op_display_label("") == ""

    assert kernel._condition_name(ast.parse("enabled").body[0].value) == "enabled"
    assert (
        kernel._condition_name(ast.parse("not enabled").body[0].value) == "not enabled"
    )
    assert "is not None" in kernel._condition_name(
        ast.parse("state is not None").body[0].value
    )
    assert kernel._condition_name(ast.parse("obj.enabled").body[0].value) is None

    step = KernelPipelineStep(
        "a", "first", "KernelOp", "a", [], tensor_inputs=frozenset({"q"})
    )
    fallback = KernelPipelineStep("b", "second", "KernelOp", "b", [])
    assert kernel.compute_tensor_step_targets(
        ["kwarg: q=query", "kwarg: k=key"], [step, fallback]
    ) == {"q": "first"}
    assert kernel.compute_tensor_step_targets(["kwarg: q=query"], [fallback]) == {
        "q": "second"
    }
    assert kernel.compute_tensor_step_targets([], [fallback]) == {}


def test_kernel_module_resolution_and_search_roots(tmp_path: Path, monkeypatch):
    root = tmp_path / "code"
    package = root / "pkg"
    package.mkdir(parents=True)
    module_file = package / "ops.py"
    module_file.write_text("def run():\n    return 1\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(kernel, "_KERNEL_SEARCH_ROOTS", [])
    monkeypatch.setattr(kernel, "_KERNEL_FIXTURE_ROOT", None)

    kernel.register_kernel_search_root(module_file)
    kernel.register_kernel_search_root(root)
    assert kernel._kernel_search_roots() == [root, package]
    assert kernel._search_root_module_file("pkg.ops") == module_file
    assert kernel._search_root_module_file("pkg") == package / "__init__.py"
    assert kernel._module_file_path("pkg.ops") == module_file
    assert kernel._read_module_source("pkg.ops") == (
        module_file.read_text(encoding="utf-8"),
        "pkg.ops",
    )


def test_kernel_symbol_import_following(monkeypatch):
    sources = {
        "pkg.wrapper": "from pkg.impl import run as execute\n",
        "pkg.impl": "def run(x):\n    return x\n\nclass Op:\n    def forward(self, x):\n        return x\n",
    }
    monkeypatch.setattr(
        kernel,
        "_read_module_source",
        lambda module: (sources[module], module) if module in sources else None,
    )
    found = kernel._find_symbol_definition("pkg.wrapper", "execute")
    assert found is not None and found[1:] == ("run", "pkg.impl")
    found_class = kernel._find_symbol_definition("pkg.impl", "Op")
    assert found_class is not None and found_class[1] == "Op.forward"
    assert kernel._find_symbol_definition("pkg.impl", "missing") is None
    assert (
        kernel._resolve_relative_module("impl", "pkg.wrapper", 1) == "pkg.wrapper.impl"
    )
    assert kernel._resolve_relative_module("", "pkg.sub.wrapper", 2) == "pkg"
    assert kernel._should_follow_import("pkg.ops", "pkg.wrapper")
    assert not kernel._should_follow_import("other.ops", "pkg.wrapper")


def test_extract_pipeline_tracks_branches_buffers_and_factories():
    text = """
def forward(q, k, enabled=True):
    out = empty_like(q)
    qn = l2norm_fwd(q)
    if enabled:
        gate = gate_fwd(k)
    else:
        gate = fallback_fwd(k)
    builder = sparse_kernel(qn)
    builder(qn, gate, out)
    finish_fwd(out)
"""
    func = _function(text)
    steps = kernel._extract_pipeline_from_function(
        func,
        source=text,
        prefix="forward",
        skip_calls=set(),
        tensor_ports={"q", "k"},
        flags={"enabled": True},
    )
    assert [step.call_name for step in steps] == [
        "empty_like",
        "l2norm_fwd",
        "gate_fwd",
        "sparse_kernel",
        "finish_fwd",
    ]
    assert steps[1].tensor_inputs == frozenset({"q"})
    assert steps[2].condition == "enabled"
    assert steps[-1].predecessors == frozenset({steps[-2].attr_name})
    assert "fallback_fwd" not in {step.call_name for step in steps}


def test_kernel_computation_decomposition_and_recurrence(monkeypatch):
    expr = ast.parse("sigmoid(x) * scale + (1 / sqrt(y))", mode="eval").body
    ops = kernel._decompose_computation_expr(expr)
    assert [op.label for op in ops] == ["Sigmoid", "× scale", "Sqrt", "÷", "+"]

    rstd = ast.parse("x * rstd", mode="eval").body
    assert kernel._decompose_computation_expr(rstd)[-1].second_operand == "input"
    assert (
        kernel._decompose_computation_expr(ast.parse("x * x", mode="eval").body) == []
    )
    assert (
        kernel._decompose_computation_expr(ast.parse("-softplus(x)", mode="eval").body)[
            -1
        ].label
        == "Softplus"
    )

    triton = _function(
        """
@triton.jit
def fused_kernel(x, scale):
    a = tl.sigmoid(x)
    b = a * scale
    if x:
        c = tl.cumsum(b)
""",
        "fused_kernel",
    )
    assert kernel._function_has_triton_jit_decorator(triton)
    assert kernel._extract_triton_computation_labels(triton) == [
        "Sigmoid",
        "× scale",
        "CumSum",
    ]
    assert (
        kernel._extract_recurrence_from_kernel_source(
            "b_v = tl.load(p_v) - b_v\nb_h1 *= tl.exp2(g)\nx = tl.dot(k, b_h1)"
        )
        == "v_new = v − W @ h\nh = exp2(g) · h\nh = h + k @ v_new"
    )
    assert kernel._extract_recurrence_from_kernel_source("x = 1") is None


def test_kernel_op_substeps_and_pipeline_partition(monkeypatch):
    source_text = """
def helper(x):
    fused_kernel[(1,)](x)

@triton.jit
def fused_kernel(x):
    a = tl.sigmoid(x)
    b = a * tl.exp(x)
"""
    monkeypatch.setattr(
        kernel,
        "_resolve_implementation",
        lambda *args, **kwargs: (source_text, "helper", "pkg.ops"),
    )
    children = kernel.introspect_kernel_op_substeps(
        "helper", {}, "pkg.ops", parent_attr="parent"
    )
    assert [child.label for child in children] == ["Sigmoid", "Exp", "×"]
    assert children[-1].second_operand == "parent_sub_0"
    assert (
        kernel.introspect_kernel_op_substeps(
            "chunk_kda_fwd_intra", {}, "pkg.ops", parent_attr="parent"
        )
        == ()
    )

    steps = [
        KernelPipelineStep("prep_fwd", "prep", "KernelOp", "prep", []),
        KernelPipelineStep(
            "recurrent_fwd",
            "out",
            "KernelOp",
            "out",
            [],
            predecessors=frozenset({"prep", "removed"}),
        ),
    ]
    filtered = kernel._filter_step_predecessors(steps, {"prep", "out"})
    assert filtered[1].predecessors == frozenset({"prep"})
    assert kernel._is_output_pipeline_step("recurrent_fwd")
    assert kernel._is_output_pipeline_step("chunk_gla_fwd_o_gk")
    assert not kernel._is_output_pipeline_step("l2norm_fwd")


def test_model_graph_serialization_saving_and_validation(tmp_path: Path):
    child = ModelGraph(
        "child",
        nodes=[
            ModelGraphNode(
                "bad",
                NodeKind.LEAF,
                "Mystery",
                OperationKind.UNKNOWN,
                {"why": "unknown"},
            )
        ],
    )
    graph = ModelGraph(
        "root",
        nodes=[
            ModelGraphNode(
                "sub",
                NodeKind.SUBGRAPH,
                "Child",
                OperationKind.COMPOSITE,
                {"subgraph_key": "child"},
            ),
            ModelGraphNode("input", NodeKind.LEAF, "Input", OperationKind.SYNTHETIC),
        ],
        edges=[GraphEdge("input", "sub", label="hidden")],
        inline_frames=[InlineFrame("frame", "Frame", ["input"], "details")],
        subgraphs={"child": child},
    )
    payload = graph.to_dict()
    assert payload["edges"][0]["label"] == "hidden"
    assert payload["inline_frames"][0]["sublabel"] == "details"
    assert json.loads(graph.to_json())["subgraphs"]["child"]["title"] == "child"

    path = model_graph.save_model_graph(graph, tmp_path / "nested" / "graph.json")
    assert path.is_file()
    architecture = model_graph.save_architecture_model_graphs(
        {"name": "demo"}, tmp_path / "architecture.json"
    )
    assert json.loads(architecture.read_text(encoding="utf-8")) == {"name": "demo"}

    issues = model_graph.collect_non_reduced_operations(graph)
    assert any(issue.node_id == "bad" for issue in issues)
    with pytest.raises(AssertionError, match="could not classify"):
        model_graph.assert_operations_reduced(graph)


def test_model_graph_helpers_and_classification():
    assert model_graph.is_torch_primitive_label("Log Softmax")
    assert model_graph.is_torch_primitive_label("× scale")
    assert not model_graph.is_torch_primitive_label("Fused sigmoid")
    assert model_graph.classify_operation(None) == OperationKind.UNKNOWN
    assert (
        model_graph.classify_operation(
            _block(children=[_block(attr_name="inner")], is_basic=False)
        )
        == OperationKind.COMPOSITE
    )
    assert (
        model_graph.classify_operation(
            _block(class_name="CustomLinearWrapper", is_basic=False)
        )
        == OperationKind.UNKNOWN
    )

    spec = GraphNodeSpec(
        key="node",
        block=_block(
            attr_name="@forward_op",
            class_name="Custom",
            details=["detail"],
            external_inputs=["mask"],
        ),
        label="Displayed",
        port_label="q",
        port_style="inline",
        synthetic="@tensor",
    )
    metadata = model_graph._minimal_metadata(spec)
    assert metadata["class_name"] == "Custom"
    assert metadata["port_label"] == "q"
    assert metadata["synthetic"] == "@tensor"

    frames = model_graph._convert_inline_frames(
        [InlineFrameSpec("f", "Frame", "sub", [0, 9])], {0: "node"}
    )
    assert frames == [InlineFrame("f", "Frame", ["node"], "sub")]


def test_loader_builds_filters_and_delegates(monkeypatch):
    assert loader.resolve_checkpoint_arg(
        checkpoint=Path("checkpoint"), source="source"
    ) == Path("checkpoint")
    detailed = loader.build_detailed_basic_ops(add=["custom"], remove=[])
    assert detailed.is_basic("Linear")
    assert detailed.is_basic("custom")

    captured: dict[str, object] = {}
    fake_spec = types.SimpleNamespace(
        class_registry={"Layer": object()}, export_block_trees=[1]
    )

    def load_architecture(checkpoint, **kwargs):
        captured.update({"checkpoint": checkpoint, **kwargs})
        return fake_spec

    monkeypatch.setattr(loader, "load_architecture", load_architecture)
    assert (
        loader.load_model_spec(
            "model",
            github="github:huggingface/transformers",
            analyze_code=False,
            require_code=True,
            allow_github_repos=["amd/repo"],
        )
        is fake_spec
    )
    assert captured["analyze_code"] is True
    assert captured["detailed"] is True
    assert captured["allow_github_repos"] == ["amd/repo"]


def test_loader_require_code_errors(monkeypatch):
    monkeypatch.setattr(
        loader,
        "load_architecture",
        lambda *args, **kwargs: types.SimpleNamespace(
            class_registry={}, export_block_trees=[]
        ),
    )
    with pytest.raises(FileNotFoundError, match="No modeling source"):
        loader.load_model_spec("model", require_code=True)

    monkeypatch.setattr(
        loader,
        "load_architecture",
        lambda *args, **kwargs: types.SimpleNamespace(
            class_registry={"Layer": object()}, export_block_trees=[]
        ),
    )
    with pytest.raises(ValueError, match="no computation block trees"):
        loader.load_model_spec("model", require_code=True)
