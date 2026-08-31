"""Tests for GitHub URL parsing and split checkpoint/code sources."""

from pathlib import Path

import pytest

from visualizer.extract import load_architecture
from visualizer.github import parse_github_url
from visualizer.source import resolve_github_files, resolve_source_files
from visualizer.source_policy import SourcePolicy, set_source_policy


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _allow_fixture_github_repos():
    set_source_policy(SourcePolicy.from_env_and_cli(["acme/custom", "acme/models"]))
    yield


def test_parse_github_web_url_with_subpath():
    ref = parse_github_url("https://github.com/acme/models/tree/develop/src/llm")
    assert ref.owner == "acme"
    assert ref.repo == "models"
    assert ref.ref == "develop"
    assert ref.subpath == "src/llm"


def test_parse_github_shorthand():
    ref = parse_github_url("github:acme/models@main:src")
    assert ref.display == "github://acme/models@main/src"


def test_resolve_github_files_from_local_fixture(tmp_path: Path, monkeypatch):
    fixture_dir = FIXTURES / "custom_model"

    def fake_fetch(ref, cache_root=None, **kwargs):
        return fixture_dir

    monkeypatch.setattr("visualizer.source.fetch_github_source", fake_fetch)

    files, label = resolve_github_files("github:acme/custom@main")
    assert len(files) == 1
    assert files[0].name == "modeling_custom.py"
    assert label.startswith("github://acme/custom")


def test_split_checkpoint_and_github(monkeypatch):
    fixture_dir = FIXTURES / "custom_model"
    config = FIXTURES / "llama_like" / "config.json"

    def fake_fetch(ref, cache_root=None, **kwargs):
        return fixture_dir

    monkeypatch.setattr("visualizer.source.fetch_github_source", fake_fetch)

    spec = load_architecture(
        checkpoint=config,
        github="github:acme/custom@main",
        name="Split sources",
    )

    assert spec.attention_type == "MLA"
    assert spec.decoder_class == "CustomDecoderLayer"
    assert "config.json" in spec.checkpoint_source
    assert spec.github_source == "github://acme/custom@main"
    assert spec.num_hidden_layers == 16


def test_resolve_source_files_prefers_github_over_hf(monkeypatch):
    def fake_fetch(ref, cache_root=None, **kwargs):
        return FIXTURES / "custom_model"

    monkeypatch.setattr("visualizer.source.fetch_github_source", fake_fetch)

    files, labels = resolve_source_files(
        "gpt2",
        {"model_type": "gpt2"},
        github="github:acme/custom@main",
    )
    assert files[0].name == "modeling_custom.py"
    assert labels[0].startswith("github://")


def test_find_modeling_files_walks_nested_python(tmp_path: Path):
    from visualizer.github import find_modeling_files

    (tmp_path / "inference").mkdir()
    nested = tmp_path / "inference" / "model.py"
    nested.write_text("class Decoder: pass\n", encoding="utf-8")
    sibling = tmp_path / "inference" / "kernel.py"
    sibling.write_text("def sparse_attn(): pass\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "modeling_top.py").write_text("class Top: pass\n", encoding="utf-8")

    files = find_modeling_files(tmp_path)
    names = [path.name for path in files]
    assert names[0] == "modeling_top.py"
    assert {path.name for path in files} == {"modeling_top.py", "model.py", "kernel.py"}
    assert all("__pycache__" not in path.parts for path in files)


def test_find_modeling_files_keeps_snapshot_symlink_names(tmp_path: Path):
    from visualizer.github import find_modeling_files

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "abc123"
    blob.write_text("class MLA: pass\n", encoding="utf-8")
    inference = tmp_path / "snapshots" / "rev" / "inference"
    inference.mkdir(parents=True)
    linked = inference / "model.py"
    linked.symlink_to(blob)

    files = find_modeling_files(tmp_path / "snapshots" / "rev")
    assert [path.name for path in files] == ["model.py"]
    assert files[0].suffix == ".py"


def test_resolve_source_files_finds_nested_snapshot_python(tmp_path: Path):
    snapshot = tmp_path / "inference"
    snapshot.mkdir()
    (snapshot / "model.py").write_text("class MLA: pass\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"model_type": "deepseek_v4"}', encoding="utf-8")

    files, labels = resolve_source_files(tmp_path, {"model_type": "deepseek_v4"})
    assert any(path.name == "model.py" for path in files)
    assert labels


def test_resolve_source_files_uses_cached_hub_snapshot(tmp_path: Path, monkeypatch):
    snapshot = tmp_path / "snapshots" / "abc"
    (snapshot / "inference").mkdir(parents=True)
    (snapshot / "inference" / "model.py").write_text("class MLA: pass\n", encoding="utf-8")
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "main").write_text("abc\n", encoding="utf-8")

    monkeypatch.setattr("visualizer.source._hub_cache_root", lambda: tmp_path.parent)
    monkeypatch.setattr(
        "visualizer.source._hub_snapshot_root",
        lambda model_id: snapshot,
    )

    files, labels = resolve_source_files(
        "deepseek-ai/DeepSeek-V4-Flash",
        {"model_type": "deepseek_v4"},
    )
    assert any(path.name == "model.py" for path in files)
    assert labels[0].startswith("hf://")


def _stub_upstream_transformers(monkeypatch, tmp_path: Path, available: set[str]) -> list[str]:
    """Serve upstream modeling files for `available` model types, recording lookups."""
    requested: list[str] = []

    def fake_fetch(ref, **kwargs):
        requested.append(ref.subpath)
        model_type = ref.subpath.split("/")[-2]
        if model_type not in available:
            raise FileNotFoundError(ref.subpath)
        path = tmp_path / f"modeling_{model_type}.py"
        path.write_text(f"class {model_type}DecoderLayer: pass\n", encoding="utf-8")
        return path

    monkeypatch.setattr("visualizer.source.fetch_github_source", fake_fetch)
    monkeypatch.setattr("visualizer.source._hub_snapshot_root", lambda model_id: None)
    monkeypatch.setattr("visualizer.source._list_repo_python_files", lambda model_id: [])
    monkeypatch.setattr(
        "visualizer.source._download_repo_files",
        lambda model_id, filenames: [],
    )
    return requested


def test_resolve_source_files_reads_transformers_native_model_from_github(tmp_path, monkeypatch):
    """Qwen3-style checkpoints ship no modeling code, so upstream source is used."""
    requested = _stub_upstream_transformers(monkeypatch, tmp_path, {"qwen3_moe"})

    files, labels = resolve_source_files("Qwen/Qwen3-235B-A22B", {"model_type": "qwen3_moe"})

    assert [path.name for path in files] == ["modeling_qwen3_moe.py"]
    assert requested == ["src/transformers/models/qwen3_moe/modeling_qwen3_moe.py"]
    assert labels == [
        "github://huggingface/transformers@main/src/transformers/models/qwen3_moe/"
        "modeling_qwen3_moe.py"
    ]


def test_resolve_source_files_falls_back_to_nested_text_config_model_type(tmp_path, monkeypatch):
    """A multimodal wrapper without upstream code still finds its text backbone."""
    requested = _stub_upstream_transformers(monkeypatch, tmp_path, {"minimax_m2"})

    files, _labels = resolve_source_files(
        "MiniMaxAI/MiniMax-M3",
        {"model_type": "minimax_m3_vl", "text_config": {"model_type": "minimax_m2"}},
    )

    assert [path.name for path in files] == ["modeling_minimax_m2.py"]
    assert len(requested) == 2


def test_resolve_source_files_keeps_checkpoint_modeling_code(tmp_path, monkeypatch):
    """Source shipped with the checkpoint wins; no upstream lookup is attempted."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "modeling_custom.py").write_text("class Block: pass\n", encoding="utf-8")
    requested = _stub_upstream_transformers(monkeypatch, tmp_path, {"custom"})
    monkeypatch.setattr("visualizer.source._hub_snapshot_root", lambda model_id: snapshot)

    files, _labels = resolve_source_files("acme/custom", {"model_type": "custom"})

    assert [path.name for path in files] == ["modeling_custom.py"]
    assert requested == []


def test_resolve_source_files_orders_modeling_before_config_helpers(tmp_path, monkeypatch):
    """Analysis reads files in order, so modeling code must come first."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "configuration_minimax_m3_vl.py").write_text("class Cfg: pass\n", encoding="utf-8")
    (snapshot / "processing_minimax.py").write_text("class Proc: pass\n", encoding="utf-8")
    _stub_upstream_transformers(monkeypatch, tmp_path, {"minimax_m3_vl"})
    monkeypatch.setattr("visualizer.source._hub_snapshot_root", lambda model_id: snapshot)

    files, _labels = resolve_source_files("MiniMaxAI/MiniMax-M3", {"model_type": "minimax_m3_vl"})

    assert [path.name for path in files][0] == "modeling_minimax_m3_vl.py"
