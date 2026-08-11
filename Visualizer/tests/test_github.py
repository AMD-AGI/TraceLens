"""Tests for GitHub URL parsing and split checkpoint/code sources."""

from pathlib import Path

import pytest

from visualizer.extract import load_architecture
from visualizer.github import parse_github_url
from visualizer.source import resolve_github_files, resolve_source_files


FIXTURES = Path(__file__).parent / "fixtures"


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

    def fake_fetch(ref, cache_root=None):
        return fixture_dir

    monkeypatch.setattr("visualizer.source.fetch_github_source", fake_fetch)

    files, label = resolve_github_files("github:acme/custom@main")
    assert len(files) == 1
    assert files[0].name == "modeling_custom.py"
    assert label.startswith("github://acme/custom")


def test_split_checkpoint_and_github(monkeypatch):
    fixture_dir = FIXTURES / "custom_model"
    config = FIXTURES / "llama_like" / "config.json"

    def fake_fetch(ref, cache_root=None):
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
    def fake_fetch(ref, cache_root=None):
        return FIXTURES / "custom_model"

    monkeypatch.setattr("visualizer.source.fetch_github_source", fake_fetch)

    files, labels = resolve_source_files(
        "gpt2",
        {"model_type": "gpt2"},
        github="github:acme/custom@main",
    )
    assert files[0].name == "modeling_custom.py"
    assert labels[0].startswith("github://")
