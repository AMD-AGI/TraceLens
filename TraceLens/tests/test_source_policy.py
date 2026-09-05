###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for GitHub source introspection whitelist policy."""

from pathlib import Path

import pytest

from TraceLens.ModelUtils.extract import load_architecture
from TraceLens.ModelUtils.github import fetch_github_source, parse_github_url
from TraceLens.ModelUtils.source import resolve_github_files
from TraceLens.ModelUtils.source_policy import SourcePolicy, set_source_policy

FIXTURES = Path(__file__).parent / "fixtures"


def test_default_policy_allows_transformers():
    policy = SourcePolicy.from_env_and_cli()
    assert policy.is_github_repo_allowed("huggingface", "transformers")
    assert policy.is_github_repo_allowed("HuggingFace", "Transformers")


def test_default_policy_allows_fla_kernel_repo():
    policy = SourcePolicy.from_env_and_cli()
    assert policy.is_github_repo_allowed("fla-org", "flash-linear-attention")


def test_default_policy_blocks_unknown_repo():
    policy = SourcePolicy.from_env_and_cli()
    assert not policy.is_github_repo_allowed("acme", "custom")


def test_cli_allow_repo_extends_whitelist():
    policy = SourcePolicy.from_env_and_cli(["acme/custom"])
    assert policy.is_github_repo_allowed("acme", "custom")
    assert policy.is_github_repo_allowed("huggingface", "transformers")


def test_fetch_github_source_enforces_policy(monkeypatch):
    set_source_policy(SourcePolicy.from_env_and_cli())
    ref = parse_github_url("github:acme/custom@main")

    with pytest.raises(PermissionError, match="not whitelisted"):
        fetch_github_source(ref)


def test_fetch_github_source_allows_whitelisted_repo(monkeypatch, tmp_path: Path):
    set_source_policy(SourcePolicy.from_env_and_cli(["acme/custom"]))

    def fake_fetch_archive(ref, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        modeling = cache_dir / "modeling_custom.py"
        modeling.write_text("class Block: pass\n", encoding="utf-8")
        return cache_dir

    monkeypatch.setattr("TraceLens.ModelUtils.github._fetch_archive", fake_fetch_archive)

    ref = parse_github_url("github:acme/custom@main")
    root = fetch_github_source(ref, cache_root=tmp_path / "cache")
    assert root.is_dir()


def test_resolve_github_files_requires_whitelist():
    set_source_policy(SourcePolicy.from_env_and_cli())

    with pytest.raises(PermissionError, match="not whitelisted"):
        resolve_github_files("github:acme/custom@main")


def test_load_architecture_allow_repo_passes_through(monkeypatch):
    config = FIXTURES / "llama_like" / "config.json"
    fixture_dir = FIXTURES / "custom_model"

    def fake_fetch(ref, **kwargs):
        return fixture_dir

    monkeypatch.setattr("TraceLens.ModelUtils.source.fetch_github_source", fake_fetch)

    spec = load_architecture(
        checkpoint=config,
        github="github:acme/custom@main",
        allow_github_repos=["acme/custom"],
    )
    assert spec.github_source == "github://acme/custom@main"


def test_local_checkpoint_does_not_require_github_whitelist():
    set_source_policy(SourcePolicy.from_env_and_cli())

    spec = load_architecture(FIXTURES / "custom_model", analyze_code=True)
    assert spec.decoder_class == "CustomDecoderLayer"
