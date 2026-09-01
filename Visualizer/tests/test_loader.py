###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for shared HF/local model loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from visualizer.basic_ops import BasicOpFilter
from visualizer.loader import load_model_spec, resolve_checkpoint_arg

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_resolve_checkpoint_arg_prefers_flag():
    assert resolve_checkpoint_arg(checkpoint="flag", source="pos") == "flag"
    assert resolve_checkpoint_arg(source="pos") == "pos"


def test_load_model_spec_custom_fixture():
    spec = load_model_spec(
        FIXTURES / "custom_model",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
        require_code=True,
    )
    assert spec.export_block_trees
    assert spec.code_sources
    assert any("custom_model" in src or "modeling" in src for src in spec.code_sources)


def test_load_model_spec_requires_modeling_source(tmp_path: Path):
    config_dir = tmp_path / "config_only"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        '{"model_type": "nonexistent_model_xyz", "hidden_size": 128, "vocab_size": 1000}',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="No modeling source"):
        load_model_spec(config_dir, require_code=True)


def test_load_model_spec_hf_kimi():
    pytest.importorskip("huggingface_hub")
    spec = load_model_spec(
        "moonshotai/Kimi-K3",
        detailed=True,
        basic_ops=BasicOpFilter.for_detailed(),
        require_code=True,
    )
    assert spec.checkpoint_source
    assert any(src.startswith("hf://") for src in spec.code_sources)
    assert len(spec.export_block_trees) >= 4
    titles = [title for title, _ in spec.export_block_trees]
    assert any("Attn" in title for title in titles)
