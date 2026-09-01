###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for fact sheet source URL formatting."""

from __future__ import annotations

import pytest

from model_explorer_export.fact_sheet import (
    build_fact_sheet_viewer,
    checkpoint_source_url,
    github_source_url,
)
from visualizer.extract import ArchitectureSpec


def test_checkpoint_source_url_maps_hf_scheme_to_huggingface():
    assert checkpoint_source_url("hf://moonshotai/Kimi-K3/config.json") == (
        "https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json"
    )


def test_github_source_url_maps_display_scheme_to_github():
    assert github_source_url("github://acme/custom@main/modeling_custom.py") == (
        "https://github.com/acme/custom/blob/main/modeling_custom.py"
    )
    assert github_source_url("github://acme/custom@main") == (
        "https://github.com/acme/custom/tree/main"
    )


def test_build_fact_sheet_viewer_uses_https_links_in_html():
    spec = ArchitectureSpec(
        name="Test",
        model_type="test",
        checkpoint_source="hf://org/model/config.json",
        github_source="github://org/repo@v1/model.py",
    )
    viewer = build_fact_sheet_viewer(spec)

    assert "hf://" not in viewer["body"]
    assert "github://" not in viewer["body"]
    assert "https://huggingface.co/org/model/blob/main/config.json" in viewer["body"]
    assert "https://github.com/org/repo/blob/v1/model.py" in viewer["body"]
    assert (
        'href="https://huggingface.co/org/model/blob/main/config.json"'
        in viewer["bodyHtml"]
    )
    assert 'href="https://github.com/org/repo/blob/v1/model.py"' in viewer["bodyHtml"]


def test_build_fact_sheet_omits_raw_forward_op_ids():
    pytest.importorskip("huggingface_hub")
    from visualizer.loader import load_model_spec

    spec = load_model_spec("zai-org/GLM-5.3-Flash", detailed=True)
    viewer = build_fact_sheet_viewer(spec)

    assert "@op_" not in viewer["body"]
    assert "@op_" not in viewer["bodyHtml"]
    forward_line = next(
        line for line in viewer["body"].splitlines() if line.startswith("- Forward:")
    )
    assert "MatMul" in forward_line
    assert "Multiply" in forward_line
