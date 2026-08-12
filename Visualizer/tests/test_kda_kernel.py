"""Tests for kernel pipeline introspection."""

from __future__ import annotations

from pathlib import Path

import pytest

from visualizer.ast_analyze import (
    SYNTHETIC_ATTENTION,
    analyze_source,
    kernel_kwarg_ports,
    tensor_input_label_order,
)
from visualizer.kernel_pipeline import (
    introspect_kernel_op_substeps,
    introspect_kernel_pipeline,
    parse_kernel_call_flags,
    parse_kernel_import,
)

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "kda_kernel"


@pytest.fixture(autouse=True)
def _kernel_fixture_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACELENS_KERNEL_FIXTURE_ROOT", str(_FIXTURE_ROOT))


def test_parse_kernel_call_flags_reads_modeling_kwargs():
    details = [
        "kernel: chunk_kda",
        "kwarg: use_qk_l2norm_in_kernel=True",
        "kwarg: use_gate_in_kernel=True",
        "kwarg: use_beta_sigmoid_in_kernel=True",
        "kwarg: A_log=self.A_log",
    ]
    flags = parse_kernel_call_flags(details)
    assert flags["_kernel"] == "chunk_kda"
    assert flags["use_qk_l2norm_in_kernel"] is True
    assert flags["use_gate_in_kernel"] is True
    assert flags["use_beta_sigmoid_in_kernel"] is True
    assert flags["A_log"] == "self.A_log"


def test_analyze_source_attaches_kernel_import_for_kda():
    code_path = (
        Path.home()
        / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
    )
    if not code_path.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    analysis = analyze_source(code_path.read_text(), filename="modeling_kimi_linear.py")
    assert analysis.external_imports["chunk_kda"] == "fla.ops.kda#chunk_kda"
    details = analysis.class_registry["KimiDeltaAttention"].forward_step_details[SYNTHETIC_ATTENTION]
    assert parse_kernel_import(details) == ("fla.ops.kda", "chunk_kda")


def test_kernel_kwarg_ports_read_modeling_kwargs():
    details = [
        "kernel: chunk_kda",
        "kwarg: q=q",
        "kwarg: k=k",
        "kwarg: v=v",
        "kwarg: g=g",
        "kwarg: beta=beta",
    ]
    assert kernel_kwarg_ports(details) == {
        "q": "q",
        "k": "k",
        "v": "v",
        "g": "g",
        "beta": "beta",
    }
    ordered = tensor_input_label_order(
        details,
        {
            "q": ["q_proj"],
            "k": ["k_proj"],
            "v": ["v_proj"],
            "g": ["f_a_proj"],
            "beta": ["b_proj"],
        },
    )
    assert ordered == ["q", "k", "v", "g", "beta"]


def test_introspect_kernel_pipeline_discovers_steps_from_imported_kernel():
    details = [
        "kernel: chunk_kda",
        "import: fla.ops.kda#chunk_kda",
        "kwarg: use_qk_l2norm_in_kernel=True",
        "kwarg: use_gate_in_kernel=True",
        "kwarg: use_beta_sigmoid_in_kernel=True",
    ]
    pipeline_steps, output_steps = introspect_kernel_pipeline(details)
    labels = [step.label for step in pipeline_steps]
    assert labels.count("l2norm_fwd") == 2
    assert any("l2norm_fwd" in step.computation for step in pipeline_steps)
    assert any("chunk_gla_fwd_o_gk" in step.computation for step in output_steps)
    gated = next(step for step in pipeline_steps if "gated" in step.call_name.lower())
    assert gated.call_name == "chunk_gated_delta_rule_fwd_h"


def test_introspect_kernel_op_substeps_expand_fused_beta_sigmoid():
    from visualizer.kernel_pipeline import (
        _collect_import_map,
        _discover_pipeline_entrypoints,
        _find_symbol_definition,
        _parse_module,
    )

    details = [
        "kernel: chunk_kda",
        "import: fla.ops.kda#chunk_kda",
    ]
    import_ref = parse_kernel_import(details)
    assert import_ref is not None
    module, symbol = import_ref
    entrypoints, _ = _discover_pipeline_entrypoints(module, symbol)
    merged_imports = {}
    owning_module = module
    for source, _ in entrypoints:
        tree = _parse_module(source)
        resolved = _find_symbol_definition(module, symbol)
        entry_module = resolved[2] if resolved is not None else module
        merged_imports.update(_collect_import_map(tree, entry_module))
        owning_module = entry_module

    substeps = introspect_kernel_op_substeps(
        "fused_beta_sigmoid",
        merged_imports,
        owning_module,
        parent_attr="beta_step",
    )
    assert [step.label for step in substeps] == ["Sigmoid", "× scale"]


def test_introspect_kernel_pipeline_expands_helper_kernels():
    details = [
        "kernel: chunk_kda",
        "import: fla.ops.kda#chunk_kda",
        "kwarg: use_qk_l2norm_in_kernel=True",
        "kwarg: use_beta_sigmoid_in_kernel=True",
        "kwarg: use_gate_in_kernel=True",
    ]
    pipeline_steps, _ = introspect_kernel_pipeline(details)
    beta = next(step for step in pipeline_steps if step.call_name == "fused_beta_sigmoid")
    assert [child.label for child in beta.children] == ["Sigmoid", "× scale"]
    l2norm = next(step for step in pipeline_steps if step.call_name == "l2norm_fwd")
    assert "Sum" in [child.label for child in l2norm.children]
    gate = next(step for step in pipeline_steps if step.call_name == "kda_gate_chunk_cumsum")
    assert "CumSum" in [child.label for child in gate.children]
