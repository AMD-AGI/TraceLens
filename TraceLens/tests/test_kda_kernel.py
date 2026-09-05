###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for kernel pipeline introspection."""

from __future__ import annotations

from pathlib import Path

import pytest

from TraceLens.ModelUtils.ast_analyze import (
    SYNTHETIC_ATTENTION,
    analyze_source,
    kernel_kwarg_ports,
    tensor_input_label_order,
)
from TraceLens.ModelUtils.kernel_pipeline import (
    introspect_kernel_op_substeps,
    introspect_kernel_pipeline,
    parse_kernel_call_flags,
    parse_kernel_import,
)

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "kda_kernel"
_KIMI_CODE_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py"
)


def _load_chunk_kda_pipeline():
    from TraceLens.ModelUtils.basic_ops import BasicOpFilter
    from TraceLens.ModelUtils.block_tree import build_block_node

    if not _KIMI_CODE_PATH.exists():
        pytest.skip("Kimi-K3 modeling file not cached locally")

    basic = BasicOpFilter.from_cli(add=[r"(?i)^Linear$", r"(?i)^RMSNorm$"])
    analysis = analyze_source(
        _KIMI_CODE_PATH.read_text(), filename="modeling_kimi_linear.py"
    )
    attn = build_block_node(
        attr_name="self_attn",
        class_name="KimiDeltaAttention",
        registry=analysis.class_registry,
        basic_ops=basic,
    )
    pipeline = next(
        child for child in attn.children if child.class_name == "KernelPipeline"
    )
    return pipeline, basic


@pytest.fixture(autouse=True)
def _kernel_fixture_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACELENS_KERNEL_FIXTURE_ROOT", str(_FIXTURE_ROOT))


def test_kernel_search_root_expands_kernel_module_beside_modeling_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import TraceLens.ModelUtils.kernel_pipeline as kernel_pipeline

    (tmp_path / "model.py").write_text(
        "from sibling_kernel import sibling_attn\n", encoding="utf-8"
    )
    (tmp_path / "sibling_kernel.py").write_text(
        "import torch\n"
        "\n"
        "def build_attn_kernel(h, d):\n"
        "    return lambda *args: None\n"
        "\n"
        "def sibling_attn(q, kv):\n"
        "    b, s, h, d = q.size()\n"
        "    o = torch.empty_like(q)\n"
        "    kernel = build_attn_kernel(q.size(2), d)\n"
        "    kernel(q, kv, o)\n"
        "    o = o.narrow(2, 0, h).contiguous()\n"
        "    return o\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kernel_pipeline, "_KERNEL_SEARCH_ROOTS", [])
    monkeypatch.setattr(kernel_pipeline, "_KERNEL_FIXTURE_ROOT", None)

    details = ["kernel: sibling_attn", "import: sibling_kernel#sibling_attn"]
    # A kernel module on no import path reads as opaque until its directory is known.
    assert introspect_kernel_pipeline(details) == ([], [])

    kernel_pipeline.register_kernel_search_root(tmp_path / "model.py")
    pipeline_steps, _output_steps = introspect_kernel_pipeline(details)
    labels = [step.call_name for step in pipeline_steps]

    assert (
        "build_attn_kernel" in labels
    ), "the compiled kernel call takes its builder's name"
    assert "contiguous" in labels
    assert "size" not in labels, "shape queries are not pipeline stages"


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
    details = analysis.class_registry["KimiDeltaAttention"].forward_step_details[
        SYNTHETIC_ATTENTION
    ]
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
    from TraceLens.ModelUtils.kernel_pipeline import (
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


def test_introspect_l2norm_substeps_use_real_second_operands():
    from TraceLens.ModelUtils.kernel_pipeline import (
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
        "l2norm_fwd",
        merged_imports,
        owning_module,
        parent_attr="forward_l2norm_fwd_q",
    )
    assert [(step.label, step.second_operand) for step in substeps] == [
        ("Sum", None),
        ("Sqrt", None),
        ("÷", None),
        ("×", "input"),
    ]


def test_l2norm_inv_sqrt_wires_only_real_operands():
    from TraceLens.ModelUtils.computation_graph import build_computation_graph

    pipeline, _ = _load_chunk_kda_pipeline()
    graph = build_computation_graph(pipeline)
    for frame in graph.inline_frames:
        if "l2norm_fwd" not in frame.frame_id:
            continue
        members = set(frame.node_indices)
        inv_sqrt = next(i for i in members if graph.nodes[i].label == "÷")
        normalize = next(i for i in members if graph.nodes[i].label == "×")
        inv_sources = [
            graph.nodes[src].label for src, tgt in graph.links if tgt == inv_sqrt
        ]
        norm_sources = {
            graph.nodes[src].label for src, tgt in graph.links if tgt == normalize
        }
        assert inv_sources == ["Sqrt"]
        assert "÷" in norm_sources
        assert any(label in norm_sources for label in {"q", "k"})
    from TraceLens.ModelUtils.kernel_pipeline import (
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
        "kda_gate_chunk_cumsum",
        merged_imports,
        owning_module,
        parent_attr="gate_step",
    )
    multiply = next(step for step in substeps if step.label == "×")
    assert multiply.second_operand == "gate_step_sub_0"


def test_introspect_kernel_pipeline_expands_helper_kernels():
    details = [
        "kernel: chunk_kda",
        "import: fla.ops.kda#chunk_kda",
        "kwarg: use_qk_l2norm_in_kernel=True",
        "kwarg: use_beta_sigmoid_in_kernel=True",
        "kwarg: use_gate_in_kernel=True",
    ]
    pipeline_steps, _ = introspect_kernel_pipeline(details)
    beta = next(
        step for step in pipeline_steps if step.call_name == "fused_beta_sigmoid"
    )
    assert [child.label for child in beta.children] == ["Sigmoid", "× scale"]
    l2norm = next(step for step in pipeline_steps if step.call_name == "l2norm_fwd")
    assert "Sum" in [child.label for child in l2norm.children]
    gate = next(
        step for step in pipeline_steps if step.call_name == "kda_gate_chunk_cumsum"
    )
    assert "CumSum" in [child.label for child in gate.children]
