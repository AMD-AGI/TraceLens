###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 index and unpad shapes on the exported graph."""

from __future__ import annotations

# --- test_kimi_k3_index_select.py ---


def test_kimi_index_first_axis_is_no_longer_an_opaque_leaf(kimi_packed_nodes):
    nodes = kimi_packed_nodes
    opaque = [n for n in nodes if n.get("label") == "Index first axis"]
    assert not opaque, [n["id"] for n in opaque]


def test_kimi_index_select_reads_base_and_indices_and_feeds_unsqueeze(
    kimi_packed_nodes,
):
    nodes = kimi_packed_nodes
    by_id = {n["id"]: n for n in nodes}
    selects = [n for n in nodes if n.get("label") == "Index select"]
    assert selects, "expected at least one Index select node"

    node = selects[0]
    # Two real tensor operands: the rearranged base and the gathered indices. The
    # ``indices`` operand now flows from the *expanded* ``get_unpad_data`` body
    # (its ``torch.nonzero(...).flatten()`` int64 ``[nnz]`` result) rather than the
    # former opaque ``Get unpad data`` tile.
    edges = {
        edge["metadata"].get("port_label"): edge
        for edge in (node.get("incomingEdges") or [])
    }
    assert by_id.get(edges["x"]["sourceNodeId"], {}).get("label") == "Rearrange"
    indices_src = by_id.get(edges["indices"]["sourceNodeId"], {})
    assert "get_unpad_data" in indices_src.get("id", "")

    # The ``.unsqueeze(0)`` chained on the call result reconnects to the op (no
    # orphaned consumer left behind by the inlining).
    consumers = [
        m.get("label")
        for m in nodes
        if any(
            edge.get("sourceNodeId") == node["id"]
            for edge in (m.get("incomingEdges") or [])
        )
    ]
    assert "Unsqueeze" in consumers


# --- test_kimi_k3_get_unpad_data.py ---


def _is_cpu(node) -> bool:
    return any(
        a.get("key") == "device" and a.get("value") == "cpu"
        for a in node.get("attrs", [])
    )


def test_kimi_get_unpad_data_is_no_longer_an_opaque_leaf(kimi_packed_nodes):
    opaque = [n for n in kimi_packed_nodes if n.get("label") == "Get unpad data"]
    assert not opaque, [n["id"] for n in opaque]


def test_kimi_get_unpad_data_ops_are_device_not_cpu(kimi_packed_nodes):
    unpad = [n for n in kimi_packed_nodes if "get_unpad_data" in n.get("id", "")]
    labels = {n.get("label") for n in unpad}
    # The device work is visible...
    assert "Nonzero" in labels
    # ...and none of the expanded ops is tagged device: cpu (the only host crossing
    # is the untraced ``.item()`` scalar read).
    cpu_ops = [n.get("label") for n in unpad if _is_cpu(n)]
    assert cpu_ops == [], cpu_ops
    # ``max_seqlen = lens.max().item()`` is a host scalar, so it has no tensor
    # consumer and is correctly pruned.


def test_kimi_unpad_publishes_its_real_return_names(kimi_packed_nodes):
    """The helper returns ``indices`` and ``cu_seqlens``, and says so.

    Only reachable in Kimi's own environment: the helper this interpreter would
    otherwise parse has no ``_prepare_cu_seqlens_from_mask`` arm at all.
    """
    unpad = [n for n in kimi_packed_nodes if "get_unpad_data" in n.get("id", "")]
    labels = {n.get("label") for n in unpad}
    assert {"indices", "cu_seqlens"} <= labels, sorted(labels)


def test_kimi_cu_seqlens_is_one_longer_than_the_batch(kimi_packed_nodes):
    """``F.pad(cumsum(lens), (1, 0))`` prepends a zero, so the extent is B + 1.

    int32, not int64: ``_get_unpad_data`` writes
    ``torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32)`` and the flash
    kernels it feeds require int32 offsets. This asserted int64 while the export
    ignored a ``dtype=`` outside a tensor constructor.
    """
    tiles = [
        n
        for n in kimi_packed_nodes
        if n.get("label") == "cu_seqlens" and "get_unpad_data" in n.get("id", "")
    ]
    assert tiles
    shapes = {
        a.get("value")
        for n in tiles
        for a in n.get("attrs", [])
        if a.get("key") == "output_shape"
    }
    assert shapes == {"[B + 1] int32"}, shapes


def test_kimi_index_select_shape_leads_with_nnz(kimi_packed_nodes):
    """With ``get_unpad_data`` expanded, ``indices`` carries an int64 ``[nnz]``
    shape, so the ``Index select`` output leads with the symbolic ``nnz`` gathered
    row count instead of the base-passthrough batch dim."""
    selects = [n for n in kimi_packed_nodes if n.get("label") == "Index select"]
    assert selects
    node = selects[0]
    meta = node.get("outputsMetadata") or []
    shape = next(
        (a.get("value") for a in meta[0].get("attrs", []) if a.get("key") == "shape"),
        "",
    )
    assert shape.startswith("[nnz"), shape


def test_no_device_cpu_nodes_remain_after_per_op_hostness(model_graph_nodes):
    """Expanded helpers do not leave a ``device: cpu`` tile behind."""
    for model_id in (
        "moonshotai/Kimi-K3",
        "zai-org/GLM-5.3-Flash",
        "MiniMaxAI/MiniMax-M3",
    ):
        nodes = model_graph_nodes(model_id)
        cpu = [n.get("label") for n in nodes if _is_cpu(n)]
        assert cpu == [], (model_id, cpu)


def test_a_rectangular_run_does_not_unpad_at_all(kimi_nodes):
    """The complement: with padding kept, none of this apparatus exists.

    ``if attention_mask is not None`` asks whether this batch was padded, and
    nothing in the config answers it. Drawing both arms made the whole
    attention body carry ``[1, nnz, H]`` -- the packed answer -- while its
    other arm carries ``[B, S, H]``, so every shape below was true of one run
    only. The default run keeps the rectangular batch, and then there is no
    padding to strip: no ``get_unpad_data``, no ``index_first_axis``, no
    ``pad_input`` putting it back, and no phi left over to disagree about it.
    """
    for fragment in ("get_unpad_data", "index_first_axis", "pad_input"):
        assert not [n for n in kimi_nodes if fragment in n.get("id", "")], fragment
    assert not [n for n in kimi_nodes if n.get("label") == "Index select"]
    assert not [n for n in kimi_nodes if n.get("label") == "Nonzero"]
    body = [
        n
        for n in kimi_nodes
        if "/self_attn/" in n.get("id", "") and n.get("label") == "Linear"
    ]
    assert body, "the attention body should still be drawn"
    for node in body:
        shape = next(
            (
                a.get("value")
                for a in node.get("attrs", [])
                if a.get("key") == "output_shape"
            ),
            "",
        )
        # Rectangular: the batch axis is the batch, not a packed single row.
        assert shape.startswith("[B, S,"), (node["id"], shape)
