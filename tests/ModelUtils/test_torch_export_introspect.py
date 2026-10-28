###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``--input-spec`` tracing with ``torch.export`` on a small multi-input module."""

import re

import pytest

torch = pytest.importorskip("torch")

from TraceLens.ModelUtils.torch_introspect import (
    InputSpec,
    export_module_inventory,
    symbolic_example_sizes,
)

# mask first: torch.export reads S off whichever input it likes; the source reads ids.
SPECS = [InputSpec("mask", ("B", "S")), InputSpec("ids", ("B", "S"), "int64")]
MAX_POSITIONS = 64


class _TwoInputBlock(torch.nn.Module):
    """Shape-dependent control flow, a no_grad region, an in-place write, and
    size reads of an input and of an intermediate tensor."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 8)
        self.proj = torch.nn.Linear(8, 8)
        self.register_buffer("table", torch.ones(MAX_POSITIONS, 8))

    def forward(self, ids, mask):
        seq = ids.shape[1]
        if seq > self.table.shape[0]:
            raise ValueError("sequence longer than the position table")
        x = self.emb(ids) + self.table[:seq]
        with torch.no_grad():
            pos = torch.arange(seq).float() * self.proj.weight[0, 0]
        y = self.proj(x + pos[None, :, None]) * mask[..., None]
        y = y.clone()
        y[..., 0] = 0.0
        return self._per_row(y).sum(-1)

    def _per_row(self, h):
        rows, _, _ = h.shape
        return h * torch.arange(rows)[:, None, None]


@pytest.fixture(scope="module")
def inventory():
    return export_module_inventory(_TwoInputBlock().eval(), SPECS)


def test_fx_cannot_trace_the_module():
    with pytest.raises(torch.fx.proxy.TraceError):
        torch.fx.symbolic_trace(_TwoInputBlock())


def test_shapes_stay_symbolic(inventory):
    by_name = {info.name: info for info in inventory}
    assert by_name["ids"].shape == ("B", "S")
    assert by_name["ids"].dtype == "int64"
    assert next(i for i in inventory if i.op == "output").shape == ("B", "S")
    shapes = [info.shape for info in inventory if info.shape is not None]
    assert ("B", "S", 8) in shapes
    example_sizes = set(symbolic_example_sizes(SPECS).values())
    leaked = [s for s in shapes if example_sizes & set(s)]
    assert not leaked, leaked
    assert all(
        isinstance(dim, int) or re.fullmatch(r"[BS0-9*+\-]+", dim)
        for shape in shapes
        for dim in shape
    )


def test_every_op_is_visible_and_consumed(inventory):
    consumed = {name for info in inventory for name in info.inputs}
    dead = [i.name for i in inventory if i.op != "output" and i.name not in consumed]
    assert not dead, dead
    # A get_attr here would reference a nested graph: ops hidden in one node.
    nested = [info.name for info in inventory if info.op == "get_attr"]
    assert not nested, nested
    labels = {info.label for info in inventory}
    assert {"embedding", "linear", "arange", "sum"} <= labels


def test_size_read_docks_on_the_tensor_the_source_reads(inventory):
    positions, rows = [info for info in inventory if info.label == "arange"]
    assert positions.inputs == ["ids"]
    by_name = {info.name: info for info in inventory}
    # y as _per_row received it: after the in-place y[..., 0] = 0.0
    assert by_name[rows.inputs[0]].label == "select_scatter"


def test_weight_and_literal_reads_are_constant(inventory):
    from TraceLens.ModelUtils.torch_introspect import (
        computation_graph_from_fx_inventory,
    )

    graph = computation_graph_from_fx_inventory(inventory)
    constant = {node.label for node in graph.nodes if node.constant}
    assert "select" in constant  # self.proj.weight[0, 0]
    assert not {"arange", "embedding", "linear"} & constant
