###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A constructor sized by a shape local names the same axis as everything else.

DeepSeek's compressor writes::

    batch, _, _ = hidden_states.shape
    ...
    compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

``batch`` IS the leading axis of ``hidden_states``, which the rest of the graph
calls ``B``. Kept as the bare local it printed as ``batch``, so the phi joining
this arm to the other compared ``[batch, 0, 512]`` against ``[B, n_windows,
512]`` and read two differing axes where there were none -- one axis under two
names, and one that is genuinely zero on this branch.

Reshape arguments already resolved shape locals this way. Constructor sizes did
not, and the fix has to be ADDITIVE: the local is kept whenever the tensor it
was read from is out of scope, because ``compressed_len`` names the extent
where an unresolved ``compressed_kv.shape[2]`` would name nothing.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import (
    _constructed_shape,
    _shape_local_reads,
)


class TestTheRecordedReads:
    def test_one_local(self) -> None:
        assert _shape_local_reads(["shape_locals: batch=hidden_states.shape[0]"]) == {
            "batch": "hidden_states.shape[0]"
        }

    def test_several(self) -> None:
        details = ["shape_locals: batch=x.shape[0], seq_len=x.shape[1]"]
        assert _shape_local_reads(details) == {
            "batch": "x.shape[0]",
            "seq_len": "x.shape[1]",
        }

    def test_none_recorded(self) -> None:
        assert _shape_local_reads(["size0: 4"]) == {}


class TestAConstructorSizedByAShapeLocal:
    DETAILS = [
        "size0: batch",
        "size1: 0",
        "size2: 512",
        "shape_locals: batch=hidden_states.shape[0]",
    ]

    def test_resolves_to_the_axis_it_measured(self) -> None:
        out = _constructed_shape(
            self.DETAILS, {}, None, {"hidden_states": ("B", "S", 4096)}
        )
        assert out == ("B", 0, 512)

    def test_keeps_the_local_when_that_tensor_is_out_of_scope(self) -> None:
        # `compressed_len = compressed_kv.shape[2]` with no `compressed_kv` in
        # scope: the local names the extent, the unresolved read names nothing.
        out = _constructed_shape(
            ["size0: compressed_len", "shape_locals: compressed_len=kv.shape[2]"],
            {},
            None,
            {"hidden_states": ("B", "S", 4096)},
        )
        assert out == ("compressed_len",)

    def test_unchanged_when_nothing_was_recorded(self) -> None:
        out = _constructed_shape(["size0: batch", "size1: 4"], {}, None, None)
        assert out == ("batch", 4)

    def test_a_local_naming_a_missing_axis_is_kept(self) -> None:
        out = _constructed_shape(
            ["size0: depth", "shape_locals: depth=x.shape[7]"],
            {},
            None,
            {"x": ("B", "S")},
        )
        assert out == ("depth",)
