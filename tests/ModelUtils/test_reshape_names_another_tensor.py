###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A reshape argument that reads another tensor's axis, and who may answer it.

An ``@op_l`` id in this file is a sample key this test builds to exercise id
parsing. It is not a lookup into an upstream modeling file.


MiniMax's indexer and DeepSeek's compressor both open on a tuple unpack::

    batch, q_len, _ = hidden_states.shape
    ...
    scores.view(batch, self.num_heads, q_len, num_key_blocks, self.block_size)

The extractor substitutes the local back to what bound it, so the reshape
declares ``hidden_states.shape[0]`` and ``hidden_states.shape[1]``. Neither
names the tensor being reshaped -- ``scores`` is ``[B, H, S, S]`` and
``hidden_states`` is ``[B, S, hidden]`` -- so the axes have to come from the
tensor the model actually named, and when nothing in scope knows that tensor the
read cannot be answered at all. Printing an unresolved ``x.shape[0]`` as a
dimension states one tensor's extent as another's.

Bailing on the whole reshape is not the safe answer either: it kept the
OPERAND's rank, and the rank-4 ``[..., self.head_dim:]`` slices below then
addressed an axis that did not exist. That is the defect ``resize_dim`` /
declared-rank checks caught, three steps downstream of where it happened.

The second half of this is identity: which id is a FRAME. A step that is a bare
operation names itself on both sides
(``seq:25:@op_l580_c17_view:@op_l580_c17_view:0``), so cutting the id at its op
segment leaves the op's own id standing where a frame's name belongs. Treating
that as a frame published the op's own input under the enclosing class's
forward-parameter name -- which is how the indexer came to be told that
``hidden_states`` was the attention scores it was reshaping, and the one wrong
answer is worse than no answer because it looks resolved.
"""

from __future__ import annotations

from TraceLens.ModelUtils.shape_inference import ShapeInferencer, _snapshot_axis


def _axis(token, *, named=None, snapshots=()):
    return _snapshot_axis(token, frozenset(snapshots), named or {})


class TestTheAxisComesFromTheTensorTheModelNamed:
    def test_a_snapshot_read_of_a_known_tensor(self) -> None:
        assert (
            _axis(
                "hidden_states.shape[0]",
                named={"hidden_states": ("B", "S", 6144)},
                snapshots=("hidden_states.shape[0]",),
            )
            == "B"
        )

    def test_the_second_axis_of_the_same_unpack(self) -> None:
        assert (
            _axis(
                "hidden_states.shape[1]",
                named={"hidden_states": ("B", "S", 6144)},
                snapshots=("hidden_states.shape[1]",),
            )
            == "S"
        )

    def test_an_axis_counted_from_the_end(self) -> None:
        assert (
            _axis(
                "x.shape[-1]",
                named={"x": ("B", "S", 768)},
                snapshots=("x.shape[-1]",),
            )
            == 768
        )

    def test_a_dotted_tensor_name(self) -> None:
        assert (
            _axis(
                "self.cache.shape[0]",
                named={"self.cache": (4, 8)},
                snapshots=("self.cache.shape[0]",),
            )
            == 4
        )


class TestWhenItMustDecline:
    def test_an_inline_read_is_not_a_snapshot(self) -> None:
        """Written in the reshape itself, it reads the local as it is NOW."""
        assert _axis("x.shape[0]", named={"x": ("B", "S")}) is None

    def test_a_tensor_nothing_in_scope_knows(self) -> None:
        assert (
            _axis("q.shape[0]", named={"x": ("B",)}, snapshots=("q.shape[0]",)) is None
        )

    def test_an_axis_the_named_tensor_does_not_have(self) -> None:
        assert (
            _axis("x.shape[3]", named={"x": ("B", "S")}, snapshots=("x.shape[3]",))
            is None
        )

    def test_a_token_that_is_not_a_shape_read(self) -> None:
        assert _axis("num_key_blocks", named={}, snapshots=("num_key_blocks",)) is None

    def test_a_slice_rather_than_one_axis(self) -> None:
        assert (
            _axis("x.shape[:2]", named={"x": ("B", "S")}, snapshots=("x.shape[:2]",))
            is None
        )

    def test_an_empty_named_shape(self) -> None:
        assert _axis("x.shape[0]", named={"x": ()}, snapshots=("x.shape[0]",)) is None


class TestWhichIdIsAFrame:
    def test_a_module_step_is_a_frame(self) -> None:
        """GPT-2's ``Conv1D``: the ops inside it share ``seq:0:c_attn``."""
        assert (
            ShapeInferencer._frame_prefix("seq:0:c_attn:@op_l119_c35_view:0")
            == "seq:0:c_attn"
        )

    def test_a_free_function_frame(self) -> None:
        assert (
            ShapeInferencer._frame_prefix(
                "self_attn/seq:17:@attention:@fn_l350_repeat_kv::@op_l336_c20_expand:1"
            )
            == "self_attn/seq:17:@attention:@fn_l350_repeat_kv:"
        )

    def test_a_frame_nested_in_another_frame(self) -> None:
        assert (
            ShapeInferencer._frame_prefix(
                "@fn_l763_create_causal_mask:@fn_l860_find_packed_sequence_indices:"
                ":@op_l750_c24_slice"
            )
            == "@fn_l763_create_causal_mask:@fn_l860_find_packed_sequence_indices:"
        )

    def test_a_bare_operation_step_is_not_a_frame(self) -> None:
        """It names itself on both sides; there is no frame around it."""
        assert (
            ShapeInferencer._frame_prefix(
                "seq:25:@op_l580_c17_view:@op_l580_c17_view:0"
            )
            == ""
        )

    def test_a_constant_hanging_off_a_bare_operation_step(self) -> None:
        assert (
            ShapeInferencer._frame_prefix(
                "seq:2:@op_l157_c33_cast:@op_l157_c33_cast:0:const:weight"
            )
            == ""
        )

    def test_a_frames_own_terminal_node_still_finds_its_frame(self) -> None:
        """The return of a frame carries the frame's name, not an ``@op_``.

        It lands on the same prefix its own inner ops recorded, which is what
        lets it read the shape the frame was handed.
        """
        assert (
            ShapeInferencer._frame_prefix("seq:17:@attention:@fn_l350_repeat_kv:2")
            == ShapeInferencer._frame_prefix(
                "seq:17:@attention:@fn_l350_repeat_kv::@op_l336_c20_expand:1"
            )
            == "seq:17:@attention:@fn_l350_repeat_kv:"
        )

    def test_an_id_with_no_operation_segment(self) -> None:
        assert ShapeInferencer._frame_prefix("seq:3:q_proj:q_proj:0") == ""
