###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""An attention spelled once per branch is read from the branch that runs.

``Glm5NextVisionAttention.forward`` writes its kernel twice::

    if is_flash_attention_requested(self.config):
        attn_output, _ = attention_interface(self, q, k, v, cu_seq_lens_q=..., ...)
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits  = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]
        outs    = [attention_interface(self, a, b, c, ...)[0] for a, b, c in zip(*splits)]
        out     = torch.cat(outs, dim=1)

GLM leaves ``_attn_implementation`` unset, which means sdpa, so the ELSE arm runs
-- yet the flash arm was drawn, giving ``sdpa`` a ``cu_seqlens`` operand the model
never passes it. Four things had to hold for the live arm to render:

* the statement walk must resolve the flash predicate (not only ``_config_value``);
* a comprehension over a literal tuple is a FAN-OUT -- one clone per element --
  and each clone must bind to the name it was built from, or all three splits
  bind to one name and two are dead;
* a comprehension over ``zip(*names)`` is a real loop whose body is emitted ONCE,
  reading each of those names;
* the kernel's source position -- what ranks it among the forward's ops -- must
  come from the call that RUNS, or it sorts ahead of the splits feeding it.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import (
    _clone_per_element,
    _comprehension_over_literal,
    _expand_zipped_literal_comprehension,
    _kernel_merge_source_position,
    _rebind_fanout_comprehension,
    analyze_source,
)


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


class TestFanOutComprehension:
    def test_a_literal_tuple_is_a_fan_out(self) -> None:
        parsed = _comprehension_over_literal(
            _expr("[torch.split(t, lengths, dim=2) for t in (q, k, v)]")
        )
        assert parsed is not None
        body, param, iterable = parsed
        clones = _clone_per_element(body, param, iterable)
        assert [ast.unparse(e) for e in clones.elts] == [
            "torch.split(q, lengths, dim=2)",
            "torch.split(k, lengths, dim=2)",
            "torch.split(v, lengths, dim=2)",
        ]

    def test_a_runtime_iterable_is_a_real_loop_and_is_left_alone(self) -> None:
        assert _comprehension_over_literal(_expr("[f(x) for x in xs]")) is None

    def test_a_filtered_comprehension_is_left_alone(self) -> None:
        assert _comprehension_over_literal(_expr("[f(x) for x in (a, b) if x]")) is None

    def test_each_clone_binds_to_the_name_it_was_built_from(self) -> None:
        stmt = ast.parse(
            "splits = [torch.split(t, lengths, dim=2) for t in (q, k, v)]"
        ).body[0]
        bindings: dict[str, ast.expr] = {}
        rebound = _rebind_fanout_comprehension(stmt, bindings)
        assert ast.unparse(rebound.targets[0]) == "(q, k, v)"
        assert ast.unparse(bindings["splits"]) == "(q, k, v)"

    def test_a_non_name_element_is_not_rebound(self) -> None:
        stmt = ast.parse("xs = [f(t) for t in (a.b, c)]").body[0]
        bindings: dict[str, ast.expr] = {}
        assert _rebind_fanout_comprehension(stmt, bindings) is stmt
        assert bindings == {}


class TestZippedComprehension:
    def test_the_body_is_emitted_once_reading_each_name(self) -> None:
        bindings = {"splits": _expr("(q, k, v)")}
        body = _expand_zipped_literal_comprehension(
            _expr("[attn(self, a, b, c, scaling=1)[0] for a, b, c in zip(*splits)]"),
            bindings,
        )
        assert ast.unparse(body) == "attn(self, q, k, v, scaling=1)[0]"

    def test_an_unbound_collector_is_left_alone(self) -> None:
        assert (
            _expand_zipped_literal_comprehension(
                _expr("[f(a, b) for a, b in zip(*splits)]"), {}
            )
            is None
        )

    def test_an_arity_mismatch_is_left_alone(self) -> None:
        assert (
            _expand_zipped_literal_comprehension(
                _expr("[f(a, b, c) for a, b, c in zip(*splits)]"),
                {"splits": _expr("(q, k)")},
            )
            is None
        )


_SOURCE = textwrap.dedent("""
    class M(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.qkv = nn.Linear(8, 24)
            self.config = config

        def forward(self, hidden_states, cu_seqlens, max_seqlen=None, **kwargs):
            query_states, key_states, value_states = self.qkv(hidden_states).unbind(0)
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            if is_flash_attention_requested(self.config):
                attn_output, _ = attention_interface(
                    self, query_states, key_states, value_states,
                    cu_seq_lens_q=cu_seqlens, **kwargs,
                )
            else:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                splits = [
                    torch.split(tensor, lengths.tolist(), dim=2)
                    for tensor in (query_states, key_states, value_states)
                ]
                attn_outputs = [
                    attention_interface(self, q, k, v, is_causal=False)[0]
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(attn_outputs, dim=1)
            return attn_output
    """)


class TestTheLiveArmIsWhatRenders:
    def test_the_kernel_reads_the_splits_not_the_flash_operands(self) -> None:
        analysis = analyze_source(_SOURCE, config={"hidden_size": 8})
        structure = analysis.class_registry["M"]
        preds = structure.forward_step_predecessors.get("@attention")
        assert preds, structure.forward_step_predecessors
        assert all("split" in attr for attr in preds), preds
        # The flash arm's packed-attention argument is not a kernel operand here.
        assert "cu_seqlens" not in dict(structure.attention_inputs)

    def test_the_kernel_is_ordered_after_the_splits_that_feed_it(self) -> None:
        analysis = analyze_source(_SOURCE, config={"hidden_size": 8})
        calls = list(analysis.class_registry["M"].forward_calls)
        splits = [i for i, call in enumerate(calls) if "split" in call]
        assert splits, calls
        assert calls.index("@attention") > max(splits), calls

    def test_the_concat_reads_the_kernel(self) -> None:
        analysis = analyze_source(_SOURCE, config={"hidden_size": 8})
        structure = analysis.class_registry["M"]
        concat = next(
            op for op in structure.forward_operations.values() if op.label == "Concat"
        )
        assert "@attention" in concat.predecessors, concat.predecessors


class TestKernelSourcePosition:
    def test_the_rank_comes_from_the_live_arm(self) -> None:
        """Both arms spell the kernel, and which one ranks it is the live one."""
        module = ast.parse(_SOURCE)
        func = next(
            node
            for cls in module.body
            if isinstance(cls, ast.ClassDef)
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        # Unconfigured means sdpa, so the ELSE arm runs and the kernel ranks at
        # the comprehension -- after the splits that feed it.
        sdpa_rank = _kernel_merge_source_position(func, {})
        # Configure flash and the FLASH arm runs, which is earlier in the source.
        flash_rank = _kernel_merge_source_position(
            func, {"_attn_implementation": "flash_attention_2"}
        )
        assert sdpa_rank is not None and flash_rank is not None
        assert flash_rank < sdpa_rank, (flash_rank, sdpa_rank)
