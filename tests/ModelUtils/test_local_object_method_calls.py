###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A method called on a LOCAL object is read like any other.

DeepSeek's compressor and indexer run their KV cache's methods::

    chunk_kv, chunk_gate, pos = cache_layer.store_compression_weights("c", kv, gate)
    prior_kv, prior_gate = cache_layer.update_overlap_state("c", chunk_kv, chunk_gate, d)

Both are defined in the model's own source (``DeepseekV4HCACache`` /
``DeepseekV4CSACache``), so they are fully introspectable -- but the receiver is
not ``self``, so none of the submodule paths saw them. The calls vanished, their
results had no producer, and whatever read them fell back on the op that
happened to precede it: ``new_kv[:, 0, :ratio] = prior_kv.to(...)`` was drawn
reading the GATE CHUNK, which the model only ever passes INTO the cache, and
every phi over a cached value took the projection as its arm.

Four things have to hold together, and each was a separate layer:

* the call is recognised -- the method name resolves when exactly one class in
  this file defines it, which is a fact about the parsed source, not a list;
* the ECHO is excluded. ``key_states, value_states = past_key_values.update(
  key_states, value_states, ...)`` hands back the names it was given, which
  changes nothing drawable and is already modelled as a passthrough. Expanding
  it draws the cache's own bookkeeping with nothing reading it;
* the producer is recorded in the branch env too (``_call_step_producer``),
  or the phi over a variable assigned in both arms still takes whichever
  producer came last;
* what the method RETURNS is published like a forward's returns, or the
  caller's unpack binds to nothing and every op behind a returned value is
  left dead.
"""

from __future__ import annotations

import ast
import textwrap

from TraceLens.ModelUtils.ast_analyze import (
    LocalMethod,
    _forward_operations_from_forward,
    _local_method_call_sites,
    _uniquely_defined_methods,
    analyze_sources,
    named_method_return_slots,
)

_SOURCE = textwrap.dedent("""
    class Cache:
        def store_weights(self, name: str, kv, gate):
            buffered = self.buffer[name]
            self.buffer[name] = kv[:, 8:].clone()
            return kv[:, :8], gate[:, :8]

        def only_here(self, x):
            return x

    class Other:
        def only_here(self, x):
            return x

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.kv_proj = nn.Linear(8, 8)

        def forward(self, hidden_states, cache_layer, past_key_values):
            kv = self.kv_proj(hidden_states)
            gate = self.kv_proj(hidden_states)
            chunk_kv, chunk_gate = cache_layer.store_weights("c", kv, gate)
            kv, gate = past_key_values.update(kv, gate)
            return chunk_kv
""")


def _tree():
    return ast.parse(_SOURCE)


def _forward():
    block = next(
        n for n in _tree().body if isinstance(n, ast.ClassDef) and n.name == "Block"
    )
    return next(
        n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )


class TestWhichMethodsResolve:
    def test_a_name_only_one_class_defines(self) -> None:
        assert _uniquely_defined_methods(_tree())["store_weights"].owner == "Cache"

    def test_it_carries_the_methods_own_parameter_names(self) -> None:
        """The call binds its arguments to THESE names, not the caller's locals.

        Without them the producers are recorded positionally but mapped to no
        parameter, so the expansion's ``@method_input`` resolves by chain
        position -- the op emitted last before the call.
        """
        resolved = _uniquely_defined_methods(_tree())["store_weights"]
        assert resolved.params == ("name", "kv", "gate")

    def test_a_name_two_classes_define_resolves_to_neither(self) -> None:
        assert "only_here" not in _uniquely_defined_methods(_tree())

    def test_forward_and_dunders_are_left_to_the_paths_that_have_them(self) -> None:
        owners = _uniquely_defined_methods(_tree())
        assert "forward" not in owners and "__init__" not in owners


class TestWhichCallSitesAreTaken:
    def test_a_call_returning_new_names(self) -> None:
        sites = _local_method_call_sites(_forward(), _uniquely_defined_methods(_tree()))
        assert [resolved.owner for resolved in sites.values()] == ["Cache"]
        assert [receiver for receiver, _line in sites] == ["cache_layer"]

    def test_the_echo_is_left_alone(self) -> None:
        """``kv, gate = past_key_values.update(kv, ...)`` changes nothing drawn."""
        owners = dict(_uniquely_defined_methods(_tree()))
        owners["update"] = LocalMethod("Cache", ("kv", "gate"))
        sites = _local_method_call_sites(_forward(), owners)
        assert all(receiver != "past_key_values" for receiver, _line in sites)

    def test_a_method_no_class_here_defines(self) -> None:
        assert _local_method_call_sites(_forward(), {}) == {}


class TestWhatTheMethodHandsBack:
    def test_its_return_slots_are_published(self) -> None:
        analysis = analyze_sources({"m.py": _SOURCE})
        slots, order = named_method_return_slots(
            analysis.class_registry["Cache"], "store_weights"
        )
        assert order == ["kv", "gate"], (slots, order)
        assert set(slots) == {"kv", "gate"}

    def test_a_method_that_is_not_there(self) -> None:
        analysis = analyze_sources({"m.py": _SOURCE})
        assert named_method_return_slots(
            analysis.class_registry["Cache"], "absent"
        ) == ({}, [])


class TestTheCallIsCaptured:
    def test_it_becomes_a_forward_step_naming_its_class(self) -> None:
        analysis = analyze_sources({"m.py": _SOURCE})
        block = analysis.class_registry["Block"]
        calls = [call for call in block.forward_calls if "cache_layer" in call]
        assert len(calls) == 1, block.forward_calls
        details = block.forward_step_details[calls[0]]
        assert "method: store_weights" in details
        assert "method_class: Cache" in details

    def test_the_echoing_cache_update_is_not_a_step(self) -> None:
        analysis = analyze_sources({"m.py": _SOURCE})
        block = analysis.class_registry["Block"]
        assert not [c for c in block.forward_calls if "past_key_values" in c]


def test_a_store_into_persistent_state_is_recorded() -> None:
    """``self.buffer[name] = ...`` publishes; the op behind it is not dead."""
    from TraceLens.ModelUtils.ast_analyze import named_method_operations

    analysis = analyze_sources({"m.py": _SOURCE})
    ops = named_method_operations(analysis.class_registry["Cache"], "store_weights")
    stored = [
        detail
        for op in ops
        for detail in op.details
        if str(detail).startswith("stores:")
    ]
    assert stored == ["stores: buffer"], [op.details for op in ops]


class TestWhatTheCallHandsIn:
    def test_each_argument_binds_to_the_methods_own_parameter(self) -> None:
        """The producers alone are positional; the NAMES are what the frame docks on.

        Recorded positionally and mapped to nothing, the expansion's
        ``@method_input`` resolves by chain position -- the op emitted last
        before the call -- so DeepSeek's cache frame read the hidden state
        instead of the ``kv`` the call passes it.
        """
        caller = _forward()
        owners = _uniquely_defined_methods(_tree())
        analysis = _forward_operations_from_forward(
            caller, self_values={}, all_tensor_ops=True, local_object_methods=owners
        )
        step = next(k for k in analysis.step_predecessor_args if "cache_layer" in k)
        bound = analysis.step_predecessor_args[step]
        # The fixture calls ``self.kv_proj`` twice, so each call site carries its
        # own suffix -- and the two arguments bind to DIFFERENT ones, which is
        # the whole point: positionally they would collapse onto one producer.
        assert sorted(bound) == ["gate", "kv"]
        assert bound["kv"] != bound["gate"]
        assert all(value.startswith("kv_proj") for value in bound.values())

    def test_a_string_literal_argument_contributes_no_producer(self) -> None:
        """``"c"`` names the cache entry; it is not a tensor the frame reads."""
        caller = _forward()
        owners = _uniquely_defined_methods(_tree())
        analysis = _forward_operations_from_forward(
            caller, self_values={}, all_tensor_ops=True, local_object_methods=owners
        )
        step = next(k for k in analysis.step_predecessor_args if "cache_layer" in k)
        assert "name" not in analysis.step_predecessor_args[step]
