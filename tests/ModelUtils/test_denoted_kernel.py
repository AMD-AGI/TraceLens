###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Following a called name through wrappers to the kernel it denotes.

A library hands its kernel over through several layers, none of which is the
kernel. Transformers' flash path is four: the config names a table KEY, the
table returns a wrapper, the wrapper calls a helper in another module, and that
helper calls a local unpacked from a tuple returned by a lazy importer, which
returns the names it imported from the kernel package. Stopping anywhere along
that chain names the library boundary rather than what the GPU runs.

So follow it: a readable function is a wrapper, and what it returns is what it
really is; a local bound out of another call is that call's return slot. The
walk ends at a name no readable source defines -- which is exactly a compiled
kernel, since those have no Python body. We never go INTO it; there is nothing
there to draw.

Nothing here knows what the kernel computes or which library it comes from.
"""

from __future__ import annotations

import ast

from TraceLens.ModelUtils.ast_analyze import (
    _chain_root_call,
    _condition_holds,
    _denoted_kernel,
    _returned_at_path,
    _target_slot_path,
)


class _Registry:
    """A stand-in parse cache over source strings, keyed by module name."""

    def __init__(self, modules: dict[str, str]):
        self._modules = modules

    def load(self, module: str):
        source = self._modules.get(module)
        if source is None:
            return None
        tree = ast.parse(source)
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        bindings = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    bindings[alias.asname or alias.name] = f"{node.module}#{alias.name}"
        return funcs, bindings


class TestFollowingTheChain:
    def test_a_wrapper_resolves_to_what_it_calls(self) -> None:
        reg = _Registry(
            {
                "lib": "def wrapper(q):\n    out = kernel(q)\n    return out\n",
            }
        )
        assert _denoted_kernel("wrapper", None, "lib", reg, {"q"}) == "kernel"

    def test_it_crosses_module_boundaries(self) -> None:
        reg = _Registry(
            {
                "front": "from back import helper\n"
                "def entry(q):\n    out = helper(q)\n    return out\n",
                "back": "def helper(q):\n    out = real_kernel(q)\n    return out\n",
            }
        )
        assert _denoted_kernel("entry", None, "front", reg, {"q"}) == "real_kernel"

    def test_a_tuple_return_hands_back_its_first_value(self) -> None:
        """``return out, None`` is one real value and an empty slot."""
        reg = _Registry(
            {"lib": "def wrapper(q):\n    out = kernel(q)\n    return out, None\n"}
        )
        assert _denoted_kernel("wrapper", None, "lib", reg, {"q"}) == "kernel"

    def test_a_local_unpacked_from_a_tuple_return(self) -> None:
        reg = _Registry(
            {
                "lib": (
                    "def imports():\n"
                    "    from pkg import fast_fn, slow_fn\n"
                    "    return fast_fn, slow_fn\n"
                    "def wrapper(q):\n"
                    "    a, b = imports()\n"
                    "    out = b(q)\n"
                    "    return out\n"
                )
            }
        )
        assert _denoted_kernel("wrapper", None, "lib", reg, {"q"}) == "slow_fn"

    def test_the_arm_is_chosen_by_what_the_caller_supplied(self) -> None:
        source = (
            "def wrapper(q, mask):\n"
            "    if mask is not None:\n"
            "        out = padded_kernel(q)\n"
            "    else:\n"
            "        out = plain_kernel(q)\n"
            "    return out\n"
        )
        reg = _Registry({"lib": source})
        assert (
            _denoted_kernel("wrapper", None, "lib", reg, {"q", "mask"})
            == "padded_kernel"
        )
        assert _denoted_kernel("wrapper", None, "lib", reg, {"q"}) == "plain_kernel"

    def test_layout_work_after_the_kernel_is_walked_back_through(self) -> None:
        """The arm ends in padding; the chain STARTS at the kernel."""
        reg = _Registry(
            {
                "lib": (
                    "def wrapper(q):\n"
                    "    raw = the_kernel(q)\n"
                    "    out = pad(raw, q)\n"
                    "    return out\n"
                )
            }
        )
        assert _denoted_kernel("wrapper", None, "lib", reg, {"q"}) == "the_kernel"


class TestWhenItDeclines:
    def test_an_arm_it_cannot_judge(self) -> None:
        """Guarded on something only the data decides -- so do not guess."""
        reg = _Registry(
            {
                "lib": (
                    "def wrapper(q):\n"
                    "    if q.is_cuda:\n"
                    "        out = a_kernel(q)\n"
                    "    return out\n"
                )
            }
        )
        assert _denoted_kernel("wrapper", None, "lib", reg, {"q"}) is None

    def test_a_function_that_returns_no_name(self) -> None:
        reg = _Registry({"lib": "def wrapper(q):\n    return 3\n"})
        assert _denoted_kernel("wrapper", None, "lib", reg, {"q"}) is None

    def test_a_module_with_no_readable_source(self) -> None:
        """Where reading runs out IS the kernel -- it has no Python body."""
        reg = _Registry({})
        assert _denoted_kernel("some_kernel", None, "absent", reg, set()) == (
            "some_kernel"
        )

    def test_a_chain_that_loops(self) -> None:
        reg = _Registry({"lib": "def a(q):\n    out = a(q)\n    return out\n"})
        assert _denoted_kernel("a", None, "lib", reg, {"q"}) in (None, "a")


class TestTheConditionRule:
    def _test(self, source):
        return ast.parse(source, mode="eval").body

    def test_is_not_none_reads_the_supplied_set(self) -> None:
        assert _condition_holds(self._test("mask is not None"), {"mask"}) is True
        assert _condition_holds(self._test("mask is not None"), set()) is False

    def test_is_none_is_its_inverse(self) -> None:
        assert _condition_holds(self._test("mask is None"), set()) is True
        assert _condition_holds(self._test("mask is None"), {"mask"}) is False

    def test_not_inverts(self) -> None:
        assert _condition_holds(self._test("not (mask is not None)"), {"mask"}) is False

    def test_anything_else_is_unknown(self) -> None:
        for source in ("x > 1", "flag", "a == b", "mask is not 0"):
            assert _condition_holds(self._test(source), {"mask"}) is None


class TestTheSmallPieces:
    def test_a_nested_tuple_target_path(self) -> None:
        target = ast.parse("(a, b, c), d = f()").body[0].targets[0]
        assert _target_slot_path(target, "b") == [0, 1]
        assert _target_slot_path(target, "d") == [1]
        assert _target_slot_path(target, "zz") is None

    def test_reading_a_return_at_a_path(self) -> None:
        func = ast.parse("def f():\n    return (p, q), r\n").body[0]
        assert _returned_at_path(func, [0, 1]) == "q"
        assert _returned_at_path(func, [1]) == "r"
        assert _returned_at_path(func, [9]) is None

    def test_the_chain_root_stops_at_a_parameter(self) -> None:
        func = ast.parse("def f(x):\n    out = k(x)\n    return out\n").body[0]
        call = func.body[0].value
        assert _chain_root_call(func, call) is call
