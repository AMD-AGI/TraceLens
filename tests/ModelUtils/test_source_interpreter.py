###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The source interpreter runs a package's own code, sandboxed."""

import sys
import textwrap

import pytest

from TraceLens.ModelUtils.source_interpreter import (
    Inert,
    InertModule,
    SourceInterpreter,
)


def _package(tmp_path, files):
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    return tmp_path


def _run(tmp_path, files, module="pkg.mod"):
    root = _package(tmp_path, files)
    interpreter = SourceInterpreter({"pkg": root})
    with interpreter:
        return interpreter, interpreter.import_module(module)


def test_a_comparison_hands_back_its_value_and_only_a_chain_tests_truth(tmp_path):
    class Elementwise:
        """Compares to an array, whose truth is ambiguous, like a tensor."""

        def __ne__(self, other):
            return self

        def __lt__(self, other):
            return True

        def __bool__(self):
            raise RuntimeError("ambiguous")

    _, module = _run(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                def single(x):
                    return x != -1

                def chain(a, b, c):
                    return a < b < c
            """,
        },
    )
    value = Elementwise()
    assert module.single(value) is value
    assert module.chain(1, 2, 3) is True
    assert module.chain(3, 2, 1) is False


def test_an_inert_stdlib_child_never_replaces_the_hosts_submodule(tmp_path):
    import urllib.request  # noqa: F401 - the host's own submodule

    host_request = sys.modules["urllib.request"]
    interpreter, module = _run(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                import urllib.request
                seen = urllib.request
            """,
        },
    )
    assert isinstance(module.seen, InertModule)
    assert urllib.request is host_request


def test_a_class_decorated_by_an_inert_factory_stays_a_class(tmp_path):
    _, module = _run(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                import missing_native as native

                @native.register(1)
                class Kept:
                    value = 3

                plain = native.dumps(Kept)
            """,
        },
    )
    assert isinstance(module.Kept, type) and module.Kept.value == 3
    # A plain call on a class is not a decorator: it stays a stand-in.
    assert isinstance(module.plain, Inert)


def test_names_a_star_import_of_an_inert_module_may_bring_are_its_stand_ins(tmp_path):
    _, module = _run(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/lib.py": """
                from missing_native import *
                __all__ = ["Flags"]
                used = Flags.FAST
            """,
            "pkg/mod.py": """
                from pkg.lib import *
                from pkg.lib import Flags as Named
            """,
        },
    )
    assert isinstance(module.Flags, Inert)
    assert isinstance(module.Named, Inert)


def test_a_fetched_library_that_cannot_load_is_a_recorded_stand_in(tmp_path):
    library = _package(
        tmp_path / "lib",
        {"broken/__init__.py": "raise AssertionError('needs a built runtime')\n"},
    )
    root = _package(
        tmp_path / "src", {"pkg/__init__.py": "", "pkg/mod.py": "import broken\n"}
    )
    interpreter = SourceInterpreter({"pkg": root})
    interpreter.resolver = lambda top: library if top == "broken" else None
    with interpreter:
        module = interpreter.import_module("pkg.mod")
    assert isinstance(module.broken, InertModule)
    assert "needs a built runtime" in interpreter.unmodelled["broken"]


def test_the_engine_s_own_failures_are_not_hidden(tmp_path):
    root = _package(
        tmp_path,
        {"pkg/__init__.py": "", "pkg/mod.py": "raise ValueError('a real bug')\n"},
    )
    interpreter = SourceInterpreter({"pkg": root})
    with interpreter, pytest.raises(ValueError, match="a real bug"):
        interpreter.import_module("pkg.mod")


def test_scratch_trees_are_removed_and_moved_but_real_files_are_not(tmp_path):
    interpreter, module = _run(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                import os, shutil
                os.makedirs("/scratch/a", exist_ok=True)
                with open("/scratch/a/x.txt", "w") as f:
                    f.write("x")
                shutil.move("/scratch/a", "/scratch/b")
                moved = os.path.exists("/scratch/b/x.txt") and not os.path.exists("/scratch/a/x.txt")
                shutil.rmtree("/scratch/b")
                removed = not os.path.exists("/scratch/b/x.txt")
                shutil.rmtree("/nowhere", ignore_errors=True)
            """,
        },
    )
    assert module.moved and module.removed
    with interpreter, pytest.raises(PermissionError):
        interpreter.modules["shutil"].rmtree(str(tmp_path / "pkg"))


def test_a_library_that_cannot_load_is_a_stand_in_down_to_its_submodules(tmp_path):
    library = _package(
        tmp_path / "lib",
        {
            "broken/__init__.py": "raise AssertionError('needs a built runtime')\n",
            "broken/language.py": "from broken.native import Thing\n",
        },
    )
    root = _package(
        tmp_path / "src",
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": "import broken\nfrom broken.language import Thing\n",
        },
    )
    interpreter = SourceInterpreter({"pkg": root})
    interpreter.resolver = lambda top: library if top == "broken" else None
    with interpreter:
        module = interpreter.import_module("pkg.mod")
    assert isinstance(module.Thing, Inert)


def test_a_stand_in_s_signature_has_stand_in_parameters(tmp_path):
    _, module = _run(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                import inspect
                from missing_native import dialects

                keyword = inspect.signature(dialects.LoadOp).parameters["aux"].kind is inspect.Parameter.KEYWORD_ONLY
            """,
        },
    )
    assert module.keyword is False


def test_the_options_a_parser_declares_are_noted(tmp_path):
    interpreter, _ = _run(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                import argparse
                parser = argparse.ArgumentParser()
                group = parser.add_argument_group("kernels")
                group.add_argument("--enable-fast-path", action="store_true")
                parser.add_argument("--backend", default="auto")
                parser.add_argument("--max-requests", type=int)
                args = parser.parse_args([])
            """,
        },
    )
    assert interpreter.environment.options == {
        "help": ("--help", True),
        "enable_fast_path": ("--enable-fast-path", True),
        "backend": ("--backend", False),
        "max_requests": ("--max-requests", False),
    }
    assert interpreter.environment.option_numbers == {"max_requests"}


def test_a_launch_knows_which_settings_decided_it(tmp_path):
    import types as host_types

    from TraceLens.ModelUtils.source_interpreter import Environment

    root = _package(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                import os
                import probe

                def fused():
                    return os.environ.get("FUSED", "0") == "1"

                class Layer:
                    def __init__(self, width):
                        self.shared = 0 if not fused() else 1
                        self.width = width

                    def forward(self, x):
                        bound = int(os.environ.get("BOUND", "256"))
                        if self.shared > 0:
                            probe.launch("append")
                        if x < bound:
                            probe.launch("small")
                        probe.launch_sized(self.width)

                layer = Layer(int(os.environ.get("WIDTH", "64")))
                layer.forward(3)
            """,
        },
    )
    seen = []
    probe = host_types.ModuleType("probe")
    interpreter = SourceInterpreter(
        {"pkg": root}, Environment(), host_modules={"probe": probe}
    )
    probe.launch = lambda name: seen.append((name, interpreter.control_settings()))
    probe.launch_sized = lambda width: seen.append(
        ("sized", interpreter.argument_settings())
    )
    with interpreter:
        interpreter.import_module("pkg.mod")
    # FUSED decided the layer's attribute when it was built; BOUND the
    # branch taken at the call; WIDTH shaped the argument.
    assert seen == [("small", ("BOUND",)), ("sized", ("WIDTH",))]
    interpreter = SourceInterpreter(
        {"pkg": root},
        Environment(settings={"FUSED": "1"}),
        host_modules={"probe": probe},
    )
    seen.clear()
    with interpreter:
        interpreter.import_module("pkg.mod")
    assert seen[0] == ("append", ("FUSED",))


def test_break_and_continue_leave_loops_through_try_with_and_match(tmp_path):
    _, module = _run(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                import contextlib

                def run():
                    seen = []
                    for i in range(10):
                        try:
                            if i == 1:
                                continue
                            with contextlib.nullcontext():
                                if i == 6:
                                    break
                        finally:
                            seen.append(-i)
                        match i:
                            case 3:
                                continue
                        seen.append(i)
                    else:
                        seen.append("no break")
                    while True:
                        break
                    return seen

                def gen():
                    for i in range(4):
                        if i == 2:
                            continue
                        yield i

                values = run()
                generated = list(gen())
            """,
        },
    )
    assert module.values == [0, 0, -1, -2, 2, -3, -4, 4, -5, 5, -6]
    assert module.generated == [0, 1, 3]


def test_a_cached_helper_keeps_the_settings_its_value_came_from(tmp_path):
    import types as host_types

    from TraceLens.ModelUtils.source_interpreter import Environment

    root = _package(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/mod.py": """
                import functools, os
                import probe

                @functools.lru_cache(maxsize=1)
                def use_fp8():
                    return os.environ.get("KV_FP8") == "1"

                for _ in range(2):
                    if not use_fp8():
                        probe.launch()
            """,
        },
    )
    seen = []
    probe = host_types.ModuleType("probe")
    interpreter = SourceInterpreter(
        {"pkg": root}, Environment(), host_modules={"probe": probe}
    )
    probe.launch = lambda: seen.append(interpreter.control_settings())
    with interpreter:
        interpreter.import_module("pkg.mod")
    # The second call hits the cache and still says what decided it.
    assert seen == [("KV_FP8",), ("KV_FP8",)]
