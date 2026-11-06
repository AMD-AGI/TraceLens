###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Constants and defaults an inference engine's own source defines.

Some launch shapes follow from a number the engine fixes in a module
(``_Q_SCALE_SHAPE = (1, 4, 16, 4)``) or from the default of an environment
switch (``SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)``). They are not in the
checkpoint. This module reads them from the same pinned checkout the
architecture loader reads, so the engine package need not be installed and the
value cannot drift from the revision the model source was read at.

A fact the source does not give is absent. Callers treat an absent fact as
unknown and leave the shapes that depend on it out.
"""

from __future__ import annotations

import ast
import bisect
import operator
import re
from pathlib import Path

from TraceLens.ModelUtils.github import fetch_github_file, parse_github_url

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
}

# Fact name -> (kind, module, symbol, engine). ``const`` is a module-level
# assignment, ``class`` a ``Class.attribute`` assignment, ``env`` a class
# attribute of ``Envs`` built by ``EnvBool`` or ``EnvInt`` with a literal
# default, ``default`` the literal default of ``function.parameter``, and
# ``pattern`` the integer a regular expression captures from the module text.
_MEMORY_POOL = "sglang.srt.mem_cache.deepseek_v4_memory_pool"
_HIP_BACKEND = "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix"
_FP4_INDEXER = "sglang.kernels.ops.attention.dsv4.fp4_indexer_hip"
ENGINE_FACTS = {
    "fuse_wqa_wkv": ("env", "sglang.srt.environ", "SGLANG_OPT_FUSE_WQA_WKV", "sglang"),
    "indexer_q_scale_shape": ("const", _FP4_INDEXER, "_Q_SCALE_SHAPE", "sglang"),
    "indexer_kv_block": ("const", _FP4_INDEXER, "_KV_BLOCK_SIZE", "sglang"),
    "indexer_cta_target": ("const", _FP4_INDEXER, "_DECODE_BASE_CTA_TARGET", "sglang"),
    "indexer_ctas_per_query": (
        "const",
        _FP4_INDEXER,
        "_DECODE_CTAS_PER_QUERY",
        "sglang",
    ),
    "indexer_cta_info_width": (
        "const",
        _FP4_INDEXER,
        "_DECODE_CTA_INFO_WIDTH",
        "sglang",
    ),
    "indexer_page_bucket": (
        "default",
        _FP4_INDEXER,
        "_guarded_pages.page_table_bucket",
        "sglang",
    ),
    "page_index_align": ("const", _HIP_BACKEND, "PAGE_INDEX_ALIGNED_SIZE", "sglang"),
    "page_size": (
        "pattern",
        "sglang.srt.layers.attention.deepseek_v4_backend",
        r"assert self\.page_size == (\d+)",
        "sglang",
    ),
    "topk_plan_ints": (
        "const",
        "sglang.kernels.ops.attention.dsv4.topk",
        "_PLAN_METADATA_INTS_PER_BATCH",
        "sglang",
    ),
    "kv_blocks_per_c128": (
        "class",
        _MEMORY_POOL,
        "DeepSeekV4UnifiedKVPool.K_PER_BLOCK",
        "sglang",
    ),
}

# Ring rows each request holds in a compress state pool, per ratio. The engine
# computes them in ``get_compress_state_ring_size``, which reads
# ``ONLINE_C128`` (off on ROCm, by its own definition).
_RING_RATIOS = (4, 128)


class _Unresolved(Exception):
    """An expression the evaluator does not model."""


def _evaluate(node: ast.AST, known: dict):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple | ast.List):
        return tuple(_evaluate(item, known) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _evaluate(key, known): _evaluate(value, known)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, ast.Name):
        if node.id in known:
            return known[node.id]
        raise _Unresolved(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand, known)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand, known)
    if isinstance(node, ast.BoolOp):
        # Short circuit, so an operand the source leaves to the runtime (an
        # environment read) is never evaluated once an earlier one decides.
        is_and = isinstance(node.op, ast.And)
        result = is_and
        for operand in node.values:
            result = _evaluate(operand, known)
            if bool(result) != is_and:
                return result
        return result
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](
            _evaluate(node.left, known), _evaluate(node.right, known)
        )
    raise _Unresolved(type(node).__name__)


def module_constant(source: str, name: str, known: dict | None = None):
    """Value of the module-level ``name = <expression>`` in ``source``.

    ``known`` seeds names the source takes from its environment, such as the
    platform flag a ``not _is_hip and ...`` expression reads.
    """
    known = dict(known or {})
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                known[target.id] = _evaluate(value, known)
            except _Unresolved:
                continue
    if name not in known:
        raise KeyError(name)
    return known[name]


def class_constant(source: str, container: str, name: str):
    """Value of the class attribute ``container.name = <expression>``."""
    for statement in ast.parse(source).body:
        if not (isinstance(statement, ast.ClassDef) and statement.name == container):
            continue
        for member in statement.body:
            if isinstance(member, ast.Assign):
                targets, value = member.targets, member.value
            elif isinstance(member, ast.AnnAssign) and member.value is not None:
                targets, value = [member.target], member.value
            else:
                continue
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                try:
                    return _evaluate(value, {})
                except _Unresolved:
                    break
    raise KeyError(name)


def function_default(source: str, function: str, parameter: str):
    """Literal default of ``parameter`` in the module-level ``def function``."""
    for statement in ast.parse(source).body:
        if not (isinstance(statement, ast.FunctionDef) and statement.name == function):
            continue
        args = statement.args
        positional = args.posonlyargs + args.args
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        pairs = list(zip(positional, defaults, strict=True))
        pairs += list(zip(args.kwonlyargs, args.kw_defaults, strict=True))
        for argument, default in pairs:
            if argument.arg == parameter and default is not None:
                return _evaluate(default, {})
    raise KeyError(f"{function}.{parameter}")


def function_result(source: str, function: str, arguments: tuple, scope: dict):
    """Result of calling the module-level ``def function`` from ``source``.

    Only that function is compiled, with ``scope`` as its globals. A function
    that reads a name ``scope`` lacks raises ``KeyError``.
    """
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.FunctionDef) and statement.name == function:
            module = ast.Module(body=[statement], type_ignores=[])
            namespace = dict(scope)
            exec(
                compile(ast.fix_missing_locations(module), function, "exec"), namespace
            )
            try:
                return namespace[function](*arguments)
            except NameError as error:
                raise KeyError(str(error)) from error
    raise KeyError(function)


def environment_default(source: str, name: str, container: str = "Envs"):
    """Literal default of ``name = EnvBool(<default>)`` inside ``class Envs``."""
    for statement in ast.parse(source).body:
        if not (isinstance(statement, ast.ClassDef) and statement.name == container):
            continue
        for member in statement.body:
            if isinstance(member, ast.Assign):
                targets, value = member.targets, member.value
            elif isinstance(member, ast.AnnAssign) and member.value is not None:
                targets, value = [member.target], member.value
            else:
                continue
            if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
                continue
            if isinstance(value, ast.Call) and value.args:
                try:
                    return _evaluate(value.args[0], {})
                except _Unresolved:
                    break
    raise KeyError(name)


def _module_text(engine_source: str, module: str) -> str | None:
    """Source of ``module`` in the pinned checkout, fetched into the cache."""
    ref = parse_github_url(engine_source)
    relative = Path(*module.split("."))
    for candidate in (relative.with_suffix(".py"), relative / "__init__.py"):
        found = fetch_github_file(ref, f"python/{candidate.as_posix()}")
        if found is not None:
            return found.read_text(encoding="utf-8")
    return None


def _read_fact(text: str, kind: str, symbol: str):
    if kind == "const":
        return module_constant(text, symbol)
    if kind == "env":
        return environment_default(text, symbol)
    if kind == "class":
        container, _, name = symbol.partition(".")
        return class_constant(text, container, name)
    if kind == "default":
        function, _, parameter = symbol.partition(".")
        return function_default(text, function, parameter)
    if kind == "pattern":
        found = re.search(symbol, text)
        if found is None:
            raise KeyError(symbol)
        return int(found.group(1))
    raise KeyError(kind)


def _ring_facts(text: str, hip: bool) -> list[tuple[str, object]]:
    """Ring rows per request for each compress ratio, from the engine's function."""
    try:
        online = module_constant(text, "ONLINE_C128", {"_is_hip": hip})
        return [
            (
                f"compress_state_ring_{ratio}",
                function_result(
                    text,
                    "get_compress_state_ring_size",
                    (ratio,),
                    {"ONLINE_C128": online},
                ),
            )
            for ratio in _RING_RATIOS
        ]
    except (KeyError, SyntaxError, OSError):
        return []


def load_engine_facts(
    engine: str | None, hip: bool = True
) -> tuple[tuple[str, object], ...]:
    """The facts ``engine``'s pinned source gives, as ``(name, value)`` pairs."""
    engine_source = engine_source_of(engine)
    if engine_source is None:
        return ()
    facts: list[tuple[str, object]] = []
    for fact, (kind, module, symbol, owner) in ENGINE_FACTS.items():
        if owner != engine:
            continue
        try:
            text = _module_text(engine_source, module)
            if text is None:
                continue
            facts.append((fact, _read_fact(text, kind, symbol)))
        except (KeyError, SyntaxError, OSError):
            continue
    if engine == "sglang":
        pool = _module_text(engine_source, _MEMORY_POOL)
        if pool is not None:
            facts.extend(_ring_facts(pool, hip))
    return tuple(facts)


# Engine name -> source the run is read at, once a trace has placed it.
_TRACED_SOURCES: dict[str, str] = {}


def engine_source_of(engine: str | None) -> str | None:
    """The source ``engine`` is read at: the release the trace ran, else the pin."""
    from TraceLens.ModelUtils.source import ENGINE_GITHUB_SOURCES

    if engine is None:
        return None
    return _TRACED_SOURCES.get(engine) or ENGINE_GITHUB_SOURCES.get(engine)


_FRAME = re.compile(r"/python/(sglang/srt/[\w/]+\.py)\((\d+)\): (\w+)$")
_RELEASE = re.compile(r"^(.*@v?)(\d+)\.(\d+)\.(\d+)$")
_SAMPLE_FILES = 6
_SAMPLE_FRAMES = 8
_EARLIER_RELEASES = 5


def _traced_frames(events) -> dict[str, list[tuple[int, str]]]:
    """``{file: [(line, function)]}`` of engine frames the profile recorded."""
    seen: dict[str, set[tuple[int, str]]] = {}
    for event in events:
        if event.get("cat") != "python_function":
            continue
        found = _FRAME.search(event.get("name") or "")
        if found:
            seen.setdefault(found.group(1), set()).add(
                (int(found.group(2)), found.group(3))
            )
    ranked = sorted(seen.items(), key=lambda item: (-len(item[1]), item[0]))
    return {
        path: sorted(frames)[:_SAMPLE_FRAMES] for path, frames in ranked[:_SAMPLE_FILES]
    }


def _frames_defined_at(source: str, frames: dict[str, list[tuple[int, str]]]) -> int:
    """How many recorded frames are ``def`` lines at that line in ``source``."""
    ref = parse_github_url(source)
    matched = 0
    for path, entries in frames.items():
        found = fetch_github_file(ref, f"python/{path}")
        if found is None:
            continue
        lines = found.read_text(encoding="utf-8").splitlines()
        for line, function in entries:
            if line <= len(lines) and re.match(
                rf"\s*(async\s+)?def\s+{function}\b", lines[line - 1]
            ):
                matched += 1
    return matched


def traced_engine_source(engine: str | None, events) -> str | None:
    """The release of ``engine`` whose source the trace's python frames match.

    A profile records ``file(line): function`` for every engine frame, and a
    ``def`` sits on one line in one release. The pinned release is the default.
    An earlier patch release that matches more of the sampled frames is the one
    the run used.
    """
    from TraceLens.ModelUtils.source import ENGINE_GITHUB_SOURCES

    pinned = ENGINE_GITHUB_SOURCES.get(engine or "")
    if pinned is None:
        return None
    frames = _traced_frames(events)
    release = _RELEASE.match(pinned)
    if not frames or release is None:
        return pinned
    prefix, major, minor, patch = release.groups()
    candidates = [pinned] + [
        f"{prefix}{major}.{minor}.{earlier}"
        for earlier in range(
            int(patch) - 1, max(-1, int(patch) - 1 - _EARLIER_RELEASES), -1
        )
    ]
    best, best_score = pinned, -1
    for candidate in candidates:
        score = _frames_defined_at(candidate, frames)
        if score > best_score:
            best, best_score = candidate, score
    return best


def use_traced_engine_source(engine: str | None, events) -> str | None:
    """Read ``engine`` at the release the trace ran, for the rest of the run."""
    source = traced_engine_source(engine, events)
    if engine is not None and source is not None:
        _TRACED_SOURCES[engine] = source
    return source


def _first_dim(args: dict, rank: int):
    dims = args.get("Input Dims") or []
    shape = tuple(dims[0]) if dims else ()
    if len(shape) != rank:
        return None
    return shape[0] if rank == 1 else shape


def _slice_end(args: dict):
    """End of an ``aten::slice(self, dim, start, end, step)``."""
    inputs = args.get("Concrete Inputs") or []
    return (
        int(inputs[3]) if len(inputs) > 3 and inputs[3].lstrip("-").isdigit() else None
    )


# Fact -> (python function, aten op, reader of that op's args, one value or
# the set seen). The functions are where the engine touches the pools and the
# graph replay buffers, so the eager ops inside them carry the run's sizes.
_RUNTIME_PROBES = {
    "kv_mapping_rows": (
        "translate_loc_from_full_to_swa",
        "aten::index",
        lambda args: _first_dim(args, 1),
        False,
    ),
    "req_to_token_shape": (
        "alloc_for_decode",
        "aten::index",
        lambda args: _first_dim(args, 2),
        False,
    ),
    # The replay view slices each static input buffer to the padded token
    # count the captured graph was built for.
    "replay_tokens": ("build_replay_fb_view", "aten::slice", _slice_end, True),
}


def trace_runtime_facts(events) -> tuple[tuple[str, object], ...]:
    """Sizes the profile's eager ops record about the run.

    ``kv_mapping_rows`` is the length of the allocator's full-to-SWA map
    (``size + page_size + 1``). ``req_to_token_shape`` is the request-to-token
    table, ``(request slots, context + 4)``. ``replay_tokens`` are the padded
    token counts the decode graphs replayed at.
    """
    events = list(events)
    windows: dict[str, list[dict]] = {fact: [] for fact in _RUNTIME_PROBES}
    wanted = {probe[0]: fact for fact, probe in _RUNTIME_PROBES.items()}
    for event in events:
        if event.get("cat") != "python_function":
            continue
        function = (event.get("name") or "").rpartition(": ")[2]
        if function in wanted:
            windows[wanted[function]].append(event)
    if not any(windows.values()):
        return ()
    ops = sorted(
        (event for event in events if event.get("cat") == "cpu_op"),
        key=lambda event: event["ts"],
    )
    starts = [event["ts"] for event in ops]
    facts: list[tuple[str, object]] = []
    for fact, frames in windows.items():
        _, op_name, read, collect = _RUNTIME_PROBES[fact]
        seen = []
        for frame in frames:
            end = frame["ts"] + frame.get("dur", 0)
            for event in ops[bisect.bisect_left(starts, frame["ts"]) :]:
                if event["ts"] > end:
                    break
                if event.get("tid") != frame.get("tid") or event["name"] != op_name:
                    continue
                value = read(event.get("args") or {})
                if value is not None:
                    seen.append(value)
                    break
            if seen and not collect:
                break
        if seen:
            facts.append((fact, tuple(sorted(set(seen))) if collect else seen[0]))
    return tuple(facts)


def dockerfile_pins(text: str, variable: str) -> dict[str, str]:
    """``{stage: value}`` of ``ENV <variable>="value"`` for each ``FROM .. AS stage``."""
    pins: dict[str, str] = {}
    stage = None
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 4 and words[0].upper() == "FROM" and words[-2].upper() == "AS":
            stage = words[-1]
        elif words and words[0].upper() == "ENV" and stage is not None:
            key, _, value = " ".join(words[1:]).partition("=")
            if key == variable and value:
                pins[stage] = value.strip().strip('"').strip("'")
    return pins


def kernel_library_commit(
    engine: str, variable: str, device_prefix: str = "gfx9"
) -> str | None:
    """The commit of a kernel library the engine's own container build pins.

    The engine's ROCm Dockerfile sets ``<variable>`` per device stage. The
    commit the stages of this device family agree on is the one the engine's
    kernels were built against.
    """
    engine_source = engine_source_of(engine)
    if engine_source is None:
        return None
    ref = parse_github_url(engine_source)
    found = fetch_github_file(ref, "docker/rocm.Dockerfile")
    if found is None:
        return None
    pins = dockerfile_pins(found.read_text(encoding="utf-8"), variable)
    family = {value for stage, value in pins.items() if stage.startswith(device_prefix)}
    return family.pop() if len(family) == 1 else None
