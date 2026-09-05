###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Derive kernel computation pipelines from modeling AST details and kernel source code."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from TraceLens.ModelUtils.ast_analyze import kernel_kwarg_ports, kernel_name_from_step_details

_KERNEL_SOURCE_CACHE = Path.home() / ".cache" / "tracelens" / "kernel_sources"
_KERNEL_FIXTURE_ROOT = os.environ.get("TRACELENS_KERNEL_FIXTURE_ROOT")

# Directories of the modeling files under analysis, searched for imported kernel sources.
_KERNEL_SEARCH_ROOTS: list[Path] = []

# Top-level import roots -> (github org/repo, branch) for source fetch when the package is not installed.
_PACKAGE_SOURCE_REPOS: dict[str, tuple[str, str]] = {
    "fla": ("fla-org/flash-linear-attention", "main"),
}

_BUILTIN_SKIP_CALLS = frozenset(
    {
        "len",
        "list",
        "get",
        "min",
        "max",
        "pop",
        "warn",
        "int",
        "float",
        "str",
        "tuple",
        "set",
        "dict",
        "range",
        "enumerate",
        "zip",
        "map",
        "any",
        "all",
        "super",
        "isinstance",
        "ValueError",
        "AssertionError",
        "TypeError",
        "RuntimeError",
        "NotImplementedError",
    }
)

# Allocations produce an output buffer that a later kernel call writes into.
_ALLOCATION_CALLS = frozenset(
    {
        "empty",
        "empty_like",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "full",
        "full_like",
        "new_empty",
        "new_zeros",
        "new_ones",
        "new_full",
    }
)

# Tensor metadata queries: they read shape or layout rather than producing a stage.
_SKIP_TENSOR_QUERY_CALLS = frozenset(
    {
        "size",
        "dim",
        "numel",
        "stride",
        "element_size",
        "get_default_dtype",
    }
)

_SKIP_KERNEL_CALLS = frozenset(
    {
        "apply",
        "type_as",
        "pop",
        "assert",
        "raise",
        "warn",
        "delete",
        "save_for_backward",
        "prepare_chunk_indices",
        "compress_h0",
        "autocast_custom_fwd",
        "autocast_custom_bwd",
        "input_guard",
        "dispatch",
    }
)

# Display hints keyed by function names discovered in kernel source (not model-specific).
_OP_LABEL_OVERRIDES: dict[str, str] = {
    "l2norm_fwd": "L2Norm",
    "chunk_gla_fwd_o_gk": "Output o",
    "chunk_gated_delta_rule_fwd_h": "Gated delta rule h",
    "kda_gate_chunk_cumsum": "Gate cumsum",
    "chunk_kda_fwd_intra": "Intra-chunk WY",
    "fused_beta_sigmoid": "Fused beta sigmoid",
    "chunk_local_cumsum": "Chunk local cumsum",
}


TENSOR_PORT_KERNEL_ATTR_RE = re.compile(r"forward_(?P<stem>.+)_fwd_(?P<port>[a-z])$")


def tensor_port_kernel_frame_label(attr_name: str) -> str | None:
    """Namespace segment for a kernel expanded once per tensor port (e.g. q/k)."""
    match = TENSOR_PORT_KERNEL_ATTR_RE.match(attr_name)
    if match is None:
        return None
    return f"{match.group('stem')}_fwd_{match.group('port')}"


@dataclass(frozen=True)
class KernelPipelineStep:
    call_name: str
    attr_name: str
    class_name: str
    label: str
    details: list[str]
    condition: str | None = None
    tensor_inputs: frozenset[str] = frozenset()
    computation: str = ""
    predecessors: frozenset[str] = frozenset()
    children: tuple[KernelPipelineStep, ...] = ()
    second_operand: str | None = None


@dataclass(frozen=True)
class ComputationOp:
    label: str
    second_operand: int | Literal["input"] | None = None


@dataclass(frozen=True)
class _ImportTarget:
    module: str
    symbol: str


def parse_kernel_call_flags(details: list[str]) -> dict[str, str | bool]:
    """Parse ``forward_step_details`` lines captured from the modeling forward call."""
    flags: dict[str, str | bool] = {}
    for line in details:
        if line.startswith("kernel:"):
            flags["_kernel"] = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("kwarg:"):
            continue
        payload = line.split(":", 1)[1].strip()
        if "=" not in payload:
            continue
        name, raw_value = payload.split("=", 1)
        name = name.strip()
        value = raw_value.strip()
        if value in {"True", "False"}:
            flags[name] = value == "True"
        else:
            flags[name] = value
    return flags


def parse_kernel_import(details: list[str]) -> tuple[str, str] | None:
    """Return ``(module, symbol)`` from an ``import:`` detail line."""
    for line in details:
        if not line.startswith("import:"):
            continue
        payload = line.split(":", 1)[1].strip()
        if "#" in payload:
            module, symbol = payload.split("#", 1)
            return module.strip(), symbol.strip()
        if "." in payload:
            module, symbol = payload.rsplit(".", 1)
            return module.strip(), symbol.strip()
        return payload, payload
    return None


def _resolve_relative_module(
    relative_module: str, current_module: str, level: int
) -> str:
    if level == 0:
        return relative_module or current_module

    package_parts = current_module.split(".")
    base = package_parts[: max(0, len(package_parts) - (level - 1))]
    if not relative_module:
        return ".".join(package_parts[: max(0, len(package_parts) - level)])
    return ".".join(base + relative_module.split("."))


def _parse_module(source: str) -> ast.Module:
    return ast.parse(source)


def _function_def(module: ast.Module, qualname: str) -> ast.FunctionDef | None:
    parts = qualname.split(".")
    if len(parts) == 1:
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == parts[0]:
                return node
        return None

    class_name, method_name = parts[0], parts[1]
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def _expr_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _call_name(call: ast.Call) -> str:
    name = _expr_name(call.func) or "op"
    return name.split(".")[-1]


def _iter_calls(func: ast.FunctionDef) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            calls.append(node)
    return calls


def _collect_import_map(
    module: ast.Module, module_name: str
) -> dict[str, _ImportTarget]:
    imports: dict[str, _ImportTarget] = {}

    def walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.ImportFrom):
                resolved = _resolve_relative_module(
                    stmt.module or "", module_name, stmt.level
                )
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    imports[alias.asname or alias.name] = _ImportTarget(
                        resolved, alias.name
                    )
            elif isinstance(stmt, ast.Try):
                walk(stmt.body)
                for handler in stmt.handlers:
                    walk(handler.body)

    walk(module.body)
    return imports


def register_kernel_search_root(path: str | Path) -> None:
    """Resolve kernel imports that ship beside the modeling source being analyzed.

    A checkpoint like ``inference/model.py`` does ``from kernel import sparse_attn``,
    naming a sibling file that is on no import path, so without its directory the
    kernel reads as opaque and its pipeline never expands.
    """
    root = Path(path).expanduser()
    if root.is_file():
        root = root.parent
    if root.is_dir() and root not in _KERNEL_SEARCH_ROOTS:
        _KERNEL_SEARCH_ROOTS.insert(0, root)


def _kernel_search_roots() -> list[Path]:
    roots = list(_KERNEL_SEARCH_ROOTS)
    if _KERNEL_FIXTURE_ROOT:
        roots.insert(0, Path(_KERNEL_FIXTURE_ROOT))
    return roots


def _search_root_module_file(module: str) -> Path | None:
    """Locate a module file within a registered search root."""
    relative = Path(*module.split("."))
    for root in _kernel_search_roots():
        candidate = root / relative.with_suffix(".py")
        if candidate.is_file():
            return candidate
        package_init = root / relative / "__init__.py"
        if package_init.is_file():
            return package_init
    return None


def _module_file_path(module: str) -> Path | None:
    from_root = _search_root_module_file(module)
    if from_root is not None:
        return from_root

    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    if spec is not None and spec.origin and spec.origin != "namespace":
        origin = Path(spec.origin)
        if origin.is_file():
            return origin

    cache_file = _KERNEL_SOURCE_CACHE / Path(*module.split(".")).with_suffix(".py")
    if cache_file.is_file():
        return cache_file
    cache_init = _KERNEL_SOURCE_CACHE / Path(*module.split(".")) / "__init__.py"
    if cache_init.is_file():
        return cache_init

    top_level = module.split(".", 1)[0]
    repo = _PACKAGE_SOURCE_REPOS.get(top_level)
    if repo is None:
        return None
    org_repo, branch = repo
    owner, repo_name = org_repo.split("/", 1)
    from TraceLens.ModelUtils.source_policy import get_source_policy

    get_source_policy().require_github_repo_allowed(owner, repo_name)
    relative = Path(*module.split(".")).with_suffix(".py")
    url = f"https://raw.githubusercontent.com/{org_repo}/{branch}/{relative.as_posix()}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            text = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        package_init = Path(*module.split(".")) / "__init__.py"
        url = f"https://raw.githubusercontent.com/{org_repo}/{branch}/{package_init.as_posix()}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                text = response.read().decode("utf-8")
        except (OSError, urllib.error.URLError):
            return None
        cache_init.parent.mkdir(parents=True, exist_ok=True)
        cache_init.write_text(text, encoding="utf-8")
        return cache_init

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return cache_file


def _read_module_source(module: str) -> tuple[str, str] | None:
    # A sibling of the modeling file outranks any same-named installed module.
    from_root = _search_root_module_file(module)
    if from_root is not None:
        return from_root.read_text(encoding="utf-8"), module

    try:
        imported = importlib.import_module(module)
        file_path = inspect.getfile(imported)
        if Path(file_path).is_file():
            return Path(file_path).read_text(encoding="utf-8"), module
    except Exception:
        pass

    file_path = _module_file_path(module)
    if file_path is None:
        return None
    return file_path.read_text(encoding="utf-8"), module


def _find_symbol_definition(
    module: str,
    symbol: str,
) -> tuple[str, str, str] | None:
    """Return ``(source, qualname, owning_module)`` for a module symbol."""
    loaded = _read_module_source(module)
    if loaded is None:
        return None
    source, resolved_module = loaded
    tree = _parse_module(source)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == symbol:
            return source, node.name, resolved_module
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            forward = _function_def(tree, f"{symbol}.forward")
            if forward is not None:
                return source, f"{symbol}.forward", resolved_module

    imports = _collect_import_map(tree, resolved_module)
    target = imports.get(symbol)
    if target is not None and (
        target.module != resolved_module or target.symbol != symbol
    ):
        return _find_symbol_definition(target.module, target.symbol)

    try:
        imported = importlib.import_module(resolved_module)
        obj = getattr(imported, symbol, None)
        if obj is not None:
            file_path = inspect.getsourcefile(obj) or inspect.getfile(obj)
            if file_path and Path(file_path).is_file():
                obj_source = Path(file_path).read_text(encoding="utf-8")
                obj_module = getattr(obj, "__module__", resolved_module)
                obj_tree = _parse_module(obj_source)
                if isinstance(obj, type):
                    forward = _function_def(obj_tree, f"{symbol}.forward")
                    if forward is not None:
                        return obj_source, f"{symbol}.forward", obj_module
                for node in obj_tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == symbol:
                        return obj_source, node.name, obj_module
    except Exception:
        pass

    return None


def _package_root(module: str) -> str:
    return module.split(".", 1)[0]


def _should_follow_import(module: str, root_module: str) -> bool:
    return _package_root(module) == _package_root(root_module)


def _is_pipeline_handoff(callee: str, imported: _ImportTarget) -> bool:
    """True when an imported call delegates to a multi-stage forward pipeline function."""
    name = imported.symbol
    if not name.endswith("_fwd"):
        return False
    # Skip small utility forwards (e.g. l2norm_fwd) that are steps within the wrapper forward.
    if name.count("_") < 2:
        return False
    return True


def _discover_pipeline_entrypoints(
    module: str,
    symbol: str,
) -> tuple[list[tuple[str, str]], set[str]]:
    """Walk the imported kernel implementation starting from the modeling import."""
    root = _find_symbol_definition(module, symbol)
    if root is None:
        return [], set()

    source, qualname, owning_module = root
    tree = _parse_module(source)
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    handoff_calls: set[str] = set()

    def add(src: str, qn: str) -> None:
        key = (src, qn)
        if key not in seen:
            seen.add(key)
            ordered.append(key)

    wrapper = _function_def(tree, qualname)
    if wrapper is None:
        return ordered, handoff_calls

    forward_qn: str | None = None
    for call in _iter_calls(wrapper):
        if _call_name(call) == "apply" and isinstance(call.func, ast.Attribute):
            class_name = _expr_name(call.func.value)
            if class_name:
                forward_qn = f"{class_name.split('.')[-1]}.forward"
                add(source, forward_qn)
            break

    if forward_qn is None:
        add(source, qualname)
        forward_qn = qualname

    forward_func = _function_def(tree, forward_qn)
    if forward_func is None:
        return ordered, handoff_calls

    imports = _collect_import_map(tree, owning_module)
    for call in _iter_calls(forward_func):
        callee = _call_name(call)
        imported = imports.get(callee)
        if imported is None or not _should_follow_import(
            imported.module, owning_module
        ):
            continue
        if not _is_pipeline_handoff(callee, imported):
            continue
        handoff_calls.add(callee)
        resolved = _find_symbol_definition(imported.module, imported.symbol)
        if resolved is not None:
            add(resolved[0], resolved[1])

    return ordered, handoff_calls


def kernel_op_display_label(name: str) -> str:
    """Short diagram label for a kernel call tile."""
    return _label_from_call_name(name)


def _label_from_call_name(name: str) -> str:
    if name in _OP_LABEL_OVERRIDES:
        return _OP_LABEL_OVERRIDES[name]
    if name.endswith("_fwd") or name.endswith("_bwd"):
        stem = name.rsplit("_", 1)[0]
        if stem.endswith("_intra") or stem.endswith("_h") or "chunk" in stem:
            cleaned = stem.replace("_", " ")
        else:
            cleaned = re.sub(r"_(fwd|bwd|kernel)$", "", name).replace("_", " ")
    else:
        cleaned = name.replace("_", " ")
    titled = cleaned[:1].upper() + cleaned[1:] if cleaned else name
    if "l2norm" in name.lower():
        return "L2Norm"
    return titled


def modeling_tensor_port_names(details: list[str]) -> set[str]:
    """Return kwarg tensor parameter names passed into a kernel call from modeling code."""
    return {
        param
        for param, value in kernel_kwarg_ports(details).items()
        if value and not value.startswith("self.") and value not in {"True", "False"}
    }


def _call_references_port_names(call: ast.Call, port_names: set[str]) -> frozenset[str]:
    """Map a kernel call AST node to modeling tensor port names it consumes."""
    aliases = set(port_names)
    for name in port_names:
        aliases.update({f"{name}_raw", f"{name}_org", f"{name}_input"})
    if "g" in port_names:
        aliases.add("g_input")
    if "beta" in port_names:
        aliases.add("beta_raw")

    matched: set[str] = set()
    for node in ast.walk(call):
        if not isinstance(node, ast.Name):
            continue
        token = node.id
        if token in port_names:
            matched.add(token)
            continue
        for port in port_names:
            if token in {f"{port}_raw", f"{port}_org", f"{port}_input"}:
                matched.add(port)
                break
    return frozenset(matched)


def compute_tensor_step_targets(
    details: list[str],
    pipeline_steps: list[KernelPipelineStep],
) -> dict[str, str]:
    """Map each modeling tensor port to the first pipeline step that consumes it."""
    port_names = modeling_tensor_port_names(details)
    if not port_names or not pipeline_steps:
        return {}

    targets: dict[str, str] = {}
    for step in pipeline_steps:
        for port in step.tensor_inputs:
            if port in port_names and port not in targets:
                targets[port] = step.attr_name
    if targets:
        return targets

    for step in pipeline_steps:
        for port in port_names:
            if port not in targets:
                targets[port] = step.attr_name
    return targets


def _is_output_pipeline_step(call_name: str) -> bool:
    lowered = call_name.lower()
    if "_o_" in lowered or lowered.endswith("_o_gk"):
        return True
    if "recurrent" in lowered and lowered.endswith("_fwd"):
        return True
    return False


def _condition_name(test: ast.AST) -> str | None:
    if isinstance(test, ast.Name):
        return test.id
    if (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Name)
    ):
        return f"not {test.operand.id}"
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
        return ast.unparse(test)
    return None


def _step_matches_flags(step: KernelPipelineStep, flags: dict[str, str | bool]) -> bool:
    if step.condition is None:
        return True
    if step.condition.startswith("not "):
        key = step.condition.removeprefix("not ")
        return flags.get(key) is False
    if " is not None" in step.condition:
        return False
    value = flags.get(step.condition)
    if isinstance(value, bool):
        return value
    return True


def _details_for_call(source: str, call_name: str) -> list[str]:
    """Extract display detail strings from kernel source when present."""
    del source, call_name
    return []


def _statement_computation(stmt: ast.stmt) -> str:
    """Return the source-level computation performed by one forward statement."""
    try:
        return ast.unparse(stmt).strip()
    except Exception:
        return ""


def _merge_computation_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right or right in left:
        return left
    return f"{left}\n{right}"


def _docstring_computes_line(module: str, symbol: str) -> str | None:
    """Return a ``Computes: ...`` docstring line from an imported kernel symbol."""
    resolved = _find_symbol_definition(module, symbol)
    if resolved is None:
        return None
    source, qualname, _ = resolved
    func = _function_def(_parse_module(source), qualname)
    if func is None:
        return None
    doc = ast.get_docstring(func) or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("computes:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _extract_recurrence_from_kernel_source(source: str) -> str | None:
    """Derive recurrence update lines from a gated-delta kernel implementation."""
    parts: list[str] = []
    if re.search(r"b_v = tl\.load\(p_v[^;\n]+- b_v", source):
        parts.append("v_new = v − W @ h")
    if re.search(r"b_h1 \*=", source) and "exp2" in source:
        parts.append("h = exp2(g) · h")
    if re.search(r"tl\.dot\([^)]*b_h1", source):
        parts.append("h = h + k @ v_new")
    if not parts:
        return None
    return "\n".join(parts)


def _recurrence_for_call(
    call_name: str,
    imports: dict[str, _ImportTarget],
) -> str | None:
    """Follow a pipeline call import and extract recurrence relations from its kernel source."""
    if "gated_delta" not in call_name.lower():
        return None
    imported = imports.get(call_name)
    if imported is None:
        return None
    resolved = _find_symbol_definition(imported.module, imported.symbol)
    if resolved is None:
        return None
    source, _, _ = resolved
    return _extract_recurrence_from_kernel_source(source)


def _computation_for_statement(
    stmt: ast.stmt,
    call: ast.Call,
    call_name: str,
    *,
    imports: dict[str, _ImportTarget],
) -> str:
    """Build the primary block label for one kernel pipeline step from AST/source."""
    lines: list[str] = []
    statement_text = _statement_computation(stmt)
    if statement_text:
        lines.append(statement_text)

    imported = imports.get(call_name)
    if imported is not None:
        doc_formula = _docstring_computes_line(imported.module, imported.symbol)
        if doc_formula and doc_formula not in statement_text:
            lines.append(doc_formula)

    recurrence = _recurrence_for_call(call_name, imports)
    if recurrence:
        existing = "\n".join(lines)
        for part in recurrence.splitlines():
            if part not in existing:
                lines.append(part)

    return "\n".join(lines)


def _assignment_target_names(stmt: ast.stmt) -> list[str]:
    if not isinstance(stmt, ast.Assign):
        return []
    names: list[str] = []
    for target in stmt.targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
    return names


def _bind_assignment(
    stmt: ast.stmt, step_attr: str, var_producer: dict[str, str]
) -> None:
    for name in _assignment_target_names(stmt):
        var_producer[name] = step_attr


def _bind_out_parameters(
    call: ast.Call,
    buffer_names: set[str],
    step_attr: str,
    var_producer: dict[str, str],
) -> None:
    """Credit a step with the pre-allocated buffers it writes into.

    Kernels return their result through an out-parameter, so without this the stages
    reading that buffer have no producer and the pipeline lays out as disconnected.
    """
    for arg in call.args:
        if isinstance(arg, ast.Name) and arg.id in buffer_names:
            var_producer[arg.id] = step_attr


def _predecessor_attr_names(
    call: ast.Call, var_producer: dict[str, str]
) -> frozenset[str]:
    predecessors: set[str] = set()
    for node in ast.walk(call):
        if isinstance(node, ast.Name) and node.id in var_producer:
            predecessors.add(var_producer[node.id])
    return frozenset(predecessors)


def _should_bind_step(condition: str | None, flags: dict[str, str | bool]) -> bool:
    """True when an AST branch matches modeling kwargs and may update variable producers."""
    if not flags:
        return True
    return _step_matches_flags(
        KernelPipelineStep(
            call_name="",
            attr_name="",
            class_name="",
            label="",
            details=[],
            condition=condition,
        ),
        flags,
    )


def _filter_step_predecessors(
    steps: list[KernelPipelineStep],
    active_attrs: set[str],
) -> list[KernelPipelineStep]:
    """Drop predecessor links to pipeline steps removed by flag filtering."""
    return [
        KernelPipelineStep(
            call_name=step.call_name,
            attr_name=step.attr_name,
            class_name=step.class_name,
            label=step.label,
            details=step.details,
            condition=step.condition,
            tensor_inputs=step.tensor_inputs,
            computation=step.computation,
            predecessors=frozenset(
                pred for pred in step.predecessors if pred in active_attrs
            ),
            children=step.children,
        )
        for step in steps
    ]


def _effective_port_refs(
    call: ast.Call, port_refs: frozenset[str], active_ports: set[str]
) -> frozenset[str]:
    """Resolve tensor port names for a call, including direct argument names."""
    if port_refs:
        return port_refs
    if call.args and isinstance(call.args[0], ast.Name):
        arg = call.args[0].id
        if not active_ports or arg in active_ports:
            return frozenset({arg})
    return frozenset()


def _call_instance_key(call: ast.Call, port_refs: frozenset[str]) -> str:
    """Distinguish repeated calls to the same kernel (e.g. l2norm_fwd(q) vs l2norm_fwd(k))."""
    if port_refs:
        return ",".join(sorted(port_refs))
    if call.args and isinstance(call.args[0], ast.Name):
        return call.args[0].id
    return ""


def _step_dedupe_key(
    call_name: str,
    condition: str | None,
    port_refs: frozenset[str],
    *,
    call: ast.Call | None = None,
) -> str:
    """Unique key for one kernel call instance (split duplicate ops on different tensor ports)."""
    port_key = (
        _call_instance_key(call, port_refs)
        if call is not None
        else ",".join(sorted(port_refs))
    )
    cond = condition or ""
    return f"{call_name}|{cond}|{port_key}"


def _step_attr_name(
    prefix: str,
    call_name: str,
    port_refs: frozenset[str],
    *,
    used: set[str],
    call: ast.Call | None = None,
) -> str:
    """Build a stable, unique attr_name for a pipeline step."""
    attr_suffix = re.sub(r"[^a-zA-Z0-9_]+", "_", call_name).strip("_")
    base = f"{prefix}_{attr_suffix}"
    instance = _call_instance_key(call, port_refs) if call is not None else ""
    if "," not in instance and instance:
        base = f"{base}_{instance}"
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _port_predecessor_attr_names(
    call: ast.Call,
    port_producer: dict[str, str],
    tensor_ports: set[str],
) -> frozenset[str]:
    """Map kernel call argument names to earlier steps that processed modeling tensor ports."""
    if not tensor_ports or not port_producer:
        return frozenset()
    aliases: dict[str, str] = {}
    for port in tensor_ports:
        aliases[port] = port
        for suffix in ("_raw", "_org", "_input"):
            aliases[f"{port}{suffix}"] = port
    if "g" in tensor_ports:
        aliases["g_input"] = "g"
    predecessors: set[str] = set()
    for node in ast.walk(call):
        if not isinstance(node, ast.Name):
            continue
        port = aliases.get(node.id)
        if port is None:
            continue
        producer = port_producer.get(port)
        if producer is not None:
            predecessors.add(producer)
    return frozenset(predecessors)


def _invoked_local_names(func: ast.FunctionDef) -> dict[str, str]:
    """Local names that are called, mapped to the call that produced them.

    ``kernel = sparse_attn_kernel(...)`` followed by ``kernel(...)`` names a compiled
    kernel, so the build is not a pipeline stage and the invocation carries its name.
    """
    produced: dict[str, str] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        producer = _call_name(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                produced[target.id] = producer

    invoked: dict[str, str] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            producer = produced.get(node.func.id)
            if producer is not None:
                invoked[node.func.id] = producer
    return invoked


def _assigns_invoked_name(stmt: ast.stmt, factory_names: dict[str, str]) -> bool:
    """True when this statement builds a value that is later called."""
    if not isinstance(stmt, ast.Assign):
        return False
    return any(
        isinstance(target, ast.Name) and target.id in factory_names
        for target in stmt.targets
    )


def _factory_for_invocation(
    call: ast.Call, factory_names: dict[str, str]
) -> str | None:
    """The builder call name when ``call`` invokes a value built earlier."""
    if isinstance(call.func, ast.Name):
        return factory_names.get(call.func.id)
    return None


def _extract_pipeline_from_function(
    func: ast.FunctionDef,
    *,
    source: str,
    prefix: str,
    skip_calls: set[str],
    tensor_ports: set[str] | None = None,
    owning_module: str = "",
    flags: dict[str, str | bool] | None = None,
    port_producer: dict[str, str] | None = None,
) -> list[KernelPipelineStep]:
    steps: list[KernelPipelineStep] = []
    seen: set[str] = set()
    used_attr_names: set[str] = set()
    imports = _collect_import_map(_parse_module(source), owning_module)
    var_producer: dict[str, str] = {}
    factory_names = _invoked_local_names(func)
    buffer_names: set[str] = set()
    active_flags = dict(flags or {})
    shared_port_producer = port_producer if port_producer is not None else {}
    active_ports = tensor_ports or set()

    def _register_port_outputs(step: KernelPipelineStep) -> None:
        for port in step.tensor_inputs:
            shared_port_producer[port] = step.attr_name

    def walk(body: list[ast.stmt], conditions: tuple[str, ...]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.If):
                cond = _condition_name(stmt.test)
                if cond:
                    walk(stmt.body, conditions + (cond,))
                    if stmt.orelse:
                        walk(stmt.orelse, conditions + (f"not {cond}",))
                else:
                    walk(stmt.body, conditions)
                    walk(stmt.orelse, conditions)
                continue

            call: ast.Call | None = None
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                call = stmt.value
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
            if call is None:
                continue

            call_name = _call_name(call)
            if call_name in _ALLOCATION_CALLS:
                buffer_names.update(_assignment_target_names(stmt))
            if call_name in _SKIP_KERNEL_CALLS or call_name in _BUILTIN_SKIP_CALLS:
                continue
            if call_name in _SKIP_TENSOR_QUERY_CALLS:
                continue
            if call_name in skip_calls:
                continue
            # A call whose result gets invoked builds the kernel rather than running it,
            # so the stage is the invocation below, named after the builder.
            if _assigns_invoked_name(stmt, factory_names):
                continue
            factory = _factory_for_invocation(call, factory_names)
            if factory is not None:
                call_name = factory
            if call_name.endswith("_bwd"):
                continue
            if call_name.startswith("compress_"):
                continue
            if isinstance(call.func, ast.Attribute):
                base = _expr_name(call.func.value) or ""
                if base.split(".")[0] in {"torch", "triton", "warnings", "math"}:
                    continue

            condition = conditions[-1] if conditions else None
            if not _should_bind_step(condition, active_flags):
                continue
            port_refs = _call_references_port_names(call, active_ports)
            effective_ports = _effective_port_refs(call, port_refs, active_ports)
            dedupe_key = _step_dedupe_key(
                call_name, condition, effective_ports, call=call
            )
            computation = _computation_for_statement(
                stmt, call, call_name, imports=imports
            )
            pred_attrs = _predecessor_attr_names(
                call, var_producer
            ) | _port_predecessor_attr_names(call, shared_port_producer, active_ports)
            if dedupe_key in seen:
                for index, existing in enumerate(steps):
                    if (
                        _step_dedupe_key(
                            existing.call_name,
                            existing.condition,
                            existing.tensor_inputs,
                        )
                        == dedupe_key
                    ):
                        updated = KernelPipelineStep(
                            call_name=existing.call_name,
                            attr_name=existing.attr_name,
                            class_name=existing.class_name,
                            label=existing.call_name,
                            details=existing.details,
                            condition=existing.condition,
                            tensor_inputs=existing.tensor_inputs | effective_ports,
                            computation=_merge_computation_text(
                                existing.computation, computation
                            ),
                            predecessors=existing.predecessors | pred_attrs,
                        )
                        steps[index] = updated
                        if _should_bind_step(condition, active_flags):
                            _bind_assignment(stmt, existing.attr_name, var_producer)
                        _register_port_outputs(updated)
                        break
                continue
            seen.add(dedupe_key)

            attr_name = _step_attr_name(
                prefix, call_name, effective_ports, used=used_attr_names, call=call
            )
            created = KernelPipelineStep(
                call_name=call_name,
                attr_name=attr_name,
                class_name="KernelOp",
                label=call_name,
                details=[],
                condition=condition,
                tensor_inputs=effective_ports,
                computation=computation or call_name,
                predecessors=pred_attrs,
            )
            steps.append(created)
            _bind_assignment(stmt, attr_name, var_producer)
            _bind_out_parameters(call, buffer_names, attr_name, var_producer)
            _register_port_outputs(created)

    walk(func.body, ())
    return steps


def _is_scale_reference(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "scale"


def _flatten_ops(parts: list[list[ComputationOp]]) -> list[ComputationOp]:
    merged: list[ComputationOp] = []
    for part in parts:
        for op in part:
            if op.label and op.label not in {existing.label for existing in merged}:
                merged.append(op)
    return merged


def _tensor_names(expr: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)}


def _multiply_second_operand(
    expr: ast.BinOp,
    left_ops: list[ComputationOp],
    right_ops: list[ComputationOp],
) -> int | Literal["input"] | None:
    if _is_scale_reference(expr.left) or _is_scale_reference(expr.right):
        return None
    if left_ops and right_ops:
        return len(left_ops) - 1
    names = _tensor_names(expr.left) | _tensor_names(expr.right)
    if any("rstd" in name for name in names):
        return "input"
    if left_ops:
        return len(left_ops) - 1
    if right_ops:
        return len(right_ops) - 1
    return None


def _decompose_computation_expr(expr: ast.AST) -> list[ComputationOp]:
    """Turn a Triton/Python RHS expression tree into ordered operation labels."""
    if isinstance(expr, ast.Call):
        callee = _call_name(expr)
        arg_ops = _flatten_ops([_decompose_computation_expr(arg) for arg in expr.args])
        if callee == "sigmoid":
            return arg_ops + [ComputationOp("Sigmoid")]
        if callee == "sum":
            return arg_ops + [ComputationOp("Sum")]
        if callee == "sqrt":
            return arg_ops + [ComputationOp("Sqrt")]
        if callee == "cumsum":
            return arg_ops + [ComputationOp("CumSum")]
        if callee in {"exp", "exp2"}:
            return arg_ops + [ComputationOp("Exp")]
        if callee == "softplus":
            return arg_ops + [ComputationOp("Softplus")]
        if callee == "load":
            return arg_ops
        return arg_ops
    if isinstance(expr, ast.BinOp):
        left_ops = _decompose_computation_expr(expr.left)
        right_ops = _decompose_computation_expr(expr.right)
        if isinstance(expr.op, ast.Mult):
            left_name = _expr_name(expr.left)
            right_name = _expr_name(expr.right)
            if left_name and left_name == right_name:
                return []
            if _is_scale_reference(expr.left) or _is_scale_reference(expr.right):
                return left_ops + right_ops + [ComputationOp("× scale")]
            second_operand = _multiply_second_operand(expr, left_ops, right_ops)
            return (
                left_ops
                + right_ops
                + [ComputationOp("×", second_operand=second_operand)]
            )
        if isinstance(expr.op, ast.Div):
            if isinstance(expr.left, ast.Constant) and expr.left.value in {1, 1.0}:
                return right_ops + [ComputationOp("÷")]
            second_operand = len(left_ops) - 1 if left_ops else None
            return (
                left_ops
                + right_ops
                + [ComputationOp("÷", second_operand=second_operand)]
            )
        if isinstance(expr.op, ast.Add):
            second_operand = len(left_ops) - 1 if left_ops and right_ops else None
            return (
                left_ops
                + right_ops
                + [ComputationOp("+", second_operand=second_operand)]
            )
        if isinstance(expr.op, ast.Sub):
            second_operand = len(left_ops) - 1 if left_ops and right_ops else None
            return (
                left_ops
                + right_ops
                + [ComputationOp("−", second_operand=second_operand)]
            )
        if isinstance(expr.op, ast.Pow):
            second_operand = len(left_ops) - 1 if left_ops and right_ops else None
            return (
                left_ops
                + right_ops
                + [ComputationOp("^", second_operand=second_operand)]
            )
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        return _decompose_computation_expr(expr.operand)
    return []


def _function_has_triton_jit_decorator(func: ast.FunctionDef) -> bool:
    for decorator in func.decorator_list:
        name = _expr_name(decorator)
        if name and "triton" in name:
            return True
        if isinstance(decorator, ast.Call):
            call_name = _expr_name(decorator.func)
            if call_name and "triton" in call_name:
                return True
    return False


def _assign_performs_computation(stmt: ast.Assign) -> bool:
    """True when an assignment performs a math/gate op rather than indexing/bookkeeping."""
    computation_calls = frozenset(
        {"sigmoid", "sum", "sqrt", "cumsum", "exp", "exp2", "softplus"}
    )
    for node in ast.walk(stmt.value):
        if isinstance(node, ast.Call) and _call_name(node) in computation_calls:
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if _is_scale_reference(node.left) or _is_scale_reference(node.right):
                return True
    if isinstance(stmt.value, ast.BinOp) and isinstance(
        stmt.value.op, (ast.Mult, ast.Div)
    ):
        left = _expr_name(stmt.value.left)
        right = _expr_name(stmt.value.right)
        if left and right and left.startswith("b_") and right.startswith("b_"):
            return True
        names = {node.id for node in ast.walk(stmt.value) if isinstance(node, ast.Name)}
        if isinstance(stmt.value.op, ast.Mult) and any(
            "rstd" in name for name in names
        ):
            return True
        if isinstance(stmt.value.op, ast.Div) and isinstance(
            stmt.value.left, ast.Constant
        ):
            return True
    return False


def _extract_triton_computation_ops(func: ast.FunctionDef) -> list[ComputationOp]:
    """Collect ordered operation labels from a ``@triton.jit`` kernel body."""
    ops: list[ComputationOp] = []

    def walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign) and _assign_performs_computation(stmt):
                for op in _decompose_computation_expr(stmt.value):
                    if op.label not in {existing.label for existing in ops}:
                        ops.append(op)
            elif isinstance(stmt, ast.If):
                walk(stmt.body)
                walk(stmt.orelse)

    walk(func.body)
    return [op for op in ops if op.label not in {"+"}]


def _extract_triton_computation_labels(func: ast.FunctionDef) -> list[str]:
    return [op.label for op in _extract_triton_computation_ops(func)]


def _launched_triton_kernel_name(func: ast.FunctionDef) -> str | None:
    """Return the Triton kernel function name launched from a Python forward."""
    for call in _iter_calls(func):
        callee = _call_name(call)
        if callee.endswith("_kernel"):
            return callee
        if isinstance(call.func, ast.Subscript):
            sub_name = _expr_name(call.func.value)
            if sub_name and sub_name.endswith("_kernel"):
                return sub_name
    return None


def _resolve_implementation(
    call_name: str,
    imports: dict[str, _ImportTarget],
    owning_module: str,
) -> tuple[str, str, str] | None:
    """Resolve a pipeline call to its defining module symbol."""
    imported = imports.get(call_name)
    if imported is not None:
        return _find_symbol_definition(imported.module, imported.symbol)
    return _find_symbol_definition(owning_module, call_name)


def _follow_to_triton_kernel(
    source: str,
    qualname: str,
    owning_module: str,
    *,
    imports: dict[str, _ImportTarget],
    depth: int = 0,
) -> ast.FunctionDef | None:
    """Follow wrapper/fwd calls until a ``@triton.jit`` kernel function is reached."""
    if depth > 6:
        return None
    tree = _parse_module(source)
    func = _function_def(tree, qualname)
    if func is None:
        return None

    kernel_name = _launched_triton_kernel_name(func)
    if kernel_name is not None:
        kernel_func = _function_def(tree, kernel_name)
        if kernel_func is not None and _function_has_triton_jit_decorator(kernel_func):
            return kernel_func

    module_imports = _collect_import_map(tree, owning_module)
    for call in _iter_calls(func):
        if isinstance(call.func, ast.Attribute) and call.func.attr == "apply":
            class_name = _expr_name(call.func.value)
            if class_name:
                forward_qn = f"{class_name.split('.')[-1]}.forward"
                triton_func = _follow_to_triton_kernel(
                    source,
                    forward_qn,
                    owning_module,
                    imports=imports,
                    depth=depth + 1,
                )
                if triton_func is not None:
                    return triton_func
        callee = _call_name(call)
        if callee in _SKIP_KERNEL_CALLS or callee in _BUILTIN_SKIP_CALLS:
            continue
        if callee.endswith("_bwd"):
            continue
        target = module_imports.get(callee) or imports.get(callee)
        resolved = None
        if target is not None and _should_follow_import(target.module, owning_module):
            resolved = _find_symbol_definition(target.module, target.symbol)
        if resolved is None:
            resolved = _find_symbol_definition(owning_module, callee)
        if resolved is None:
            continue
        inner_source, inner_qn, inner_module = resolved
        if (
            inner_qn.endswith(".forward")
            or inner_qn.endswith("_fwd")
            or callee.endswith("_fwd")
        ):
            triton_func = _follow_to_triton_kernel(
                inner_source,
                inner_qn,
                inner_module,
                imports=imports,
                depth=depth + 1,
            )
            if triton_func is not None:
                return triton_func
    return None


def _should_expand_kernel_op(call_name: str) -> bool:
    """Skip large pipeline handoffs; expand helper kernels and fused ops."""
    if _is_output_pipeline_step(call_name):
        return False
    lowered = call_name.lower()
    if lowered.startswith("chunk_kda_fwd") or lowered.startswith("chunk_gated_delta"):
        return False
    if lowered.startswith("chunk_gla_fwd"):
        return False
    if call_name.endswith("_fwd") and call_name.count("_") >= 3:
        return False
    return True


def introspect_kernel_op_substeps(
    call_name: str,
    imports: dict[str, _ImportTarget],
    owning_module: str,
    *,
    parent_attr: str,
) -> tuple[KernelPipelineStep, ...]:
    """Expand one kernel call into inline sub-operations discovered from its AST."""
    if not _should_expand_kernel_op(call_name):
        return ()

    resolved = _resolve_implementation(call_name, imports, owning_module)
    if resolved is None:
        return ()

    source, qualname, module = resolved
    triton_func = _follow_to_triton_kernel(
        source,
        qualname,
        module,
        imports=imports,
    )
    if triton_func is None:
        return ()

    ops = _extract_triton_computation_ops(triton_func)
    if len(ops) < 2 or len(ops) > 6:
        return ()

    substeps: list[KernelPipelineStep] = []
    for index, op in enumerate(ops):
        second_operand: str | None = None
        if op.second_operand == "input":
            second_operand = "input"
        elif isinstance(op.second_operand, int):
            second_operand = substeps[op.second_operand].attr_name
        substeps.append(
            KernelPipelineStep(
                call_name=op.label,
                attr_name=f"{parent_attr}_sub_{index}",
                class_name="KernelSubOp",
                label=op.label,
                details=[],
                second_operand=second_operand,
            )
        )
    return tuple(substeps)


def _attach_kernel_op_expansions(
    steps: list[KernelPipelineStep],
    imports: dict[str, _ImportTarget],
    owning_module: str,
) -> list[KernelPipelineStep]:
    """Attach AST-derived sub-operation trees to expandable pipeline steps."""
    expanded: list[KernelPipelineStep] = []
    for step in steps:
        children = introspect_kernel_op_substeps(
            step.call_name,
            imports,
            owning_module,
            parent_attr=step.attr_name,
        )
        if children:
            expanded.append(
                KernelPipelineStep(
                    call_name=step.call_name,
                    attr_name=step.attr_name,
                    class_name=step.class_name,
                    label=step.label,
                    details=step.details,
                    condition=step.condition,
                    tensor_inputs=step.tensor_inputs,
                    computation=step.computation,
                    predecessors=step.predecessors,
                    children=children,
                )
            )
        else:
            expanded.append(step)
    return expanded


def introspect_kernel_pipeline(
    details: list[str],
) -> tuple[list[KernelPipelineStep], list[KernelPipelineStep]]:
    """Return ``(pipeline_steps, output_steps)`` from modeling call details and kernel code."""
    flags = parse_kernel_call_flags(details)
    kernel = str(flags.get("_kernel") or kernel_name_from_step_details(details) or "")
    tensor_ports = modeling_tensor_port_names(details)

    import_ref = parse_kernel_import(details)
    if import_ref is None and kernel:
        import_ref = (kernel, kernel)

    collected: list[KernelPipelineStep] = []
    port_producer: dict[str, str] = {}
    if import_ref is not None:
        module, symbol = import_ref
        entrypoints, handoff_calls = _discover_pipeline_entrypoints(module, symbol)
        for source, qualname in entrypoints:
            func = _function_def(_parse_module(source), qualname)
            if func is None:
                continue
            prefix = qualname.split(".")[-1]
            resolved = _find_symbol_definition(module, symbol)
            owning_module = resolved[2] if resolved is not None else module
            collected.extend(
                _extract_pipeline_from_function(
                    func,
                    source=source,
                    prefix=prefix,
                    skip_calls=handoff_calls,
                    tensor_ports=tensor_ports,
                    owning_module=owning_module,
                    flags=flags,
                    port_producer=port_producer,
                )
            )

    pipeline_steps = [
        step
        for step in collected
        if not _is_output_pipeline_step(step.call_name)
        and _step_matches_flags(step, flags)
    ]
    output_steps = [
        step
        for step in collected
        if _is_output_pipeline_step(step.call_name) and _step_matches_flags(step, flags)
    ]
    active_attrs = {step.attr_name for step in (*pipeline_steps, *output_steps)}
    pipeline_steps = _filter_step_predecessors(pipeline_steps, active_attrs)
    output_steps = _filter_step_predecessors(output_steps, active_attrs)

    if import_ref is not None:
        module, symbol = import_ref
        merged_imports: dict[str, _ImportTarget] = {}
        owning_module = module
        entrypoints, _ = _discover_pipeline_entrypoints(module, symbol)
        for source, _qualname in entrypoints:
            tree = _parse_module(source)
            resolved = _find_symbol_definition(module, symbol)
            entry_module = resolved[2] if resolved is not None else module
            merged_imports.update(_collect_import_map(tree, entry_module))
            owning_module = entry_module
        if not merged_imports:
            resolved = _find_symbol_definition(module, symbol)
            if resolved is not None:
                merged_imports = _collect_import_map(
                    _parse_module(resolved[0]), resolved[2]
                )
                owning_module = resolved[2]
        pipeline_steps = _attach_kernel_op_expansions(
            pipeline_steps, merged_imports, owning_module
        )
        output_steps = _attach_kernel_op_expansions(
            output_steps, merged_imports, owning_module
        )

    return pipeline_steps, output_steps
