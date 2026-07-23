###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Automatic tensor shape metadata for AITER and xDiT-specific kernels
in PyTorch profiler traces.

When enabled, kernel entry-point functions that bypass the aten dispatcher
are wrapped with ``torch.profiler.record_function`` so they appear as
``cpu_op`` events with tensor shapes embedded in the event name.

Usage:
    from kernel_shape_profiler import enable, disable
    enable()   # BEFORE torch.compile — so dynamo captures wrapper identities
    disable()  # after profiling stops

Design:
    For each function in _KERNEL_ENTRY_POINTS we wrap it with
    torch.profiler.record_function, embedding tensor shapes in the event
    name (e.g. ``aiter.ops.mha.flash_attn_func(arg0:[1,1536,24,128], ...)``).

    This approach is torch.compile-compatible: record_function is a no-op
    during dynamo tracing and the wrapper doesn't alter the function's
    return value or signature.  The previous torch.library custom op
    approach used a ``-> ()`` schema with thread-local return stashing,
    which broke torch.compile tracing.

    enable() must be called BEFORE torch.compile so dynamo captures the
    wrapper function identities as guards (no guard invalidation on
    subsequent runs).

Coverage for xDiT + FLUX.1-dev on ROCm:
    - aiter::flash_attn_func (AITER FMHA, BF16)
    - aiter::flash_attn_fp8_pertensor_func (FP8)
    - aiter::mha_fwd (CK kernel dispatch)
    - Tensile GEMMs (Cijk_*), Inductor Triton — covered by record_shapes=True
      on the parent aten::mm / Concrete Inputs; no wrapping needed here.

_KERNEL_ENTRY_POINTS targets the inner kernel looked up via module globals at
call time, not outer dispatch wrappers whose references are captured as instance
attributes at model init.
"""

import contextlib
import functools
import importlib
import inspect
import logging
import sys
import threading
from typing import Any, Callable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_enabled = False
# Each entry: (module_obj, attr_name, original_fn)
_patches: List[Tuple[Any, str, Callable]] = []
# Persistent cache of built wrappers keyed by qualified name.
_built_wrappers: dict = {}


@contextlib.contextmanager
def _preserve_global_torch_state():
    """Snapshot and restore process-global torch defaults around import-heavy regions."""
    get_default_device = getattr(torch, "get_default_device", None)
    saved_device = get_default_device() if get_default_device is not None else None
    saved_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        if saved_device is not None:
            try:
                torch.set_default_device(saved_device)
            except Exception:
                pass
        try:
            torch.set_default_dtype(saved_dtype)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registry of kernel entry points to wrap.
#
# Target the innermost function that:
#   (a) receives tensor arguments
#   (b) is resolved via module-global lookup at call time (not captured as an
#       instance attribute at model init)
#   (c) bypasses the aten dispatcher (so the profiler won't see it otherwise)
#
# xfuser's ATTENTION_FUNCTION_REGISTRY wrappers (_aiter_attn_call, etc.) are
# captured as instance attributes — targeting them won't work. Instead we
# target the inner aiter functions they call, which are resolved via the
# module-global `aiter` import on every call.
#
# NOTE: aiter.ops.mha exports flash_attn_func and mha_fwd. xfuser imports
# `from aiter import flash_attn_func as flash_attn_func_aiter` inside a
# function body (attention_backend.py:307), so it is resolved at call time
# from the aiter module dict — patchable.
# ---------------------------------------------------------------------------
_KERNEL_ENTRY_POINTS = [
    # ── AITER FMHA (BF16 / default path) ──
    ("aiter.ops.mha", "flash_attn_func"),

    # ── AITER FMHA varlen ──
    ("aiter.ops.mha", "flash_attn_varlen_func"),

    # ── AITER FP8 attention ──
    ("aiter.ops.mha", "flash_attn_fp8_pertensor_func"),
    ("aiter.ops.mha", "flash_attn_varlen_fp8_pertensor_func"),

    # ── AITER low-level MHA fwd ──
    ("aiter.ops.mha", "mha_fwd"),

    # ── AITER Triton attention (fallback path when CK unavailable) ──
    ("aiter.ops.triton.attention.mha", "flash_attn_func"),
    ("aiter.ops.triton.attention.mha_v3", "flash_attn_func"),
]

# Auto-discovery: scan modules under these prefixes for additional kernel
# entry points not in the explicit registry.
_AUTO_DISCOVER_PREFIXES: Tuple[str, ...] = (
    "aiter.ops.",
)


# ---------------------------------------------------------------------------
# record_function wrapper
# ---------------------------------------------------------------------------

def _make_record_function_wrapper(
    qualified_name: str,
    original_fn: Callable,
) -> Callable:
    """Wrap a function so it appears in profiler traces with tensor shapes.

    The event name includes tensor shapes for each argument, e.g.:
        aiter.ops.mha.flash_attn_func(arg0:[1,1536,24,128], arg1:[1,1536,24,128])

    This is torch.compile-compatible: record_function is traced as a no-op
    by dynamo, and the wrapper preserves the original function's return value.
    """
    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        if not _enabled:
            return original_fn(*args, **kwargs)
        shape_parts: List[str] = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                shape_parts.append(f"arg{i}:{list(arg.shape)}")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                shape_parts.append(f"{k}:{list(v.shape)}")
        event_name = (
            f"{qualified_name}({', '.join(shape_parts)})"
            if shape_parts else qualified_name
        )
        with torch.profiler.record_function(event_name):
            return original_fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Module resolution and sys.modules patching
# ---------------------------------------------------------------------------

def _resolve_target(module_path: str, attr_name: str):
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return None
    fn = getattr(mod, attr_name, None)
    if fn is None:
        return None
    return mod, attr_name, fn


def _patch_all_references(original_fn: Callable, wrapper_fn: Callable):
    patches = []
    for _mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            mod_dict = vars(mod)
        except TypeError:
            continue
        for attr_name in list(mod_dict.keys()):
            if attr_name.startswith("__"):
                continue
            try:
                if mod_dict[attr_name] is original_fn:
                    setattr(mod, attr_name, wrapper_fn)
                    patches.append((mod, attr_name, original_fn))
            except Exception:
                pass
    return patches


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

_KERNEL_SOURCE_INDICATORS = (
    "[grid",        # Triton kernel launch
    "torch.ops.",   # custom C++/HIP op dispatch
)


def _source_launches_kernel(fn: Callable) -> bool:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return any(marker in source for marker in _KERNEL_SOURCE_INDICATORS)


def _infer_schema_type(param: inspect.Parameter) -> Optional[str]:
    """Check if a parameter annotation indicates a Tensor type."""
    annotation = param.annotation
    if annotation is inspect._empty:
        return None
    _TYPE_MAP = {torch.Tensor: "Tensor", int: "int", float: "float", bool: "bool"}
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    # Check Optional[torch.Tensor]
    args = getattr(annotation, "__args__", ())
    if args and type(None) in args:
        for a in args:
            if a is not type(None) and a in _TYPE_MAP:
                return _TYPE_MAP[a]
    # String annotations
    if isinstance(annotation, str) and annotation in ("torch.Tensor", "Tensor"):
        return "Tensor"
    return None


def _is_likely_kernel_launcher(fn: Callable, sig: inspect.Signature) -> bool:
    has_any_annotation = False
    for param in sig.parameters.values():
        if param.annotation is inspect._empty:
            continue
        has_any_annotation = True
        stype = _infer_schema_type(param)
        if stype is not None and "Tensor" in stype:
            return True
    if has_any_annotation:
        return False
    return _source_launches_kernel(fn)


def _discover_kernel_entry_points() -> List[Tuple[str, str]]:
    import pkgutil
    results: List[Tuple[str, str]] = []
    seen_ids: set = set()

    for prefix in _AUTO_DISCOVER_PREFIXES:
        try:
            pkg = importlib.import_module(prefix.rstrip("."))
        except ImportError:
            continue
        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            continue
        with _preserve_global_torch_state():
            for _importer, mod_name, _is_pkg in pkgutil.walk_packages(
                pkg_path, prefix=prefix.rstrip(".") + "."
            ):
                if mod_name in sys.modules:
                    continue
                leaf = mod_name.rsplit(".", 1)[-1]
                if any(
                    k in leaf
                    for k in ("test", "bench", "tune", "autotune")
                ):
                    continue
                try:
                    importlib.import_module(mod_name)
                except Exception:
                    pass

    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not any(mod_name.startswith(p) for p in _AUTO_DISCOVER_PREFIXES):
            continue
        try:
            mod_dict = vars(mod)
        except TypeError:
            continue
        for attr_name in list(mod_dict.keys()):
            if attr_name.startswith("__"):
                continue
            obj = mod_dict[attr_name]
            if not inspect.isfunction(obj):
                continue
            fn_module = getattr(obj, "__module__", "") or ""
            if not any(fn_module.startswith(p) for p in _AUTO_DISCOVER_PREFIXES):
                continue
            obj_id = id(obj)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            try:
                sig = inspect.signature(obj)
            except (ValueError, TypeError):
                continue
            if not sig.parameters:
                continue
            if any(
                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                for p in sig.parameters.values()
            ):
                continue
            if not _is_likely_kernel_launcher(obj, sig):
                continue
            results.append((mod_name, attr_name))

    logger.debug(
        "Auto-discovered %d kernel candidates from %s",
        len(results),
        ", ".join(_AUTO_DISCOVER_PREFIXES),
    )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enable():
    """Patch registered kernel entry points to appear in profiler traces.

    Must be called BEFORE torch.compile so dynamo captures wrapper identities
    as guards.  _discover_kernel_entry_points() force-imports aiter.ops.*
    subpackages, handling lazy module loading.
    """
    global _enabled
    with _lock:
        if _enabled:
            return

        _patches.clear()
        _wrapped_ids: set = set()

        with _preserve_global_torch_state():
            all_entry_points = list(_KERNEL_ENTRY_POINTS) + _discover_kernel_entry_points()

            for module_path, attr_name in all_entry_points:
                resolved = _resolve_target(module_path, attr_name)
                if resolved is None:
                    continue

                container, name, original_fn = resolved
                fn_id = id(original_fn)
                if fn_id in _wrapped_ids:
                    continue
                _wrapped_ids.add(fn_id)

                qualified_name = f"{module_path}.{name}"

                cached = _built_wrappers.get(qualified_name)
                if cached is not None and cached[1] is original_fn:
                    wrapper = cached[0]
                else:
                    wrapper = _make_record_function_wrapper(qualified_name, original_fn)
                    logger.debug("Registered %s via record_function", qualified_name)
                    _built_wrappers[qualified_name] = (wrapper, original_fn)

                ref_patches = _patch_all_references(original_fn, wrapper)
                _patches.extend(ref_patches)
                if not ref_patches:
                    setattr(container, name, wrapper)
                    _patches.append((container, name, original_fn))

        _enabled = True
        logger.info(
            "kernel_shape_profiler enabled: %d references patched across %d entry points",
            len(_patches),
            len(all_entry_points),
        )


def disable():
    """Restore all patched functions to originals."""
    global _enabled
    with _lock:
        if not _enabled:
            return
        _enabled = False
        for container, name, original_fn in reversed(_patches):
            try:
                setattr(container, name, original_fn)
            except Exception:
                pass
        _patches.clear()
        logger.info("kernel_shape_profiler disabled: all patches restored")
