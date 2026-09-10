"""Tensor shape metadata for Triton / FlashInfer / aiter kernels in profiler traces.

Registered kernel launchers are wrapped as torch custom ops so they appear as
``cpu_op`` events carrying ``Input Dims`` / ``Input type``. ``sitecustomize.py``
drives ``enable()`` / ``disable()`` from the profiler window. See README.md.
"""

import contextlib
import functools
import importlib
import inspect
import logging
import pkgutil
import sys
import threading
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.library import Library

logger = logging.getLogger(__name__)


def _active_default_device_override():
    """Active ``set_default_device`` override, or ``None`` if unset.

    Reads the raw ``CURRENT_DEVICE`` sentinel, not ``get_default_device()``
    (always concrete): restoring ``None`` clears a device mode a module leaked
    at import time, whereas restoring a concrete ``cpu`` would *install* one.
    """
    device_mod = sys.modules.get("torch.utils._device")
    if device_mod is None:
        try:
            import torch.utils._device as device_mod
        except Exception:
            return None
    return getattr(device_mod, "CURRENT_DEVICE", None)


@contextlib.contextmanager
def _preserve_global_torch_state():
    """Snapshot and restore global torch default device & dtype.

    ``enable()`` imports modules that may mutate global state (e.g.
    ``set_default_device("cuda")``); a leaked default device corrupts later CPU
    tensor creation (``Buffer seq_lens_cpu has different device than before``).
    """
    saved_device = _active_default_device_override()
    saved_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        try:
            torch.set_default_device(saved_device)
        except Exception:
            pass
        try:
            torch.set_default_dtype(saved_dtype)
        except Exception:
            pass


_lock = threading.Lock()
_enabled = False
# Created once and NEVER torn down: dropping it frees the registered ops, so a
# wrapper reference that leaked past disable() would dispatch into freed memory.
# The ``if not _enabled`` guard in each wrapper routes leaked calls to the original.
_lib: Optional[Library] = None
_op_counter = 0  # monotonic, never reset, so old op names stay valid
_patches: List[Tuple[Any, str, Callable]] = []  # (module, attr, original_fn)
_built_wrappers: dict = {}  # {qualified_name: (wrapper, original_fn)}, reused across cycles


def _get_or_create_lib() -> Library:
    """Return the process-wide custom-op Library, creating it on first use."""
    global _lib
    if _lib is None:
        _lib = Library("sglang_profiler", "FRAGMENT")
    return _lib


# Registry of kernel entry points to wrap: (module_path, function_name).
# Register the inner kernel, not dispatch wrappers captured as instance attrs.
_KERNEL_ENTRY_POINTS = [
    # ── Triton attention ──
    ("sglang.srt.layers.attention.triton_ops.decode_attention", "decode_attention_fwd"),
    (
        "sglang.srt.layers.attention.triton_ops.decode_attention",
        "decode_attention_fwd_normal",
    ),
    (
        "sglang.srt.layers.attention.triton_ops.decode_attention",
        "decode_attention_fwd_grouped",
    ),
    ("sglang.srt.layers.attention.triton_ops.extend_attention", "extend_attention_fwd"),
    (
        "sglang.srt.layers.attention.triton_ops.prefill_attention",
        "context_attention_fwd",
    ),
    # ── Fused MoE ──
    ("sglang.srt.layers.moe.fused_moe_triton.fused_moe", "invoke_fused_moe_kernel"),
    ("sglang.srt.layers.moe.fused_moe_triton.fused_moe", "moe_align_block_size"),
    (
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels",
        "fused_append_shared_experts",
    ),
    # ── MoE TopK ──
    ("sglang.srt.layers.moe.topk", "biased_grouped_topk_gpu"),
    # ── Layer norm (rmsnorm / fused_add_rmsnorm from sgl_kernel or aiter) ──
    ("sglang.srt.layers.layernorm", "rmsnorm"),
    ("sglang.srt.layers.layernorm", "fused_add_rmsnorm"),
    ("sglang.srt.layers.layernorm", "gemma_rmsnorm"),
    ("sglang.srt.layers.layernorm", "gemma_fused_add_rmsnorm"),
    # ── FP8 quantization ──
    ("sglang.srt.layers.quantization.fp8_utils", "per_token_group_quant_fp8"),
    ("sglang.srt.layers.quantization.fp8_utils", "scaled_fp8_quant"),
    # inner kernels looked up from the module __dict__ on every call
    ("sglang.srt.layers.quantization.fp8_utils", "w8a8_block_fp8_matmul_triton"),
    ("sglang.srt.layers.quantization.fp8_utils", "gemm_a8w8_blockscale"),
    # ── LoRA Triton ──
    ("sglang.srt.lora.triton_ops.sgemm_lora_a", "sgemm_lora_a_fwd"),
    ("sglang.srt.lora.triton_ops.sgemm_lora_b", "sgemm_lora_b_fwd"),
    # ── aiter (AMD) ops ──
    ("aiter.ops.triton.gemm_a8w8_blockscale", "gemm_a8w8_blockscale"),
    ("aiter.ops.triton.batched_gemm_a8w8_blockscale", "batched_gemm_a8w8_blockscale"),
    ("aiter.ops.norm", "rms_norm"),
    ("aiter.ops.norm", "fused_add_rms_norm"),
    # ── FlashInfer MoE (cutedsl) ──
    ("flashinfer.moe", "moe_gemm_fp8_nt_groupwise"),
    # Current-layout paths (SGLang 0.5.18+ moved kernels to sglang.kernels.ops.*;
    # aiter regrouped its ops). Unresolved legacy paths above are skipped silently.
    # ── SGLang Triton attention ──
    ("sglang.kernels.ops.attention.decode_attention", "decode_attention_fwd"),
    ("sglang.kernels.ops.attention.decode_attention", "decode_attention_fwd_normal"),
    ("sglang.kernels.ops.attention.decode_attention", "decode_attention_fwd_grouped"),
    ("sglang.kernels.ops.attention.extend_attention", "extend_attention_fwd"),
    ("sglang.kernels.ops.attention.prefill_attention", "context_attention_fwd"),
    # ── SGLang fused MoE ──
    ("sglang.kernels.ops.moe.fused_moe_triton_kernels", "invoke_fused_moe_kernel"),
    ("sglang.kernels.ops.moe.fused_moe_triton_kernels", "fused_append_shared_experts"),
    ("sglang.kernels.ops.moe.moe_align", "moe_align_block_size"),
    # ── SGLang layer norm ──
    ("sglang.kernels.ops.layernorm.norm", "rmsnorm"),
    ("sglang.kernels.ops.layernorm.norm", "fused_add_rmsnorm"),
    ("sglang.kernels.ops.layernorm.minimax_m3_rmsnorm", "gemma_rmsnorm"),
    ("sglang.kernels.ops.layernorm", "gemma_fused_add_rmsnorm"),
    # ── SGLang LoRA Triton ──
    ("sglang.kernels.ops.gemm.sgemm_lora_a", "sgemm_lora_a_fwd"),
    ("sglang.kernels.ops.gemm.sgemm_lora_b", "sgemm_lora_b_fwd"),
    # ── aiter regrouped Triton ops ──
    ("aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale", "gemm_a8w8_blockscale"),
    ("aiter.ops.triton.normalization.rmsnorm", "rms_norm"),
    # aiter's batched blockscale GEMM was renamed (not just moved); left to
    # auto-discovery rather than pinned to its long current name here.
]

# Auto-discovery prefixes: enable() scans loaded modules under these and wraps
# functions that look like kernel launchers (signature/source heuristics).
_AUTO_DISCOVER_PREFIXES: Tuple[str, ...] = (
    "flashinfer.",
    "sglang.kernels.ops.",
    "sglang.srt.",
    "aiter.ops.",
)


# Schema building — works with or without type annotations.
# Python type → torch schema type.
_TYPE_MAP = {
    torch.Tensor: "Tensor",
    Optional[torch.Tensor]: "Tensor?",
    int: "int",
    float: "float",
    bool: "bool",
    str: "str",
    torch.dtype: "ScalarType",
}

# Same, for PEP 563 string annotations.
_STRING_TYPE_MAP = {
    "torch.Tensor": "Tensor",
    "Tensor": "Tensor",
    "Optional[torch.Tensor]": "Tensor?",
    "Optional[Tensor]": "Tensor?",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "str": "str",
    "torch.dtype": "ScalarType",
}


def _infer_schema_type(param: inspect.Parameter) -> Optional[str]:
    """Map a parameter's annotation to a torch schema type string."""
    annotation = param.annotation
    if annotation is inspect._empty:
        return None

    # String annotations (PEP 563)
    if isinstance(annotation, str):
        if annotation in _STRING_TYPE_MAP:
            return _STRING_TYPE_MAP[annotation]
        if annotation.startswith("Optional[") and annotation.endswith("]"):
            inner = annotation[len("Optional[") : -1]
            base = _STRING_TYPE_MAP.get(inner)
            if base is not None:
                return base if base.endswith("?") else base + "?"
        return None

    # Real type annotations: direct match, then Optional[X] / Union[X, None]
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    origin = getattr(annotation, "__origin__", None)
    if origin is type(None):
        return None
    args = getattr(annotation, "__args__", ())
    if args and type(None) in args:
        for a in args:
            if a is not type(None) and a in _TYPE_MAP:
                return _TYPE_MAP[a] + "?"
    return None


def _build_schema_from_sig(
    sig: inspect.Signature,
    skip_self: bool = False,
) -> Optional[Tuple[str, List[str], List[str]]]:
    """Build a schema from signature annotations.

    Returns ``(schema_str, tensor_params, non_tensor_params)``, or ``None`` if
    there are no tensor params or the signature can't be mapped.
    """
    tensor_params: List[str] = []
    non_tensor_params: List[str] = []
    schema_parts: List[str] = []

    for name, param in sig.parameters.items():
        if skip_self and name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return None

        stype = _infer_schema_type(param)
        if stype is not None and "Tensor" in stype:
            tensor_params.append(name)
            schema_parts.append(f"{stype} {name}")
        else:
            non_tensor_params.append(name)

    if not tensor_params:
        return None

    schema_str = f"({', '.join(schema_parts)}) -> ()"
    return schema_str, tensor_params, non_tensor_params


# Thread-local side channel for non-tensor args and return values.
_tls = threading.local()


def _stash_non_tensor_args(op_name: str, values: dict):
    if not hasattr(_tls, "stash"):
        _tls.stash = {}
    _tls.stash[op_name] = values


def _pop_non_tensor_args(op_name: str) -> dict:
    if not hasattr(_tls, "stash"):
        return {}
    return _tls.stash.pop(op_name, {})


def _stash_return_value(op_name: str, value: Any):
    if not hasattr(_tls, "returns"):
        _tls.returns = {}
    _tls.returns[op_name] = value


def _pop_return_value(op_name: str) -> Any:
    if not hasattr(_tls, "returns"):
        return None
    return _tls.returns.pop(op_name, None)


def _next_op_name(base: str) -> str:
    global _op_counter
    sanitized = base.replace(".", "_").replace("::", "_").replace("-", "_")
    name = f"{sanitized}_{_op_counter}"
    _op_counter += 1
    return name


def _register_op(
    op_name: str,
    schema_str: str,
    original_fn: Callable,
    tensor_param_names: List[str],
    non_tensor_param_names: List[str],
    sig: inspect.Signature,
    skip_self: bool = False,
) -> Optional[Callable]:
    """Register a function as a torch custom op and return a dispatch wrapper, or None on failure."""
    try:
        lib = _get_or_create_lib()
        lib.define(op_name + schema_str)

        def impl(*tensor_args):
            nt_args = _pop_non_tensor_args(op_name)
            full_kwargs = {}
            t_idx = 0
            for pname, param in sig.parameters.items():
                if skip_self and pname == "self":
                    continue
                if pname in tensor_param_names:
                    full_kwargs[pname] = tensor_args[t_idx]
                    t_idx += 1
                elif pname in non_tensor_param_names:
                    if pname in nt_args:
                        full_kwargs[pname] = nt_args[pname]
                    elif param.default is not inspect._empty:
                        full_kwargs[pname] = param.default
            result = original_fn(**full_kwargs)
            # Schema is -> () so stash the real return for the caller.
            _stash_return_value(op_name, result)

        lib.impl(op_name, impl, dispatch_key="CompositeExplicitAutograd")

        torch_op = getattr(torch.ops.sglang_profiler, op_name)

        @functools.wraps(original_fn)
        def dispatch_wrapper(*args, **kwargs):
            # A leaked reference must be a no-op when profiling is inactive.
            if not _enabled:
                return original_fn(*args, **kwargs)
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                return original_fn(*args, **kwargs)

            tensor_args = []
            nt_vals = {}
            for pname, val in bound.arguments.items():
                if skip_self and pname == "self":
                    continue
                if pname in tensor_param_names:
                    tensor_args.append(val)
                elif pname in non_tensor_param_names:
                    nt_vals[pname] = val

            # torch dispatch fails with "no tensor arguments" if all are None.
            if not any(isinstance(t, torch.Tensor) for t in tensor_args):
                return original_fn(*args, **kwargs)

            _stash_non_tensor_args(op_name, nt_vals)
            try:
                torch_op(*tensor_args)
                return _pop_return_value(op_name)
            except Exception:
                # Dispatch failed: clear the stash and fall back to the original.
                _pop_non_tensor_args(op_name)
                _pop_return_value(op_name)
                return original_fn(*args, **kwargs)

        # Lets enable() recognise its own wrappers and never re-wrap one.
        dispatch_wrapper._kernel_shape_wrapper = True
        return dispatch_wrapper

    except Exception as e:
        logger.debug("Failed to register %s: %s", op_name, e)
        return None


def _resolve_target(module_path: str, attr_name: str):
    """Resolve *attr_name* (``"func"`` or ``"Class.method"``) in *module_path*.

    Returns ``(container, attr_name, original_fn, is_method)`` or ``None``.
    """
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return None

    if "." in attr_name:
        cls_name, method_name = attr_name.split(".", 1)
        cls = getattr(mod, cls_name, None)
        if cls is None:
            return None
        fn = getattr(cls, method_name, None)
        if fn is None:
            return None
        return cls, method_name, fn, True
    else:
        fn = getattr(mod, attr_name, None)
        if fn is None:
            return None
        return mod, attr_name, fn, False


def _patch_all_references(original_fn: Callable, wrapper_fn: Callable):
    """Rebind every ``sys.modules`` reference to *original_fn* to *wrapper_fn*.

    Handles the ``from X import Y`` pattern. Returns ``(module, attr, original_fn)``
    tuples for later restoration.
    """
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


def _make_record_function_wrapper(
    qualified_name: str,
    original_fn: Callable,
) -> Callable:
    """Fallback wrapper: emit a ``record_function`` event with shapes in the name.

    Used when a ``torch.library`` schema can't be built (no annotations, ``*args``).
    """

    @functools.wraps(original_fn)
    def wrapper(*args, **kwargs):
        # A leaked reference must be a no-op when profiling is inactive.
        if not _enabled:
            return original_fn(*args, **kwargs)
        shape_parts: List[str] = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                shape_parts.append(f"arg{i}:{list(arg.shape)}")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                shape_parts.append(f"{k}:{list(v.shape)}")
        if shape_parts:
            event_name = f"{qualified_name}({', '.join(shape_parts)})"
        else:
            event_name = qualified_name
        with torch.profiler.record_function(event_name):
            return original_fn(*args, **kwargs)

    wrapper._kernel_shape_wrapper = True
    return wrapper


# Kernel-launch detection heuristics.
# Substrings that strongly indicate a function launches a GPU kernel.
_KERNEL_SOURCE_INDICATORS = (
    "[grid",  # Triton launch pattern: kernel[grid](...)
    "torch.ops.",  # Custom C++/CUDA op dispatch
    "sgl_kernel.",  # sgl-kernel extension entry points
)


def _source_launches_kernel(fn: Callable) -> bool:
    """Return True if *fn* source contains known kernel-launch patterns."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return any(marker in source for marker in _KERNEL_SOURCE_INDICATORS)


def _is_likely_kernel_launcher(fn: Callable, sig: inspect.Signature) -> bool:
    """Decide whether *fn* likely launches a GPU kernel.

    A Tensor annotation includes it; non-tensor-only annotations exclude it; no
    annotations falls back to source pattern matching.
    """
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


def _force_import_submodules(prefix: str) -> None:
    """Recursively import submodules under *prefix* into ``sys.modules``.

    *prefix* is a package name without a trailing dot (e.g. ``"sglang.srt"``).
    """
    try:
        pkg = importlib.import_module(prefix)
    except ImportError:
        return

    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return

    # Restore global defaults after every import so a module that mutates them
    # can't taint later imports or the serving path.
    saved_device = _active_default_device_override()
    saved_dtype = torch.get_default_dtype()

    def _restore_defaults():
        try:
            torch.set_default_device(saved_device)
        except Exception:
            pass
        try:
            torch.set_default_dtype(saved_dtype)
        except Exception:
            pass

    for _importer, mod_name, _is_pkg in pkgutil.walk_packages(
        pkg_path, prefix=prefix + "."
    ):
        if mod_name in sys.modules:
            continue
        # Skip test / benchmark / autotune modules: not entry points, and some
        # set the default device at import.
        leaf = mod_name.rsplit(".", 1)[-1]
        if (
            "test" in leaf
            or leaf.startswith("test_")
            or leaf.startswith("bench_")
            or leaf.endswith("_test")
            or leaf.endswith("_tune")
        ):
            continue
        try:
            importlib.import_module(mod_name)
        except Exception:
            # Optional modules may fail to import depending on environment.
            pass
        finally:
            _restore_defaults()


def _discover_kernel_entry_points() -> List[Tuple[str, str]]:
    """Scan ``sys.modules`` under ``_AUTO_DISCOVER_PREFIXES`` for likely kernel launchers.

    Returns a list of ``(module_path, function_name)`` pairs.
    """
    # Force-import submodules first so deeper kernels appear in sys.modules.
    for prefix in _AUTO_DISCOVER_PREFIXES:
        _force_import_submodules(prefix.rstrip("."))

    results: List[Tuple[str, str]] = []
    seen_ids: set = set()

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

            # Only plain Python functions. @triton.jit objects must NOT be wrapped:
            # rebinding them breaks Triton's device-side global resolution.
            if not inspect.isfunction(obj):
                continue

            # Only functions defined within a target namespace.
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

            # Skip signatures we can't map to a torch schema.
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


def enable():
    """Patch registered kernel entry points to appear as cpu_op."""
    global _enabled
    with _lock:
        if _enabled:
            return

        # _lib / _op_counter are process-persistent; only per-cycle patches rebuild.
        _patches.clear()
        _wrapped_ids: set = set()  # function ids, to avoid double-wrapping

        # Import-heavy region may mutate global torch defaults; restore on exit.
        with _preserve_global_torch_state():
            all_entry_points = (
                list(_KERNEL_ENTRY_POINTS) + _discover_kernel_entry_points()
            )

            for module_path, attr_name in all_entry_points:
                resolved = _resolve_target(module_path, attr_name)
                if resolved is None:
                    continue

                container, name, original_fn, is_method = resolved
                is_plain_function = not is_method

                # Already our own wrapper (a compat re-export resolved to it).
                # Re-wrapping would nest annotations and double-count the call.
                if getattr(original_fn, "_kernel_shape_wrapper", False):
                    logger.debug(
                        "Skipping already-wrapped %s.%s", module_path, attr_name
                    )
                    continue

                fn_id = id(original_fn)
                if fn_id in _wrapped_ids:
                    logger.debug("Skipping duplicate %s.%s", module_path, attr_name)
                    continue
                _wrapped_ids.add(fn_id)

                qualified_name = f"{module_path}.{name}"

                # Reuse a prior-cycle wrapper if the function is unchanged, so
                # each op is defined once and leaked refs stay live.
                wrapper = None
                cached = _built_wrappers.get(qualified_name)
                if cached is not None and cached[1] is original_fn:
                    wrapper = cached[0]

                if wrapper is None:
                    try:
                        sig = inspect.signature(original_fn)
                    except (ValueError, TypeError):
                        logger.debug(
                            "Cannot inspect signature of %s — skipping",
                            qualified_name,
                        )
                        continue

                    base = f"{module_path.split('.')[-1]}_{name}"
                    schema_info = _build_schema_from_sig(sig, skip_self=is_method)

                    if schema_info is not None:
                        # Full tensor annotations → torch.library custom op
                        schema_str, t_names, nt_names = schema_info
                        op_name = _next_op_name(base)
                        wrapper = _register_op(
                            op_name,
                            schema_str,
                            original_fn,
                            t_names,
                            nt_names,
                            sig,
                            skip_self=is_method,
                        )
                        if wrapper is not None:
                            logger.debug(
                                "Registered %s as custom op %s", qualified_name, op_name
                            )

                    if wrapper is None:
                        # No annotations / registration failed → record_function
                        wrapper = _make_record_function_wrapper(
                            qualified_name,
                            original_fn,
                        )
                        logger.debug(
                            "Registered %s via record_function", qualified_name
                        )

                    _built_wrappers[qualified_name] = (wrapper, original_fn)

                # --- Apply patches ---
                if is_plain_function:
                    ref_patches = _patch_all_references(original_fn, wrapper)
                    _patches.extend(ref_patches)
                    if not ref_patches:
                        setattr(container, name, wrapper)
                        _patches.append((container, name, original_fn))
                else:
                    setattr(container, name, wrapper)
                    _patches.append((container, name, original_fn))

            n_discovered = len(all_entry_points) - len(_KERNEL_ENTRY_POINTS)

        _enabled = True
        logger.info(
            "kernel_shape_profiler enabled: %d references patched across "
            "%d entry points (%d explicit + %d auto-discovered)",
            len(_patches),
            len(_KERNEL_ENTRY_POINTS) + n_discovered,
            len(_KERNEL_ENTRY_POINTS),
            n_discovered,
        )


def disable():
    """Restore all patched functions to originals."""
    global _enabled
    with _lock:
        if not _enabled:
            return
        # Flip the flag first so any leaked wrapper short-circuits to the original.
        _enabled = False
        for container, name, original_fn in reversed(_patches):
            try:
                setattr(container, name, original_fn)
            except Exception:
                pass
        _patches.clear()
        # Keep _lib / _op_counter / _built_wrappers alive (see _lib docs).
        logger.info("kernel_shape_profiler disabled: all patches restored")


def is_enabled() -> bool:
    return _enabled
