###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Right now everything is very manually done, but maybe it can be improved
# checkout this as it might be useful: https://github.com/pytorch/pytorch/blob/main/torch/fx/operator_schemas.py

from pprint import pprint
from typing import Dict, Any, List, Optional, Tuple
import logging
import re
import warnings
import time

from .utils import (
    _get_torch_or_raise,
    TensorCfg,
    build_tensor,
    list_profile_tensor_types,
)
from .custom_inits import CustomInit, PagedAttentionInit, MoeRoutingInit

logger = logging.getLogger(__name__)

# -- Known defaults for string arguments the profiler drops ----------------
_STR_ARG_DEFAULTS: Dict[str, str] = {
    "kv_cache_dtype": "auto",
}

# -- Op-name aliases -------------------------------------------------------
_OP_NAME_ALIASES: Dict[str, List[str]] = {
    "_rocm_C::wvSplitK": ["_rocm_C::wvSpltK"],
}

# -- Auto-import registry --------------------------------------------------
_NAMESPACE_IMPORTS: Dict[str, List[str]] = {
    "aiter": ["aiter"],
    "_rocm_C": ["vllm._rocm_C"],
    "_C": ["vllm._C"],
    "_C_cache_ops": ["vllm._C"],
    "vllm": ["vllm._C", "vllm._rocm_C"],
}

_auto_import_attempted: set = set()


def _try_auto_import(op_name: str, verbose: bool = False) -> bool:
    """Try to import the library that registers a custom op's schema.

    Returns True if at least one new module was successfully imported.
    """
    namespace = op_name.split("::")[0] if "::" in op_name else ""
    if not namespace or namespace == "aten":
        return False
    if namespace in _auto_import_attempted:
        return False
    _auto_import_attempted.add(namespace)

    modules = _NAMESPACE_IMPORTS.get(namespace, [namespace])
    imported_any = False
    for mod in modules:
        try:
            __import__(mod)
            print(f"[EventReplayer] Auto-imported '{mod}' for op '{op_name}'")
            imported_any = True
        except ImportError:
            if verbose:
                print(f"[EventReplayer] Could not import '{mod}' for namespace '{namespace}'")
    return imported_any


def _try_resolve(op_name: str):
    """Attempt JIT + torch.ops + module resolution for a single op name.
    Returns (func, source_str) or (None, None).
    """
    torch = _get_torch_or_raise()
    import importlib

    # 1. JIT registry
    try:
        func, _ = torch._C._jit_get_operation(op_name)
        if func is not None:
            return func, "jit"
    except RuntimeError:
        pass

    if "::" in op_name:
        ns, func_name = op_name.split("::", 1)

        # 2. torch.ops namespace
        ns_obj = getattr(torch.ops, ns, None)
        if ns_obj is not None:
            func_obj = getattr(ns_obj, func_name, None)
            if callable(func_obj):
                return func_obj, "torch.ops"

        # 3. Direct Python module lookup
        try:
            mod = importlib.import_module(ns)
            func_obj = getattr(mod, func_name, None)
            if callable(func_obj):
                return func_obj, f"module:{ns}"
        except ImportError:
            pass

    return None, None


def _resolve_op_func(op_name: str, verbose: bool = False):
    """Resolve an op name to a callable, with auto-import on failure.

    Tries: JIT registry -> torch.ops -> module import -> auto-import -> aliases.
    Returns (func, source_str, resolved_name) or raises RuntimeError.
    """
    for attempt in range(2):
        func, source = _try_resolve(op_name)
        if func is not None:
            return func, source, op_name

        for alias in _OP_NAME_ALIASES.get(op_name, []):
            func, source = _try_resolve(alias)
            if func is not None:
                logger.warning(
                    "Op '%s' resolved via alias '%s' (%s).",
                    op_name, alias, source,
                )
                return func, source, alias

        if attempt == 0 and _try_auto_import(op_name, verbose):
            continue
        break

    ns = op_name.split("::")[0] if "::" in op_name else ""
    hint = ""
    if ns and ns != "aten":
        known = _NAMESPACE_IMPORTS.get(ns)
        if known:
            hint = f" Try: {', '.join(f'import {m}' for m in known)}"
        else:
            hint = (f" The namespace '{ns}' is not in the auto-import registry."
                    f" Use EventReplayer.register_namespace('{ns}', ['your.module'])"
                    f" to add it.")

    raise RuntimeError(
        f"Cannot resolve op '{op_name}'.{hint} "
        f"Ensure the library that defines it is imported."
    )


def _search_schemas(op_name: str, verbose: bool = False):
    """Return all registered FunctionSchemas for *op_name*,
    with auto-import on empty results.
    """
    torch = _get_torch_or_raise()

    for attempt in range(2):
        schemas: list = []
        seen_strs: set = set()

        for s in torch._C._jit_get_all_schemas():
            if s.name == op_name:
                s_str = str(s)
                if s_str not in seen_strs:
                    schemas.append(s)
                    seen_strs.add(s_str)

        if "::" in op_name:
            ns, func_name = op_name.split("::", 1)
            ns_obj = getattr(torch.ops, ns, None)
            if ns_obj is not None:
                op_obj = getattr(ns_obj, func_name, None)
                if op_obj is not None:
                    try:
                        for overload_name in op_obj.overloads():
                            overload = getattr(op_obj, overload_name)
                            s = overload._schema
                            s_str = str(s)
                            if s_str not in seen_strs:
                                schemas.append(s)
                                seen_strs.add(s_str)
                    except Exception:
                        try:
                            s = op_obj.default._schema
                            s_str = str(s)
                            if s_str not in seen_strs:
                                schemas.append(s)
                                seen_strs.add(s_str)
                        except Exception:
                            pass

        if schemas or attempt > 0:
            break
        if not _try_auto_import(op_name, verbose):
            break

    if verbose:
        print(f"Found {len(schemas)} schemas for {op_name}:")
        for s in schemas:
            pprint(str(s))
        print("-" * 80)

    return schemas


class EventReplayer:
    _custom_init_registry: List[CustomInit] = [
        PagedAttentionInit(),
        MoeRoutingInit(),
    ]

    @classmethod
    def register_custom_init(cls, init: CustomInit):
        """Add a custom initializer to the registry."""
        cls._custom_init_registry.append(init)

    @classmethod
    def register_namespace(cls, namespace: str, modules: List[str]):
        """Register a namespace-to-module mapping for auto-import."""
        _NAMESPACE_IMPORTS[namespace] = modules

    @classmethod
    def list_custom_inits(cls) -> List[Tuple[str, List[str]]]:
        """List all registered custom initializers and their op patterns."""
        return [(type(i).__name__, i.op_patterns) for i in cls._custom_init_registry]

    def __init__(
        self,
        event: Dict[str, Any],
        device: str = "cuda",
        lazy: bool = False,
        verbose: bool = False,
        auto_init: bool = True,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the EventReplayer with the event data and device type.

        Args:
            event (Dict[str, Any]): From the pytorch profile json data['traceEvents']
            device (str): The device type ('cuda' or 'cpu').
            lazy (bool): If True, defer tensor creation until replay().
            verbose (bool): Flag to enable verbose output.
            auto_init (bool): If True, automatically apply custom initializers
                for ops that need realistic tensor content (e.g., paged attention
                block tables, MoE routing tensors).
            init_kwargs (Dict[str, Any]): Parameters passed to custom initializers
                (e.g., {"moe_distribution": "zipf", "moe_zipf_s": 1.5}).
        """
        self.event = event
        self.device = device
        self.lazy = lazy
        self.verbose = verbose
        self._auto_init = auto_init
        self._init_kwargs = init_kwargs or {}
        self._inits_applied = False
        self._func = None
        self._setup()

    def _setup(self):
        """
        Setup the event replayer by extracting relevant information from the event.
        """
        if self.verbose:
            print(f"Preparing {self.event['name']} event for replay")

        self._func, self._func_source, self._resolved_name = _resolve_op_func(
            self.event["name"], verbose=self.verbose
        )
        if self.verbose:
            print(f"Resolved op via {self._func_source}")
            if self._resolved_name != self.event["name"]:
                print(f"  (aliased from '{self.event['name']}' -> '{self._resolved_name}')")

        try:
            self.matched_schema = EventReplayer._search_schema(
                self.event, self._resolved_name, self.verbose
            )
            self._schemaless = False
        except ValueError:
            if self.verbose:
                print(
                    "No schema found; falling back to schemaless replay "
                    "(all args treated as positional, types inferred from profile)"
                )
            self.matched_schema = None
            self._schemaless = True

        if self._schemaless:
            self.event_replay_IR = EventReplayer._get_event_replay_IR_schemaless(
                self.event, self.verbose, resolved_name=self._resolved_name
            )
        else:
            self.event_replay_IR = EventReplayer._get_event_replay_IR(
                self.event, self.matched_schema, self.verbose
            )
        if not self.lazy:
            if self.verbose:
                print("setting up args and kwargs")
            self.args, self.kwargs = EventReplayer._get_args_kwargs(
                self.event_replay_IR, device=self.device
            )

    def replay(self):
        """
        Replay the event using the matched schema and event replay IR.
        """
        if self.lazy:
            args, kwargs = EventReplayer._get_args_kwargs(
                self.event_replay_IR, device=self.device
            )
        else:
            args, kwargs = self.args, self.kwargs

        if not self._inits_applied and self._auto_init:
            self._apply_custom_inits()

        self._func(*args, **kwargs)

    def _apply_custom_inits(self):
        """Run all applicable custom initializers on this replayer's tensors."""
        for custom_init in self._custom_init_registry:
            if custom_init.applies_to(self):
                try:
                    summary = custom_init.initialize(self, **self._init_kwargs)
                    if summary:
                        print(summary)
                except Exception as e:
                    warnings.warn(
                        f"[custom init] {type(custom_init).__name__} failed: {e}"
                    )
        self._inits_applied = True

    @staticmethod
    def _search_schema(
        event: Dict[str, Any],
        resolved_name: Optional[str] = None,
        verbose: bool = False,
    ) -> Optional["torch._C.FunctionSchema"]:
        name = resolved_name or event["name"]
        op_schemas = _search_schemas(name, verbose=verbose)

        for schema in op_schemas:
            if verbose:
                print(f"Checking schema:")
                pprint(str(schema))
            if EventReplayer._is_schema_match(event, schema, verbose):
                if verbose:
                    print(f"Schema matched successfully")
                    print("-" * 80)
                return schema
            if verbose:
                print("-" * 80)

        raise ValueError(
            f"Cannot find matching schema for {name}. "
            f"Searched {len(op_schemas)} candidate(s). "
            f"Please check the event data and ensure the op's library is imported."
        )

    @staticmethod
    def _is_schema_match(
        event: Dict[str, Any], schema: "torch._C.FunctionSchema", verbose: bool = False
    ) -> bool:
        """
        Check if the event matches the schema.
        """
        op_name, pos_args_schema, kwargs_schema, return_type = (
            EventReplayer.parse_schema_string(schema)
        )
        full_args_schema = pos_args_schema + kwargs_schema
        if len(event["args"]["Input type"]) != len(full_args_schema):
            return False
        for idx in range(len(event["args"]["Input type"])):
            profiled_type = event["args"]["Input type"][idx]
            schema_type = full_args_schema[idx]["arg_type"]
            if verbose:
                print(f"Checking arg {idx}:")
                print(f"\tSchema type: {schema_type}")
                print(f"\tProfiled type: {profiled_type}")

            is_match = True
            if schema_type.endswith("?"):
                schema_type = schema_type[:-1]
                if profiled_type == "":
                    continue
                elif (
                    profiled_type == "ScalarList"
                    and event["args"]["Concrete Inputs"][idx] == "[]"
                ):
                    continue
            if EventReplayer._is_tensor_schema_type(schema_type):
                if profiled_type not in list_profile_tensor_types:
                    is_match = False
            elif schema_type == "bool":
                profiled_value = event["args"]["Concrete Inputs"][idx]
                if profiled_value.lower() not in ("true", "false"):
                    is_match = False
            elif schema_type in ("int", "SymInt"):
                if profiled_type != "Scalar":
                    is_match = False
                profiled_value = event["args"]["Concrete Inputs"][idx]
                if not profiled_value.lstrip("-").isdigit():
                    is_match = False
            elif schema_type in ("float", "Scalar"):
                if profiled_type != "Scalar":
                    is_match = False
                profiled_value = event["args"]["Concrete Inputs"][idx]
                try:
                    float(profiled_value)
                except ValueError:
                    is_match = False
            elif schema_type.startswith("int[") or schema_type.startswith("SymInt["):
                if profiled_type != "ScalarList":
                    is_match = False
                profiled_value = event["args"]["Concrete Inputs"][idx]
                profiled_value_cleaned = [
                    x.strip() for x in profiled_value.strip()[1:-1].split(",")
                ]
                if not all(x.lstrip("-").isdigit() for x in profiled_value_cleaned):
                    is_match = False
            elif schema_type.startswith("bool["):
                if profiled_type != "ScalarList":
                    is_match = False
                profiled_value = event["args"]["Concrete Inputs"][idx]
                profiled_value_cleaned = [
                    x.strip() for x in profiled_value.strip()[1:-1].split(",")
                ]
                if not all(
                    x.lower() in ("true", "false") for x in profiled_value_cleaned
                ):
                    is_match = False
            elif schema_type.startswith("Tensor["):
                raise ValueError(
                    f"Tensor list type not supported: {schema_type} as the "
                    f"tensor shapes are not provided in the event"
                )
            elif schema_type == "str":
                is_match = profiled_type == "Scalar" or profiled_type == ""
            elif schema_type == "ScalarType":
                is_match = profiled_type == "Scalar" or profiled_type == ""
            elif schema_type == "Layout":
                is_match = profiled_type == "Scalar" or profiled_type == ""
            elif schema_type == "Device":
                is_match = profiled_type == "Scalar" or profiled_type == ""
            elif schema_type == "MemoryFormat":
                is_match = profiled_type == "Scalar" or profiled_type == ""
            elif schema_type == "Generator":
                is_match = profiled_type == "" or profiled_type == "Scalar"
            else:
                warnings.warn(
                    f"Unknown schema type: {schema_type}. Skipping this case."
                )
                is_match = False
            if not is_match:
                if verbose:
                    print(
                        f"Schema type {schema_type} does not match profiled type {profiled_type}"
                    )
                return False
        return True

    @staticmethod
    def _is_tensor_schema_type(schema_type: str) -> bool:
        """Check if a schema type string represents a Tensor argument."""
        if schema_type in ("Tensor", "Tensor?"):
            return True
        if schema_type.startswith("Tensor("):
            return True
        return False

    @staticmethod
    def _should_skip_tensor_init(evt_name: str, arg_name: str, arg_idx: int) -> bool:
        """Determine whether a tensor argument is an output-only buffer."""
        if evt_name.endswith("_") and arg_idx == 0:
            return True
        if arg_name == "out":
            return True
        if evt_name == "aten::copy_" and arg_name != "src":
            return True
        return False

    @staticmethod
    def _get_event_replay_IR(
        event: Dict[str, Any], schema: "torch._C.FunctionSchema", verbose: bool = False
    ) -> Dict[str, Any]:
        """Get the event replay IR from the event and schema."""
        evt_name = event["name"]
        op_name, pos_args_schema, kwargs_schema, return_type = (
            EventReplayer.parse_schema_string(schema)
        )
        full_args_schema = pos_args_schema + kwargs_schema
        list_pos_args = []
        list_kwargs = []
        for idx in range(len(event["args"]["Input type"])):
            arg_name = full_args_schema[idx]["arg_name"]
            arg_type = full_args_schema[idx]["arg_type"]

            if verbose:
                print(f"Processing arg {idx}: {arg_name} ({arg_type})")
                print(f"Profiled args type: {event['args']['Input type'][idx]}")
                print(f"Profiled args dims: {event['args']['Input Dims'][idx]}")
                print(f"Profiled args strides: {event['args']['Input Strides'][idx]}")
                print(f"Concrete Inputs: {event['args']['Concrete Inputs'][idx]}")

            if arg_type.endswith("?") and event["args"]["Input type"][idx] == "":
                if arg_type.startswith("str"):
                    default = full_args_schema[idx].get("default")
                    if default is None or default == "None":
                        default = _STR_ARG_DEFAULTS.get(arg_name)
                        if default is not None:
                            logger.warning(
                                "%s arg '%s' (position %d): profiler dropped "
                                "the string value. Using known default '%s'.",
                                evt_name, arg_name, idx, default,
                            )
                    value = "" if default is None else default
                else:
                    value = None
            elif (
                arg_type.endswith("?") and event["args"]["Concrete Inputs"][idx] == "[]"
            ):
                value = []
            else:
                if EventReplayer._is_tensor_schema_type(arg_type):
                    init = "normal"
                    if EventReplayer._should_skip_tensor_init(evt_name, arg_name, idx):
                        init = None
                    profiled_dtype = event["args"]["Input type"][idx]
                    if profiled_dtype in ("long", "long int", "int", "bool", "unsigned char"):
                        init = "zeros" if init == "normal" else init
                    value = TensorCfg(
                        shape=event["args"]["Input Dims"][idx],
                        dtype=profiled_dtype,
                        strides=event["args"]["Input Strides"][idx],
                        init=init,
                    )
                else:
                    arg_str = event["args"]["Concrete Inputs"][idx]
                    if arg_type in ("bool", "bool?"):
                        value = arg_str.lower() == "true"
                    elif arg_type in ("int", "int?", "SymInt", "SymInt?"):
                        value = int(arg_str)
                    elif arg_type in ("Scalar", "Scalar?"):
                        if arg_str.lstrip("-").isdigit():
                            value = int(arg_str)
                        else:
                            value = float(arg_str)
                    elif arg_type in ("float", "float?"):
                        value = float(arg_str)
                    elif arg_type in ("str", "str?"):
                        if not arg_str and arg_name in _STR_ARG_DEFAULTS:
                            value = _STR_ARG_DEFAULTS[arg_name]
                            logger.warning(
                                "%s arg '%s' (position %d): profiler dropped "
                                "the string value. Using known default '%s'.",
                                evt_name, arg_name, idx, value,
                            )
                        else:
                            value = arg_str
                    elif arg_type.startswith("int[") or arg_type.startswith("SymInt["):
                        value = [
                            int(x.strip()) for x in arg_str.strip()[1:-1].split(",")
                        ]
                    elif arg_type.startswith("bool["):
                        value = [
                            x.strip().lower() == "true"
                            for x in arg_str.strip()[1:-1].split(",")
                        ]
                    else:
                        raise ValueError(f"Unsupported arg type: {arg_type}")
            if verbose:
                print(f"Parsed value: {value}")
                print(
                    f"Positional/Keyword: {'Positional' if idx < len(pos_args_schema) else 'Keyword'}"
                )
                print("-" * 80)
            if idx < len(pos_args_schema):
                list_pos_args.append(
                    {"arg_name": arg_name, "arg_type": arg_type, "value": value}
                )
            else:
                list_kwargs.append(
                    {"arg_name": arg_name, "arg_type": arg_type, "value": value}
                )
        return {"list_pos_args": list_pos_args, "list_kwargs": list_kwargs}

    @staticmethod
    def _get_schema_arg_names(op_name: str) -> List[str]:
        """Best-effort: return a list of arg names from any available schema."""
        schemas = _search_schemas(op_name, verbose=False)
        if not schemas:
            return []
        try:
            _, pos_args, kw_args, _ = EventReplayer.parse_schema_string(schemas[0])
            return [a["arg_name"] for a in pos_args + kw_args]
        except Exception:
            return []

    @staticmethod
    def _get_event_replay_IR_schemaless(
        event: Dict[str, Any],
        verbose: bool = False,
        resolved_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a replay IR without a schema by inferring types from profile data."""
        evt_name = event["name"]
        schema_arg_names = EventReplayer._get_schema_arg_names(
            resolved_name or evt_name
        )

        list_pos_args = []
        n_args = len(event["args"]["Input type"])
        for idx in range(n_args):
            profiled_type = event["args"]["Input type"][idx]
            profiled_dims = event["args"]["Input Dims"][idx]
            profiled_strides = event["args"]["Input Strides"][idx]
            concrete = event["args"]["Concrete Inputs"][idx]

            if verbose:
                print(
                    f"Schemaless arg {idx}: type={profiled_type!r} "
                    f"dims={profiled_dims} concrete={concrete!r}"
                )

            if profiled_type in list_profile_tensor_types:
                init = "normal"
                if profiled_type in (
                    "long", "long int", "int", "bool", "unsigned char",
                ):
                    init = "zeros"
                value = TensorCfg(
                    shape=profiled_dims,
                    dtype=profiled_type,
                    strides=profiled_strides,
                    init=init,
                )
                arg_type = "Tensor"
            elif profiled_type == "Scalar" and concrete:
                if concrete.lower() in ("true", "false"):
                    value = concrete.lower() == "true"
                    arg_type = "bool"
                elif concrete.lstrip("-").isdigit():
                    value = int(concrete)
                    arg_type = "int"
                else:
                    try:
                        value = float(concrete)
                        arg_type = "float"
                    except ValueError:
                        value = concrete
                        arg_type = "str"
            elif profiled_type == "" and concrete == "":
                hint_name = (
                    schema_arg_names[idx] if idx < len(schema_arg_names) else None
                )
                default = _STR_ARG_DEFAULTS.get(hint_name) if hint_name else None
                if default is not None:
                    value = default
                    arg_type = "str"
                    logger.warning(
                        "%s arg '%s' (position %d): profiler dropped the "
                        "string value. Using known default '%s'.",
                        evt_name, hint_name, idx, default,
                    )
                else:
                    value = None
                    arg_type = "None"
            elif profiled_type == "ScalarList" and concrete:
                items = [x.strip() for x in concrete.strip()[1:-1].split(",") if x.strip()]
                if all(x.lstrip("-").isdigit() for x in items):
                    value = [int(x) for x in items]
                else:
                    value = [float(x) for x in items]
                arg_type = "list"
            else:
                value = None
                arg_type = "unknown"
                if verbose:
                    print(f"  -> defaulting to None for unknown type")

            inferred_name = (
                schema_arg_names[idx] if idx < len(schema_arg_names) else f"arg{idx}"
            )
            list_pos_args.append(
                {"arg_name": inferred_name, "arg_type": arg_type, "value": value}
            )
            if verbose:
                print(f"  -> {inferred_name}: {arg_type} = {value}")
                print("-" * 80)

        return {"list_pos_args": list_pos_args, "list_kwargs": []}

    @staticmethod
    def _get_args_kwargs(
        event_replay_IR: Dict[str, Any], device: str = "cuda"
    ) -> tuple[List["torch.Tensor"], Dict[str, Any]]:
        """Get the arguments and keyword arguments from the event replay IR."""
        pos_args = []
        for arg in event_replay_IR["list_pos_args"]:
            value = arg["value"]
            if isinstance(value, TensorCfg):
                pos_args.append(build_tensor(value, device=device))
            else:
                pos_args.append(value)
        kwargs = {}
        for arg in event_replay_IR["list_kwargs"]:
            value = arg["value"]
            if isinstance(value, TensorCfg):
                kwargs[arg["arg_name"]] = build_tensor(value, device=device)
            else:
                kwargs[arg["arg_name"]] = value
        return pos_args, kwargs

    @staticmethod
    def parse_schema_string(
        schema,
    ) -> Tuple[str, List[Tuple[str, str, Optional[str], bool]], str]:
        schema_str = str(schema)
        match = re.match(r"^([^\(]+)\((.*)\)\s*->\s*(.*)$", schema_str.strip())
        if not match:
            raise ValueError(f"Cannot parse schema string: {schema_str}")
        op_name, args_str, return_type = match.groups()
        parts = args_str.split("*")
        pos_part = parts[0].rstrip(",").strip()
        kwarg_part = parts[1].lstrip(",").strip() if len(parts) > 1 else ""

        def _parse_arg(raw_arg: str) -> Tuple[str, str, Optional[str], bool]:
            # Greedy (.+) consumes everything up to the last whitespace before
            # a valid identifier, so "Tensor($0! -> ) key_cache" parses correctly.
            m = re.match(
                r"^(.+)\s+([A-Za-z_]\w*(?:=.*)?)$", raw_arg.strip()
            )
            if not m:
                raise ValueError(f"Invalid arg: {raw_arg}")
            arg_type = m.group(1).strip()
            name_default = m.group(2)
            m2 = re.match(r"^([A-Za-z_]\w*)(?:=(.*))?$", name_default)
            if not m2:
                raise ValueError(f"Invalid arg name/default: {name_default}")
            arg_name = m2.group(1)
            default = m2.group(2).strip() if m2.group(2) else None
            return arg_type, arg_name, default

        args = []
        for item in [x.strip() for x in pos_part.split(",") if x.strip()]:
            arg_type, arg_name, default = _parse_arg(item)
            args.append(
                {"arg_type": arg_type, "arg_name": arg_name, "default": default}
            )
        kwargs = []
        for item in [x.strip() for x in kwarg_part.split(",") if x.strip()]:
            arg_type, arg_name, default = _parse_arg(item)
            kwargs.append(
                {"arg_type": arg_type, "arg_name": arg_name, "default": default}
            )

        return op_name.strip(), args, kwargs, return_type.strip()

    def get_repro_info(self) -> Dict[str, Any]:
        """
        Extracts the minimal, serializable information needed to reproduce the event call.
        """
        dict_repro_info = {}
        dict_repro_info["op_name"] = self.event["name"]
        list_pos_args, list_kwargs = (
            self.event_replay_IR["list_pos_args"],
            self.event_replay_IR["list_kwargs"],
        )
        list_pos_args_copy, list_kwargs_copy = list_pos_args.copy(), list_kwargs.copy()
        for idx, val in enumerate(list_pos_args_copy):
            if isinstance(val["value"], TensorCfg):
                list_pos_args_copy[idx]["value"] = val["value"].__dict__
        for idx, val in enumerate(list_kwargs_copy):
            if isinstance(val["value"], TensorCfg):
                list_kwargs_copy[idx]["value"] = val["value"].__dict__
        dict_repro_info["replay_ir"] = {
            "list_pos_args": list_pos_args_copy,
            "list_kwargs": list_kwargs_copy,
        }
        return dict_repro_info
