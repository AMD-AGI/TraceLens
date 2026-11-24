###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stand-ins for compiled third-party libraries that engine source builds on.

Interpreted engine code (:mod:`TraceLens.ModelUtils.source_interpreter`) may
declare its data with a compiled library rather than the standard library.
Nothing from such a library runs here; each stand-in reproduces the part of
its semantics that declaring and using that data needs, on top of the
standard library.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import types


def msgspec_module() -> dict[str, types.ModuleType]:
    """``msgspec``: ``Struct`` classes are dataclasses with the same options."""
    module = types.ModuleType("msgspec")
    structs = types.ModuleType("msgspec.structs")
    msgpack = types.ModuleType("msgspec.msgpack")
    json_module = types.ModuleType("msgspec.json")

    class _NoDefault:
        def __repr__(self):
            return "NODEFAULT"

    nodefault = _NoDefault()
    unset = type("UnsetType", (), {"__bool__": lambda self: False})()

    def field(*, default=nodefault, default_factory=nodefault, name=None):
        if default_factory is not nodefault:
            return dataclasses.field(default_factory=default_factory)
        if default is not nodefault:
            if isinstance(default, list | dict | set):
                value = default
                return dataclasses.field(default_factory=lambda: copy.copy(value))
            return dataclasses.field(default=default)
        return dataclasses.field()

    class Struct:
        """A ``msgspec.Struct``: annotated fields, defaults, optional freezing."""

        def __init_subclass__(
            cls,
            *,
            frozen=False,
            kw_only=False,
            eq=True,
            order=False,
            omit_defaults=False,
            forbid_unknown_fields=False,
            tag=None,
            tag_field=None,
            rename=None,
            repr_omit_defaults=False,
            array_like=False,
            gc=True,
            weakref=False,
            cache_hash=False,
            **kwargs,
        ):
            kwargs.pop("dict", None)
            super().__init_subclass__(**kwargs)
            inherited = getattr(cls, "__struct_options__", {})
            options = {
                "frozen": frozen or inherited.get("frozen", False),
                "kw_only": kw_only or inherited.get("kw_only", False),
            }
            cls.__struct_options__ = options
            for name, value in list(vars(cls).items()):
                if isinstance(value, list | dict | set) and name in getattr(
                    cls, "__annotations__", {}
                ):
                    setattr(
                        cls,
                        name,
                        dataclasses.field(
                            default_factory=lambda value=value: copy.copy(value)
                        ),
                    )
            dataclasses.dataclass(
                cls,
                frozen=options["frozen"],
                kw_only=options["kw_only"],
                eq=eq,
                order=order,
            )
            cls.__struct_fields__ = tuple(item.name for item in dataclasses.fields(cls))

        def __copy__(self):
            return dataclasses.replace(self)

    def defstruct(name, fields, *, bases=None, module=None, namespace=None, **options):
        """A ``Struct`` subclass named ``name`` with ``fields``, built at runtime."""
        annotations = {}
        body = dict(namespace or {})
        for item in fields:
            if isinstance(item, str):
                annotations[item] = object
            elif len(item) == 2:
                annotations[item[0]] = item[1]
            else:
                annotations[item[0]] = item[1]
                body[item[0]] = item[2]
        body["__annotations__"] = annotations
        if module is not None:
            body["__module__"] = module
        return types.new_class(
            name,
            tuple(bases or (Struct,)),
            options,
            lambda namespace_: namespace_.update(body),
        )

    def replace(struct, **changes):
        return dataclasses.replace(struct, **changes)

    def fields(struct):
        return tuple(
            types.SimpleNamespace(
                name=item.name,
                encode_name=item.name,
                type=item.type,
                default=(
                    nodefault if item.default is dataclasses.MISSING else item.default
                ),
                default_factory=(
                    nodefault
                    if item.default_factory is dataclasses.MISSING
                    else item.default_factory
                ),
                required=item.default is dataclasses.MISSING
                and item.default_factory is dataclasses.MISSING,
            )
            for item in dataclasses.fields(struct)
        )

    def asdict(struct):
        return {
            item.name: getattr(struct, item.name) for item in dataclasses.fields(struct)
        }

    def astuple(struct):
        return tuple(asdict(struct).values())

    def force_setattr(struct, name, value):
        object.__setattr__(struct, name, value)

    def to_builtins(value, **kwargs):
        if dataclasses.is_dataclass(value):
            return {key: to_builtins(item) for key, item in asdict(value).items()}
        if isinstance(value, list | tuple):
            return [to_builtins(item) for item in value]
        if isinstance(value, dict):
            return {key: to_builtins(item) for key, item in value.items()}
        return value

    def convert(value, type_=None, **kwargs):
        if type_ is not None and isinstance(value, dict):
            return type_(**value)
        return value

    class DecodeError(ValueError):
        pass

    class ValidationError(DecodeError):
        pass

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, value):
            return json.dumps(to_builtins(value), default=str).encode()

    class Decoder:
        def __init__(self, type_=None, *args, **kwargs):
            self.type = type_

        def decode(self, data):
            return json.loads(data)

    class Ext:
        def __init__(self, code, data):
            self.code = code
            self.data = data

    def encode(value, *args, **kwargs):
        return Encoder().encode(value)

    def decode(data, *args, type=None, **kwargs):  # noqa: A002 - msgspec's name
        return Decoder(type).decode(data)

    structs.replace = replace
    structs.fields = fields
    structs.asdict = asdict
    structs.astuple = astuple
    structs.force_setattr = force_setattr
    for target in (msgpack, json_module):
        target.encode = encode
        target.decode = decode
        target.Encoder = Encoder
        target.Decoder = Decoder
    msgpack.Ext = Ext
    module.Struct = Struct
    module.defstruct = defstruct
    module.field = field
    module.NODEFAULT = nodefault
    module.UNSET = unset
    module.UnsetType = type(unset)
    module.Meta = lambda *args, **kwargs: None
    module.Raw = bytes
    module.DecodeError = DecodeError
    module.ValidationError = ValidationError
    module.MsgspecError = DecodeError
    module.EncodeError = ValueError
    module.to_builtins = to_builtins
    module.convert = convert
    module.structs = structs
    module.msgpack = msgpack
    module.json = json_module
    return {
        "msgspec": module,
        "msgspec.structs": structs,
        "msgspec.msgpack": msgpack,
        "msgspec.json": json_module,
    }


def tqdm_module() -> dict[str, types.ModuleType]:
    """``tqdm``: a progress bar is its iterable, reported to no one."""
    module = types.ModuleType("tqdm")
    auto = types.ModuleType("tqdm.auto")

    class tqdm:  # noqa: N801 - the library's name
        def __init__(self, iterable=None, *args, total=None, **kwargs):
            self.iterable = iterable if iterable is not None else range(total or 0)
            self.n = 0
            self.total = total

        def __iter__(self):
            for item in self.iterable:
                self.n += 1
                yield item

        def __len__(self):
            return len(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, count=1):
            self.n += count

        def set_description(self, *args, **kwargs):
            pass

        set_postfix = set_postfix_str = refresh = set_description

        def close(self):
            pass

        @staticmethod
        def write(*args, **kwargs):
            pass

        @staticmethod
        def _get_free_pos(instance=None):
            return 0

    module.tqdm = auto.tqdm = tqdm
    module.trange = auto.trange = lambda *args, **kwargs: tqdm(range(*args), **kwargs)
    module.auto = auto
    return {"tqdm": module, "tqdm.auto": auto}


def tvm_ffi_modules(load_native) -> dict[str, types.ModuleType]:
    """``tvm_ffi``: an FFI that loads compiled modules, like ``ctypes.CDLL``.

    ``load_native(name)`` gives the stand-in for a compiled module, so a
    function of a module ``load_module`` returns launches what it launches.
    """
    import os

    module = types.ModuleType("tvm_ffi")
    libinfo = types.ModuleType("tvm_ffi.libinfo")
    module_module = types.ModuleType("tvm_ffi.module")
    cpp = types.ModuleType("tvm_ffi.cpp")

    class Object:
        pass

    class Module:
        pass

    def load_module(path, *args, **kwargs):
        return load_native(os.path.basename(str(path)).split(".")[0])

    def load_inline(name, *args, **kwargs):
        return load_native(name)

    def register_object(*args, **kwargs):
        if args and isinstance(args[0], type):
            return args[0]
        return lambda cls: cls

    prefix = "/usr/local/lib/python3/site-packages/tvm_ffi"
    libinfo.find_libtvm_ffi = lambda: f"{prefix}/lib/libtvm_ffi.so"
    libinfo.find_include_path = lambda: f"{prefix}/include"
    libinfo.find_dlpack_include_path = lambda: f"{prefix}/include"
    module.Object = Object
    module.Module = module_module.Module = Module
    module.load_module = load_module
    module.register_object = register_object
    module.register_func = lambda *args, **kwargs: (lambda function: function)
    module.get_global_func = lambda name, allow_missing=False: None
    module.__version__ = "0"
    module.libinfo = libinfo
    module.module = module_module
    module.cpp = cpp
    cpp.load_inline = load_inline
    return {
        "tvm_ffi": module,
        "tvm_ffi.libinfo": libinfo,
        "tvm_ffi.module": module_module,
        "tvm_ffi.cpp": cpp,
    }


def safetensors_modules(checkpoint_for) -> dict[str, types.ModuleType]:
    """``safetensors``: a weight file of the run opens and holds no weights.

    ``checkpoint_for(path)`` gives the
    :class:`~TraceLens.ModelUtils.checkpoint_index.CheckpointFile` a weight
    file path stands for. Shapes come from the model, so nothing is loaded;
    a file holds only the placeholder entries the run states it holds.
    """
    module = types.ModuleType("safetensors")
    torch_module = types.ModuleType("safetensors.torch")

    def checkpoint(path):
        found = checkpoint_for(path)
        if found is None:
            raise FileNotFoundError(f"No such file or directory: {path!r}")
        return found

    class safe_open:  # noqa: N801 - the library's name
        def __init__(self, filename, framework="pt", device="cpu"):
            self.file = checkpoint(filename)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def keys(self):
            return list(self.file.tensors)

        def offset_keys(self):
            return list(self.file.tensors)

        def metadata(self):
            return dict(self.file.metadata)

        def get_tensor(self, name):
            if name not in self.file.tensors:
                raise OSError(f"no tensor {name!r}: weights are not loaded")
            return self.file.tensors[name]

        get_slice = get_tensor

    def load_file(filename, device="cpu"):
        return dict(checkpoint(filename).tensors)

    def refused(*args, **kwargs):
        raise PermissionError("the interpreted source may not write a checkpoint")

    module.safe_open = torch_module.safe_open = safe_open
    module.SafetensorError = torch_module.SafetensorError = OSError
    torch_module.load_file = load_file
    torch_module.save_file = torch_module.save = torch_module.save_model = refused
    module.torch = torch_module
    module.__version__ = "0.4.5"
    return {"safetensors": module, "safetensors.torch": torch_module}


def library_modules(
    load_native=None, checkpoint_for=None
) -> dict[str, types.ModuleType]:
    """Every stand-in, by module name.

    ``load_native`` gives the stand-in of a compiled module by name; the FFI
    loaders need it. ``checkpoint_for`` gives the checkpoint file a weight
    file path stands for; the weight readers need it.
    """
    modules = {**msgspec_module(), **tqdm_module()}
    if load_native is not None:
        modules.update(tvm_ffi_modules(load_native))
    if checkpoint_for is not None:
        modules.update(safetensors_modules(checkpoint_for))
    return modules
