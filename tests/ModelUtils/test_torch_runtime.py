###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The run's torch: GPU tensors are meta, and what the run records matches the profiler."""

import ast
import ctypes
import textwrap

import pytest
import torch

from TraceLens.ModelUtils.torch_runtime import (
    DeviceFacts,
    DeviceMemory,
    TorchRuntime,
    _c_result,
    _C_INT_DEFAULT,
    _schema_values,
    _symmetric_collective,
    argument_record,
    content_controlled_parameters,
    stored_parameters,
)


@pytest.fixture
def runtime():
    return TorchRuntime(
        DeviceFacts(name="gpu", total_memory=1 << 34, gfx_target="gfx950")
    )


def test_arguments_are_recorded_as_the_profiler_records_them():
    x, index = torch.ones(5), torch.tensor([1, 2])
    # ``Tensor?[]`` is a generic list to the profiler; ``Tensor[]`` is not.
    record = argument_record(
        _schema_values(torch.ops.aten.index.Tensor, (x, [index]), {})
    )
    assert record.types == ("float", "")
    record = argument_record(_schema_values(torch.ops.aten.cat.default, ([x, x],), {}))
    assert record.types == ("TensorList", "Scalar")
    # A Python number in a Tensor slot is the dispatcher's 0-dim tensor.
    record = argument_record(_schema_values(torch.ops.aten.sub_.Tensor, (x, 3), {}))
    assert record.types == ("float", "long int", "Scalar") and record.dims[1] == ()
    # A string is recorded with neither type nor value.
    record = argument_record(
        _schema_values(
            torch.ops.aten.div.Tensor_mode, (x, x), {"rounding_mode": "floor"}
        )
    )
    assert record.types[-1] == "" and record.concrete[-1] == ""


def test_free_device_memory_is_what_live_tensors_leave(runtime):
    total, reserved = 1 << 34, 1 << 30
    runtime.device = DeviceFacts(
        name="gpu", total_memory=total, gfx_target="gfx950", reserved_memory=reserved
    )
    with runtime:
        cuda = runtime.modules["torch.cuda"]
        held = torch.empty(1000, device="cuda")
        free_with, _ = cuda.mem_get_info()
        del held
        free_without, _ = cuda.mem_get_info()
    assert free_without - free_with == 4096  # 4000 bytes, in 512-byte blocks
    assert free_without <= total - 0 and runtime.memory.allocated == 0


def test_device_tensors_have_distinct_addresses_and_views_offset_into_them(runtime):
    with runtime:
        a, b = torch.empty(100, device="cuda"), torch.empty(100, device="cuda")
        assert a.data_ptr() != b.data_ptr()
        assert a[10:].data_ptr() == a.data_ptr() + 40
        assert a.data_ptr() >= DeviceMemory.BASE


def test_lazy_contents_are_read_only_when_needed(runtime):
    fetched = []
    with runtime:
        source = torch.empty(4, 3, device="meta")
        runtime.shadows.set_lazy(
            source, lambda: (fetched.append(1), torch.arange(12.0).reshape(4, 3))[1]
        )
        param = torch.empty(8, 3, device="cuda")
        param.narrow(0, 4, 4).copy_(source)
        doubled = param * 2
        assert fetched == []
        assert doubled[4:].sum().item() == 132.0
        assert fetched == [1]
        unknown = torch.empty(2, device="cuda")
        runtime.shadows.drop(unknown)
        unknown.add_(1)
        with pytest.raises(RuntimeError, match="does not determine"):
            unknown.sum().item()


def _kernel(source):
    return ast.parse(textwrap.dedent(source)).body[0]


_COPY_BY_INDEX = """
def kernel(src, dst, idx, lens, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = tl.load(idx + pid)
    count = tl.load(lens + pid)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + row * n + offsets, mask=offsets < count)
    tl.store(dst + pid * n + offsets, values)
"""


def test_a_triton_kernel_writes_only_through_what_it_stores_to():
    written = stored_parameters(_kernel(_COPY_BY_INDEX))
    # Scalars that only offset a stored pointer may be listed; they hold no contents.
    assert "dst" in written and not written & {"src", "idx", "lens"}


def test_a_triton_kernel_is_steered_by_what_it_loads_into_pointers_and_masks():
    # ``src``'s contents are only copied; ``idx`` and ``lens`` decide where.
    assert content_controlled_parameters(_kernel(_COPY_BY_INDEX)) == frozenset(
        {"idx", "lens"}
    )


def test_collectives_act_as_every_rank_holding_the_same_values():
    data = torch.tensor([3.0, 4.0])
    assert _symmetric_collective("all_reduce", (data,), {}, 4, None)
    assert data.tolist() == [12.0, 16.0]
    # Each rank filling its own slot: every slot gets what this rank put in its own.
    table = torch.tensor([1, 0, 0, 0])
    assert _symmetric_collective("all_reduce", (table,), {}, 4, None)
    assert table.tolist() == [1, 1, 1, 1]
    gathered = [torch.zeros(2), torch.zeros(2)]
    assert _symmetric_collective(
        "all_gather", (gathered, torch.tensor([5.0, 6.0])), {}, 2, None
    )
    assert [g.tolist() for g in gathered] == [[5.0, 6.0], [5.0, 6.0]]


def test_a_c_call_returns_success_in_its_declared_type():
    assert _c_result(_C_INT_DEFAULT) == 0
    assert _c_result(ctypes.c_char_p) == b""
    assert _c_result(ctypes.c_double) == 0.0
    assert _c_result(None) is None


def test_a_dsl_compiled_function_launches_under_the_op_that_calls_it(runtime):
    from TraceLens.ModelUtils.torch_runtime import _dsl_compiler_module

    compiler = _dsl_compiler_module("dsl.compiler", runtime.log)

    def gemm(out, a, b):
        pass

    with runtime:
        a = torch.empty(4, 8, device="cuda")
        kernel = compiler.jit(gemm)
        runtime.log.begin_capture()
        with runtime.log.op("lib::gemm", [a]):
            # Compiling with arguments runs it once too, as the DSL does.
            compiler.compile(kernel, a)(
                compiler.from_dlpack(a).mark_layout_dynamic(leading_dim=1), a, a
            )
        runtime.log.end_capture()
    assert len(runtime.log.launches) == 2
    assert all(
        launch.op_name == "lib::gemm" and launch.record.dims == ((4, 8),)
        for launch in runtime.log.launches
    )


def test_a_dsl_launcher_launches_its_kernels_named_as_the_dsl_names_them(runtime):
    from TraceLens.ModelUtils.torch_runtime import _dsl_compiler_module

    compiler = _dsl_compiler_module("dsl.compiler", runtime.log)

    @compiler.kernel
    def moe_reduction_kernel(x, y):
        pass

    @compiler.kernel(name="stated_name")
    def other_kernel(x):
        pass

    @compiler.jit
    def launch(x, y):
        moe_reduction_kernel(x, y).launch(grid=(1, 1, 1))
        other_kernel(x).launch(grid=(1, 1, 1))

    with runtime:
        x = torch.empty(4, 8, device="cuda")
        runtime.log.begin_capture()
        with runtime.log.op("lib::moe", [x]):
            launch(x, x)
        runtime.log.end_capture()
    assert [item.kernels for item in runtime.log.launches] == [
        ("moe_reduction_kernel_0",),
        ("stated_name",),
    ]

    # A launch whose grid needs the DSL's own types still launches its kernel.
    @compiler.jit
    def launch_typed(x, y):
        moe_reduction_kernel(x, y).launch(grid=(dsl_int64(x), 1, 1))

    def dsl_int64(value):
        raise RuntimeError("needs the DSL's runtime")

    with runtime:
        runtime.log.begin_capture()
        with runtime.log.op("lib::moe", [x]):
            launch_typed(x, x)
        runtime.log.end_capture()
    assert runtime.log.launches[-1].kernels == ("moe_reduction_kernel_0",)


def test_a_native_call_launches_each_kernel_its_generated_source_launches(runtime):
    source = (
        'extern "C" void pa_x(void* out, int n) {\n'
        "    qkv_kernel<1><<<n, 64>>>(out);\n"
        "    reduce_kernel<<<n, 64>>>(out);\n"
        "}\n"
    )
    runtime.generated_sources = lambda: {"/build/pa_x.cpp": source}
    module = runtime.native_module("pa_x")
    module.c_abi = True
    with runtime:
        out = torch.empty(16, device="cuda")
        runtime.log.begin_capture()
        assert module.pa_x(ctypes.c_void_p(out.data_ptr()), 4) == 0
        runtime.log.end_capture()
    assert [launch.kernels for launch in runtime.log.launches] == [
        ("qkv_kernel",),
        ("reduce_kernel",),
    ]


def test_a_c_call_handed_only_host_memory_is_host_code(runtime):
    module = runtime.native_module("rccl")
    module.c_abi = True
    module.machine_functions = {}
    with runtime:
        runtime.log.begin_capture()
        version = ctypes.c_int()
        assert module.ncclGetVersion(ctypes.byref(version)) == 0
        runtime.log.end_capture()
    assert runtime.log.launches == []


def test_a_parameter_wrapping_a_known_tensor_has_its_values(runtime):
    with runtime:
        with torch.device("cuda"):
            scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        assert scale.is_cuda and scale.item() == -1.0
        # An unknown write to any tensor over the storage reaches them all.
        runtime.shadows.drop(scale.data)
        with pytest.raises(RuntimeError, match="does not determine"):
            scale.item()


def test_a_kernel_compiled_ahead_launches_with_its_other_arguments(runtime):
    triton = runtime.modules["triton"]

    def scale_rows(x, out, n, BLOCK):
        pass

    source = triton.compiler.ASTSource(
        fn=triton.jit(scale_rows), signature={}, constexprs={"BLOCK": 64}
    )
    kernel = triton.compile(source, target=None, options=None)
    with runtime:
        x = torch.empty(4, 8, device="cuda")
        runtime.log.begin_capture()
        kernel[(4, 1, 1)](x, torch.empty_like(x), 8)
        runtime.log.end_capture()
    (launch,) = runtime.log.launches
    assert launch.kernels == ("scale_rows",)
    # Gluon kernels launch the same way.
    assert isinstance(
        runtime.modules["triton.experimental.gluon"].jit(scale_rows), type(source.fn)
    )


def test_a_kernel_named_by_its_repr_is_named_for_its_constants(runtime):
    triton = runtime.modules["triton"]

    def quant(x, out, BLOCK: "tl.constexpr" = 128, GROUP: "tl.constexpr" = 32):
        pass

    quant._node = _kernel(
        "def quant(x, out, BLOCK: tl.constexpr = 128, GROUP: tl.constexpr = 32):\n    pass\n"
    )
    named = lambda spec: "quant_" + "_".join(  # noqa: E731
        f"{k}_{spec.constants[k]}" for k in ("BLOCK", "GROUP")
    )
    kernel = triton.jit(repr=named)(quant)
    with runtime:
        x = torch.empty(4, 8, device="cuda")
        runtime.log.begin_capture()
        kernel[(4,)](x, torch.empty_like(x), BLOCK=256)
        runtime.log.end_capture()
    assert runtime.log.launches[0].kernels == ("quant_BLOCK_256_GROUP_32",)


def test_a_heuristic_constant_is_computed_from_the_launch(runtime):
    triton = runtime.modules["triton"]

    def quant(x, n, EVEN: "tl.constexpr"):
        pass

    quant._node = _kernel("def quant(x, n, EVEN: tl.constexpr):\n    pass\n")
    named = lambda spec: f"quant_EVEN_{spec.constants['EVEN']}"  # noqa: E731
    kernel = triton.heuristics({"EVEN": lambda args: args["n"] % 128 == 0})(
        triton.jit(repr=named)(quant)
    )
    with runtime:
        runtime.log.begin_capture()
        kernel[(1,)](torch.empty(4, device="cuda"), 256)
        runtime.log.end_capture()
    assert runtime.log.launches[0].kernels == ("quant_EVEN_True",)


def test_the_store_reads_back_what_every_rank_wrote():
    from TraceLens.ModelUtils.torch_runtime import _TCPStore

    store = _TCPStore(world_size=4)
    store.set("ipc/p0/rank0", b"handle")
    # Another rank's key holds what this rank wrote under its own.
    assert store.get("ipc/p0/rank3") == b"handle"
    assert store.add("barrier", 1) == 4
    with pytest.raises(KeyError):
        store.get("never/written")


def test_a_host_function_allocating_device_memory_by_hand_returns_an_address(runtime):
    from TraceLens.ModelUtils.native_host import HostFunction
    from TraceLens.ModelUtils.native_index import NativeIndex, NativeSignature

    index = NativeIndex()
    index.signatures["alloc_flags"] = NativeSignature(
        "alloc_flags", [("size", "int")], "int"
    )
    index.signatures["free_flags"] = NativeSignature(
        "free_flags", [("ptr", "int")], "None"
    )
    functions = {
        "alloc_flags": HostFunction(
            "alloc_flags",
            ["size"],
            "void* buffer;\n HIP_CALL(hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));\n"
            " return (int64_t)buffer;",
        ),
        "free_flags": HostFunction(
            "free_flags", ["ptr"], "HIP_CALL(hipFree((void*)ptr));"
        ),
    }

    class Host:
        """Host functions the evaluator cannot translate."""

        def __init__(self):
            self.functions = functions

        def __call__(self, name, args):
            raise LookupError(name)

    module = runtime.native_module("ar", index, Host())
    with runtime:
        free = runtime.memory.raw
        first, second = module.alloc_flags(4096), module.alloc_flags(100)
        assert isinstance(first, int) and first != second and first >= DeviceMemory.BASE
        assert runtime.memory.raw == free + 4096 + DeviceMemory.BLOCK
        module.free_flags(first)
        assert runtime.memory.raw == free + DeviceMemory.BLOCK


def test_a_native_op_handed_no_tensor_is_host_bookkeeping(runtime):
    from TraceLens.ModelUtils.native_index import NativeIndex

    index = NativeIndex()
    index.op_schemas[("hostlib", "init_comm")] = "init_comm(int _0, int _1) -> int"
    index.op_schemas[("hostlib", "comm_size")] = "comm_size() -> int"
    index.op_functions[("hostlib", "init_comm")] = "init_comm"
    index.op_functions[("hostlib", "comm_size")] = "comm_size"

    def host(name, args):
        if name == "comm_size":
            return 4096
        raise LookupError(name)

    runtime.register_native_ops(index, host)
    with runtime:
        runtime.log.begin_capture()
        ops = runtime.registry.get("hostlib", "init_comm"), runtime.registry.get(
            "hostlib", "comm_size"
        )
        assert ops[0].impls["CUDA"](0, 8) == 0
        assert ops[1].impls["CUDA"]() == 4096
        runtime.log.end_capture()
    assert runtime.log.launches == []


def test_a_number_a_native_op_leaves_open_is_its_schema_type_s_default(runtime):
    from TraceLens.ModelUtils.torch_runtime import NATIVE_RESULT

    op = runtime.registry.get("lib", "buffer_count")
    op.schema = "(Tensor dummy, int fa) -> (Tensor, int)"
    marker = torch.empty(1)
    op.impls["CUDA"] = lambda dummy, fa: (marker, NATIVE_RESULT)
    # The fake runs the library's stub, whose body is ``...``.
    op.fake = lambda dummy, fa: (torch.empty(1), None)
    with runtime:
        tensor, count = runtime.registry.call(
            "lib", "buffer_count", (torch.empty(1), 7), {}
        )
    assert tensor is marker and count == 0


def test_a_native_call_launching_through_its_calls_records_one_launch_of_its_alternatives(
    runtime,
):
    from TraceLens.ModelUtils.native_index import NativeIndex

    index = NativeIndex()
    index.kernels["/src/ar.cuh"] = {"reduce_1stage", "reduce_2stage"}
    index.named = {
        "all_reduce": {"_dispatch"},
        "_dispatch": {"reduce_1stage", "reduce_2stage"},
    }
    module = runtime.native_module("ar", index)
    with runtime:
        x = torch.empty(16, device="cuda")
        runtime.log.begin_capture()
        module.all_reduce(0, x, x)
        runtime.log.end_capture()
    assert [launch.kernels for launch in runtime.log.launches] == [
        ("reduce_1stage", "reduce_2stage")
    ]


def test_a_triton_launch_repeated_on_the_same_contents_is_emulated_once(
    runtime, tmp_path, monkeypatch
):
    import triton.runtime.interpreter as interpreter_module

    from TraceLens.ModelUtils.source_interpreter import Environment, SourceInterpreter

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "kern.py").write_text(textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def double(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offsets = tl.arange(0, BLOCK)
            tl.store(out_ptr + offsets, tl.load(x_ptr + offsets) * 2)

        def run():
            x = torch.arange(8, device="cuda")
            outputs = []
            for source in (x, x, x, x + 1):
                out = torch.zeros(8, dtype=x.dtype, device="cuda")
                double[(1,)](source, out, BLOCK=8)
                outputs.append(out)
            return [out.cpu().tolist() for out in outputs]
    """))
    runs = []
    original = interpreter_module.GridExecutor.__call__
    monkeypatch.setattr(
        interpreter_module.GridExecutor,
        "__call__",
        lambda self, *a, **k: (runs.append(1), original(self, *a, **k))[1],
    )
    interpreter = SourceInterpreter(
        {"pkg": tmp_path}, Environment(), host_modules=runtime.modules
    )
    with interpreter, runtime:
        outputs = interpreter.import_module("pkg.kern").run()
    assert outputs == [[0, 2, 4, 6, 8, 10, 12, 14]] * 3 + [[2, 4, 6, 8, 10, 12, 14, 16]]
    # Run once per distinct contents.
    assert len(runs) == 2


def test_a_c_call_handed_a_tensor_descriptor_is_a_launch(runtime):
    class TensorDescriptor(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("numel", ctypes.c_int64)]

    module = runtime.native_module("asm_attention")
    module.c_abi = True
    module.machine_functions = {}
    with runtime:
        q = torch.empty(16, device="cuda")
        descriptor = TensorDescriptor(q.data_ptr(), 16)
        runtime.log.begin_capture()
        assert module.attention_fwd(ctypes.pointer(descriptor), 4) == 0
        runtime.log.end_capture()
    assert len(runtime.log.launches) == 1


def test_a_fake_that_cannot_run_leaves_the_kernel_s_result(runtime):
    from TraceLens.ModelUtils.torch_runtime import NATIVE_RESULT

    op = runtime.registry.get("lib", "decode")
    op.schema = "(Tensor dummy, Tensor x, int n) -> ()"
    op.impls["CUDA"] = lambda dummy, x, n: NATIVE_RESULT

    def fake(dummy, x):  # written for fewer arguments than the op takes
        return None

    op.fake = fake
    with runtime:
        assert (
            runtime.registry.call(
                "lib", "decode", (torch.empty(1), torch.empty(2), 3), {}
            )
            is NATIVE_RESULT
        )


def test_indexing_by_an_int32_tensor_first_copies_it_to_int64(runtime):
    with runtime:
        table = torch.empty(9, 64, dtype=torch.int32, device="cuda")
        rows = torch.zeros(4, 1, dtype=torch.int64, device="cuda")
        offsets = torch.zeros(4, 16, dtype=torch.int32, device="cuda")
        runtime.log.begin_capture()
        table[rows, offsets]
        runtime.log.end_capture()
    names = [launch.op_name for launch in runtime.log.launches]
    # The int32 offsets converted, then the stretched int64 rows made whole.
    assert names == ["aten::copy_", "aten::copy_", "aten::index"]
    converted, stretched = runtime.log.launches[:2]
    assert converted.record.dims[:2] == ((4, 16), (4, 16)) and converted.record.types[
        :2
    ] == ("long int", "int")
    assert stretched.record.types[:2] == ("long int", "long int")


def test_a_triton_launch_that_cannot_be_emulated_is_not_tried_again(
    runtime, tmp_path, monkeypatch
):
    import triton.runtime.interpreter as interpreter_module

    from TraceLens.ModelUtils.source_interpreter import Environment, SourceInterpreter

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "kern.py").write_text(textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def broken(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offsets = tl.arange(0, BLOCK)
            tl.store(out_ptr + offsets, not_a_function(tl.load(x_ptr + offsets)))

        def run():
            x = torch.arange(8, device="cuda")
            for _ in range(3):
                # A part of a buffer: its writes are emulated as launched.
                out = torch.zeros(16, dtype=x.dtype, device="cuda")[:8]
                broken[(1,)](x, out, BLOCK=8)
    """))
    runs = []
    original = interpreter_module.GridExecutor.__call__
    monkeypatch.setattr(
        interpreter_module.GridExecutor,
        "__call__",
        lambda self, *a, **k: (runs.append(1), original(self, *a, **k))[1],
    )
    interpreter = SourceInterpreter(
        {"pkg": tmp_path}, Environment(), host_modules=runtime.modules
    )
    with interpreter, runtime:
        interpreter.import_module("pkg.kern").run()
    assert len(runs) == 1


def test_a_triton_launch_whose_writes_nothing_reads_is_never_emulated(
    runtime, tmp_path, monkeypatch
):
    import triton.runtime.interpreter as interpreter_module

    from TraceLens.ModelUtils.source_interpreter import Environment, SourceInterpreter

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "kern.py").write_text(textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def double(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offsets = tl.arange(0, BLOCK)
            tl.store(out_ptr + offsets, tl.load(x_ptr + offsets) * 2)

        def run():
            x = torch.arange(8, device="cuda")
            unread = torch.zeros(8, dtype=x.dtype, device="cuda")
            double[(1,)](x, unread, BLOCK=8)
            read = torch.zeros(8, dtype=x.dtype, device="cuda")
            double[(1,)](x + 1, read, BLOCK=8)
            # Changed after the launch: the launch saw it as it was.
            x.add_(100)
            return read.cpu().tolist()
    """))
    runs = []
    original = interpreter_module.GridExecutor.__call__
    monkeypatch.setattr(
        interpreter_module.GridExecutor,
        "__call__",
        lambda self, *a, **k: (runs.append(1), original(self, *a, **k))[1],
    )
    interpreter = SourceInterpreter(
        {"pkg": tmp_path}, Environment(), host_modules=runtime.modules
    )
    with interpreter, runtime:
        assert interpreter.import_module("pkg.kern").run() == [
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
        ]
    assert len(runs) == 1


def test_a_triton_launch_on_unknown_contents_is_emulated_when_read(
    runtime, tmp_path, monkeypatch
):
    import triton.runtime.interpreter as interpreter_module

    from TraceLens.ModelUtils.source_interpreter import Environment, SourceInterpreter

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "kern.py").write_text(textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def split(x_ptr, u_ptr, known_ptr, mixed_ptr, BLOCK: tl.constexpr):
            offsets = tl.arange(0, BLOCK)
            tl.store(known_ptr + offsets, tl.load(x_ptr + offsets) * 2)
            tl.store(mixed_ptr + offsets, tl.load(u_ptr + offsets) + 1)

        def run():
            x = torch.arange(8, device="cuda")
            unknown = torch.empty(8, dtype=x.dtype, device="cuda")
            outputs = []
            for _ in range(2):
                # Made empty: the launch writes it whole.
                known = torch.empty(8, dtype=x.dtype, device="cuda")
                mixed = torch.zeros(8, dtype=x.dtype, device="cuda")
                split[(1,)](x, unknown, known, mixed, BLOCK=8)
                outputs.append((known, mixed))
            # Only the second launch's writes are read.
            known, mixed = outputs[1]
            return known.cpu().tolist(), mixed
    """))
    runs = []
    original = interpreter_module.GridExecutor.__call__
    monkeypatch.setattr(
        interpreter_module.GridExecutor,
        "__call__",
        lambda self, *a, **k: (runs.append(1), original(self, *a, **k))[1],
    )
    interpreter = SourceInterpreter(
        {"pkg": tmp_path}, Environment(), host_modules=runtime.modules
    )
    with interpreter, runtime:
        known, mixed = interpreter.import_module("pkg.kern").run()
        # What the unknown contents decide stays unknown.
        assert runtime.log.shadows.get(mixed) is None
    assert known == [0, 2, 4, 6, 8, 10, 12, 14]
    # The read launch ran over both fillings; the unread one never ran.
    assert len(runs) == 2


def test_a_triton_launch_reading_another_s_results_runs_neither(
    runtime, tmp_path, monkeypatch
):
    import triton.runtime.interpreter as interpreter_module

    from TraceLens.ModelUtils.source_interpreter import Environment, SourceInterpreter

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "kern.py").write_text(textwrap.dedent("""
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def double(x_ptr, out_ptr, BLOCK: tl.constexpr):
            offsets = tl.arange(0, BLOCK)
            tl.store(out_ptr + offsets, tl.load(x_ptr + offsets) * 2)

        def run(read):
            x = torch.arange(8, device="cuda")
            first = torch.empty(8, dtype=x.dtype, device="cuda")
            double[(1,)](x, first, BLOCK=8)
            second = torch.empty(8, dtype=x.dtype, device="cuda")
            double[(1,)](first, second, BLOCK=8)
            return second.cpu().tolist() if read else None
    """))
    runs = []
    original = interpreter_module.GridExecutor.__call__
    monkeypatch.setattr(
        interpreter_module.GridExecutor,
        "__call__",
        lambda self, *a, **k: (runs.append(1), original(self, *a, **k))[1],
    )
    interpreter = SourceInterpreter(
        {"pkg": tmp_path}, Environment(), host_modules=runtime.modules
    )
    with interpreter, runtime:
        module = interpreter.import_module("pkg.kern")
        # Only kernels read the first launch's results: nothing runs.
        assert module.run(False) is None
        assert runs == []
        # The host reads the second's: both run, once each.
        assert module.run(True) == [0, 4, 8, 12, 16, 20, 24, 28]
    assert len(runs) == 2
