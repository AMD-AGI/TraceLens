###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""A host function's kernels are the launches its C++ body makes, in order."""

from TraceLens.ModelUtils.native_index import launch_sequences

_SOURCE = """
extern "C" void attention(void* out, int n, void* stream) {
    dim3 grid(n);
    attention_kernel<0, __hip_bfloat16, 128><<<grid, 256, 0, (hipStream_t)stream>>>(out);
    if (n > 1) {
        reduce_kernel<__hip_bfloat16><<<grid, 128>>>(out);
    }
}

void plain(int n) { int m = n * 2; }

void wrapped(float* x) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(scale_kernel), dim3(1), dim3(64), 0, 0, x);
}
"""


def test_launches_are_read_in_order_through_templates_and_branches():
    sequences = launch_sequences(_SOURCE)
    assert sequences["attention"] == ["attention_kernel", "reduce_kernel"]
    assert sequences["wrapped"] == ["scale_kernel"]
    # A function that launches nothing has no sequence.
    assert "plain" not in sequences


def test_a_launcher_object_called_with_the_kernel_is_a_launch():
    source = """
    inline tvm::ffi::Tensor plan(const tvm::ffi::TensorView x) {
      LaunchKernel(div_ceil(n, 256), 256, device)(plan_kernel, params);
      return out;
    }
    """
    assert launch_sequences(source)["plan"] == ["plan_kernel"]


def test_a_macro_that_defines_an_entry_point_exports_the_name_it_is_given():
    from TraceLens.ModelUtils.native_index import _macro_exports, exporting_macros

    header = (
        "#define ENTRYPOINT(func, DECL_ARGS, CALL_ARGS) "
        "static void func##_impl DECL_ARGS; AITER_C_ITFS int func DECL_ARGS { return 0; }"
    )
    source = "ENTRYPOINT(gemm_asm, (void* a, int n), (a, n)) { launch(a); }"
    macros = exporting_macros([header])
    assert macros == {"ENTRYPOINT": 0}
    assert _macro_exports(source, macros) == {"gemm_asm"}


def test_an_op_bound_to_a_function_takes_the_schema_torch_infers_from_it(tmp_path):
    from TraceLens.ModelUtils.native_index import build_native_index

    (tmp_path / "ops.h").write_text(
        "using fptr_t = int64_t;\n"
        "fptr_t init_comm(int64_t rank, int64_t world, std::optional<int64_t> size = std::nullopt);\n"
        "void open_handles(fptr_t fa, const std::vector<torch::Tensor>& handles);\n"
        "void nested(fptr_t fa, const std::vector<std::vector<int64_t>>& offsets);\n"
    )
    (tmp_path / "bind.cc").write_text(
        "TORCH_LIBRARY_EXPAND(lib, m) {\n"
        '  m.def("init_comm", &init_comm);\n'
        '  m.def("open_handles", &open_handles);\n'
        '  m.def("nested", &nested);\n'
        '  m.def("stated(int x) -> int");\n'
        "}\n"
    )
    index = build_native_index([tmp_path])
    assert (
        index.op_schemas[("lib", "init_comm")]
        == "init_comm(int _0, int _1, int? _2) -> int"
    )
    assert (
        index.op_schemas[("lib", "open_handles")]
        == "open_handles(int _0, Tensor[] _1) -> ()"
    )
    # A type torch cannot infer a schema type for: no schema.
    assert ("lib", "nested") not in index.op_schemas
    assert index.op_functions[("lib", "init_comm")] == "init_comm"
    assert index.op_schemas[("lib", "stated")] == "stated(int x) -> int"


def test_a_binding_takes_the_overload_with_as_many_parameters_as_it_lists(tmp_path):
    from TraceLens.ModelUtils.native_index import build_native_index

    (tmp_path / "internal.h").write_text(
        "float prefill(prefill_args args, std::string dtype);\n"
    )
    (tmp_path / "torch.h").write_text(
        "std::vector<at::Tensor> prefill(at::Tensor& q, at::Tensor& k, at::Tensor& v);\n"
    )
    (tmp_path / "bind.cpp").write_text(
        'm.def("prefill", &torch_itfs::prefill, py::arg("q"), py::arg("k"), py::arg("v"));\n'
    )
    index = build_native_index([tmp_path])
    assert [kind for _, kind, *_ in index.signatures["prefill"].parameters] == [
        "torch.Tensor"
    ] * 3


def test_a_function_launching_through_its_calls_has_their_kernels_as_alternatives(
    tmp_path,
):
    from TraceLens.ModelUtils.native_index import build_native_index

    (tmp_path / "ar.cuh").write_text(
        "__global__ void __launch_bounds__(512, 1) reduce_1stage(float* x) {}\n"
        "__global__ void reduce_2stage(float* x) {}\n"
        "__global__ void unrelated(float* x) {}\n"
        "struct Comm {\n"
        "  void allreduce(float* x, int bytes,\n"
        "#ifndef USE_ROCM\n"
        "                 int limit = 20){\n"
        "#else\n"
        "                 int limit = 16)\n"
        "  {\n"
        "#endif\n"
        "#define KL(name) name<<<1, 512>>>(x)\n"
        "    if (bytes < 1024) { KL(reduce_1stage); } else { KL(reduce_2stage); }\n"
        "  }\n"
        "};\n"
    )
    (tmp_path / "ar.cu").write_text(
        "static void _dispatch(long fa, float* x, int n) { reinterpret_cast<Comm*>(fa)->allreduce(x, n); }\n"
        "void all_reduce(long fa, float* x, int n) { _dispatch(fa, x, n); }\n"
    )
    index = build_native_index([tmp_path])
    assert index.kernel_alternatives("all_reduce") == ("reduce_1stage", "reduce_2stage")
    assert index.kernel_alternatives("unknown") == ()


def test_launches_are_read_as_a_rocm_host_pass_reads_the_source():
    from TraceLens.ModelUtils.native_index import launch_sequences

    source = (
        "#ifdef __HIP_DEVICE_COMPILE__\n"
        "__global__ void device_only() {}\n"
        "#else\n"
        "void run_host(float* x) { host_kernel<<<1, 64>>>(x); }\n"
        "#endif\n"
        "#ifdef LAUNCH_IMPL\n"
        "void impl(float* x) { impl_kernel<<<1, 64>>>(x); }\n"
        "#endif\n"
        "void act(float* x) {\n"
        "#ifndef USE_ROCM\n"
        "  cuda_kernel<<<1, 64>>>(x);\n"
        "#else\n"
        "  rocm_kernel<<<1, 64>>>(x);\n"
        "#endif\n"
        "}\n"
        "void dispatch(float* x) {\n"
        "#define KL(name) name<float><<<1, 64>>>(x)\n"
        "  KL(reduce_kernel);\n"
        "}\n"
    )
    found = launch_sequences(source)
    # The host pass's arm; a switch the including file sets keeps its body;
    # the ROCm arm; a macro's parameter is no kernel's name.
    assert found["run_host"] == ["host_kernel"]
    assert found["impl"] == ["impl_kernel"]
    assert found["act"] == ["rocm_kernel"]
    assert "dispatch" not in found


def test_kernel_names_skip_launch_bounds_and_template_arguments():
    from TraceLens.ModelUtils.native_index import _kernel_names

    assert _kernel_names(
        "__global__ void __launch_bounds__(512, 1) reduce_1stage(float* x);\n"
        "__global__ __launch_bounds__(256) void k3(int);\n"
        "template <typename T> __global__ void k4<float>(float);\n"
        "#define G __global__\n"
    ) == ["reduce_1stage", "k3", "k4"]


def test_a_launched_parameter_names_no_kernel():
    from TraceLens.ModelUtils.native_index import launch_sequences

    source = "void run(Kernel kernelName, float* x) {\n  kernelName<<<1, 64>>>(x);\n  fixed_kernel<<<1, 64>>>(x);\n}\n"
    assert launch_sequences(source)["run"] == ["fixed_kernel"]


def test_a_module_invoking_a_binding_macro_inline_exports_its_functions(tmp_path):
    from TraceLens.ModelUtils.native_index import build_native_index

    (tmp_path / "ops.hpp").write_text(
        '#define GEMM_PYBIND \\\n    m.def("gemm_tile", &gemm_tile, py::arg("x"));\n'
    )
    (tmp_path / "gemm_pybind.cu").write_text(
        '#include "ops.hpp"\nPYBIND11_MODULE(NAME, m) { GEMM_PYBIND; }\n'
    )
    index = build_native_index([tmp_path])
    assert index.exports[str(tmp_path / "gemm_pybind.cu")] == {"gemm_tile"}


def test_a_launched_local_and_a_string_s_words_name_no_kernel():
    from TraceLens.ModelUtils.native_index import launch_sequences

    source = (
        "void store(float* x, int mode) {\n"
        '  static_assert(true, "only defined<<<for>>> the indexer");\n'
        "  const auto kernel = mode == 1 ? select_kernel<1>() : select_kernel<2>();\n"
        "  LaunchKernel(4, 64, device)(kernel, params);\n"
        "  fixed_kernel<<<1, 64>>>(x);\n"
        "}\n"
    )
    assert launch_sequences(source)["store"] == ["fixed_kernel"]


def test_a_bound_enum_member_reads_as_pybind_writes_it():
    from TraceLens.ModelUtils.native_index import NativeIndex

    index = NativeIndex()
    index.enums["QuantType"] = [("No", 0), ("per_1x32", 3)]
    member = index.enum_classes["QuantType"].per_1x32
    assert str(member) == "QuantType.per_1x32" and f"{member}" == "QuantType.per_1x32"
    assert member == 3 and int(member) == 3
