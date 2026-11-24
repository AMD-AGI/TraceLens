###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""What a C++ host function returns is read from its body."""

import torch

import pytest

from TraceLens.ModelUtils.native_host import (
    TensorResults,
    struct_sizes,
    tensor_host_functions,
)
from TraceLens.ModelUtils.source_interpreter import SourceInterpreter

_HEADER = """
struct alignas(16) Plan {
  uint32_t seq_len;
  int32_t write_loc;
  int32_t page;
};
struct Pair { int64_t a; int32_t b; };
using PlanD = device::Plan;
static_assert(sizeof(Stated) == 24);
"""

_SOURCE = """
inline tvm::ffi::Tensor make_plan(
    const tvm::ffi::TensorView indices,
    const tvm::ffi::TensorView table,
    const int32_t ratio) {
  auto B = SymbolicSize{"batch_size"};
  TensorMatcher({B}).with_dtype<int64_t>().verify(indices);
  TensorMatcher({-1, -1}).verify(table);
  const auto batch_size = static_cast<uint32_t>(B.unwrap());
  const auto rows = batch_size * ratio + table.size(1);
  auto D = ffi::empty({rows, sizeof(PlanD)}, kDLUInt8, device);
  LaunchKernel(1, 256, device)(plan_kernel, params);
  return D;
}
"""


def test_struct_sizes_follow_the_c_layout_or_what_the_source_states():
    sizes = struct_sizes([_HEADER])
    assert sizes["Plan"] == 16 and sizes["PlanD"] == 16
    assert sizes["Pair"] == 16 and sizes["Stated"] == 24


def test_the_returned_tensor_is_sized_from_the_arguments_bindings():
    function = tensor_host_functions(_SOURCE)["make_plan"]
    results = TensorResults(SourceInterpreter({}), struct_sizes([_HEADER]))
    indices = torch.empty(4, dtype=torch.int64, device="meta")
    table = torch.empty(10, 7, device="meta")
    assert results.output(function, (indices, table, 2)) == ((15, 16), "uint8")


def test_a_namespaced_return_and_preprocessor_arms_are_read(tmp_path):
    from TraceLens.ModelUtils.native_host import HostEvaluator, host_functions
    from TraceLens.ModelUtils.source_interpreter import SourceInterpreter

    text = """
#define FUSE_SEM 1
OPUS_H opus::index_t mesh_size(int tokens_) { return tokens_ * 4; }
OPUS_H opus::index_t workspace_size(int tokens_, int policy_)
{
#if 1
    if(policy_ == 1)
    {
        return 0;
    }
    return mesh_size(tokens_)
#if FUSE_SEM
        + 8;
#else
        ;
#endif
#else
    return -1;
#endif
}
"""
    functions = host_functions(text)
    interpreter = SourceInterpreter({})
    with interpreter:
        evaluate = HostEvaluator(functions, interpreter)
        assert evaluate("workspace_size", (10, 0)) == 48
        assert evaluate("workspace_size", (10, 1)) == 0


def test_lambdas_tuples_unnamed_parameters_and_unknown_calls(tmp_path):
    from TraceLens.ModelUtils.native_host import HostEvaluator, host_functions
    from TraceLens.ModelUtils.source_interpreter import SourceInterpreter

    text = """
OPUS_H constexpr auto row_col(int tokens_, opus::index_t /*experts_*/)
{
    int rows = [&](){
        if(tokens_ > 1000)
            throw std::runtime_error("too many");
        return min(tokens_, 64);
    }();
    return opus::make_tuple(rows, 8);
}
OPUS_H opus::index_t rows_of(int tokens_) { auto rc = row_col(tokens_, 0); return opus::get<0>(rc); }
OPUS_H int opaque(int tokens_) { return not_a_host_function(tokens_); }
"""
    functions = host_functions(text)
    interpreter = SourceInterpreter({})
    with interpreter:
        evaluate = HostEvaluator(functions, interpreter)
        assert evaluate("rows_of", (10,)) == 10
        assert evaluate("rows_of", (100,)) == 64
        # A call it cannot make leaves the value unknown.
        with pytest.raises(LookupError):
            evaluate("opaque", (1,))


_PLAN_PAIR = """
inline PrefillPlan plan_pair(
    const tvm::ffi::TensorView seq_lens,
    const int32_t num_q_tokens) {
  auto B = SymbolicSize{"batch_size"};
  TensorMatcher({B}).verify(seq_lens);
  if (seq_lens.device().device_type == kDLGPU) {
    auto C = ffi::empty({num_q_tokens, sizeof(Plan)}, kDLUInt8, device);
    auto W = ffi::empty({num_q_tokens, 8}, kDLUInt8, device);
    LaunchKernel(1, 256, device)(plan_kernel, params);
    return PrefillPlan{std::move(C), std::move(W)};
  }
  auto counts = count_host(seq_lens);
  return PrefillPlan{std::move(counts.c), std::move(counts.w)};
}
"""


def test_a_returned_struct_of_tensors_made_in_a_branch_is_each_tensor():
    function = tensor_host_functions(_PLAN_PAIR)["plan_pair"]
    results = TensorResults(SourceInterpreter({}), struct_sizes([_HEADER]))
    seq_lens = torch.empty(2, dtype=torch.int64, device="meta")
    # The path whose tensors are known: the GPU one.
    assert results.output(function, (seq_lens, 32)) == [
        ((32, 16), "uint8"),
        ((32, 8), "uint8"),
    ]
