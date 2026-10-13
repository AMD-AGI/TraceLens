###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the TraceLens Agent Analysis per-category analyzers.

Covers every module under category_analyses/ plus the shared kernel classifier:

- utils/classify_kernels: regex-driven kernel classifier.
- Per-category classifiers and extract_category_specific for convolution,
  cpu_idle, elementwise, gemm, moe, norm, other, reduce, sdpa, triton.
- kernel_fusion: fused forms, labeler and pricer; kernel_fusion_analysis: its CLI.
- multi_kernel_analysis: severity classifiers + timeline cross-validation.
- sdpa_analysis: perf-param and kernel-breakdown parsing plus the flash /
  paged attention detectors.
- The per-module main() drivers, exercised end-to-end over tmp_path fixtures.
"""

import json, os, sys, pandas as pd, pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(ANALYSIS_DIR, "category_analyses"))
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Dict, List
from TraceLens.Agent.Analysis.utils.classify_kernels import (
    KERNEL_TYPE_RULES,
    KERNEL_TYPES,
    PRIMITIVES,
    TAXONOMY,
    classify_kernel,
    run_assertions,
)
from TraceLens.Agent.Analysis.utils.kernel_fusion import (
    MIN_IMPACT_SCORE,
    OVERLAP_EFFICIENCY,
    FusionCandidate,
    FusionForms,
    FusionLabeler,
    FusionPricer,
    KernelRecord,
)
from TraceLens.Agent.Analysis.category_analyses.multi_kernel_analysis import (
    classify_memcpy_severity,
    classify_nccl_blocking_severity,
    classify_overlap_severity,
    cross_validate_with_timeline,
)
from TraceLens.Agent.Analysis.category_analyses.sdpa_analysis import (
    classify_sdpa_operation,
    detect_flash_attention,
    detect_paged_attention,
    extract_category_specific as sdpa_extract,
    parse_kernel_breakdown,
    parse_perf_params,
)
from TraceLens.Agent.Analysis.category_analyses.convolution_analysis import (
    extract_category_specific as conv_extract,
)
from TraceLens.Agent.Analysis.category_analyses.cpu_idle_analysis import (
    analyze_kernel_patterns,
    load_gpu_timeline,
    load_ops_summary,
)
from TraceLens.Agent.Analysis.category_analyses.elementwise_analysis import (
    extract_category_specific as elementwise_extract,
)
from TraceLens.Agent.Analysis.category_analyses.norm_analysis import (
    extract_category_specific as norm_extract,
)
from TraceLens.Agent.Analysis.category_analyses.moe_analysis import (
    _check_moe_data,
    extract_category_specific as moe_extract,
)
from TraceLens.Agent.Analysis.category_analyses.triton_analysis import (
    classify_triton_operation,
    extract_category_specific as triton_extract,
)
from TraceLens.Agent.Analysis.category_analyses.gemm_analysis import (
    classify_gemm_operation,
    detect_quantized_gemm,
    extract_category_specific as gemm_extract,
)
from TraceLens.Agent.Analysis.category_analyses.reduce_analysis import (
    detect_softmax,
    extract_category_specific as reduce_extract,
)
from TraceLens.Agent.Analysis.category_analyses.other_analysis import (
    _classify_other_op,
    classify_other_operation,
    extract_category_specific as other_extract,
)
from TraceLens.Agent.Analysis.category_analyses import (
    analysis_utils as au,
    convolution_analysis,
    cpu_idle_analysis,
    elementwise_analysis,
    gemm_analysis,
    kernel_fusion_analysis,
    moe_analysis,
    multi_kernel_analysis,
    norm_analysis,
    other_analysis,
    reduce_analysis,
    sdpa_analysis,
    triton_analysis,
)
from TraceLens.Agent.Analysis.utils import arch_utils
from TraceLens.PerfModel import perf_model
from TraceLens.PerfModel.extensions import rmsnorm_perf_model_extensions as rms_ext
from TraceLens.Trace2Tree.extensions.moe_aiter_pseudo_ops import (
    _create_pseudo_op_moe_fused_aiter,
    is_aiter_fused_moe_kernel,
)
from TraceLens.Trace2Tree.extensions.moe_flydsl_pseudo_ops import (
    FUSED_MOE_PARENT,
    create_pseudo_ops_moe_flydsl,
)
from TraceLens.Trace2Tree.extensions.moe_gptq_awq_pseudo_ops import (
    _extract_topk_from_outplace,
    create_pseudo_ops_moe_gptq_awq,
)
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from tests.test_conv_backward_bytes import (
    _conv_bias_bwd_event,
    _conv_bias_fwd_event,
    _conv_bias_relu_bwd_event,
    _conv_bias_relu_fwd_event,
)
from tests.test_dit_fused_ln_modulate import _fused_ln_fwd_event
from tests.test_evoformer_attention_ops import _event as _evoformer_event
from tests.test_trace2tree import _add_gpu_chain, _mk_event

# Constants
# Perf row keys: name, the four args columns, then the three thread columns.
_MM_KEY = ["aten::mm", "((16, 64), (64, 8))", "('bf16', 'bf16')", "z"] + [""] * 4
_ADD_KEY = ["aten::add", "((16, 8), (16, 8))", "('bf16', 'bf16')", "z"] + [""] * 4

# Real kernel names, shared by the taxonomy and labeler tests.
GEMM = "nvjet_tst_64x152_64x7_2x1_v_bz_NNT"
ADD = "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<long>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<long>, std::array<char*, 3ul>)"
MUL = "void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)"
SILU = "void at::native::vectorized_elementwise_kernel<8, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)"
SIGMOID = "void at::native::vectorized_elementwise_kernel<8, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)"
SOFTMAX = "void at::native::(anonymous namespace)::cunn_SoftMaxForwardReg<c10::BFloat16, float, c10::BFloat16, at::native::(anonymous namespace)::SoftMaxForwardEpilogue, long, 6>(c10::BFloat16*, c10::BFloat16 const*, long)"
TOPK = "void aiter::grouped_topk_opt_sort_kernel<c10::BFloat16, float __vector(4), 8, true, true, false>(c10::BFloat16*, c10::BFloat16 const*, float*, int*, unsigned long, int, int, int, int, float)"
LAYERNORM = "void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float>(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, float*, float*, c10::BFloat16*)"
BATCHNORM = "void at::native::batch_norm_collect_statistics_kernel<at::native::Var, c10::BFloat16, c10::BFloat16, float, int>(at::GenericPackedTensorAccessor<c10::BFloat16 const, 3ul, at::RestrictPtrTraits, int>, float, float, at::GenericPackedTensorAccessor<float, 1ul, at::RestrictPtrTraits, int>, at::GenericPackedTensorAccessor<float, 1ul, at::RestrictPtrTraits, int>)"
QUANT = "void aiter::dynamic_per_group_scaled_quant_kernel<std::bfloat16_t, opus::fp4_t, 32, 32, true, 64, true>(opus::fp4_t*, float*, std::bfloat16_t const*, float const*, long, int, int, long, int const*, int)"
# ck_tile RMSNorm built with FusedAddEnumE0 / FusedQuantEnumE0: a plain norm.
RMSNORM = "_ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwdINS_27Rmsnorm2dFwdPipelineOnePassINS_27Rmsnorm2dFwdPipelineProblemIDF16bDF16bfDF16bDF16bDF16bffNS_19Generic2dBlockShapeINS_8sequenceIJLi1ELi3072EEEENS5_IJLi1ELi128EEEENS5_IJLi1ELi8EEEEEENS_18Rmsnorm2dFwdTraitsILb1ELb0ELb0ELb0ELNS_21Rmsnorm2dFusedAddEnumE0ELNS_23Rmsnorm2dFusedQuantEnumE0ELNS_22Rmsnorm2dSensitiveEnumE0EEEEENS_33Rmsnorm2dFwdPipelineDefaultPolicyEEENS_17Default2DEpilogueINS_24Default2DEpilogueProblemIfDF16bLb0ELb1ELb0EEEvEEEEJNSM_5KargsEEEEvDpT1_"
ADD_RMSNORM_QUANT = "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib"
AR_STORE = "void aiter::reduce_scatter_cross_device_store<std::bfloat16_t, 2>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, int, int, int, int)"
AR_RMSNORM = "void aiter::local_device_load_rmsnorm<std::bfloat16_t, 512, 2>(aiter::RankSignals, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, float, int, int, int, int)"
# A compiled kernel the taxonomy types as a plain (unfused) elementwise.
TRITON = "triton_poi_fused_embedding_0"
MEMCPY = "Memcpy HtoD (Host -> Device)"
NCCL = "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
UNKNOWN_KERNEL = "mystery_kernel"
CONV = "igemm_bwd_gtcx3_nhwc_bf16_bx0_ex1_bt128x128x32_wt32x32x8_ws1x1_wr2x2_ta1x8x2x1_1x4x1x64_tb1x8x1x2_1x4x1x64_mh_gkgs"
TRANSPOSE = "batched_transpose_4x64_half"
MOE_GEMM = (
    "mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_persist_cu256"
)
LABELER = FusionLabeler(TAXONOMY, FusionForms(TAXONOMY))


def _op_group(perf_key, kernel_names, time_us, count):
    return {
        "perf_key": perf_key,
        "kernel_names": kernel_names,
        "time_us": time_us,
        "count": count,
    }


def _name_key(name):
    return [name, "", ""]


# ----- KernelTaxonomy / classify_kernel -----


@pytest.mark.parametrize(
    "name,expected_type,expected_cat,expected_covers",
    [
        ("Memcpy HtoD (Host -> Device)", "MemCpy", "MemCpy", ("copy",)),
        ("Memset (Device)", "MemSet", "MemCpy", ("copy",)),
        (
            "_combined_routing_memset",
            "MoE Routing Memset",
            "Elementwise-MoE",
            ("route",),
        ),
        ("nvjet_tst_64x152_64x7_2x1_v_bz_NNT", "GEMM", "GEMM", ("gemm",)),
        (
            "_matmul_ogs_NNT_bf16xbf16xmxfp4_16x128x256x1_swiglu",
            "MoE GateUp+SwiGLU GEMM",
            "GEMM-MoE",
            ("moe_gemm", "act", "mul"),
        ),
        (
            "_ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwdINS_27Rmsnorm2dFwdPipelineOnePassINS_27Rmsnorm2dFwdPipelineProblemIDF16bDF16bfDF16bDF16bDF16bffNS_19Generic2dBlockShapeINS_8sequenceIJLi1ELi3072EEEENS5_IJLi1ELi128EEEENS5_IJLi1ELi8EEEEEENS_18Rmsnorm2dFwdTraitsILb1ELb0ELb0ELb0ELNS_21Rmsnorm2dFusedAddEnumE0ELNS_23Rmsnorm2dFusedQuantEnumE0ELNS_22Rmsnorm2dSensitiveEnumE0EEEEENS_33Rmsnorm2dFwdPipelineDefaultPolicyEEENS_17Default2DEpilogueINS_24Default2DEpilogueProblemIfDF16bLb0ELb1ELb0EEEvEEEEJNSM_5KargsEEEEvDpT1_",
            "RMSNorm",
            "Normalization",
            ("norm",),
        ),
        (
            "_ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwdINS_27Rmsnorm2dFwdPipelineOnePassINS_27Rmsnorm2dFwdPipelineProblemIDF16bDF16bfDF16bDF16bDF16bffNS_19Generic2dBlockShapeINS_8sequenceIJLi1ELi3072EEEENS5_IJLi1ELi128EEEENS5_IJLi1ELi8EEEEEENS_18Rmsnorm2dFwdTraitsILb1ELb0ELb0ELb0ELNS_21Rmsnorm2dFusedAddEnumE1ELNS_23Rmsnorm2dFusedQuantEnumE0ELNS_22Rmsnorm2dSensitiveEnumE0EEEEENS_33Rmsnorm2dFwdPipelineDefaultPolicyEEENS_17Default2DEpilogueINS_24Default2DEpilogueProblemIfDF16bLb0ELb1ELb0EEEvEEEEJNSM_5KargsEEEEvDpT1_",
            "Add+RMSNorm",
            "Normalization",
            ("add", "norm"),
        ),
        (
            "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib",
            "Add+RMSNorm+Quant",
            "Normalization",
            ("add", "norm", "quant"),
        ),
        (
            "_fused_rms_mxfp4_quant_kernel",
            "RMSNorm+Quant",
            "Normalization",
            ("norm", "quant"),
        ),
        (
            "void aiter::local_device_load_rmsnorm<std::bfloat16_t, 512, 2>(aiter::RankSignals, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, float, int, int, int, int)",
            "AllReduce+RMSNorm",
            "Normalization",
            ("collective", "norm"),
        ),
        (
            "void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float>(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, float*, float*, c10::BFloat16*)",
            "LayerNorm",
            "Normalization",
            ("norm",),
        ),
        (
            "void at::native::batch_norm_collect_statistics_kernel<at::native::Var, c10::BFloat16, c10::BFloat16, float, int>(at::GenericPackedTensorAccessor<c10::BFloat16 const, 3ul, at::RestrictPtrTraits, int>, float, float, at::GenericPackedTensorAccessor<float, 1ul, at::RestrictPtrTraits, int>, at::GenericPackedTensorAccessor<float, 1ul, at::RestrictPtrTraits, int>)",
            "BatchNorm",
            "Normalization",
            ("batchnorm",),
        ),
        ("attn_fwd", "Attention (flash)", "SDPA", ("gemm", "softmax", "gemm")),
        (
            "aiter::mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps",
            "Attention (MLA)",
            "SDPA",
            ("gemm", "softmax", "gemm"),
        ),
        (
            "_ZN5aiter49fuse_qk_rope_concat_and_cache_mla_per_head_kernelIDF16bDB8_S1_LN4vllm18Fp8KVCacheDataTypeE1ELS3_1ELi8EEEvPKT_S6_S6_S6_PT0_PT1_PKlSC_S6_S6_iiiiiiiiiiiiiiPKfSE_bb",
            "RoPE+KV Store",
            "Elementwise",
            ("rope", "kv_store"),
        ),
        (
            "void aiter::grouped_topk_opt_sort_kernel<c10::BFloat16, float __vector(4), 8, true, true, false>(c10::BFloat16*, c10::BFloat16 const*, float*, int*, unsigned long, int, int, int, int, float)",
            "TopK",
            "Elementwise",
            ("topk",),
        ),
        (
            "_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_ES3_S5_EEEEvPS3_PS4_i",
            "Act+Mul",
            "Elementwise",
            ("act", "mul"),
        ),
        (
            "void vllm::rotary_embedding_kernel<c10::BFloat16, true>(long const*, c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, int, long, long, long, int, int, int)",
            "Rotary Embedding",
            "Elementwise",
            ("rope",),
        ),
        (
            "void aiter::allgather_lastdim<std::bfloat16_t, 2>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int, int)",
            "Collective",
            "Communication",
            ("collective",),
        ),
        (
            "void at::native::(anonymous namespace)::cunn_SoftMaxForwardReg<c10::BFloat16, float, c10::BFloat16, at::native::(anonymous namespace)::SoftMaxForwardEpilogue, long, 6>(c10::BFloat16*, c10::BFloat16 const*, long)",
            "Softmax",
            "Elementwise",
            ("softmax",),
        ),
        (
            "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4> >(at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4>)",
            "Reduce",
            "Elementwise",
            ("reduce",),
        ),
        (
            "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<long>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<long>, std::array<char*, 3ul>)",
            "Add",
            "Elementwise",
            ("add",),
        ),
        (
            "void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)",
            "Mul",
            "Elementwise",
            ("mul",),
        ),
        (
            TRANSPOSE,
            "LayoutTransform",
            "Elementwise",
            ("pointwise",),
        ),
        (
            "_ZN2ck16tensor_operation6device12_GLOBAL__N_139kernel_grouped_conv_fwd_xdl_cshuffle_v3INS_28GridwiseGemm_xdl_cshuffle_v3INS_13tensor_layout4gemm8RowMajorENS6_11ColumnMajorES7_ttfttNS0_12element_wise11PassThroughESA_SA_LNS1_18GemmSpecializationE7ELi256ELi64ELi64ELi64ELi8ELi8ELi32ELi32ELi1ELi1ENS_8SequenceIJLi8ELi32ELi1EEEENSC_IJLi1ELi0ELi2EEEESE_Li2ELi8ELi8ELb0ELi0ESD_SE_SE_Li2ELi8ELi8ELb0ELi0ELi1ELi1ENSC_IJLi1ELi32ELi1ELi8EEEELi8ELNS_26BlockGemmPipelineSchedulerE0ELNS_24BlockGemmPipelineVersionE2EttLb0ELb0EEENS_16TensorDescriptorINS_5TupleIJNS_5EmbedINSK_IJiiiiEEESM_Lb0EEENS_11PassThroughIiEENS_3PadIiiiLb0EEESR_SP_SP_NSL_INSK_IJiiEEESS_Lb0EEEST_SP_NS_23Merge_v2_magic_divisionINSK_IJiiiEEEEESW_NS_8RightPadIiiLb0EEESY_NS_7UnMergeISS_Lb0EEESP_EEENSK_IJNSC_IJLi0EEEENSC_IJLi1EEEENSC_IJLi2EEEENSC_IJLi3EEEENSC_IJLi4EEEENSC_IJLi5EEEENSC_IJLi6EEEENSC_IJLi7EEEENSC_IJLi8EEEENSC_IJLi9ELi11ELi13EEEENSC_IJLi10ELi12ELi14EEEENSC_IJLi15EEEENSC_IJLi16EEEENSC_IJLi18EEEENSC_IJLi17EEEEEEENSK_IJNSC_IJLi1ELi2ELi3ELi4EEEES17_S18_S19_S1A_NSC_IJLi9EEEENSC_IJLi10ELi11EEEENSC_IJLi12ELi13EEEENSC_IJLi14EEEES1D_S1E_S1G_S1F_NSC_IJLi19ELi20EEEENSC_IJLi21EEEEEEENSC_IJLi19ELi21ELi20EEEElEENSJ_INSK_IJS10_SY_SY_S10_SP_EEENSK_IJS12_S13_S14_S16_S15_EEENSK_IJNSC_IJLi1ELi2EEEES15_S16_NSC_IJLi5ELi6EEEES19_EEENSC_IJLi5ELi7ELi6EEEElEENSJ_INSK_IJST_SY_SY_NSZ_INSK_IJiNS_17integral_constantIiLi64EEEEEELb0EEES22_EEENSK_IJS12_S13_S14_S15_S16_EEENSK_IJS1U_S15_S16_S1V_NSC_IJLi7ELi8EEEEEEENSC_IJLi5ELi6ELi7ELi8EEEElEENS1_30ComputePtrOffsetOfStridedBatchILi1ELi1ELi0EvEELb1ELNS_25InMemoryDataOperationEnumE0ELi1ELNS_10TailNumberE10EEEvNT_8ArgumentET0_T1_T2_T3_S2I_i",
            "Convolution",
            "Convolution",
            ("gemm",),
        ),
        (
            CONV,
            "Convolution",
            "Convolution",
            ("gemm",),
        ),
        (
            "void (anonymous namespace)::multimem_all_reduce_kernel<c10::BFloat16, 16>(c10::BFloat16*, unsigned long, unsigned int**, unsigned long, unsigned long)",
            "Library Collective",
            "Communication",
            ("library_collective",),
        ),
        ("triton_poi_fused_0", "Compiled", "Elementwise", ("pointwise", "reduce")),
        ("__amd_rocclr_copyBuffer", "Elementwise", "Elementwise", ("pointwise",)),
        (
            "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
            "Library Collective",
            "Communication",
            ("library_collective",),
        ),
        (
            "void rcclGenericKernel<2, false>(ncclDevKernelArgsStorage<4096ul>)",
            "Library Collective",
            "Communication",
            ("library_collective",),
        ),
        (
            SILU,
            "Activation (SiLU)",
            "Elementwise",
            ("act",),
        ),
        (
            "_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb0ELi3EEEvPT0_PS2_if",
            "Act+Mul",
            "Elementwise",
            ("act", "mul"),
        ),
        (
            "mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32",
            "MoE GEMM+SiLU (aiter)",
            "GEMM-MoE",
            ("moe_gemm", "act", "mul"),
        ),
        (
            "mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_persist_cu256",
            "MoE GEMM (CK)",
            "GEMM-MoE",
            ("moe_gemm",),
        ),
        (
            "kernel_paged_attention_2d",
            "Attention (AMD unified)",
            "SDPA",
            ("gemm", "softmax", "gemm"),
        ),
        ("_causal_conv1d_update_kernel", "Causal Conv1D", "SDPA", ("pointwise",)),
        (
            "triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0",
            "RMSNorm (fused triton)",
            "Normalization",
            ("add", "norm"),
        ),
        (
            "triton_poi_fused__to_copy_add_mean_moe_forward_shared_pow_rsqrt_0",
            "RMSNorm (shared expert + norm)",
            "Normalization",
            ("add", "norm"),
        ),
        (
            "triton_poi_fused_mul_sigmoid_0",
            "Activation (SiLU)",
            "Elementwise",
            ("act",),
        ),
    ],
)
def test_taxonomy_classify(name, expected_type, expected_cat, expected_covers):
    kernel = TAXONOMY.classify(name)
    assert (kernel.type, kernel.perf_category, kernel.covers) == (
        expected_type,
        expected_cat,
        expected_covers,
    )


def test_classify_kernel_reports_winning_priority():
    """A cudnn SDPA kernel also matches the convolution rule (fprop, 5); the
    flash-attention rule (20) wins and its priority is the reported confidence."""
    assert classify_kernel(
        "cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_kernel0_0"
    ) == ("Attention (flash)", "SDPA", 20)


def test_taxonomy_types_closed_over_primitives():
    assert {ktype for _, ktype, _ in KERNEL_TYPE_RULES} <= set(KERNEL_TYPES)
    for kernel in TAXONOMY.types():
        assert kernel.perf_category
        assert set(kernel.covers) <= PRIMITIVES
        assert TAXONOMY.by_type(kernel.type) == kernel


# ----- run_assertions -----


def _classified(name, ktype, index=0):
    return {
        "index": index,
        "name": name,
        "kernel_type": ktype,
    }


def test_run_assertions_clean():
    classified = [_classified("Cijk_x", "GEMM", i) for i in range(5)]
    assert run_assertions(classified) == []


def test_run_assertions_flags_unknown():
    classified = [
        _classified("mystery", "Unknown", 0),
        _classified("Cijk_x", "GEMM", 1),
    ]
    errors = run_assertions(classified)
    assert any("Unknown" in e for e in errors)


# ----- FusionForms / FusionLabeler -----


def test_fusion_forms_derived_from_fused_types():
    forms = FusionForms(TAXONOMY).forms
    assert {f.kind for f in forms} == {
        "add_norm_quant",
        "attention",
        "add_norm",
        "norm_quant",
        "allreduce_norm",
        "rope_kv_store",
        "routing_topk",
        "act_mul",
    }
    assert [f.kind for f in forms if f.replaces_span] == ["attention"]
    assert len({f.elements for f in forms}) == len(forms)


@pytest.mark.parametrize(
    "names,expected",
    [
        ([GEMM, ADD, SOFTMAX, GEMM], ("attention", "high", (0, 4))),
        ([MUL, GEMM, ADD, SOFTMAX, GEMM, ADD], ("attention", "high", (1, 5))),
        ([ADD] * 17 + [GEMM, SOFTMAX, GEMM], ("attention", "high", (17, 20))),
        # The span starts at the GEMM nearest the softmax, not at one before another GEMM.
        ([GEMM, GEMM, SOFTMAX, GEMM], ("attention", "high", (1, 4))),
        ([ADD, LAYERNORM], ("add_norm", "high", None)),
        ([LAYERNORM, QUANT], ("norm_quant", "high", None)),
        ([ADD, LAYERNORM, QUANT], ("add_norm_quant", "high", None)),
        ([AR_STORE, RMSNORM], ("allreduce_norm", "high", None)),
        ([SOFTMAX, TOPK], ("routing_topk", "high", None)),
        ([SILU, MUL], ("act_mul", "high", None)),
        ([SIGMOID, MUL], ("elementwise_chain", "high", None)),
        ([ADD, BATCHNORM], ("elementwise_chain", "high", None)),
        ([ADD, MUL, MUL, LAYERNORM], ("elementwise_chain", "high", None)),
        ([ADD, MUL, GEMM], ("gemm_epilogue", "medium", None)),
        ([GEMM, ADD], ("gemm_epilogue", "medium", None)),
        ([ADD, MUL, UNKNOWN_KERNEL], ("elementwise_chain", "low", None)),
    ],
)
def test_label_kind_confidence_span(names, expected):
    label = LABELER.label(names, [None] * len(names))
    assert (label.kind, label.confidence, label.span) == expected


@pytest.mark.parametrize(
    "names,op_keys",
    [
        pytest.param([ADD, ADD_RMSNORM_QUANT], None, id="fused_kernel"),
        pytest.param([AR_STORE, AR_RMSNORM], None, id="allreduce_rmsnorm_pair"),
        pytest.param([ADD, TRITON], None, id="compiled"),
        pytest.param([ADD, MEMCPY], None, id="copy_engine"),
        pytest.param([ADD, NCCL], None, id="library_collective"),
        pytest.param([ADD] * 16, None, id="over_cap"),
        pytest.param([GEMM, GEMM], None, id="all_matrix"),
        pytest.param([GEMM, LAYERNORM], None, id="matrix_norm_only"),
        # An op that launches a convolution owns its layout and bookkeeping kernels.
        pytest.param(
            [UNKNOWN_KERNEL, TRANSPOSE, CONV], [_MM_KEY] * 3, id="matrix_op_launch"
        ),
    ],
)
def test_label_excludes_group(names, op_keys):
    assert LABELER.label(names, op_keys or [None] * len(names)) is None


def test_matrix_mask_groups_kernels_by_op_key():
    names = (UNKNOWN_KERNEL, CONV, TRANSPOSE, ADD, MOE_GEMM)
    classes = [TAXONOMY.classify(n) for n in names]
    forms = FusionForms(TAXONOMY)
    keys = [_MM_KEY, _MM_KEY, _ADD_KEY, None, None]
    assert forms.matrix_mask(classes, keys) == [True, True, False, False, True]
    assert forms.matrix_mask(classes, [None] * 5) == [False, True, False, False, True]


# ----- FusionCandidate -----


def _standalone_candidate(
    kernels, total_us, ops, kind="gemm_epilogue", confidence="medium", name="mlp_block"
):
    return FusionCandidate(
        module_name=name,
        base_name=name,
        parent_chain=[],
        instance_count=1,
        kernel_count=len(kernels),
        kernels=tuple(kernels),
        kernel_type_signature=[k.type for k in kernels],
        kernel_type_summary=dict(Counter(k.type for k in kernels)),
        fusion_kind=kind,
        confidence=confidence,
        total_kernel_time_us=total_us,
        ops=ops,
    )


def _comparative_candidate(name, type1, time1_us, time2_us):
    kernels = (KernelRecord("k1", type1, time1_us),)
    return FusionCandidate(
        module_name=name,
        base_name=name,
        parent_chain=[],
        instance_count=1,
        kernel_count_trace1=1,
        kernel_count_trace2=0,
        delta=1,
        kernels_trace1=kernels,
        kernels_trace2=(),
        kernel_type_signature=[type1],
        kernel_type_summary={type1: 1},
        fusion_kind="elementwise_chain",
        confidence="high",
        total_kernel_time_us_trace1=time1_us,
        total_kernel_time_us_trace2=time2_us,
        ops_trace1=[],
        ops_trace2=[],
    )


def _golden_candidate(total_us=200):
    kernels = [
        KernelRecord("Cijk_gemm", "GEMM", 100, _name_key("Cijk_gemm")),
        KernelRecord("ew_add", "Elementwise", 100, _name_key("ew_add")),
    ]
    ops = [_op_group(k.perf_key, [k.name], 100.0, 1) for k in kernels]
    return _standalone_candidate(kernels, total_us, ops)


@pytest.mark.parametrize(
    "candidate",
    [_golden_candidate(), _comparative_candidate("attn", "GEMM", 10_000, 5_000)],
    ids=["standalone", "comparative"],
)
def test_fusion_candidate_json_round_trip(candidate):
    data = candidate.to_dict()
    assert not any(v is None for v in data.values())
    assert FusionCandidate.from_dict(data) == candidate


# ----- FusionPricer._split_into_subgroups -----


def test_split_gemm_epilogue():
    kernels = [{"is_matrix": True}, {"is_matrix": False}]
    typed = FusionPricer._split_into_subgroups(kernels)
    assert len(typed) == 1
    assert typed[0][0] == "gemm_epilogue"


def test_split_gemm_only():
    typed = FusionPricer._split_into_subgroups([{"is_matrix": True}])
    assert len(typed) == 1
    assert typed[0][0] == "gemm_only"


def test_split_elementwise_only():
    typed = FusionPricer._split_into_subgroups(
        [{"is_matrix": False}, {"is_matrix": False}]
    )
    assert len(typed) == 1
    assert typed[0][0] == "elementwise"


def test_split_multiple_gemm_boundaries():
    """E, GEMM, E, GEMM splits at each GEMM into three typed sub-groups."""
    kernels = [
        {"is_matrix": False},
        {"is_matrix": True},
        {"is_matrix": False},
        {"is_matrix": True},
    ]
    typed = FusionPricer._split_into_subgroups(kernels)
    assert [t for t, _ in typed] == ["elementwise", "gemm_epilogue", "gemm_only"]


# ----- FusionPricer._roofline_savings_us -----


def _pricer(perf_rows=None, baseline_ms=1.0):
    """Peak memory 1 TB/s, vector 1 TFLOPS, matrix 700 TFLOPS."""
    return FusionPricer(
        FusionForms(TAXONOMY),
        (1.0, {"matrix_bf16": 700.0, "vector_bf16": 1.0}),
        perf_rows or {},
        baseline_ms,
    )


def _ew(name, dur_us, data_in=None, data_out=None, data_moved=None, gflops=None):
    return {
        "name": name,
        "is_matrix": False,
        "dur_us": dur_us,
        "data_moved_mb": data_moved,
        "data_in_mb": data_in,
        "data_out_mb": data_out,
        "gflops": gflops,
        "has_perf_data": True,
    }


def test_roofline_savings_memory_only():
    """Single memory-bound kernel: fused == memory_time, savings = dur - memory_time."""
    enriched = [_ew("ew", dur_us=10.0, data_in=1.0, data_out=1.0, gflops=None)]
    # bytes = 2e6; peak = 1e12 B/s; frac=1.0 -> memory_time = 2.0 us; savings = 10 - 2
    savings = _pricer()._roofline_savings_us(enriched, 100.0)
    assert savings == pytest.approx(8.0)


def test_roofline_savings_no_modeled_returns_zero():
    enriched = [dict(_ew("ew", 10.0, 1.0, 1.0), has_perf_data=False)]
    assert _pricer()._roofline_savings_us(enriched, 100.0) == 0.0


def test_roofline_savings_zero_external_data_returns_zero():
    enriched = [_ew("ew", 10.0, data_in=0.0, data_out=0.0)]
    assert _pricer()._roofline_savings_us(enriched, 100.0) == 0.0


def test_roofline_savings_clamped_at_zero():
    """When current time is below fused time, savings clamp to 0."""
    enriched = [_ew("ew", dur_us=1.0, data_in=1.0, data_out=1.0)]
    # memory_time = 2.0 us > current 1.0 us -> max(0, 1 - 2) = 0
    assert _pricer()._roofline_savings_us(enriched, 100.0) == 0.0


def test_roofline_savings_unmodeled_subtracted():
    """Unmodeled kernel time is subtracted from savings, not fused."""
    enriched = [
        _ew("ew", dur_us=10.0, data_in=1.0, data_out=1.0),
        dict(_ew("unmodeled", dur_us=5.0), has_perf_data=False),
    ]
    # current = 15, memory_time = 2, unmodeled = 5 -> max(0, 15 - 2 - 5) = 8
    assert _pricer()._roofline_savings_us(enriched, 100.0) == pytest.approx(8.0)


def test_roofline_savings_overlap_blend():
    """Blends max (perfect overlap) and sum (no overlap) via OVERLAP_EFFICIENCY."""
    enriched = [
        _ew("ew1", dur_us=10.0, data_in=1.0, gflops=0.0005),
        _ew("ew2", dur_us=10.0, data_out=1.0, gflops=0.0005),
    ]
    # memory_time = 2.0; vector_time = (0.001e9)/(1e12)*1e6 = 1.0; matrix = 0
    # optimal = 2, sum = 3, fused = 2 + (1-0.85)*(3-2) = 2.15; current 20 -> 17.85
    savings = _pricer()._roofline_savings_us(enriched, 100.0)
    expected = 20.0 - (2.0 + (1.0 - OVERLAP_EFFICIENCY) * (3.0 - 2.0))
    assert savings == pytest.approx(expected)
    assert savings == pytest.approx(17.85)


# ----- FusionPricer.price -----


def _lookup(**entries):
    """Build perf_rows keyed by ``_name_key``; ``price`` keys each kernel by its name."""
    return {
        tuple(_name_key(name)): {
            "Data Moved (MB)": dm,
            "GFLOPS": gf,
            "Compute Spec": spec,
        }
        for name, (dm, gf, spec) in entries.items()
    }


def _golden_lookup():
    return _lookup(Cijk_gemm=(1.0, None, "matrix_bf16"), ew_add=(1.0, None, None))


def test_price_gemm_epilogue_full_estimate():
    """GEMM+elementwise: savings = elementwise dur; impact bands via TARGET_*."""
    est = _pricer(_golden_lookup()).price(_golden_candidate())
    assert est is not None
    # gap_high = 100/200 = 0.5; time_ms = 0.2; impact_high = 0.5*0.2/1*100 = 10.0
    assert est.impact_score_high == 10.0
    assert est.impact_score == 8.75
    assert est.impact_score_low == 7.5
    assert est.bound_type == "compute"
    assert est.estimation == "full"
    assert (est.category, est.type) == ("kernel_fusion", "kernel_fusion")
    assert (est.fusion_kind, est.confidence) == ("gemm_epilogue", "medium")


def test_price_matrix_op_launch_is_not_epilogue():
    """A kernel launched by the GEMM's own op is matrix: only ew_add is saved."""
    kernels = [
        KernelRecord("Cijk_gemm", "GEMM", 200, _name_key("Cijk_gemm")),
        KernelRecord("aux", "Unknown", 200, _name_key("Cijk_gemm")),
        KernelRecord("ew_add", "Elementwise", 200, _name_key("ew_add")),
    ]
    est = _pricer(_golden_lookup()).price(_standalone_candidate(kernels, 600, []))
    assert est.impact_score_high == 20.0


@pytest.mark.parametrize(
    "perf_rows,baseline_ms,candidate",
    [
        pytest.param(
            None,
            1.0,
            _standalone_candidate(
                [
                    KernelRecord("a", "Elementwise", 10),
                    KernelRecord("b", "Elementwise", 10),
                ],
                20,
                [],
            ),
            id="no_perf_models",
        ),
        # Priced if not for the modeled fraction: 1 of 3 modeled, below 0.5.
        pytest.param(
            _golden_lookup(),
            1.0,
            _standalone_candidate(
                [
                    KernelRecord("Cijk_gemm", "GEMM", 200, _name_key("Cijk_gemm")),
                    KernelRecord("unmodeled_a", "Elementwise", 200),
                    KernelRecord("unmodeled_b", "Elementwise", 200),
                ],
                600,
                [],
            ),
            id="below_modeled_frac",
        ),
        # A group of LONG_GROUP_KERNELS needs more: 3 of 5 modeled, below 0.75.
        pytest.param(
            _lookup(
                Cijk_gemm=(1.0, None, "matrix_bf16"),
                ew_add=(1.0, None, None),
                ew_mul=(1.0, None, None),
            ),
            1.0,
            _standalone_candidate(
                [
                    KernelRecord("Cijk_gemm", "GEMM", 200, _name_key("Cijk_gemm")),
                    KernelRecord("ew_add", "Elementwise", 200, _name_key("ew_add")),
                    KernelRecord("ew_mul", "Elementwise", 200, _name_key("ew_mul")),
                    KernelRecord("unmodeled_a", "Elementwise", 200),
                    KernelRecord("unmodeled_b", "Elementwise", 200),
                ],
                1000,
                [],
            ),
            id="below_long_group_modeled_frac",
        ),
        # A large baseline pushes impact_score_high below MIN_IMPACT_SCORE.
        pytest.param(
            _golden_lookup(), 1000.0, _golden_candidate(), id="below_min_impact_score"
        ),
        # 0.09 ms of projected savings is below MIN_SAVINGS_MS.
        pytest.param(
            _golden_lookup(),
            1.0,
            _golden_candidate(total_us=180),
            id="below_min_savings_ms",
        ),
    ],
)
def test_price_skip(perf_rows, baseline_ms, candidate):
    assert _pricer(perf_rows, baseline_ms).price(candidate) is None


# ----- FusionPricer.price_comparative -----


def test_price_comparative_measured_gap():
    est = _pricer(baseline_ms=100.0).price_comparative(
        _comparative_candidate("attn", "GEMM", 10_000, 5_000)
    )
    assert est is not None
    # gap = 5000 us -> savings_high = 5.0 ms; impact_high = 5/100*100 = 5.0
    assert est.impact_score_high == 5.0
    assert est.impact_score == 4.38  # (87.5/100)*5 = 4.375 -> 4.38
    assert est.impact_score_low == 3.75
    assert est.bound_type == "compute"
    assert est.estimation == "measured"
    assert est.type == "kernel_fusion"


def test_price_comparative_skip_no_gap():
    """trace2 not faster than trace1 -> no savings -> skip."""
    candidate = _comparative_candidate("attn", "GEMM", 5_000, 6_000)
    assert _pricer(baseline_ms=100.0).price_comparative(candidate) is None


# ----- FusionPricer.shortlist / price_all -----


def test_shortlist_drops_below_baseline_floor():
    """baseline_ms>0 sets a min total time of baseline*10*MIN_IMPACT_SCORE us."""
    floor = 1.0 * 10 * MIN_IMPACT_SCORE  # 20 us
    candidates = [
        _standalone_candidate([], floor - 1, [], name="tiny"),
        _standalone_candidate([], floor, [], name="edge"),
        _standalone_candidate([], floor + 100, [], name="keep"),
        _comparative_candidate("tiny_cmp", "GEMM", floor - 1, 0),
        _comparative_candidate("keep_cmp", "GEMM", floor + 100, 0),
    ]
    out = FusionPricer.shortlist(candidates, baseline_ms=1.0)
    assert [c.base_name for c in out] == ["edge", "keep", "keep_cmp"]


def test_price_all_invalid_baseline_returns_empty(capsys):
    out = _pricer(baseline_ms=0).price_all([_golden_candidate()])
    assert out == []
    assert "invalid" in capsys.readouterr().err


def test_price_all_sorts_by_impact_and_dispatches():
    """Standalone and comparative candidates each go to their own pricer, best first."""
    pricer = _pricer(_golden_lookup(), baseline_ms=100.0)
    small = _comparative_candidate("small", "Elementwise", 6_000, 3_000)
    large = _comparative_candidate("large", "GEMM", 20_000, 5_000)
    pairs = pricer.price_all([small, large, _golden_candidate(total_us=20_000)])
    assert [e.operation for _, e in pairs] == ["large", "mlp_block", "small"]
    assert [e.estimation for _, e in pairs] == ["measured", "full", "measured"]


def test_build_rows():
    perf = pd.Series(
        {
            "entry_point": "torch.mm",
            "FLOPS/Byte": 12.345,
            "Pct Roofline_mean": 50.0,
            "Roofline Bound": "COMPUTE_BOUND",
        }
    )
    ops = [_op_group(_MM_KEY, ["k1", "k2"], 2500.0, 3), _op_group(None, ["k"], 10.0, 1)]
    pricer = FusionPricer(
        FusionForms(TAXONOMY), (5.3, {"matrix_bf16": 1300.0}), {}, 100.0
    )
    matched, unmatched = pricer.build_rows(ops, {tuple(_MM_KEY): perf}, 100.0)
    assert matched["operation"] == "aten::mm"
    assert matched["args"] == "(16,64) bf16<br>(64,8) bf16"
    assert (matched["time_ms"], matched["pct_e2e"], matched["count"]) == (
        "2.500",
        "2.50",
        3,
    )
    assert (matched["kernel_path"], matched["flops_per_byte"]) == ("torch.mm", "12.35")
    assert matched["efficiency"] == "50.00% of 1300 TFLOPS"
    assert matched["bound"] == "compute-bound"
    assert unmatched["operation"] == unmatched["args"] == "—"
    assert unmatched["efficiency"] == unmatched["bound"] == "—"
    assert unmatched["kernel_path"] == "Not found"


# ----- classify_memcpy_severity -----


def test_memcpy_no_timeline_guard():
    """total_time_ms <= 0 short-circuits to not-flagged."""
    r = classify_memcpy_severity({"total_time_us": 5000, "total_count": 3}, 0)
    assert r["flagged"] is False
    assert "No timeline" in r["details"]


def test_memcpy_flagged_by_percent():
    """D2H > 5% of total time is flagged."""
    summary = {
        "total_time_us": 100_000,
        "total_count": 5,
        "by_direction": {
            "D2H": {"count": 3, "total_time_us": 100_000, "avg_bytes": 1024},
        },
    }
    r = classify_memcpy_severity(summary, total_time_ms=1000.0)
    # D2H time = 100 ms of 1000 ms = 10% > 5% -> flagged
    assert r["flagged"] is True
    assert len(r["issues"]) == 1
    assert r["issues"][0]["direction"] == "D2H"


def test_memcpy_flagged_by_count():
    """H2D count > 10 flags even when the time percentage is tiny."""
    summary = {
        "total_time_us": 1_000,
        "total_count": 11,
        "by_direction": {
            "H2D": {"count": 11, "total_time_us": 1_000, "avg_bytes": 64},
        },
    }
    r = classify_memcpy_severity(summary, total_time_ms=1_000_000.0)
    assert r["flagged"] is True
    assert r["issues"][0]["direction"] == "H2D"


def test_memcpy_not_flagged_below_thresholds():
    summary = {
        "total_time_us": 1_000,
        "total_count": 2,
        "by_direction": {
            "D2H": {"count": 2, "total_time_us": 1_000, "avg_bytes": 64},
        },
    }
    r = classify_memcpy_severity(summary, total_time_ms=1_000.0)
    # 1 ms of 1000 ms = 0.1% and count 2 -> not flagged
    assert r["flagged"] is False
    assert r["issues"] == []


def test_memcpy_zero_count_direction_skipped():
    summary = {
        "total_time_us": 0,
        "total_count": 0,
        "by_direction": {"D2H": {"count": 0, "total_time_us": 0}},
    }
    r = classify_memcpy_severity(summary, total_time_ms=1_000.0)
    assert r["flagged"] is False


# ----- classify_nccl_blocking_severity -----


def test_nccl_blocking_no_comm_guard():
    r = classify_nccl_blocking_severity({"total_comm_time_us": 0})
    assert r["flagged"] is False
    assert "No communication" in r["details"]


def test_nccl_blocking_flagged_above_5pct():
    r = classify_nccl_blocking_severity(
        {
            "total_comm_time_us": 10_000,
            "exposed_comm_time_us": 8_000,
            "comm_percent_of_total": 8.0,
        }
    )
    assert r["flagged"] is True
    assert r["exposed_percent_of_total"] == 8.0


def test_nccl_blocking_not_flagged_at_5pct():
    r = classify_nccl_blocking_severity(
        {
            "total_comm_time_us": 10_000,
            "exposed_comm_time_us": 5_000,
            "comm_percent_of_total": 5.0,
        }
    )
    # strictly > 5 required
    assert r["flagged"] is False


# ----- classify_overlap_severity -----


def test_overlap_insufficient_data_guard():
    r = classify_overlap_severity({"comm_overlap_ratio": None, "total_comm_time_us": 0})
    assert r["flagged"] is False
    assert "Insufficient" in r["details"]


def test_overlap_low_comm_time_guard():
    """< 100 us total comm is treated as insufficient data."""
    r = classify_overlap_severity({"comm_overlap_ratio": 0.1, "total_comm_time_us": 50})
    assert r["flagged"] is False


def test_overlap_flagged_below_70pct():
    r = classify_overlap_severity(
        {"comm_overlap_ratio": 0.5, "total_comm_time_us": 10_000}
    )
    assert r["flagged"] is True
    assert r["overlap_percent"] == 50.0
    assert r["target_percent"] == 70


def test_overlap_not_flagged_at_70pct():
    r = classify_overlap_severity(
        {"comm_overlap_ratio": 0.7, "total_comm_time_us": 10_000}
    )
    assert r["flagged"] is False


# ----- cross_validate_with_timeline -----


def _timeline(rows):
    return pd.DataFrame(rows)


def test_cross_validate_pass_within_tolerance():
    overlap = {
        "computation_time_us": 100_000,
        "exposed_comm_time_us": 5_000,
        "exposed_memcpy_time_us": 1_000,
        "total_time_us": 200_000,
    }
    tl = _timeline(
        [
            {"type": "computation_time", "time ms": 100.0, "percent": 50.0},
            {"type": "exposed_comm_time", "time ms": 5.0, "percent": 2.5},
            {"type": "exposed_memcpy_time", "time ms": 1.0, "percent": 0.5},
            {"type": "total_time", "time ms": 200.0, "percent": 100.0},
        ]
    )
    v = cross_validate_with_timeline(overlap, tl, tolerance_pct=2.0)
    assert v["status"] == "PASS"
    assert all(c["status"] == "PASS" for c in v["checks"])
    assert v["warnings"] == []


def test_cross_validate_warn_beyond_tolerance():
    overlap = {"total_time_us": 200_000}
    tl = _timeline([{"type": "total_time", "time ms": 100.0, "percent": 100.0}])
    v = cross_validate_with_timeline(overlap, tl, tolerance_pct=2.0)
    # mk=200 ms vs tl=100 ms -> 50% diff > 2% -> WARN
    assert v["status"] == "WARN"
    total_check = next(c for c in v["checks"] if c["metric"] == "Total GPU time")
    assert total_check["status"] == "WARN"
    assert len(v["warnings"]) == 1


def test_cross_validate_missing_type_skips():
    overlap = {"computation_time_us": 100_000}
    tl = _timeline([{"type": "total_time", "time ms": 100.0, "percent": 100.0}])
    v = cross_validate_with_timeline(overlap, tl)
    comp_check = next(c for c in v["checks"] if c["metric"] == "Computation time")
    assert comp_check["status"] == "SKIP"


def test_cross_validate_both_zero_pass():
    overlap = {"exposed_comm_time_us": 0}
    tl = _timeline([{"type": "exposed_comm_time", "time ms": 0, "percent": 0}])
    v = cross_validate_with_timeline(overlap, tl)
    comm_check = next(
        c for c in v["checks"] if c["metric"] == "Exposed communication time"
    )
    assert comm_check["status"] == "PASS"
    assert comm_check["diff_ms"] == 0


# ----- Unit tests: sdpa_analysis.parse_perf_params -----


@pytest.mark.parametrize(
    "ctx,gen,expected",
    [
        (900, 100, "prefill_heavy"),  # ctx_ratio 0.9 > 0.8
        (500, 500, "mixed"),  # ctx_ratio 0.5 within [0.2, 0.8]
        (100, 900, "decode_heavy"),  # ctx_ratio 0.1 < 0.2
    ],
)
def test_parse_perf_params_workload_type(ctx, gen, expected):
    params = {"sum_ctx_tokens": ctx, "sum_gen_tokens": gen}
    r = parse_perf_params(str(params))
    assert r["workload_type"] == expected


def test_parse_perf_params_workload_type_boundaries():
    """Boundary values 0.8 and 0.2 fall into the 'mixed' band (strict >/<)."""
    at_high = parse_perf_params(str({"sum_ctx_tokens": 800, "sum_gen_tokens": 200}))
    assert at_high["ctx_ratio"] == 0.8
    assert at_high["workload_type"] == "mixed"
    at_low = parse_perf_params(str({"sum_ctx_tokens": 200, "sum_gen_tokens": 800}))
    assert at_low["ctx_ratio"] == 0.2
    assert at_low["workload_type"] == "mixed"


def test_parse_perf_params_no_tokens_unknown():
    r = parse_perf_params(str({"B": 2}))
    assert r["workload_type"] == "unknown"
    assert r["ctx_ratio"] is None


def test_parse_perf_params_gqa():
    r = parse_perf_params(str({"H_Q": 8, "H_KV": 2}))
    assert r["attention_pattern"] == "GQA"
    assert r["gqa_ratio"] == 4


def test_parse_perf_params_mha():
    r = parse_perf_params(str({"H_Q": 8, "H_KV": 8}))
    assert r["attention_pattern"] == "MHA"
    assert "gqa_ratio" not in r


def test_parse_perf_params_empty_returns_defaults():
    r = parse_perf_params("")
    assert r["workload_type"] == "unknown"
    assert r["batch_size"] is None


def test_parse_perf_params_malformed_returns_defaults():
    r = parse_perf_params("not a dict literal {")
    assert r["workload_type"] == "unknown"


# ----- Unit tests: sdpa_analysis.parse_kernel_breakdown -----


def test_parse_kernel_breakdown_regex_and_percent():
    kd = str(
        [
            {"name": "paged_attention_2d", "mean_duration_us": 10.0},
            {"name": "reshape_and_cache", "mean_duration_us": 30.0},
        ]
    )
    r = parse_kernel_breakdown(kd)
    assert r["total_kernel_time_us"] == 40.0
    assert r["has_paged_attention"] is True
    assert r["has_reshape_cache"] is True
    pcts = {k["name"]: k["percent"] for k in r["kernels"]}
    assert pcts["paged_attention_2d"] == 25.0
    assert pcts["reshape_and_cache"] == 75.0


def test_parse_kernel_breakdown_fwd_kernel_type():
    kd = str([{"name": "attn_fwd_kernel", "mean_duration_us": 5.0}])
    r = parse_kernel_breakdown(kd)
    assert r["has_fwd_kernel"] is True
    assert r["kernels"][0]["kernel_type"] == "fwd_kernel"
    assert r["kernels"][0]["percent"] == 100.0


def test_parse_kernel_breakdown_strips_np_float64():
    kd = "[{'name': 'k', 'mean_duration_us': np.float64(12.5)}]"
    r = parse_kernel_breakdown(kd)
    assert r["total_kernel_time_us"] == 12.5


def test_parse_kernel_breakdown_empty():
    r = parse_kernel_breakdown("")
    assert r["kernels"] == []
    assert r["total_kernel_time_us"] == 0


# ----- Unit tests: sdpa_analysis detect_* helpers -----


def test_detect_flash_attention():
    assert detect_flash_attention("flash_attn::_flash_attn_forward") is True
    assert detect_flash_attention("aten::mm") is False


def test_detect_paged_attention():
    assert detect_paged_attention("paged_attention_2d") is True
    assert detect_paged_attention("aten::mm") is False


# ----- shared metadata fixture for get_peak_specs consumers -----


def _metadata(peak_mem_bw_tbs=5.3, matrix_bf16=1300.0):
    """Metadata dict as produced by the orchestrator; get_peak_specs reads it."""
    return {
        "max_achievable_tflops": {"matrix_bf16": matrix_bf16},
        "peak_mem_bw_tbs": peak_mem_bw_tbs,
    }


# ----- convolution_analysis.extract_category_specific -----


def test_conv_extract_with_transpose():
    df = pd.DataFrame(
        {
            "name": ["conv2d", "aten::transpose", "TransPose_kernel"],
            "Kernel Time (µs)_sum": [800.0, 100.0, 100.0],
        }
    )
    r = conv_extract(df, _metadata())
    assert r["transpose_count"] == 2
    assert r["transpose_time_ms"] == pytest.approx(0.2)
    # 200 us of 1000 us total = 20%
    assert r["transpose_overhead_percent"] == pytest.approx(20.0)
    assert r["peak_mem_bw_tbs"] == 5.3
    assert r["peak_maf_tflops"] == 1300.0


def test_conv_extract_no_transpose():
    df = pd.DataFrame(
        {"name": ["conv2d", "relu"], "Kernel Time (µs)_sum": [500.0, 500.0]}
    )
    r = conv_extract(df, _metadata())
    assert r["transpose_count"] == 0
    assert r["transpose_time_ms"] == 0
    assert r["transpose_overhead_percent"] == 0


def test_conv_extract_missing_kernel_time_column():
    """No 'Kernel Time (µs)_sum' column -> total 0 -> zero-div guard, 0% overhead."""
    df = pd.DataFrame({"name": ["conv2d", "relu"]})
    r = conv_extract(df, _metadata())
    assert r["transpose_count"] == 0
    assert r["transpose_overhead_percent"] == 0


# ----- cpu_idle_analysis.analyze_kernel_patterns -----


def test_analyze_kernel_patterns_none():
    r = analyze_kernel_patterns(None)
    assert r == {
        "short_kernel_count": 0,
        "total_kernel_count": 0,
        "avg_kernel_time_us": 0,
        "kernel_count_by_category": {},
    }


def test_analyze_kernel_patterns_empty_df():
    r = analyze_kernel_patterns(pd.DataFrame())
    assert r["total_kernel_count"] == 0
    assert r["kernel_count_by_category"] == {}


def test_analyze_kernel_patterns_sum_and_count_no_mean():
    df = pd.DataFrame({"Kernel Time (µs)_sum": [100.0, 300.0], "Count": [2, 2]})
    r = analyze_kernel_patterns(df)
    assert r["total_kernel_count"] == 4
    assert r["avg_kernel_time_us"] == pytest.approx(100.0)
    # no mean column -> short count stays zero
    assert r["short_kernel_count"] == 0


def test_analyze_kernel_patterns_with_mean_short_kernels():
    df = pd.DataFrame(
        {
            "Kernel Time (µs)_sum": [10.0, 500.0],
            "Count": [5, 1],
            "Kernel Time (µs)_mean": [2.0, 500.0],
        }
    )
    r = analyze_kernel_patterns(df)
    assert r["total_kernel_count"] == 6
    # only the 2.0us-mean row (<10) counts its 5 kernels as short
    assert r["short_kernel_count"] == 5


def test_analyze_kernel_patterns_by_category():
    df = pd.DataFrame(
        {
            "Kernel Time (µs)_sum": [10.0, 20.0, 30.0],
            "Count": [1, 2, 3],
            "op category": ["GEMM", "GEMM", "Elementwise"],
        }
    )
    r = analyze_kernel_patterns(df)
    assert r["kernel_count_by_category"] == {"GEMM": 3, "Elementwise": 3}


# ----- cpu_idle_analysis load helpers -----


def test_load_ops_summary_missing_returns_none(tmp_path):
    assert load_ops_summary(str(tmp_path)) is None


def test_load_ops_summary_reads_csv(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    pd.DataFrame({"Count": [1, 2]}).to_csv(csv_dir / "ops_summary.csv", index=False)
    df = load_ops_summary(str(tmp_path))
    assert list(df["Count"]) == [1, 2]


def test_load_gpu_timeline_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_gpu_timeline(str(tmp_path))


def test_load_gpu_timeline_reads_rows(tmp_path):
    csv_dir = tmp_path / "perf_report_csvs"
    csv_dir.mkdir()
    pd.DataFrame(
        {
            "type": ["idle_time", "total_time"],
            "time ms": [5.0, 100.0],
            "percent": [5.0, 100.0],
        }
    ).to_csv(csv_dir / "gpu_timeline.csv", index=False)
    tl = load_gpu_timeline(str(tmp_path))
    assert tl["idle_time"]["percent"] == 5.0
    assert tl["total_time"]["time_ms"] == 100.0


# ----- elementwise / norm one-liner extractors -----


def test_elementwise_extract():
    assert elementwise_extract(pd.DataFrame(), _metadata())["peak_mem_bw_tbs"] == 5.3


def test_norm_extract():
    assert (
        norm_extract(pd.DataFrame(), _metadata(peak_mem_bw_tbs=8.0))["peak_mem_bw_tbs"]
        == 8.0
    )


# ----- moe_analysis -----


def test_moe_extract_delegates_to_peak_specs():
    r = moe_extract(pd.DataFrame(), _metadata())
    assert r["peak_mem_bw_tbs"] == 5.3
    assert r["peak_maf_tflops"] == 1300.0


def test_check_moe_data_absent_returns_no_data(tmp_path):
    r = _check_moe_data(str(tmp_path), "moe_fused", "standalone")
    assert r["status"] == "NO_DATA"
    assert r["operation_count"] == 0
    assert r["category"] == "moe_fused"


def test_check_moe_data_present_returns_none(tmp_path):
    cat_dir = tmp_path / "category_data"
    cat_dir.mkdir()
    (cat_dir / "moe_fused_ops.csv").write_text("name\n")
    assert _check_moe_data(str(tmp_path), "moe_fused", "standalone") is None


# ----- triton_analysis -----


@pytest.mark.parametrize(
    "name,expected",
    [
        ("triton_poi_fused_add_0", "pointwise"),
        ("triton_red_sum_1", "reduction"),
        ("triton_per_fused_2", "persistent"),
        ("some_other_kernel", "other"),
        ("TRITON_POI_UPPER", "pointwise"),
    ],
)
def test_classify_triton_operation(name, expected):
    assert classify_triton_operation(name, None)["kernel_type"] == expected


def test_triton_extract_tallies_and_merges_specs():
    df = pd.DataFrame(
        {
            "name": [
                "triton_poi_a",
                "triton_poi_b",
                "triton_red_c",
                "triton_per_d",
                "misc_e",
            ]
        }
    )
    r = triton_extract(df, _metadata())
    assert r["pointwise_count"] == 2
    assert r["reduction_count"] == 1
    assert r["persistent_count"] == 1
    assert r["other_count"] == 1
    assert r["peak_mem_bw_tbs"] == 5.3


# ----- gemm_analysis -----


@pytest.mark.parametrize(
    "name,quantized",
    [
        ("gemm_w8a8", True),
        ("fp8_matmul", True),
        ("mxfp4_gemm", True),
        ("Cijk_regular", False),
    ],
)
def test_detect_quantized_gemm(name, quantized):
    assert detect_quantized_gemm(name) is quantized


def test_classify_gemm_operation():
    q = classify_gemm_operation("gemm_int8", None)
    assert q == {"is_quantized": True, "gemm_type": "quantized"}
    r = classify_gemm_operation("Cijk_plain", None)
    assert r == {"is_quantized": False, "gemm_type": "regular"}


def test_gemm_extract_counts_quant_and_missing_perf():
    df = pd.DataFrame(
        {
            "name": ["gemm_fp8", "Cijk_plain", "gemm_w4a16"],
            "TFLOPS/s_mean": [100.0, float("nan"), float("nan")],
        }
    )
    r = gemm_extract(df, _metadata())
    assert r["quantized_count"] == 2
    assert r["missing_perf_model_count"] == 2


def test_gemm_extract_missing_tflops_column():
    df = pd.DataFrame({"name": ["Cijk_plain"]})
    r = gemm_extract(df, _metadata())
    assert r["quantized_count"] == 0
    assert r["missing_perf_model_count"] == 0


# ----- reduce_analysis -----


def test_detect_softmax():
    assert detect_softmax("aten::_softmax") is True
    assert detect_softmax("aten::sum") is False


def test_reduce_extract_softmax_count():
    df = pd.DataFrame({"name": ["softmax_kernel", "aten::sum", "log_softmax"]})
    r = reduce_extract(df, _metadata())
    assert r["softmax_count"] == 2
    assert r["peak_mem_bw_tbs"] == 5.3


def test_reduce_extract_empty():
    r = reduce_extract(pd.DataFrame({"name": []}), _metadata())
    assert r["softmax_count"] == 0


# ----- other_analysis -----


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ncclKernel_AllReduce", "communication"),
        ("rccl_broadcast", "communication"),
        ("hipGraphLaunch", "graph"),
        ("aten::some_misc_op", "miscellaneous"),
    ],
)
def test_classify_other_operation(name, expected):
    assert classify_other_operation(name) == expected


def test_classify_other_op_callback():
    assert _classify_other_op("hipgraph_exec", None) == {"sub_category": "graph"}


def test_other_extract_counts_graph_and_misc():
    df = pd.DataFrame({"name": ["cudagraph_launch", "aten::foo", "aten::bar"]})
    r = other_extract(df, _metadata())
    assert r["graph_count"] == 1
    assert r["miscellaneous_count"] == 2
    assert r["communication_count"] == 0
    assert "communication_ops_skipped" not in r


def test_other_extract_with_skipped_comm_ops():
    df = pd.DataFrame({"name": ["aten::foo"]})
    skipped = {"count": 3, "op_names": ["a", "b", "c"], "message": "x"}
    r = other_extract(df, _metadata(), skipped_comm_ops=skipped)
    assert r["communication_ops_skipped"] == skipped


def test_other_extract_empty_skipped_comm_ops_omitted():
    df = pd.DataFrame({"name": ["aten::foo"]})
    r = other_extract(df, _metadata(), skipped_comm_ops={"count": 0})
    assert "communication_ops_skipped" not in r


# ----- sdpa_analysis.classify_sdpa_operation -----


def test_classify_sdpa_flash():
    row = pd.Series({"has_perf_model": True})
    r = classify_sdpa_operation("flash_attn::fwd", row)
    assert r["is_flash_attention"] is True
    assert r["is_paged_attention"] is False
    assert r["attention_type"] == "flash"
    assert r["has_perf_model"] is True


def test_classify_sdpa_standard():
    r = classify_sdpa_operation(
        "aten::scaled_dot_product_attention", pd.Series(dtype=object)
    )
    assert r["attention_type"] == "standard"
    assert r["has_perf_model"] is False


def test_classify_sdpa_paged_with_kernel_breakdown():
    kd = str(
        [
            {"name": "paged_attention_2d", "mean_duration_us": 30.0},
            {"name": "reshape_and_cache", "mean_duration_us": 10.0},
        ]
    )
    row = pd.Series({"kernel_details_summary": kd})
    r = classify_sdpa_operation("paged_attention_2d", row)
    assert r["is_paged_attention"] is True
    assert r["attention_type"] == "paged"
    assert r["kernel_breakdown"]["has_paged_attention_kernel"] is True
    assert r["kernel_breakdown"]["has_reshape_cache"] is True


def test_classify_sdpa_with_perf_params_gqa():
    row = pd.Series({"perf_params": str({"H_Q": 8, "H_KV": 2})})
    r = classify_sdpa_operation("flash_attn::fwd", row)
    assert r["workload_profile"]["attention_pattern"] == "GQA"
    assert r["workload_profile"]["gqa_ratio"] == 4


# ----- sdpa_analysis.extract_category_specific -----


def test_sdpa_extract_all_flash():
    df = pd.DataFrame(
        {"name": ["flash_attn::fwd", "flash_attn::bwd"], "has_perf_model": [True, True]}
    )
    r = sdpa_extract(df, _metadata())
    assert r["flash_attention_count"] == 2
    assert r["paged_attention_count"] == 0
    assert r["flash_attention_detected"] is True
    assert r["paged_attention_detected"] is False
    assert r["has_perf_model_count"] == 2
    assert "workload_profile" not in r


def test_sdpa_extract_paged_breakdown_and_workload():
    kd = str(
        [
            {"name": "paged_attention_2d", "mean_duration_us": 80.0},
            {"name": "reshape_and_cache", "mean_duration_us": 20.0},
        ]
    )
    df = pd.DataFrame(
        {
            "name": ["paged_attention_2d"],
            "kernel_details_summary": [kd],
            "perf_params": [str({"sum_ctx_tokens": 100, "sum_gen_tokens": 900})],
        }
    )
    r = sdpa_extract(df, _metadata())
    assert r["paged_attention_count"] == 1
    assert r["kernel_breakdown_avg"]["avg_paged_attention_percent"] == pytest.approx(
        80.0
    )
    assert r["kernel_breakdown_avg"]["avg_reshape_cache_percent"] == pytest.approx(20.0)
    # ctx_ratio 0.1 < 0.2 -> decode_heavy
    assert r["workload_profile"]["profile_type"] == "decode_heavy"
    assert r["workload_profile"]["total_gen_tokens"] == 900


def test_sdpa_extract_mixed_workload():
    df = pd.DataFrame(
        {
            "name": ["flash_attn::fwd"],
            "perf_params": [str({"sum_ctx_tokens": 500, "sum_gen_tokens": 500})],
        }
    )
    r = sdpa_extract(df, _metadata())
    assert r["workload_profile"]["profile_type"] == "mixed"


def test_sdpa_extract_empty():
    r = sdpa_extract(pd.DataFrame({"name": []}), _metadata())
    assert r["flash_attention_count"] == 0
    assert r["paged_attention_count"] == 0
    assert "workload_profile" not in r


# ----- driver main() end-to-end smoke tests -----
#
# These exercise each module's argparse main() over a minimal fixture tree so the
# run_category_analysis wiring (load -> build -> extract -> write) is covered.


def _write_category_inputs(base, category, ops_df):
    """Lay out the {category}_ops.csv + {category}_metadata.json that
    load_category_data expects under an output dir."""
    cat_dir = os.path.join(base, "category_data")
    meta_dir = os.path.join(base, "metadata")
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    ops_df.to_csv(os.path.join(cat_dir, f"{category}_ops.csv"), index=False)
    metadata = {
        "max_achievable_tflops": {"matrix_bf16": 1300.0},
        "peak_mem_bw_tbs": 5.3,
        "gpu_utilization": {"total_time_ms": 100.0},
    }
    with open(os.path.join(meta_dir, f"{category}_metadata.json"), "w") as f:
        json.dump(metadata, f)


def _read_metrics(base, category):
    with open(os.path.join(base, "category_data", f"{category}_metrics.json")) as f:
        return json.load(f)


def _simple_ops():
    return pd.DataFrame(
        {
            "name": ["op_a", "op_b"],
            "count": [1, 2],
            "Kernel Time (µs)_sum": [500.0, 300.0],
        }
    )


@pytest.mark.parametrize(
    "mod,category",
    [
        (convolution_analysis, "convolution"),
        (elementwise_analysis, "elementwise"),
        (norm_analysis, "norm"),
        (reduce_analysis, "reduce"),
        (triton_analysis, "triton"),
        (gemm_analysis, "gemm"),
        (sdpa_analysis, "sdpa_fwd"),
    ],
)
def test_driver_main_writes_ok_metrics(mod, category, tmp_path, monkeypatch):
    base = str(tmp_path)
    _write_category_inputs(base, category, _simple_ops())
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    mod.main()
    metrics = _read_metrics(base, category)
    assert metrics["status"] == "OK"
    assert metrics["category"] == category
    assert metrics["operation_count"] == 2


def test_driver_other_main_writes_ok(tmp_path, monkeypatch):
    base = str(tmp_path)
    df = pd.DataFrame(
        {
            "name": ["ncclKernel_x", "aten::misc", "hipgraph_launch"],
            "count": [1, 1, 1],
            "Kernel Time (µs)_sum": [100.0, 200.0, 300.0],
        }
    )
    _write_category_inputs(base, "other", df)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    other_analysis.main()
    metrics = _read_metrics(base, "other")
    assert metrics["status"] == "OK"
    # the nccl op is stripped into skipped_comm_ops
    assert metrics["category_specific"]["communication_ops_skipped"]["count"] == 1


def test_driver_other_main_missing_csv_errors(tmp_path, monkeypatch):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    with pytest.raises(SystemExit):
        other_analysis.main()
    metrics = _read_metrics(base, "other")
    assert metrics["status"] == "ERROR"


def test_driver_moe_main_no_data(tmp_path, monkeypatch):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    moe_analysis.main()
    metrics = _read_metrics(base, "moe_fused")
    assert metrics["status"] == "NO_DATA"


def test_driver_moe_main_with_data(tmp_path, monkeypatch):
    base = str(tmp_path)
    _write_category_inputs(base, "moe_fused", _simple_ops())
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    moe_analysis.main()
    metrics = _read_metrics(base, "moe_fused")
    assert metrics["status"] == "OK"


def test_driver_cpu_idle_main(tmp_path, monkeypatch):
    base = str(tmp_path)
    csv_dir = os.path.join(base, "perf_report_csvs")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    pd.DataFrame(
        {
            "type": ["idle_time", "computation_time", "total_time"],
            "time ms": [20.0, 80.0, 100.0],
            "percent": [20.0, 80.0, 100.0],
        }
    ).to_csv(os.path.join(csv_dir, "gpu_timeline.csv"), index=False)
    pd.DataFrame({"Kernel Time (µs)_sum": [50.0], "Count": [5]}).to_csv(
        os.path.join(csv_dir, "ops_summary.csv"), index=False
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    cpu_idle_analysis.main()
    metrics = _read_metrics(base, "cpu_idle")
    assert metrics["status"] == "OK"
    # idle 20% > 15% threshold
    assert metrics["idle_flagged"] is True


# ----- kernel_fusion_analysis.main() end-to-end -----


def _write_fusion_inputs(base, candidates):
    cat_dir = os.path.join(base, "category_data")
    meta_dir = os.path.join(base, "metadata")
    csv_dir = os.path.join(base, "perf_report_csvs")
    for d in (cat_dir, meta_dir, csv_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(cat_dir, "fusion_candidates.json"), "w") as f:
        json.dump(candidates, f)
    with open(os.path.join(cat_dir, "category_manifest.json"), "w") as f:
        json.dump(
            {"platform": "MI300X", "gpu_utilization": {"total_time_ms": 100.0}}, f
        )
    with open(os.path.join(meta_dir, "gemm_metadata.json"), "w") as f:
        json.dump(
            {"peak_mem_bw_tbs": 5.3, "max_achievable_tflops": {"matrix_bf16": 1300.0}},
            f,
        )
    pd.DataFrame(
        {
            "name": [_MM_KEY[0], _ADD_KEY[0]],
            "Input Dims": [_MM_KEY[1], _ADD_KEY[1]],
            "Input type": [_MM_KEY[2], _ADD_KEY[2]],
            "Input Strides": [_MM_KEY[3], _ADD_KEY[3]],
            "Data Moved (MB)": [1.0, 1.0],
            "GFLOPS": [None, None],
            "FLOPS/Byte": [None, None],
            "Compute Spec": ["matrix_bf16", None],
        }
    ).to_csv(os.path.join(csv_dir, "unified_perf_summary.csv"), index=False)


def test_driver_fusion_main_missing_files_errors(tmp_path, monkeypatch):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    with pytest.raises(SystemExit):
        kernel_fusion_analysis.main()
    metrics = _read_metrics(base, "kernel_fusion")
    assert metrics["status"] == "ERROR"


def test_driver_fusion_main_no_candidates(tmp_path, monkeypatch):
    base = str(tmp_path)
    _write_fusion_inputs(base, [])
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    kernel_fusion_analysis.main()
    metrics = _read_metrics(base, "kernel_fusion")
    assert metrics["status"] == "NO_DATA"
    assert metrics["candidate_count"] == 0


def _main_candidate(name, kind, confidence, total_us):
    """A GEMM + elementwise candidate keyed to the perf rows of _write_fusion_inputs."""
    half = total_us / 2
    kernels = [
        KernelRecord("gemm_a", "GEMM", half, _MM_KEY),
        KernelRecord("ew_a", "Elementwise", half, _ADD_KEY),
    ]
    ops = [
        _op_group(_MM_KEY, ["gemm_a"], half, 1),
        _op_group(_ADD_KEY, ["ew_a"], half, 1),
    ]
    return _standalone_candidate(
        kernels, total_us, ops, kind, confidence, name
    ).to_dict()


def test_driver_fusion_main_ok_with_estimate(tmp_path, monkeypatch):
    base = str(tmp_path)
    candidate = _main_candidate("mlp_block", "gemm_epilogue", "medium", 30000.0)
    _write_fusion_inputs(base, [candidate])
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    kernel_fusion_analysis.main()
    metrics = _read_metrics(base, "kernel_fusion")
    assert metrics["status"] == "OK"
    assert metrics["candidate_count"] == 1
    assert metrics["platform"] == "MI300X"
    (estimate,) = metrics["impact_estimates"]
    assert (estimate["fusion_kind"], estimate["confidence"]) == (
        "gemm_epilogue",
        "medium",
    )
    assert metrics["high_confidence_op_keys"] == []


def test_driver_fusion_main_suppresses_ops_of_high_confidence_findings(
    tmp_path, monkeypatch
):
    base = str(tmp_path)
    # Only high-confidence findings that clear the impact gate suppress ops. An
    # attention kernel replaces its GEMMs; an act_mul fusion leaves the GEMM running.
    _write_fusion_inputs(
        base,
        [
            _main_candidate("attention_high", "attention", "high", 30000.0),
            _main_candidate("act_mul_high", "act_mul", "high", 20000.0),
            _main_candidate("mlp_medium", "gemm_epilogue", "medium", 30000.0),
            _main_candidate("act_mul_gated", "act_mul", "high", 3000.0),
        ],
    )
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    kernel_fusion_analysis.main()
    metrics = _read_metrics(base, "kernel_fusion")
    assert {e["operation"] for e in metrics["impact_estimates"]} == {
        "attention_high",
        "act_mul_high",
        "mlp_medium",
    }
    assert metrics["high_confidence_op_keys"] == [_MM_KEY, _ADD_KEY, _ADD_KEY]


# ----- multi_kernel_analysis.main() end-to-end -----


def test_driver_multi_kernel_main_missing_data(tmp_path, monkeypatch):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "category_data"), exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    multi_kernel_analysis.main()
    metrics = _read_metrics(base, "multi_kernel")
    assert metrics["status"] == "ERROR"
    assert metrics["memcpy_assessment"]["flagged"] is False


def test_driver_multi_kernel_main_success(tmp_path, monkeypatch):
    base = str(tmp_path)
    cat_dir = os.path.join(base, "category_data")
    csv_dir = os.path.join(base, "perf_report_csvs")
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    mk_data = {
        "memcpy_summary": {
            "total_time_us": 100_000,
            "total_count": 5,
            "by_direction": {
                "D2H": {"count": 3, "total_time_us": 100_000, "avg_bytes": 1024}
            },
        },
        "overlap_analysis": {
            "total_comm_time_us": 10_000,
            "exposed_comm_time_us": 8_000,
            "comm_percent_of_total": 8.0,
            "comm_overlap_ratio": 0.2,
            "total_time_us": 200_000,
        },
        "nccl_summary": {"total_count": 2, "total_time_us": 10_000, "top_ops": []},
    }
    with open(os.path.join(cat_dir, "multi_kernel_data.json"), "w") as f:
        json.dump(mk_data, f)
    pd.DataFrame(
        {
            "type": ["total_time", "exposed_comm_time"],
            "time ms": [200.0, 8.0],
            "percent": [100.0, 4.0],
        }
    ).to_csv(os.path.join(csv_dir, "gpu_timeline.csv"), index=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", base])
    multi_kernel_analysis.main()
    metrics = _read_metrics(base, "multi_kernel")
    assert metrics["status"] == "SUCCESS"
    assert metrics["memcpy_assessment"]["flagged"] is True
    assert metrics["overlap_assessment"]["flagged"] is True
    assert len(metrics["patterns_detected"]) >= 1


class TestArchUtils:
    def test_list_and_load_platform(self):
        platforms = arch_utils.list_platforms()
        assert platforms
        for name in platforms[:3]:
            arch = arch_utils.load_arch(name)
            assert "name" in arch or "mem_bw_gbps" in arch

    def test_tl_extension_override(self, tmp_path, monkeypatch):
        ext_pkg = tmp_path / "fake_ext_pkg"
        arch_dir = ext_pkg / "Agent" / "Analysis" / "utils" / "arch"
        arch_dir.mkdir(parents=True)
        custom = {"name": "CUSTOM", "mem_bw_gbps": 1000, "memory_gb": 80}
        (arch_dir / "CUSTOM.json").write_text(json.dumps(custom))

        init_py = ext_pkg / "__init__.py"
        init_py.write_text("")

        fake_mod = ModuleType("fake_ext_pkg")
        fake_mod.__file__ = str(init_py)
        monkeypatch.setitem(sys.modules, "fake_ext_pkg", fake_mod)
        monkeypatch.setenv("TL_EXTENSION", "fake_ext_pkg")

        mapping = arch_utils._collect_arch_jsons()
        assert "CUSTOM" in mapping
        assert arch_utils.load_arch("CUSTOM")["mem_bw_gbps"] == 1000


class TestCategoryAnalysisHelpers:
    _META = {"peak_mem_bw_tbs": 5.3, "peak_maf_tflops": {"matrix_fp16": 654}}

    def test_gemm_classifiers(self):
        assert gemm_analysis.detect_quantized_gemm("aten::w8a8_mm")
        info = gemm_analysis.classify_gemm_operation("aten::mm", None)
        assert info["gemm_type"] == "regular"
        qinfo = gemm_analysis.classify_gemm_operation("aten::fp8_mm", None)
        assert qinfo["is_quantized"] is True
        ops = pd.DataFrame(
            {
                "name": ["aten::mm", "aten::fp8_mm"],
                "TFLOPS/s_mean": [100.0, None],
            }
        )
        extra = gemm_analysis.extract_category_specific(ops, self._META)
        assert extra["quantized_count"] == 1
        assert extra["missing_perf_model_count"] == 1

    def test_elementwise_extract(self):
        out = elementwise_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::add"]}), self._META
        )
        assert out["peak_mem_bw_tbs"] == 5.3

    def test_norm_extract(self):
        out = norm_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::layer_norm"]}), self._META
        )
        assert out["peak_mem_bw_tbs"] == 5.3

    def test_reduce_extract(self):
        out = reduce_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::sum"]}), self._META
        )
        assert "peak_mem_bw_tbs" in out

    def test_convolution_extract_with_transpose(self):
        ops = pd.DataFrame(
            {
                "name": ["aten::conv2d", "aten::conv_transpose2d"],
                "Kernel Time (µs)_sum": [1000.0, 500.0],
            }
        )
        out = convolution_analysis.extract_category_specific(ops, self._META)
        assert out["transpose_count"] == 1
        assert out["transpose_time_ms"] == pytest.approx(0.5)

    def test_triton_classifiers(self):
        assert (
            triton_analysis.classify_triton_operation("triton_poi_add", None)[
                "kernel_type"
            ]
            == "pointwise"
        )
        assert (
            triton_analysis.classify_triton_operation("triton_red_sum", None)[
                "kernel_type"
            ]
            == "reduction"
        )
        assert (
            triton_analysis.classify_triton_operation("triton_per_mm", None)[
                "kernel_type"
            ]
            == "persistent"
        )
        assert (
            triton_analysis.classify_triton_operation("other", None)["kernel_type"]
            == "other"
        )
        out = triton_analysis.extract_category_specific(
            pd.DataFrame(
                {
                    "name": [
                        "triton_poi_a",
                        "triton_red_b",
                        "triton_per_c",
                        "other",
                    ]
                }
            ),
            self._META,
        )
        assert out["pointwise_count"] == 1
        assert out["reduction_count"] == 1

    def test_reduce_softmax_detect(self):
        assert reduce_analysis.detect_softmax("aten::_softmax")
        out = reduce_analysis.extract_category_specific(
            pd.DataFrame({"name": ["aten::_softmax", "aten::sum"]}), self._META
        )
        assert out["softmax_count"] == 1

    def test_moe_extract_and_no_data_check(self, tmp_path):
        out = moe_analysis.extract_category_specific(
            pd.DataFrame({"name": ["moe_dispatch"]}), self._META
        )
        assert "peak_mem_bw_tbs" in out
        missing = moe_analysis._check_moe_data(str(tmp_path), "moe_fused", "standalone")
        assert missing["status"] == "NO_DATA"


class TestPerfModelNormAndConvDeep:
    def test_batch_norm_bwd_full(self):
        event = {
            "name": "aten::miopen_batch_norm_backward",
            "args": {
                "Input Dims": [
                    (8, 16, 32, 32),
                    (8, 16, 32, 32),
                    (16,),
                    (16,),
                    (16,),
                    (16,),
                    (16,),
                    (),
                ],
                "Input type": ["float"] * 7 + ["Scalar"],
                "Input Strides": [(16384, 1024, 32, 1)] * 2 + [(1,)] * 5 + [()],
                "Concrete Inputs": ["", "", "", "", "", "", "", "1e-5"],
            },
        }
        model = perf_model.BatchNormBwd(event)
        assert model.flops() > 0
        assert model.bytes() > 0

    def test_group_norm_bwd(self):
        event = {
            "args": {
                "Input Dims": [
                    None,
                    (4, 8, 32, 32),
                    (8,),
                    (8,),
                    (8,),
                    (8,),
                    (4, 8, 32, 32),
                    (),
                ],
                "Input type": ["c10::BFloat16"] * 7 + ["Scalar"],
                "Input Strides": [(), (8192, 1024, 32, 1), (1,)] * 2
                + [(8192, 1024, 32, 1)] * 2
                + [(), ()],
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "8",
                    "8",
                    "[True, True]",
                ],
            }
        }
        model = perf_model.GroupNormBwd(event)
        assert model.flops() > 0

    def test_conv_bias_bwd_with_forward_cache(self):
        fwd = perf_model.ConvBias_(_conv_bias_fwd_event())
        assert fwd.flops() > 0
        bwd = perf_model.ConvBias_Backward(_conv_bias_bwd_event())
        assert bwd.flops_bwd() > 0


def _build_moe_tree(events: List[Dict], add_python_func: bool = False) -> TraceToTree:
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree(add_python_func=add_python_func)
    return tree


def _setup_gemm_output_dir(tmp_path):
    out = tmp_path / "analysis_output"
    (out / "category_data").mkdir(parents=True)
    (out / "metadata").mkdir()
    df = pd.DataFrame(
        {
            "name": ["aten::mm", "aten::addmm"],
            "count": [12, 3],
            "Kernel Time (µs)_sum": [100_000.0, 50_000.0],
            "Kernel Time (µs)_mean": [8000.0, 16000.0],
            "Kernel Time (µs)_std": [500.0, 100.0],
            "TFLOPS/s_mean": [400.0, 350.0],
            "TB/s_mean": [0.5, 0.4],
            "FLOPS/Byte": [2000.0, 1800.0],
            "Roofline Bound": ["COMPUTE_BOUND", "COMPUTE_BOUND"],
            "Compute Spec": ["matrix_bf16", "matrix_bf16"],
            "kernel_details_summary": ["[{'name': 'Cijk_a'}]", "[{'name': 'Cijk_b'}]"],
            "call_stack_full": ["['aten::mm', 'Linear']", "['aten::addmm']"],
            "Input Dims": ["[[32, 64], [64, 128]]", "[[32, 64], [64, 128], [32, 128]]"],
            "Input type": ["['fp16', 'fp16']", "['fp16', 'fp16', 'fp16']"],
        }
    )
    df.to_csv(out / "category_data" / "gemm_ops.csv", index=False)
    meta = {
        "platform": "MI300X",
        "peak_mem_bw_tbs": 5.3,
        "max_achievable_tflops": {"matrix_bf16": 708},
        "gpu_utilization": {"total_time_ms": 1000.0},
        "output_dir": str(out),
    }
    (out / "metadata" / "gemm_metadata.json").write_text(json.dumps(meta))
    return str(out)


class TestAnalysisUtilsRunCategoryPhase9:
    def test_run_category_analysis_success(self, tmp_path):
        out = _setup_gemm_output_dir(tmp_path)
        au.run_category_analysis("gemm", out, {}, lambda ops_df, _m: {"n": len(ops_df)})
        metrics = json.loads(
            Path(out, "category_data", "gemm_metrics.json").read_text()
        )
        assert metrics["status"] == "OK"

    def test_run_category_analysis_no_data(self, tmp_path):
        out = tmp_path / "empty"
        (out / "category_data").mkdir(parents=True)
        au.run_category_analysis(
            "gemm",
            str(out),
            {},
            lambda _o, _m: {},
            no_data_check_fn=lambda _o, c, _s: {"category": c, "status": "NO_DATA"},
        )
        assert (
            json.loads((out / "category_data" / "gemm_metrics.json").read_text())[
                "status"
            ]
            == "NO_DATA"
        )

    def test_run_category_analysis_missing_csv_exits(self, tmp_path):
        out = tmp_path / "missing"
        (out / "category_data").mkdir(parents=True)
        with pytest.raises(SystemExit):
            au.run_category_analysis("gemm", str(out), {}, lambda _o, _m: {})


class TestArchAndMoePhase9:
    def test_arch_utils(self, monkeypatch):
        assert arch_utils.list_platforms()
        monkeypatch.setenv("TL_EXTENSION", "not_a_real_package_xyz")
        assert isinstance(arch_utils._collect_arch_jsons(), dict)

    def test_moe_pseudo_op_edges(self):
        assert not is_aiter_fused_moe_kernel(
            {"cat": "kernel", "name": "aiter::quant_fmoe"}
        )
        tree = _build_moe_tree([])
        _create_pseudo_op_moe_fused_aiter(tree, {"name": "wrong", "UID": 0})
        assert _extract_topk_from_outplace({"UID": 1, "args": {}}) == 8

        events = []
        moe = _mk_event(
            "cpu_op",
            "vllm::outplace_fused_experts",
            100,
            200,
            1,
            1,
            {
                "Input Dims": [
                    [128, 4096],
                    [8, 4096, 512],
                    [8, 4096, 512],
                    [128, 6],
                    [128, 6],
                ],
                "Sequence number": 2,
            },
        )
        events.append(moe)
        _add_gpu_chain(events, moe, 20, "fused_moe_kernel_gptq_awq_up", 110, 150)
        _add_gpu_chain(events, moe, 21, "fused_moe_kernel_gptq_awq_down", 160, 190)
        tree2 = _build_moe_tree(events)
        create_pseudo_ops_moe_gptq_awq(tree2)
        assert any(e.get("args", {}).get("Pseudo op") for e in tree2.events)

        fly_events = [
            _mk_event("cpu_op", FUSED_MOE_PARENT, 0, 500, 1, 1, {"Sequence number": 9}),
            _mk_event(
                "python_function", "flydsl.py(10): flydsl_moe_stage1", 50, 100, 1, 1, {}
            ),
        ]
        create_pseudo_ops_moe_flydsl(_build_moe_tree(fly_events, add_python_func=True))


class TestPerfModelConvAndNormBoost:
    @pytest.mark.parametrize(
        "cls,fwd_factory,bwd_cls,bwd_factory",
        [
            (
                perf_model.ConvBias_,
                _conv_bias_fwd_event,
                perf_model.ConvBias_Backward,
                _conv_bias_bwd_event,
            ),
            (
                perf_model.ConvBiasReLU_,
                _conv_bias_relu_fwd_event,
                perf_model.ConvBiasReLU_Backward,
                _conv_bias_relu_bwd_event,
            ),
        ],
    )
    def test_conv_bias_family(self, cls, fwd_factory, bwd_cls, bwd_factory):
        fwd = cls(fwd_factory())
        assert fwd.flops() > 0
        assert fwd.bytes() > 0
        bwd = bwd_cls(bwd_factory())
        assert bwd.flops_bwd() > 0
        assert bwd.bytes_bwd() > 0

    def test_fused_ln_modulate(self):
        fwd = perf_model.FusedLnModulate(_fused_ln_fwd_event())
        assert fwd.flops() > 0
        assert fwd.bytes() > 0

    def test_evoformer_attention(self):
        evo = perf_model.evoformer_attention(_evoformer_event())
        assert evo.flops() > 0
        assert evo.bytes() > 0

    def test_reduce_and_grouped_gemm(self):
        reduce_evt = {
            "name": "aten::mean",
            "args": {
                "Input Dims": [(4, 256)],
                "Input type": ["c10::BFloat16"],
                "Concrete Inputs": ["", "[1]", "True"],
            },
        }
        model = perf_model.aten_reduce(reduce_evt)
        assert model.flops() > 0
        gg_event = {
            "args": {
                "Input Dims": [
                    [4, 128],
                    [8, 256, 128],
                    [8, 256],
                    [8],
                    [8],
                    [8],
                    [8],
                    [8],
                    [4, 4],
                ],
                "Input type": [
                    "c10::BFloat16",
                    "c10::Float8_e4m3fn",
                    "c10::Float",
                    "c10::Int",
                ]
                + ["c10::Int"] * 5,
            }
        }
        g = perf_model.primus_turbo_grouped_gemm(gg_event)
        assert g.flops() > 0


class TestRmsNormExtensionsBytes:
    def test_rmsnorm_family_bytes(self):
        base = {
            "args": {
                "Input Dims": [(4, 256), (256,), (256,)],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
                "Input Strides": [(256, 1), (1,), (1,)],
            }
        }
        for cls in (rms_ext.aiter_rmsnorm,):
            model = cls(base)
            b = model.bytes()
            assert b is None or b > 0
            assert model.get_compute_precision() in (None, "bf16", "fp8")
