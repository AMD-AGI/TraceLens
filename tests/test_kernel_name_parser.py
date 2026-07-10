###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Integration tests using verbatim kernel names extracted from trace files.

Trace sources:
  Tensile  — MI300 Qwen inference trace
  iGEMM    — MI300 ResNet training trace
  cuBLAS   — H100 Qwen, Wan-AI, Falconsai inference traces
  CUTLASS  — H100 BERT, Falconsai, Wan-AI inference traces
  CK       — MI300 vLLM FP8 inference trace
  Triton   — MI300 SGLang and vLLM FP8 inference traces
  nvjet    — B200 and H100 graph capture traces
"""

from TraceLens.PerfModel.kernel_name_parser import gemm_name_parser


class TestRealTensile:
    # Source: MI300 Qwen inference trace

    def test_mt256x128x64_tn(self):
        name = (
            "Cijk_Alik_Bljk_B_BS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x64_MI16x16x1_SN_LDSB1_"
            "AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_"
            "GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA1024_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_"
            "LWPMn1_MIAV0_MIWT8_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS16_NLCA1_NLCB1_"
            "ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW8_SK0_SKXCCM0_TLDS1_ULSGRO0_"
            "USL1_UIOFGRO0_USFGROn1_VSn1_VWA8_VWB4_WSGRA0_WSGRB0_WS64_WG32_8_1"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 256
        assert result["mt_n"] == 128
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)

    def test_mt128x224x64_tn(self):
        name = (
            "Cijk_Alik_Bljk_B_BS_BH_Bias_HA_S_SAV_UserArgs_MT128x224x64_MI16x16x1_SN_LDSB1_"
            "AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_"
            "GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA512_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_"
            "LWPMn1_MIAV0_MIWT4_7_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS16_NLCA1_NLCB1_"
            "ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKXCCM0_TLDS1_ULSGRO0_"
            "USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 224
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)

    def test_mt16x16x128_nn(self):
        name = (
            "Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x128_MI16x16x1_SN_LDSB1_AFC1_"
            "AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA2_GRVWB2_GSUAMB_"
            "GLS0_ISA942_IU1_K1_LBSPPA256_LBSPPB512_LBSPPM0_LPA16_LPB8_LPM0_LRVW4_LWPMn1_"
            "MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_ONLL1_"
            "PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_"
            "UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 16
        assert result["mt_k"] == 128
        assert result["transpose"] == (False, False)

    def test_mt16x16x256_tn(self):
        # Source: MI300 Qwen inference trace
        name = (
            "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x256_MI16x16x1_SN_LDSB1_"
            "AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_"
            "GSUAMBSK_GLS0_ISA942_IU1_K1_LBSPPA512_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_"
            "LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_"
            "ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_"
            "USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG16_4_4"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 16
        assert result["mt_k"] == 256
        assert result["transpose"] == (True, False)

    def test_mt64x32x256_tn(self):
        # Source: MI300 Qwen inference trace
        name = (
            "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x32x256_MI16x16x1_SN_LDSB1_"
            "AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_"
            "GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA1024_LBSPPB512_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_"
            "LWPMn1_MIAV0_MIWT2_1_MO40_NTn1_NTA4_NTB0_NTC0_NTD4_NTM0_NEPBS16_NLCA1_NLCB1_"
            "ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW2_SK0_SKXCCM0_TLDS1_ULSGRO0_"
            "USL1_UIOFGRO0_USFGROn1_VSn1_VWA2_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 64
        assert result["mt_n"] == 32
        assert result["mt_k"] == 256
        assert result["transpose"] == (True, False)


class TestRealIgemm:
    # Source: MI300 ResNet training trace

    def test_wrw_64x32x32(self):
        name = (
            "igemm_wrw_gtcx3_nhwc_bf16_bx0_ex1_bt64x32x32_wt16x16x16_ws1x1_wr2x1"
            "_ta1x4x1x2_1x8x1x32_tb1x4x1x1_1x8x1x32_vs1_gkgs"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 64
        assert result["mt_n"] == 32
        assert result["mt_k"] == 32
        assert result["transpose"] == (None, None)

    def test_fwd_256x64x8(self):
        name = (
            "igemm_fwd_gtcx3_nhwc_bf16_bx0_ex1_bt256x64x8_wt64x16x4_ws1x1_wr2x2"
            "_ta1x1x8x1_1x8x1x32_tb1x1x2x1_1x8x1x32_me"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 256
        assert result["mt_n"] == 64
        assert result["mt_k"] == 8
        assert result["transpose"] == (None, None)

    def test_fwd_64x128x32(self):
        name = (
            "igemm_fwd_gtcx3_nhwc_bf16_bx0_ex1_bt64x128x32_wt32x32x8_ws1x1_wr2x1"
            "_ta1x8x1x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 64
        assert result["mt_n"] == 128
        assert result["mt_k"] == 32
        assert result["transpose"] == (None, None)

    def test_fwd_128x128x32(self):
        name = (
            "igemm_fwd_gtcx3_nhwc_bf16_bx0_ex1_bt128x128x32_wt32x32x8_ws1x1_wr2x2"
            "_ta1x8x2x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 128
        assert result["mt_k"] == 32
        assert result["transpose"] == (None, None)

    def test_wrw_256x128x16(self):
        name = (
            "igemm_wrw_gtcx3_nhwc_bf16_bx0_ex1_bt256x128x16_wt32x32x8_ws2x1_wr2x2"
            "_ta1x4x1x4_1x4x1x64_tb1x4x1x2_1x4x1x64_gkgs"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 256
        assert result["mt_n"] == 128
        assert result["mt_k"] == 16
        assert result["transpose"] == (None, None)


class TestRealCublasXmma:
    # Source: H100 Qwen, Wan-AI, and Falconsai inference traces

    def test_sm90_128x128x64_tn(self):
        name = (
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n"
            "_tilesize128x128x64_warpgroupsize1x1x1"
            "_execute_segment_k_off_kernel__5x_cublas"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 128
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)

    def test_sm90_64x64x64_tn(self):
        name = (
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n"
            "_tilesize64x64x64_warpgroupsize1x1x1"
            "_execute_segment_k_off_kernel__5x_cublas"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 64
        assert result["mt_n"] == 64
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)

    def test_sm90_128x64x64_tn(self):
        name = (
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n"
            "_tilesize128x64x64_warpgroupsize1x1x1"
            "_execute_segment_k_off_kernel__5x_cublas"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 64
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)

    def test_sm90_256x128x64_tn(self):
        # Source: H100 Wan-AI inference trace
        name = (
            "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n"
            "_tilesize256x128x64_warpgroupsize2x1x1"
            "_execute_segment_k_off_kernel__5x_cublas"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 256
        assert result["mt_n"] == 128
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)

    def test_sm90_implicit_gemm_128x128x64(self):
        # Source: H100 Wan-AI inference trace
        name = (
            "sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc"
            "_tilesize128x128x64_warpgroupsize1x1x1_g1"
            "_execute_segment_k_off_kernel__5x_cudnn"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 128
        assert result["mt_k"] == 64


class TestRealCutlass:
    # Source: H100 BERT, Falconsai, and Wan-AI inference traces

    def test_wmma_32x32x128_tn(self):
        name = (
            "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16"
            "_32x32_128x2_tn_align8>"
            "(cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8::Params)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 32
        assert result["mt_n"] == 32
        assert result["mt_k"] == 128
        assert result["transpose"] == (True, False)

    def test_tensorop_64x64x64_tn(self):
        name = (
            "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16"
            "_64x64_64x6_tn_align8>"
            "(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8::Params)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 64
        assert result["mt_n"] == 64
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)

    def test_tensorop_256x128x64_tn(self):
        name = (
            "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16"
            "_256x128_64x3_tn_align2>"
            "(cutlass_80_tensorop_bf16_s16816gemm_bf16_256x128_64x3_tn_align2::Params)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 256
        assert result["mt_n"] == 128
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)

    def test_cudnn_fprop_256x64x32(self):
        # cuDNN conv variant — no transpose in name
        name = (
            "void cutlass__5x_cudnn::Kernel<cutlass_tensorop_bf16_s16816fprop_optimized_bf16"
            "_256x64_32x4_nhwc_align8>"
            "(cutlass_tensorop_bf16_s16816fprop_optimized_bf16_256x64_32x4_nhwc_align8::Params)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 256
        assert result["mt_n"] == 64
        assert result["mt_k"] == 32

    def test_tensorop_128x64x64_tn(self):
        # Source: H100 Falconsai inference trace
        name = (
            "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16"
            "_128x64_64x6_tn_align8>"
            "(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x6_tn_align8::Params)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 64
        assert result["mt_k"] == 64
        assert result["transpose"] == (True, False)


class TestRealCKDemangled:
    # Source: MI300 vLLM FP8 inference trace

    def test_kernel_gemm_xdl_abscale(self):
        # ABScale layout: BlockSize=256, ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128,
        # MPerBlock=16, NPerBlock=128, KPerBlock=256
        name = (
            "void ck::kernel_gemm_xdl_cshuffle_v3<ck::GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<>, ck::tensor_layout::gemm::RowMajor, ck::f8_fnuz_t, ck::f8_fnuz_t, "
            "float, float, ck::Tuple<>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "(ck::tensor_operation::device::GemmSpecialization)2, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 16, 1, 16>, ck::Sequence<8>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t>, "
            "true, (ck::InMemoryDataOperationEnum)0, 2, (ck::TailNumber)10>"
            "(ck::GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<>, ck::tensor_layout::gemm::RowMajor, ck::f8_fnuz_t, ck::f8_fnuz_t, "
            "float, float, ck::Tuple<>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "(ck::tensor_operation::device::GemmSpecialization)2, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 16, 1, 16>, ck::Sequence<8>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t>::Argument)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 128
        assert result["mt_k"] == 256
        assert result["transpose"] == (False, True)  # RowMajor A, ColumnMajor B

    def test_kernel_gemm_xdl_abscale_gemspec0(self):
        # Same class, different GemmSpecialization enum value (0 vs 2)
        name = (
            "void ck::kernel_gemm_xdl_cshuffle_v3<ck::GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<>, ck::tensor_layout::gemm::RowMajor, ck::f8_fnuz_t, ck::f8_fnuz_t, "
            "float, float, ck::Tuple<>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "(ck::tensor_operation::device::GemmSpecialization)0, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 16, 1, 16>, ck::Sequence<8>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t>, "
            "true, (ck::InMemoryDataOperationEnum)0, 2, (ck::TailNumber)10>"
            "(ck::GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<>, ck::tensor_layout::gemm::RowMajor, ck::f8_fnuz_t, ck::f8_fnuz_t, "
            "float, float, ck::Tuple<>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "(ck::tensor_operation::device::GemmSpecialization)0, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 16, 1, 16>, ck::Sequence<8>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t>::Argument)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 128
        assert result["mt_k"] == 256

    def test_kernel_moe_gemm_blockscale(self):
        # MoeGemmBlockScale: BlockSize=256, ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128,
        # MPerBlock=16, NPerBlock=128, KPerBlock=256
        name = (
            "void ck::kernel_moe_gemm<ck::GridwiseMoeGemmBlockScale<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<ck::tensor_layout::gemm::RowMajor>, ck::tensor_layout::gemm::RowMajor, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, float, float, ck::Tuple<float>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "MulABScaleExpertWeightA8W8blkscale, "
            "(ck::tensor_operation::device::GemmSpecialization)0, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "1, false, true, false, false, int, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, true>, "
            "true, (ck::InMemoryDataOperationEnum)0, 2, (ck::TailNumber)1>"
            "(ck::GridwiseMoeGemmBlockScale<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<ck::tensor_layout::gemm::RowMajor>, ck::tensor_layout::gemm::RowMajor, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, float, float, ck::Tuple<float>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "MulABScaleExpertWeightA8W8blkscale, "
            "(ck::tensor_operation::device::GemmSpecialization)0, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "1, false, true, false, false, int, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, true>::Argument)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 128
        assert result["mt_k"] == 256
        assert result["transpose"] == (False, True)  # RowMajor A, ColumnMajor B

    def test_kernel_moe_gemm_blockscale_tailnumber0(self):
        # Same class, different TailNumber (0 vs 1) and InMemoryDataOperationEnum (1 vs 0)
        name = (
            "void ck::kernel_moe_gemm<ck::GridwiseMoeGemmBlockScale<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<ck::tensor_layout::gemm::RowMajor>, ck::tensor_layout::gemm::RowMajor, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, float, float, ck::Tuple<float>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "MulABScaleExpertWeightA8W8blkscale, "
            "(ck::tensor_operation::device::GemmSpecialization)0, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 4, 1, 64>, ck::Sequence<2, 1, 1, 1>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "0, false, false, false, true, int, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, true>, "
            "false, (ck::InMemoryDataOperationEnum)1, 2, (ck::TailNumber)0>"
            "(ck::GridwiseMoeGemmBlockScale<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<ck::tensor_layout::gemm::RowMajor>, ck::tensor_layout::gemm::RowMajor, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, float, float, ck::Tuple<float>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "MulABScaleExpertWeightA8W8blkscale, "
            "(ck::tensor_operation::device::GemmSpecialization)0, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 4, 1, 64>, ck::Sequence<2, 1, 1, 1>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "0, false, false, false, true, int, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, true>::Argument)"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 128
        assert result["mt_k"] == 256

    def test_kernel_gemm_xdl_abscale_via_sglang(self):
        # Source: MI300 SGLang FP8 inference trace — different GemmSpec variant
        name = (
            "void ck::kernel_gemm_xdl_cshuffle_v3<ck::GridwiseGemmMultiD_ABScale_xdl_cshuffle_v3<"
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, "
            "ck::Tuple<>, ck::tensor_layout::gemm::RowMajor, ck::f8_fnuz_t, ck::f8_fnuz_t, "
            "float, float, ck::Tuple<>, unsigned short, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "ck::tensor_operation::element_wise::PassThrough, "
            "(ck::tensor_operation::device::GemmSpecialization)0, "
            "256, 1, 128, 128, 16, 128, 256, 16, 16, 16, 16, 1, 2, "
            "ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, "
            "2, 16, 16, false, 0, ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, "
            "ck::Sequence<1, 0, 2>, 2, 16, 16, false, 0, 1, 2, "
            "ck::Sequence<1, 16, 1, 16>, ck::Sequence<8>, "
            "(ck::BlockGemmPipelineScheduler)0, (ck::BlockGemmPipelineVersion)0, "
            "ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t, ck::f8_fnuz_t>, "
            "true, (ck::InMemoryDataOperationEnum)0, 2, (ck::TailNumber)10>"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 128
        assert result["mt_k"] == 256


class TestRealTriton:
    # Source: MI300 SGLang and vLLM FP8 inference traces

    def test_blockscale_128x128x128_splitk_7168_grid17(self):
        name = (
            "_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128"
            "_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128"
            "_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_7168"
            "_EVEN_K_1_GRID_MN_17_cache_modifier_CG"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 128
        assert result["mt_k"] == 128
        assert result["transpose"] == (None, None)

    def test_blockscale_128x128x128_splitk_1536_grid24(self):
        name = (
            "_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128"
            "_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128"
            "_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_1536"
            "_EVEN_K_1_GRID_MN_24_cache_modifier_CG"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 128
        assert result["mt_k"] == 128

    def test_blockscale_128x128x128_splitk_2048_grid56(self):
        name = (
            "_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128"
            "_BLOCK_SIZE_M_128_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128"
            "_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_2048"
            "_EVEN_K_1_GRID_MN_56_cache_modifier_CG"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 128
        assert result["mt_n"] == 128
        assert result["mt_k"] == 128

    def test_batched_gemm_16x128x128(self):
        # Source: MI300 vLLM FP8 inference trace
        name = (
            "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel"
            "_HAS_BIAS_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N_128_BLOCK_SIZE_K_128"
            "_GROUP_SIZE_M_1_EVEN_K_1_cache_modifier_CG_GRID_MN_8"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 128
        assert result["mt_k"] == 128

    def test_batched_gemm_16x64x128(self):
        # Source: MI300 vLLM FP8 inference trace
        name = (
            "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel"
            "_HAS_BIAS_0_BLOCK_SIZE_M_16_BLOCK_SIZE_N_64_BLOCK_SIZE_K_128"
            "_GROUP_SIZE_M_1_EVEN_K_1_cache_modifier_CG_GRID_MN_4"
        )
        result = gemm_name_parser(name)
        assert result is not None
        assert result["mt_m"] == 16
        assert result["mt_n"] == 64
        assert result["mt_k"] == 128


class TestRealNvjet:
    # Source: B200 eager inference trace and H100 graph capture trace
    # Transpose is the last 3 chars: T=transposed, N=not transposed, order = A, B, C.

    def test_tst_bias_tnt(self):
        # Source: B200 eager inference trace
        result = gemm_name_parser("nvjet_tst_64x64_64x16_2x1_2cta_v_bz_bias_TNT")
        assert result is not None
        assert result["transpose"] == (True, False)

    def test_tst_bias_tnn(self):
        # Source: B200 eager inference trace
        result = gemm_name_parser("nvjet_tst_144x128_64x6_2x1_v_bz_bias_TNN")
        assert result is not None
        assert result["transpose"] == (True, False)

    def test_tst_no_bias_tnt(self):
        # Source: B200 eager inference trace
        result = gemm_name_parser("nvjet_tst_192x8_64x8_4x1_v_bz_TNT")
        assert result is not None
        assert result["transpose"] == (True, False)

    def test_hsh_coopa_tnt(self):
        # Source: H100 graph capture trace
        result = gemm_name_parser("nvjet_hsh_256x144_64x4_1x2_h_bz_coopA_TNT")
        assert result is not None
        assert result["transpose"] == (True, False)

    def test_hsh_coopa_bias_tnn(self):
        # Source: H100 graph capture trace
        result = gemm_name_parser("nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_bias_TNN")
        assert result is not None
        assert result["transpose"] == (True, False)
