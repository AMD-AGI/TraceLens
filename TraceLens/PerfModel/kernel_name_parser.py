###############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re


def gemm_name_parser(kernel_name):
    """
    Parse the kernel name to identify GEMM op details.
    Args:
        kernel_name (str): The name of the kernel.
    Returns:
        dict: A dictionary containing the GEMM operation details.
    """
    if is_tensile_gemm(kernel_name):
        return parse_tensile_gemm(kernel_name)
    elif is_triton_gemm(kernel_name):
        return parse_triton_gemm(kernel_name)
    elif is_ck_gemm(kernel_name):
        return parse_ck_gemm(kernel_name)
    elif is_igemm(kernel_name):
        return parse_igemm(kernel_name)
    elif is_cublas_xmma_gemm(kernel_name):
        return parse_cublas_xmma_gemm(kernel_name)
    elif is_cutlass_gemm(kernel_name):
        return parse_cutlass_gemm(kernel_name)
    elif is_nvjet_gemm(kernel_name):
        return parse_nvjet_gemm(kernel_name)


def is_tensile_gemm(kernel_name):
    """
    Check if a kernel name matches the more general ROCm GEMM naming pattern.
    Allows an arbitrary prefix before 'Cijk_Alik_Bljk_...' where each of C/A/B
    is followed by exactly three axis letters.
    Example matches:
      - 'Cijk_Alik_Bljk_...'
      - 'Custom_Cijk_Alik_Bljk_BBS_BH_Bias_AS_SAV_User...'
    """
    pattern = r"^.*C[a-z]{3}_A[a-z]{3}_B[a-z]{3}.*$"
    return bool(re.match(pattern, kernel_name))


def parse_tensile_gemm(kernel_name):

    # 1. Parse the transpose flags from the kernel name
    trans_a, trans_b = None, None
    if "_Ailk_" in kernel_name:
        trans_a = False
    elif "_Alik_" in kernel_name:
        trans_a = True
    if "_Bljk_" in kernel_name:
        trans_b = False
    elif "_Bjlk_" in kernel_name:
        trans_b = True

    # 2. Parse the macro tile size from the kernel name
    # Example: ''Cijk_Ailk_Bjlk_BBS_BH_Bias_HAS_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC...'
    # The macro tile size is usually represented by 'MT' followed by the tile dimensions.
    # In this example, the macro tile size is 'MT64x16x64'.
    # 64 is M tile, 16 is N tile, 64 is K loop unroll called DepthU
    macro_tile_match = re.search(r"MT(\d+)x(\d+)x(\d+)", kernel_name)
    if macro_tile_match:
        mt_m = int(macro_tile_match.group(1))
        mt_n = int(macro_tile_match.group(2))
        mt_k = int(macro_tile_match.group(3))
    else:
        mt_m, mt_n, mt_k = None, None, None  # Fallback in case pattern is not found

    # Feel free to add more details as needed.
    # https://github.com/ROCm/Tensile/wiki/Kernel-Parameters#kernel-names

    return {
        "transpose": (trans_a, trans_b),
        "mt_m": mt_m,
        "mt_n": mt_n,
        "mt_k": mt_k,
    }


def is_ck_gemm(kernel_name):
    """
    Check if a kernel name is a Composable Kernel (CK) GEMM.
    Matches demangled (void ck::kernel_*) and mangled (_ZN2ck*kernel_*) forms.
    """
    if "void ck::kernel_" in kernel_name:
        return True
    if "_ZN2ck" in kernel_name and "kernel_" in kernel_name:
        return True
    return False


def parse_ck_gemm(kernel_name):
    """
    Parse tile sizes from CK kernel names. Handles:
    - Demangled: 3-branch dispatch on Gridwise class name
    - Mangled: 3-anchor ELi integer extraction
    """

    # ── Demangled CK kernels ──
    if "GemmSpecialization)" in kernel_name:
        # Extract all comma-separated integers after GemmSpecialization
        m = re.search(r"GemmSpecialization\)\d+,\s*([\d,\s]+)", kernel_name)
        if not m:
            return None
        ints = [int(x) for x in re.findall(r"\d+", m.group(1))]

        # MoeGemmMX_BPreshuffle: ScaleBlockSize, BlockSize,
        #   MPerBlock, NPerBlock, KPerBlock, ...
        # (checked first because kernel name contains "MulABScale..." which
        #  would falsely match the ABScale branch below)
        if "MoeGemmMX" in kernel_name:
            if len(ints) < 5:
                return None
            mt_m, mt_n, mt_k = ints[2], ints[3], ints[4]
        # ABScale / MoeGemmBlockScale: BlockSize, ScaleBlockM, ScaleBlockN,
        #   ScaleBlockK, MPerBlock, NPerBlock, KPerBlock, ...
        elif "GridwiseGemmMultiD_ABScale" in kernel_name or "MoeGemmBlockScale" in kernel_name:
            if len(ints) < 7:
                return None
            mt_m, mt_n, mt_k = ints[4], ints[5], ints[6]
        # GemmMultiD_xdl (no ABScale), MoeGemm (non-BlockScale):
        #   BlockSize, MPerBlock, NPerBlock, KPerBlock, ...
        elif len(ints) >= 4:
            mt_m, mt_n, mt_k = ints[1], ints[2], ints[3]
        else:
            return None

        return {
            "transpose": (None, None),
            "mt_m": mt_m,
            "mt_n": mt_n,
            "mt_k": mt_k,
        }

    # ── Mangled CK kernels ──

    # Anchor 1: GemmSpecializationE — conv fwd variants (GridwiseGemm_xdl_cshuffle_v3)
    # Ints: BlockSize, MPerBlock, NPerBlock, KPerBlock, ...
    m = re.search(r"GemmSpecializationE\d+E((?:Li\d+E)+)", kernel_name)
    if m:
        ints = [int(x) for x in re.findall(r"Li(\d+)E", m.group(1))]
        if len(ints) >= 4:
            return {
                "transpose": (None, None),
                "mt_m": ints[1],
                "mt_n": ints[2],
                "mt_k": ints[3],
            }

    # Anchor 2: InMemoryDataOperationEnumE — conv fwd_multiple_abd, bwd_data
    #   (GridwiseGemmMultipleD_xdl_cshuffle)
    # Ints: NumGemmKPrefetchStage, BlockSize, MPerBlock, NPerBlock, KPerBlock, ...
    m = re.search(r"InMemoryDataOperationEnumE\d+E((?:Li\d+E)+)", kernel_name)
    if m:
        ints = [int(x) for x in re.findall(r"Li(\d+)E", m.group(1))]
        if len(ints) >= 5:
            return {
                "transpose": (None, None),
                "mt_m": ints[2],
                "mt_n": ints[3],
                "mt_k": ints[4],
            }

    # Anchor 3: PassThroughES{n}_S{n}_ — bwd_weight (no GemmSpecialization)
    # Ints: MPerBlock, NPerBlock, K0PerBlock, ...
    m = re.search(r"11PassThroughES\d+_S\d+_(Li\d+E(?:Li\d+E)+)", kernel_name)
    if m:
        ints = [int(x) for x in re.findall(r"Li(\d+)E", m.group(1))]
        if len(ints) >= 3:
            return {
                "transpose": (None, None),
                "mt_m": ints[0],
                "mt_n": ints[1],
                "mt_k": ints[2],
            }

    return None


def is_triton_gemm(kernel_name):
    """Check if a kernel name is a Triton GEMM (AITER/vLLM FP8 block-scale, batched)."""
    return "BLOCK_SIZE_M_" in kernel_name and "BLOCK_SIZE_N_" in kernel_name


def parse_triton_gemm(kernel_name):
    m_m = re.search(r"BLOCK_SIZE_M_(\d+)", kernel_name)
    m_n = re.search(r"BLOCK_SIZE_N_(\d+)", kernel_name)
    m_k = re.search(r"BLOCK_SIZE_K_(\d+)", kernel_name)
    mt_m = int(m_m.group(1)) if m_m else None
    mt_n = int(m_n.group(1)) if m_n else None
    mt_k = int(m_k.group(1)) if m_k else None
    return {"transpose": (None, None), "mt_m": mt_m, "mt_n": mt_n, "mt_k": mt_k}


def is_igemm(kernel_name):
    """Check if a kernel name is a MIOpen implicit GEMM convolution kernel."""
    return kernel_name.startswith("igemm_")


def parse_igemm(kernel_name):
    trans_a, trans_b = None, None
    bt = re.search(r"bt(\d+)x(\d+)x(\d+)", kernel_name)
    mt_m = int(bt.group(1)) if bt else None
    mt_n = int(bt.group(2)) if bt else None
    mt_k = int(bt.group(3)) if bt else None
    return {
        "transpose": (trans_a, trans_b),
        "mt_m": mt_m,
        "mt_n": mt_n,
        "mt_k": mt_k,
    }


def is_cublas_xmma_gemm(kernel_name):
    """Check if a kernel name is a cuBLAS XMMA GEMM (sm80/sm90 Hopper TMA/WGMMA path)."""
    return "xmma_gemm_" in kernel_name or "xmma_fprop_implicit_gemm_" in kernel_name


def parse_cublas_xmma_gemm(kernel_name):
    ts = re.search(r"tilesize(\d+)x(\d+)x(\d+)", kernel_name)
    mt_m = int(ts.group(1)) if ts else None
    mt_n = int(ts.group(2)) if ts else None
    mt_k = int(ts.group(3)) if ts else None

    trans_a, trans_b = None, None
    tr = re.search(r"_([tn])([tn])_", kernel_name)
    if tr:
        trans_a = tr.group(1) == "t"
        trans_b = tr.group(2) == "t"

    return {
        "transpose": (trans_a, trans_b),
        "mt_m": mt_m,
        "mt_n": mt_n,
        "mt_k": mt_k,
    }


def is_cutlass_gemm(kernel_name):
    """Check if a kernel name is a CUTLASS GEMM/conv kernel."""
    return "cutlass" in kernel_name and re.search(
        r"gemm|fprop|dgrad|wgrad", kernel_name
    )


def parse_cutlass_gemm(kernel_name):
    m = re.search(
        r"(?:gemm|fprop|dgrad|wgrad)\w*_(\d+)x(\d+)_(\d+)x(\d+)",
        kernel_name,
    )
    if m:
        mt_m = int(m.group(1))
        mt_n = int(m.group(2))
        mt_k = int(m.group(3))
    else:
        mt_m, mt_n, mt_k = None, None, None

    trans_a, trans_b = None, None
    tr = re.search(r"_([tn]{2})_", kernel_name)
    if tr:
        trans_a = tr.group(1)[0] == "t"
        trans_b = tr.group(1)[1] == "t"

    return {
        "transpose": (trans_a, trans_b),
        "mt_m": mt_m,
        "mt_n": mt_n,
        "mt_k": mt_k,
    }


def is_nvjet_gemm(kernel_name):
    """
    Check if a kernel name matches the NVIDIA GEMM naming pattern:
    """
    return kernel_name.startswith("nvjet")


def parse_nvjet_gemm(kernel_name):
    transpose_chars = kernel_name.split("_")[-1]
    transpose = transpose_chars[0] == "T", transpose_chars[1] == "T"
    return {"transpose": transpose}
