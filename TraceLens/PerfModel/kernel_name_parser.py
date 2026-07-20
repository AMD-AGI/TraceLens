###############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re

import itanium_demangler


def _demangle_ck(kernel_name):
    """
    Demangle a mangled CK kernel name using itanium_demangler.
    Returns the normalized demangled string (with (int)/(bool)/enum qualifiers
    stripped so bare integers remain), or None if parsing fails.
    """
    try:
        ast = itanium_demangler.parse(kernel_name)
        if ast is None:
            return None
        demangled = str(ast)
        # Strip (int) and (bool) type qualifiers from integer/bool literals
        demangled = re.sub(r"\(int\)\s*", "", demangled)
        demangled = re.sub(r"\(bool\)\s*", "", demangled)
        # Normalize enum casts: (ck::SomeEnum)7 -> SomeEnum)7
        # This preserves the EnumName)value pattern that parse_ck_gemm uses as an anchor.
        demangled = re.sub(r"\([^)]*::(\w+)\)", r"\1)", demangled)
        return demangled
    except Exception:
        return None


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


def _parse_gridwise_params(kernel_name, class_name):
    """
    Parse the top-level template parameters of a CK Gridwise class by name.
    Returns a list of parameter strings, or [] if the class is not found.
    """
    idx = kernel_name.find(class_name + "<")
    if idx < 0:
        return []
    start = idx + len(class_name) + 1
    depth = 1
    params = []
    current = []
    i = start
    while i < len(kernel_name) and depth > 0:
        c = kernel_name[i]
        if c == "<":
            depth += 1
            current.append(c)
        elif c == ">":
            depth -= 1
            if depth == 0:
                params.append("".join(current).strip())
            else:
                current.append(c)
        elif c == "," and depth == 1:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(c)
        i += 1
    return params


def parse_ck_gemm(kernel_name):
    """
    Parse tile sizes from CK kernel names (demangled form).
    For mangled names, demangle first via _demangle_ck then recurse.

    Absolute template parameter positions are verified from CK source headers and
    doxygen (docs-7.0.2, docs-7.1.0). All positions are 0-indexed top-level params
    of the Gridwise class template.
    """

    # ── Mangled CK kernels: demangle first, then recurse ──
    if kernel_name.startswith("_ZN2ck"):
        demangled = _demangle_ck(kernel_name)
        if demangled is not None:
            return parse_ck_gemm(demangled)
        return None

    # ── Demangled CK kernels: dispatch on Gridwise class name ──

    # Map of Gridwise class substring → (M_pos, N_pos, K_pos, trans_A_pos, trans_B_pos)
    # Positions are absolute indices in the top-level template parameter list.
    # trans positions are indices of the RowMajor/ColumnMajor layout type params
    # (None = not extractable for this class).
    #
    # Verified sources:
    #   GridwiseMoeGemmMX*:          docs-7.0.2/globals_defs_d.html
    #   GridwiseGemmMultiD_ABScale:  gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp
    #   GridwiseGemmMultiD_blockscale: same family as ABScale
    #   GridwiseMoeGemmBlockScale:   gridwise_moe_gemm_blockscale.hpp
    #   GridwiseGemmMultiD_xdl:      gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp
    #   GridwiseMoeGemm (no scale):  gridwise_moe_gemm.hpp
    #   GridwiseGemm_xdl_cshuffle_v3: observed in traces, layout consistent with above
    #   GridwiseGemmMultipleD_xdl_cshuffle: observed in traces
    #   GridwiseGemm_bk0mk1:         docs-7.1.0 struct reference
    #
    # Layout params (RowMajor/ColumnMajor) are the first two type params for
    # classes that have them; None for classes where layout is not type-parameterized.

    GRIDWISE_CONFIG = {
        # class substring: (M_pos, N_pos, K_pos, has_layout)
        # MoeGemmMX family: [0..15]=layouts/types/ops, [16]=ScaleBlockSize,
        #   [17]=BlockSize, [18]=M, [19]=N, [20]=K
        # Checked first — name contains "MulABScale" which would match ABScale below
        "MoeGemmMX":                     (18, 19, 20, True),
        # ABScale/blockscale/BlockScale: [0..1]=layouts, [14]=BlockSize,
        #   [15]=ScaleBlockM, [16]=ScaleBlockN, [17]=ScaleBlockK, [18]=M, [19]=N, [20]=K
        "GridwiseGemmMultiD_ABScale":    (18, 19, 20, True),
        "GridwiseGemmMultiD_blockscale": (18, 19, 20, True),
        "MoeGemmBlockScale":             (18, 19, 20, True),
        # b_preshuffle variant: GemmSpec + BlockSize both explicit before M/N/K,
        #   so [13]=GemmSpec, [14]=BlockSize, [15]=M, [16]=N, [17]=K
        # Must come before "GemmMultiD_xdl" to avoid wrong 13/14/15 match.
        "GridwiseGemmMultiD_xdl_cshuffle_v3_b_preshuffle": (15, 16, 17, True),
        # No-scale families: [0..2]=layouts (or data types), [12]=BlockSize, [13]=M, [14]=N, [15]=K
        "GemmMultiD_xdl":                (13, 14, 15, True),
        "GridwiseMoeGemm":               (13, 14, 15, True),
        "GridwiseGemm_xdl_cshuffle_v3":  (13, 14, 15, True),
        # GridwiseGemmMultipleD_xdl_cshuffle: [10]=InMemoryDataOp, [11]=prefetch,
        #   [12]=BlockSize, [13]=M, [14]=N, [15]=K; no layout type params (uses data types)
        "GridwiseGemmMultipleD_xdl_cshuffle": (13, 14, 15, False),
        # bwd_weight: [0]=BlockSize (int), [1-4]=data types, [5]=InMemoryDataOp,
        #   [6-8]=TensorDescriptors, [9-11]=PassThrough x3, [12]=M, [13]=N, [14]=K
        "GridwiseGemm_bk0mk1":           (12, 13, 14, False),
    }

    for class_substr, (m_pos, n_pos, k_pos, has_layout) in GRIDWISE_CONFIG.items():
        if class_substr not in kernel_name:
            continue

        # Find the actual class name (longest match wins to avoid prefix collisions)
        cls_match = re.search(r"(Gridwise\w*" + re.escape(class_substr.lstrip("Gridwise")) + r"\w*)<",
                              kernel_name)
        if not cls_match:
            # Fallback: search directly for the substring followed by <
            m = re.search(re.escape(class_substr) + r"(\w*)<", kernel_name)
            if not m:
                continue
            actual_class = class_substr + m.group(1)
        else:
            actual_class = cls_match.group(0).rstrip("<")

        params = _parse_gridwise_params(kernel_name, actual_class)
        if len(params) <= max(m_pos, n_pos, k_pos):
            continue

        def to_int(p):
            p = p.strip()
            m = re.match(r"^-?\d+$", p)
            return int(p) if m else None

        mt_m = to_int(params[m_pos])
        mt_n = to_int(params[n_pos])
        mt_k = to_int(params[k_pos])
        if mt_m is None or mt_n is None or mt_k is None:
            continue

        # A and B layouts: first two RowMajor/ColumnMajor type params in the name.
        # RowMajor = not transposed, ColumnMajor = transposed.
        # https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
        if has_layout:
            layouts = re.findall(r"(RowMajor|ColumnMajor)", kernel_name)
            trans_a = layouts[0] == "ColumnMajor" if len(layouts) >= 2 else None
            trans_b = layouts[1] == "ColumnMajor" if len(layouts) >= 2 else None
        else:
            trans_a, trans_b = None, None

        return {
            "transpose": (trans_a, trans_b),
            "mt_m": mt_m,
            "mt_n": mt_n,
            "mt_k": mt_k,
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
