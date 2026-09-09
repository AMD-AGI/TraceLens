###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Perf models for the gsplat 3DGS kernels (custom CUDA/HIP in
``submodules/gsplat/gsplat/cuda/csrc``, used by ``GaussianSplatRenderer``).
Core TraceLens has no model for them, so without this they report blank
roofline columns (``has_perf_model = False``).

Shapes come from the function-level autograd op (the kernels carry none):
    _FullyFusedProjectionPacked (projection_ewa_3dgs_packed_fwd)
        Dims[0]=means [N,3], Dims[4]=viewmats [C,4,4]
    _SphericalHarmonics
        Dims[1]=dirs [N,3], Dims[2]=coeffs [N,K,3], Concrete[0]=degrees_to_use
    _RasterizeToPixels
        Dims[0]=means2d [nnz,2], Dims[2]=colors [nnz,CDIM],
        Dims[9]=isect_offsets [C,tile_h,tile_w], Dims[10]=flatten_ids [n_isects],
        Concrete[6:9]=image_width, image_height, tile_size

intersect_tile / intersect_offset appear only as (Synthetic Op) children with no
shapes of their own (their counts live on the sibling _RasterizeToPixels), so
they are classify-only (see OP_CATEGORY_PATTERNS).

Reference: SphericalHarmonicsCUDA.cu, ProjectionEWA3DGSPacked.cu,
RasterizeToPixels3DGSFwd.cu, IntersectTile.cu (submodules/gsplat/.../csrc).
"""

from TraceLens.PerfModel.utils import name2bpe, optional_int, torch_dtype_map


# ---------------------------------------------------------------------------
# small trace-arg helpers
# ---------------------------------------------------------------------------
def _dims(event):
    return event.get("args", {}).get("Input Dims", []) or []


def _types(event):
    return event.get("args", {}).get("Input type", []) or []


def _concrete(event):
    return event.get("args", {}).get("Concrete Inputs", []) or []


def _dim_at(dims, i):
    """Return dims[i] as a tuple, or () if missing/empty."""
    if i < len(dims) and dims[i]:
        return tuple(dims[i])
    return ()


class _GsplatBase:
    """Common scaffolding for gsplat perf models (float32 kernels, vector MAF)."""

    category = "GaussianSplat"
    bwd_category = None
    sheet_category = "GaussianSplat"

    def __init__(self, event, arch=None, python_path=None, **kwargs):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        self.param_details = self.get_param_details(event)
        self.dtype = self.param_details.get("dtype", "float")
        self.bpe = name2bpe(self.dtype) or 4

    def get_maf_type(self):
        return "vector"

    def get_compute_precision(self):
        return torch_dtype_map(self.dtype) if self.dtype else None


class gsplat_spherical_harmonics(_GsplatBase):
    """
    ``_SphericalHarmonics`` (gsplat ``spherical_harmonics_fwd_kernel``).
    One thread per (element, channel); evaluates the SH basis to
    ``degrees_to_use`` and accumulates ``K`` coeffs. Bandwidth-bound; arithmetic
    grows with degree (default sh_degree=0 -> K=1, i.e. a scaled copy).

    FLOPs: f(deg) * N * 3.
    Bytes: dirs [N,3] (deg>=1) + coeffs [N,K,3] + mask [N] (1B) + colors [N,3].
    """

    # Multiply-adds per channel per element by SH degree
    # (sh_coeffs_to_color_fast, SphericalHarmonicsCUDA.cu).
    _FLOP_PER_DEG = {0: 1, 1: 5, 2: 15, 3: 30, 4: 50}

    @staticmethod
    def get_param_details(event):
        dims = _dims(event)
        types = _types(event)
        concrete = _concrete(event)
        dirs = _dim_at(dims, 1)  # [N, 3]
        coeffs = _dim_at(dims, 2)  # [N, K, 3]
        N = dirs[0] if dirs else (coeffs[0] if coeffs else 0)
        K = coeffs[1] if len(coeffs) >= 2 else 1
        # degrees_to_use is Concrete Inputs[0]; fall back to inferring from K.
        deg = None
        if concrete:
            deg = optional_int(str(concrete[0]).strip())
        if deg is None:
            deg = {1: 0, 4: 1, 9: 2, 16: 3, 25: 4}.get(K, 0)
        dtype = types[1] if len(types) > 1 and types[1] else "float"
        return {"N": N, "K": K, "deg": deg, "dtype": dtype}

    def flops(self):
        p = self.param_details
        return self._FLOP_PER_DEG.get(p["deg"], 50) * p["N"] * 3

    def bytes(self):
        p = self.param_details
        N, K = p["N"], p["K"]
        read_elems = (3 * N if p["deg"] >= 1 else 0) + 3 * K * N
        write_elems = 3 * N
        return (read_elems + write_elems) * self.bpe + N  # +N bool mask (1B)


class gsplat_projection_ewa_packed(_GsplatBase):
    """
    ``gsplat::projection_ewa_3dgs_packed_fwd_kernel`` (op
    ``_FullyFusedProjectionPacked``). One thread per (camera, gaussian)
    candidate; fixed per-candidate arithmetic (world->cam transform,
    quat/scale->covariance, EWA persp projection, 2x2 blur+inverse), so FLOPs
    scale with C*N.

    FLOPs: C * N * C_PROJ.
    Bytes: per candidate means[3]+quats[4]+scales[3] read, means2d[2]+depths[1]+
           conics[3] + radii/ids (int) write; viewmats/Ks amortized per camera.
    Note: nnz (survivors) unknown here, so writes cover all C*N -> slight upper bound.
    """

    # Multiply-adds per candidate for the EWA projection pipeline (mat3/mat2
    # products in posW2C + covar + persp_proj + blur + inverse; ~250-350).
    C_PROJ = 256

    @staticmethod
    def get_param_details(event):
        dims = _dims(event)
        types = _types(event)
        means = _dim_at(dims, 0)  # [N, 3]
        viewmats = _dim_at(dims, 4)  # [C, 4, 4]
        N = means[0] if means else 0
        C = viewmats[0] if viewmats else 1
        dtype = types[0] if types and types[0] else "float"
        return {"N": N, "C": C, "dtype": dtype}

    def flops(self):
        p = self.param_details
        return p["C"] * p["N"] * self.C_PROJ

    def bytes(self):
        p = self.param_details
        cand = p["C"] * p["N"]
        bpe = self.bpe
        # per-candidate float reads (means 3, quats 4, scales 3) + writes
        # (means2d 2, depths 1, conics 3)
        float_traffic = cand * (3 + 4 + 3 + 2 + 1 + 3) * bpe
        # per-camera matrices (viewmats 16, Ks 9), read once per camera
        matrix_reads = p["C"] * (16 + 9) * bpe
        # integer outputs: radii [2] int32 + 3 index arrays int64
        int_traffic = cand * (2 * 4 + 3 * 8)
        return float_traffic + matrix_reads + int_traffic


class gsplat_rasterize_to_pixels(_GsplatBase):
    """
    ``_RasterizeToPixels`` (gsplat ``rasterize_to_pixels_3dgs_fwd_kernel``).
    Block per (image, tile), thread per pixel; each pixel walks its tile's
    gaussians front-to-back until transmittance saturates. Per (pixel, gaussian)
    ~= (10 + 2*CDIM) FLOPs (delta, conic sigma, __expf, alpha, T update, color MAF).

    FLOPs: n_isects * tile_size^2 * (10 + 2*CDIM). Upper bound - ignores early-T
           termination (scale by EARLY_OUT_FACTOR for mean-case).
    Bytes: means2d[2]+conics[3]+colors[CDIM]+opacities[1] per gaussian + colors
           [CDIM]+alpha[1] per pixel.
    Compute/latency-bound (byte roofline ~100x below measured), so the FLOP
    ceiling is the meaningful bound.
    """

    # <1.0 models average-case work after early-T termination; 1.0 keeps the
    # reported value an explicit upper bound.
    EARLY_OUT_FACTOR = 1.0

    @staticmethod
    def get_param_details(event):
        dims = _dims(event)
        types = _types(event)
        concrete = _concrete(event)
        means2d = _dim_at(dims, 0)  # [nnz, 2]
        colors = _dim_at(dims, 2)  # [nnz, CDIM]
        isect_offsets = _dim_at(dims, 9)  # [C, tile_h, tile_w]
        flatten_ids = _dim_at(dims, 10)  # [n_isects]

        nnz = means2d[0] if means2d else 0
        cdim = colors[1] if len(colors) >= 2 else 3
        n_isects = flatten_ids[0] if flatten_ids else 0
        C = isect_offsets[0] if len(isect_offsets) >= 1 else 1
        tile_h = isect_offsets[1] if len(isect_offsets) >= 2 else 0
        tile_w = isect_offsets[2] if len(isect_offsets) >= 3 else 0

        # image_width, image_height, tile_size are Concrete Inputs[6:9]
        def _ci(i, default):
            if i < len(concrete):
                v = optional_int(str(concrete[i]).strip())
                if v is not None:
                    return v
            return default

        tile_size = _ci(8, 16)
        width = _ci(6, tile_w * tile_size)
        height = _ci(7, tile_h * tile_size)
        dtype = types[0] if types and types[0] else "float"
        return {
            "nnz": nnz,
            "CDIM": cdim,
            "n_isects": n_isects,
            "C": C,
            "tile_size": tile_size,
            "width": width,
            "height": height,
            "P": C * width * height,
            "dtype": dtype,
        }

    def flops(self):
        p = self.param_details
        per_eval = 10 + 2 * p["CDIM"]
        return int(
            p["n_isects"] * (p["tile_size"] ** 2) * per_eval * self.EARLY_OUT_FACTOR
        )

    def bytes(self):
        p = self.param_details
        read = p["nnz"] * (2 + 3 + p["CDIM"] + 1) * self.bpe
        write = p["P"] * (p["CDIM"] + 1) * self.bpe
        return read + write
