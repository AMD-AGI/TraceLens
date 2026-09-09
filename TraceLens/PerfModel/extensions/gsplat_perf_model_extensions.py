###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Theoretical performance models for the gsplat 3D Gaussian Splatting (3DGS)
rasterization backbone used by ``src/models/models/rasterization.py``
(``GaussianSplatRenderer`` / ``Rasterizer``).

These kernels are custom CUDA/HIP kernels shipped in
``submodules/gsplat/gsplat/cuda/csrc`` and are *not* covered by any of the
core TraceLens perf models, so before this extension they showed up in the
report with empty ``GFLOPS`` / ``Data Moved (MB)`` / ``TB/s`` / ``TFLOPS/s`` /
``Compute Spec`` columns (``has_perf_model = False``).

Parameters are sourced from the *function-level* autograd op that carries the
tensor shapes in the trace (``Input Dims`` / ``Concrete Inputs``):

    _FullyFusedProjectionPacked -> gsplat::projection_ewa_3dgs_packed_fwd_kernel
        Input Dims[0] = means [N, 3], Input Dims[4] = viewmats [C, 4, 4]
    _SphericalHarmonics
        Input Dims[1] = dirs [N, 3], Input Dims[2] = coeffs [N, K, 3]
        Concrete Inputs[0] = degrees_to_use
    _RasterizeToPixels
        Input Dims[0] = means2d [nnz, 2], Input Dims[2] = colors [nnz, CDIM]
        Input Dims[9] = isect_offsets [C, tile_h, tile_w]
        Input Dims[10] = flatten_ids [n_isects]
        Concrete Inputs[6:9] = image_width, image_height, tile_size

The two tile-intersection kernels (``intersect_tile`` / ``intersect_offset``)
surface only as ``... (Synthetic Op)`` children with no ``Input Dims`` of their
own, so their FLOPs/bytes depend on ``nnz`` / ``n_isects`` that live on the
sibling ``_RasterizeToPixels`` op. They are therefore classified only (see the
regex in ``torch_op_mapping.OP_CATEGORY_PATTERNS``); upgrading them to a full
roofline requires tree-level sibling access.

Reference kernels:
    submodules/gsplat/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu
    submodules/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu
    submodules/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu
    submodules/gsplat/gsplat/cuda/csrc/IntersectTile.cu
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
    Perf model for ``_SphericalHarmonics`` (gsplat ``spherical_harmonics_fwd_kernel``).

    One thread per ``N * 3`` (elem, color-channel). For each element it evaluates
    the SH basis up to ``degrees_to_use`` and accumulates ``K`` coefficients into
    the output color. Pure bandwidth-bound; arithmetic grows with SH degree.

    FLOPs  : f(deg) * N * 3, where f is the number of multiply-adds per channel
             for the basis evaluation (deg 0 is a single scaled coeff).
    Bytes  : read dirs [N,3] (deg>=1 only) + coeffs [N,K,3] + mask [N] (1B) +
             write colors [N,3], scaled by bytes-per-element.

    In ``GaussianSplatRenderer`` the default is ``sh_degree = 0`` (K = 1), so this
    is effectively a scaled copy; the degree table future-proofs higher orders.
    """

    # Approx FLOPs per channel per element for each SH degree (count of pSH terms
    # / multiply-adds in sh_coeffs_to_color_fast, SphericalHarmonicsCUDA.cu).
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
    Perf model for ``gsplat::projection_ewa_3dgs_packed_fwd_kernel``
    (autograd op ``_FullyFusedProjectionPacked``).

    One thread per ``C * N`` (camera, gaussian) candidate. Each valid candidate:
    world->camera transform (posW2C), quat/scale->covariance
    (quat_scale_to_covar_preci), covariance world->camera (covarW2C), perspective
    projection with the EWA Jacobian (persp_proj), 2x2 blur + inverse (add_blur /
    glm::inverse). This is a fixed, shape-independent amount of arithmetic per
    candidate, so FLOPs scale with the candidate count.

    FLOPs  : C * N * C_PROJ   (C_PROJ ~ per-gaussian projection arithmetic)
    Bytes  : reads means[3]+quats[4]+scales[3] per candidate (+viewmats/Ks
             amortized per camera) + writes means2d[2]+depths[1]+conics[3] and
             radii (2 x int32) + (batch,cam,gaussian)_ids (3 x int64).

    Notes:
      * ``nnz`` (surviving projections) is not on this event, so writes are
        estimated over all ``C * N`` candidates -> a slight upper bound.
    """

    # Multiply-adds per candidate for the EWA projection pipeline. Derived by
    # counting the mat3/mat2 products in posW2C + quat_scale_to_covar_preci +
    # covarW2C + persp_proj + add_blur + 2x2 inverse (order ~250-350).
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
    Perf model for ``_RasterizeToPixels`` (gsplat ``rasterize_to_pixels_3dgs_fwd_kernel``).

    Block per (image, tile), thread per pixel. Each pixel walks the gaussians
    that intersect its tile (front-to-back) accumulating color until the
    transmittance ``T`` saturates. Per (pixel, in-tile gaussian) evaluation:
    delta (2 sub), sigma via conic (3 mul + 2 add), __expf, alpha (mul/min),
    T update, vis, and CDIM colour multiply-adds ~= (10 + 2*CDIM) FLOPs.

    FLOPs  : n_isects * tile_size^2 * (10 + 2*CDIM)   [upper bound: ignores the
             early-T termination, so treat as a roofline ceiling; scale by
             ``EARLY_OUT_FACTOR`` for a mean-case estimate].
    Bytes  : read means2d[2]+conics[3]+colors[CDIM]+opacities[1] per gaussian
             (loaded once to shared memory) + write colors[CDIM]+alpha[1] per
             pixel.

    This kernel is strongly compute/latency-bound (the byte roofline is ~2 orders
    of magnitude below the measured throughput), so the FLOP ceiling is the
    meaningful bound.
    """

    # Set < 1.0 to model average-case work after early-T termination. Left at 1.0
    # so the reported number is an explicit upper bound.
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
