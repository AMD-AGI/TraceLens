###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the gsplat 3DGS perf model extensions.

Shapes are taken from a real inference trace of the HunyuanWorld-Mirror
GaussianSplatRenderer (src/models/models/rasterization.py):

    _RasterizeToPixels        nnz=20_879_001, CDIM=4, C=40, tiles 19x33,
                              n_isects=41_471_966, W=518, H=294, tile=16
    _SphericalHarmonics       N=21_379_014, K=1, degree=0
    _FullyFusedProjectionPacked  N=718_850 gaussians, C=40 cameras
"""

import pytest

from TraceLens.PerfModel.extensions.gsplat_perf_model_extensions import (
    gsplat_projection_ewa_packed,
    gsplat_rasterize_to_pixels,
    gsplat_spherical_harmonics,
)
from TraceLens.PerfModel.torch_op_mapping import (
    categorize_torch_op,
    resolve_perf_model_class,
)


def _event(name, dims, types, concrete, strides=None):
    args = {
        "Input Dims": dims,
        "Input type": types,
        "Concrete Inputs": concrete,
    }
    if strides is not None:
        args["Input Strides"] = strides
    return {"name": name, "args": args}


RASTERIZE_EVENT = _event(
    "_RasterizeToPixels",
    [
        [20879001, 2],  # means2d
        [20879001, 3],  # conics
        [20879001, 4],  # colors  -> CDIM=4
        [20879001],  # opacities
        [], [], [], [], [],
        [40, 19, 33],  # isect_offsets [C, tile_h, tile_w]
        [41471966],  # flatten_ids [n_isects]
        [],
    ],
    ["float", "float", "float", "float", "", "", "Scalar", "Scalar", "Scalar",
     "int", "int", "Scalar"],
    ["", "", "", "", "", "", "518", "294", "16", "", "", "False"],
)

SH_EVENT = _event(
    "_SphericalHarmonics",
    [[], [21379014, 3], [21379014, 1, 3], [21379014]],
    ["Scalar", "float", "float", "bool"],
    ["0", "", "", ""],
)

PROJ_NAME = (
    "_FullyFusedProjectionPacked->void "
    "gsplat::projection_ewa_3dgs_packed_fwd_kernel<float>(...) (Synthetic Op)"
)
PROJ_EVENT = _event(
    PROJ_NAME,
    [[718850, 3], [], [718850, 4], [718850, 3], [40, 4, 4], [40, 3, 3],
     [], [], [], [], [], [], [], [], [718850]],
    ["float", "", "float", "float", "float", "float"] + ["Scalar"] * 8 + ["float"],
    ["", "", "", "", "", "", "518", "294", "0.3", "0.01", "1e10", "0.",
     "False", "False", ""],
)

INTERSECT_TILE = (
    "cudaLaunchKernel->void gsplat::intersect_tile_kernel<float>(...) (Synthetic Op)"
)
INTERSECT_OFFSET = (
    "cudaLaunchKernel->gsplat::intersect_offset_kernel(...) (Synthetic Op)"
)


class TestGsplatResolution:
    def test_rasterize_resolves(self):
        assert resolve_perf_model_class("_RasterizeToPixels") is gsplat_rasterize_to_pixels

    def test_sh_resolves(self):
        assert resolve_perf_model_class("_SphericalHarmonics") is gsplat_spherical_harmonics

    def test_projection_resolves_via_matcher(self):
        assert resolve_perf_model_class(PROJ_NAME) is gsplat_projection_ewa_packed

    @pytest.mark.parametrize("name", [INTERSECT_TILE, INTERSECT_OFFSET])
    def test_intersect_is_category_only(self, name):
        # No perf model, but still classified (not "other").
        assert resolve_perf_model_class(name) is None

    @pytest.mark.parametrize(
        "name",
        [
            "_RasterizeToPixels",
            "_SphericalHarmonics",
            PROJ_NAME,
            INTERSECT_TILE,
            INTERSECT_OFFSET,
        ],
    )
    def test_all_categorize_as_gaussiansplat(self, name):
        assert categorize_torch_op({"name": name, "args": {}}) == "GaussianSplat"


class TestSphericalHarmonics:
    def test_degree0_is_bandwidth_bound(self):
        m = gsplat_spherical_harmonics(SH_EVENT)
        N = 21379014
        # degree 0 -> 1 madd/channel; K=1
        assert m.flops() == 1 * N * 3
        # read coeffs 3*K*N + write 3N (no dirs at deg0), *bpe(4) + mask N
        assert m.bytes() == (3 * 1 * N + 3 * N) * 4 + N
        assert m.get_compute_precision() == "fp32"
        assert m.get_maf_type() == "vector"


class TestProjection:
    def test_flops_scale_with_candidates(self):
        m = gsplat_projection_ewa_packed(PROJ_EVENT)
        N, C = 718850, 40
        assert m.flops() == C * N * gsplat_projection_ewa_packed.C_PROJ
        assert m.bytes() > 0
        assert m.get_compute_precision() == "fp32"


class TestRasterize:
    def test_compute_bound_flops_and_bytes(self):
        m = gsplat_rasterize_to_pixels(RASTERIZE_EVENT)
        n_isects, cdim, tile = 41471966, 4, 16
        assert m.flops() == n_isects * tile * tile * (10 + 2 * cdim)
        # read nnz*(2+3+CDIM+1)*bpe + write C*W*H*(CDIM+1)*bpe
        nnz, C, W, H = 20879001, 40, 518, 294
        expected = nnz * (2 + 3 + cdim + 1) * 4 + C * W * H * (cdim + 1) * 4
        assert m.bytes() == expected

    def test_flops_per_byte_is_high(self):
        m = gsplat_rasterize_to_pixels(RASTERIZE_EVENT)
        # Rasterization is compute-bound: FLOPs/Byte should be >> 1.
        assert m.flops() / m.bytes() > 50


# Bilinear upsample event (from the DPT dense heads):
#   input [6, 256, 11, 19] bf16 -> output spatial [21, 37]
BILINEAR_EVENT = _event(
    "aten::upsample_bilinear2d",
    [[6, 256, 11, 19], [], [], []],
    ["c10::BFloat16", "ScalarList", "Scalar", ""],
    ["", "[21, 37]", "True", ""],
    strides=[[53504, 1, 4864, 256], [], [], []],
)


class TestUpsampleBilinear:
    def test_resolves(self):
        from TraceLens.PerfModel.perf_model import aten_upsample_bilinear

        assert resolve_perf_model_class("aten::upsample_bilinear2d") is aten_upsample_bilinear

    def test_output_shape_and_roofline(self):
        cls = resolve_perf_model_class("aten::upsample_bilinear2d")
        m = cls(BILINEAR_EVENT)
        n_in = 6 * 256 * 11 * 19
        n_out = 6 * 256 * 21 * 37
        bpe = 2  # bf16
        assert m.flops() == 11 * n_out
        assert m.bytes() == n_in * bpe + n_out * bpe
        assert m.get_compute_precision() == "bf16"
        # bandwidth-bound: FLOPs/Byte should be small (order ~a few).
        assert m.flops() / m.bytes() < 10

    def test_categorized_as_elementwise(self):
        assert categorize_torch_op({"name": "aten::upsample_bilinear2d", "args": {}}) == "elementwise"
