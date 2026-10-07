###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Placeholder external perf model.

Pass this file with ``--extension_file`` to see ``External Time (µs)``
columns. The time is ``2*M*N*K*B`` at a fixed 100 TFLOP/s, so it is not a
device model.

A real model belongs in an extension file outside this repository that
defines ``external_perf_model`` the same way and imports its library
itself.
"""


def external_perf_model(category, params, arch):
    if category != "GEMM":
        return None
    flops = 2 * params["M"] * params["N"] * params["K"] * (params.get("B") or 1)
    return flops / 100e12 * 1e6
