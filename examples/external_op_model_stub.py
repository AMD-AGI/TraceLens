###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Placeholder external op model.

Pass this file with ``--extension_file`` to see ``External`` columns. The
time is ``2*M*N*K*B`` at a fixed 100 TFLOP/s, so it is not a device model.

``params`` holds the same values as the report's ``param:`` columns. Any key
may be missing; ``transpose``, for example, is only there when the GEMM's
kernel name parses. Return None, a time in µs, or a dict whose ``time_us``
is optional and whose other scalar values become ``External <name>``
columns.

A real model belongs in an extension file outside this repository that
defines ``external_op_model`` the same way and imports its library itself.
"""


def external_op_model(category, params, arch):
    if category != "GEMM":
        return None
    flops = 2 * params["M"] * params["N"] * params["K"] * (params.get("B") or 1)
    return {
        "time_us": flops / 100e12 * 1e6,
        "Transpose": str(params.get("transpose", "unknown")),
    }
