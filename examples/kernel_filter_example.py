###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Example kernel filter.

Pass this file with ``--extension_file`` to add ``Non-Data-Mov Kernel Time
(µs)`` and ``Non-Data-Mov TFLOPS/s`` columns: each op's busy time without its
copy and transpose kernels. ``keep_kernel(kernel)`` gets the kernel event and
returns True to count it.
"""

DATA_MOVEMENT_PATTERNS = ("at::native::direct_copy_kernel_cuda", "transpose_")


def non_data_mov_filter(kernel):
    return not any(pattern in kernel["name"] for pattern in DATA_MOVEMENT_PATTERNS)


kernel_filters = {"Non-Data-Mov": non_data_mov_filter}
