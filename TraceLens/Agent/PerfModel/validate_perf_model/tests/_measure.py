###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Measurement protocol shared by every ``test_<key>``."""

import torch

# Constants
# Larger than the L2 and on-package cache of the supported GPUs.
L2_FLUSH_BYTES = 512 * 1024 * 1024


def prepare_measured_call():
    """Flush L2 and launch a marker kernel; call right before the measured call.

    The flush makes the measured call start from a cold cache, so its read
    bytes are not lowered by data the warmup calls left in L2. The
    ``bitwise_not`` marker separates the warmup dispatches from the measured
    ones in the rocprofv3 trace. Both are ``at::native`` kernels, which the
    validator treats as infrastructure and leaves out of the results.
    """
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    flush.fill_(1)
    marker = torch.zeros(1, dtype=torch.int32, device="cuda")
    marker.bitwise_not_()
    torch.cuda.synchronize()
