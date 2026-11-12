###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Match ranges for the validator's measured-over-predicted ratios."""

# Constants

RATIO_MATCH = {"flops_ratio": (0.95, 1.05), "bytes_ratio": (0.9, 1.2)}
BAND_COLUMNS = {"flops_ratio": "flops_band", "bytes_ratio": "bytes_band"}


def ratio_band(column, ratio):
    """Return ``match`` or ``deviation`` for ``ratio``, empty when it is None."""
    if ratio is None:
        return ""
    low, high = RATIO_MATCH[column]
    return "match" if low <= ratio <= high else "deviation"
