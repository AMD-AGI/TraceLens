###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Match ranges for the validator's measured-over-predicted ratios."""

# Constants

# A model must never predict more than the hardware measured, so each range
# starts at 1.0; the upper end allows for kernel artifacts.
RATIO_MATCH = {"flops_ratio": (1.0, 1.05), "bytes_ratio": (1.0, 1.2)}
BAND_COLUMNS = {"flops_ratio": "flops_band", "bytes_ratio": "bytes_band"}
# Float rounding allowance below the 1.0 floor.
RATIO_EPS = 1e-6
MATCH = "match"
OVER_PREDICTED = "over-predicted"
DEVIATION = "deviation"
DEVIATION_SUSPECT = "deviation-suspect"
DEVIATION_EXPLAINED = "deviation-explained"
NO_PREDICTION = "no prediction"


def ratio_band(column, ratio):
    """Return the band of ``ratio``, empty when it is None.

    ``match`` inside the range, ``deviation`` above it, and ``over-predicted``
    below it (the model predicts more than was measured).
    """
    if ratio is None:
        return ""
    low, high = RATIO_MATCH[column]
    if ratio < low - RATIO_EPS:
        return OVER_PREDICTED
    return MATCH if ratio <= high else DEVIATION


def match_status(run_status, bands):
    """Combine a row's bands into one match status.

    Empty unless the run status is ``OK``; ``no prediction`` when a ratio is
    missing; ``over-predicted`` when any ratio is below its range;
    ``deviation-suspect`` when any is above it; otherwise ``match``. Only the
    investigating agent turns ``deviation-suspect`` into
    ``deviation-explained``.
    """
    if run_status != "OK":
        return ""
    if "" in bands:
        return NO_PREDICTION
    if OVER_PREDICTED in bands:
        return OVER_PREDICTED
    if DEVIATION in bands:
        return DEVIATION_SUSPECT
    return MATCH
