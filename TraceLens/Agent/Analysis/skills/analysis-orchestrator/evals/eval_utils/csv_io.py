###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared CSV writer for the eval scripts."""

import csv


def write_csv(path, rows, fieldnames, verbose=False):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if verbose:
        print("  Written: {} ({} rows)".format(path, len(rows)))
