###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-category GPU replay harnesses for validate_perf_model.

Each ``test_<op>`` function builds the input tensors and returns the kernel
call. The shared ``_runner.py`` script dispatches to the right function based
on a ``--op <registry_key>`` flag, passes through dimensional kwargs, and
measures the returned call.
"""
