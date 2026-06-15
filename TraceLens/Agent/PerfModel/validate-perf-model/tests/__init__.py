###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-category GPU test functions for validate_perf_model.

Each ``test_<op>`` function initializes input tensors and invokes the target
kernel with 3 warmup iterations + 1 measured iteration. The shared
``_runner.py`` script dispatches to the right test function based on a
``--op <registry_key>`` flag and passes through dimensional kwargs.

This package replaces the runtime Python-string codegen previously embedded in
``validate_perf_model.py``.
"""
