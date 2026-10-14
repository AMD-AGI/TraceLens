###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared trace paths and discovery helpers for TraceLens tests."""

from __future__ import annotations

import os

import pytest

_TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACES_ROOT = os.path.join(_TESTS_DIR, "traces")
INFERENCE_ROOT = os.path.join(TRACES_ROOT, "inference")
TESTS_DIR = _TESTS_DIR
ROCprof_FILE = os.path.join(_TESTS_DIR, "rocprof/908_results.json.gz")
NORM_TRACE = os.path.join(
    TRACES_ROOT, "perf_model/normalization/normalization_layer_test.json.gz"
)
RESNET_TRACE = os.path.join(TRACES_ROOT, "mi300/resnet_act_checkpoint.json.gz")
RESNET = RESNET_TRACE
TIMESFORMER1 = os.path.join(
    TRACES_ROOT, "mi300/facebook_timesformer-base-finetuned-k400__1016002.json.gz"
)
TIMESFORMER2 = os.path.join(
    TRACES_ROOT, "h100/facebook_timesformer-base-finetuned-k400__1016002.json.gz"
)
COMPARE_DIR = os.path.join(_TESTS_DIR, "traces/compare_test_ops")
JAX_PB = os.path.join(
    TRACES_ROOT,
    "mi300/jax_conv_minimal_legacy/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb",
)


def _discover_trace_gz_files():
    cases = []
    for root, _dirs, files in os.walk(TRACES_ROOT):
        for name in sorted(files):
            if not name.endswith(".json.gz"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, TESTS_DIR)
            cases.append(pytest.param(path, id=rel.replace(os.sep, "/")))
    return cases


def _discover_inference_cases():
    if not os.path.isdir(INFERENCE_ROOT):
        return []
    cases = []
    for entry in sorted(os.listdir(INFERENCE_ROOT)):
        dirpath = os.path.join(INFERENCE_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        if not gz:
            continue
        cases.append(pytest.param(dirpath, gz[0], id=entry))
    return cases
