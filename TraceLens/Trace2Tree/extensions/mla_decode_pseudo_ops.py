###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import logging
from .pseudo_ops_utils import inject_pseudo_op_wrap_children

logger = logging.getLogger(__name__)

MLA_DECODE_FWD_PATTERN = re.compile(r"aiter/mla.py\(\d+\): mla_decode_fwd")
STAGE1_KERNEL_NAME = "aiter::mla_decode_stage1_asm_fwd"


def create_pseudo_ops_mla_decode(trace_tree):
    """
    Create pseudo ops for MLA decode forward operations.

    Finds python_function events matching 'aiter/mla.py<line>: mla_decode_fwd',
    verifies they have a child op 'aiter::mla_decode_stage1_asm_fwd' with GPU
    kernels, then injects 'pseudo_mla_decode_fwd' wrapping all children.
    Shapes are inherited from the stage1 event.
    """

    mla_python_funcs = _find_mla_decode_python_funcs(trace_tree)

    if not mla_python_funcs:
        logger.warning(
            "No python_function events matching mla_decode_fwd pattern found"
        )
        return

    logger.info(f"Processing {len(mla_python_funcs)} MLA decode operations")

    for py_func_evt in mla_python_funcs:
        _create_pseudo_op_mla_decode(trace_tree, py_func_evt)


def _find_mla_decode_python_funcs(trace_tree):
    """Find all python_function events matching the mla_decode_fwd pattern."""

    matched = []
    for name, uids in trace_tree.name2event_uids.items():
        if MLA_DECODE_FWD_PATTERN.search(name):
            for uid in uids:
                evt = trace_tree.get_UID2event(uid)
                if evt.get("cat") == "python_function":
                    matched.append(evt)
    return matched


def _find_stage1_child(trace_tree, parent_evt):
    """
    Search descendants of parent_evt for 'aiter::mla_decode_stage1_asm_fwd'.
    Returns the first matching event, or None.
    """

    for child_evt in trace_tree.get_children_events(parent_evt):
        if child_evt.get("name") == STAGE1_KERNEL_NAME:
            return child_evt
        found = _find_stage1_child(trace_tree, child_evt)
        if found is not None:
            return found
    return None


def _create_pseudo_op_mla_decode(trace_tree, py_func_evt):
    """Create a single pseudo op for one MLA decode python_function event."""

    stage1_evt = _find_stage1_child(trace_tree, py_func_evt)

    if stage1_evt is None:
        logger.warning(
            f"No {STAGE1_KERNEL_NAME} child found for "
            f"'{py_func_evt['name']}' (UID {py_func_evt['UID']}), skipping"
        )
        return

    gpu_event_ids = py_func_evt.get("gpu_events", [])
    if not gpu_event_ids:
        logger.warning(
            f"No GPU events for MLA decode python_function UID {py_func_evt['UID']}"
        )
        return

    inject_pseudo_op_wrap_children(
        trace_tree,
        py_func_evt,
        "pseudo_mla_decode_fwd",
        shape_donor_evt=stage1_evt,
    )
