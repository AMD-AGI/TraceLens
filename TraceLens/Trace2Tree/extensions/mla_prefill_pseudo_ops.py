###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import logging
from .pseudo_ops_utils import inject_pseudo_op_wrap_children

logger = logging.getLogger(__name__)

MLA_PREFILL_FWD_PATTERN = re.compile(r":\s*mla_fp8_prefill_attn(\b|$)")
PREFILL_CPU_OP_NAME = "aiter::mla_prefill_ps_asm_fwd"


def create_pseudo_ops_mla_prefill(trace_tree):
    """
    Create pseudo ops for MLA fp8 prefill attention operations.

    Finds python_function events whose name ends with 'mla_fp8_prefill_attn',
    verifies they have a descendant cpu_op 'aiter::mla_prefill_ps_asm_fwd' with
    GPU kernels, then injects 'pseudo_mla_prefill_fwd' wrapping all children.
    Shapes are inherited from the cpu_op event.
    """

    prefill_python_funcs = _find_mla_prefill_python_funcs(trace_tree)

    if not prefill_python_funcs:
        logger.warning(
            "No python_function events matching mla_fp8_prefill_attn pattern found"
        )
        return

    logger.info(f"Processing {len(prefill_python_funcs)} MLA prefill operations")

    for py_func_evt in prefill_python_funcs:
        _create_pseudo_op_mla_prefill(trace_tree, py_func_evt)


def _find_mla_prefill_python_funcs(trace_tree):
    """Find all python_function events matching the mla_fp8_prefill_attn pattern."""

    matched = []
    for name, uids in trace_tree.name2event_uids.items():
        if MLA_PREFILL_FWD_PATTERN.search(name):
            for uid in uids:
                evt = trace_tree.get_UID2event(uid)
                if evt.get("cat") == "python_function":
                    matched.append(evt)
    return matched


def _find_prefill_cpu_op_child(trace_tree, parent_evt):
    """
    Search descendants of parent_evt for 'aiter::mla_prefill_ps_asm_fwd'.
    Returns the first matching event, or None.
    """

    for child_evt in trace_tree.get_children_events(parent_evt):
        if child_evt.get("name") == PREFILL_CPU_OP_NAME:
            return child_evt
        found = _find_prefill_cpu_op_child(trace_tree, child_evt)
        if found is not None:
            return found
    return None


def _create_pseudo_op_mla_prefill(trace_tree, py_func_evt):
    """Create a single pseudo op for one MLA prefill python_function event."""

    donor_evt = _find_prefill_cpu_op_child(trace_tree, py_func_evt)

    if donor_evt is None:
        logger.warning(
            f"No {PREFILL_CPU_OP_NAME} child found for "
            f"'{py_func_evt['name']}' (UID {py_func_evt['UID']}), skipping"
        )
        return

    gpu_event_ids = py_func_evt.get("gpu_events", [])
    if not gpu_event_ids:
        logger.warning(
            f"No GPU events for MLA prefill python_function UID {py_func_evt['UID']}"
        )
        return

    inject_pseudo_op_wrap_children(
        trace_tree,
        py_func_evt,
        "pseudo_mla_prefill_fwd",
        shape_donor_evt=donor_evt,
    )
