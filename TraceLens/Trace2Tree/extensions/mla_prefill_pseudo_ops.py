###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
import logging
from collections import defaultdict

from .pseudo_ops_utils import inject_pseudo_op_wrap_children

logger = logging.getLogger(__name__)

MLA_PREFILL_FWD_PATTERN = re.compile(r":\s*mla_fp8_prefill_attn(\b|$)")
PREFILL_CPU_OP_NAME = "aiter::mla_prefill_ps_asm_fwd"


def _is_mla_prefill_python(name: str) -> bool:
    return "mla_fp8_prefill_attn" in name and bool(
        MLA_PREFILL_FWD_PATTERN.search(name)
    )


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

    prefill_index = _build_mla_prefill_cpu_op_index(trace_tree)

    cpu_roots_acc = set(trace_tree.cpu_root_nodes)

    for py_func_evt in prefill_python_funcs:
        _create_pseudo_op_mla_prefill(
            trace_tree, py_func_evt, prefill_index, cpu_roots_acc
        )

    trace_tree.cpu_root_nodes[:] = sorted(
        cpu_roots_acc,
        key=lambda u: (trace_tree.get_UID2event(u).get("ts", 0), u),
    )


def _find_mla_prefill_python_funcs(trace_tree):
    """Find all python_function events matching the mla_fp8_prefill_attn pattern."""

    matched = []
    for name, uids in trace_tree.name2event_uids.items():
        if "mla_fp8_prefill_attn" not in name:
            continue
        if not MLA_PREFILL_FWD_PATTERN.search(name):
            continue
        for uid in uids:
            evt = trace_tree.get_UID2event(uid)
            if evt.get("cat") == "python_function":
                matched.append(evt)
    return matched


def _build_mla_prefill_cpu_op_index(trace_tree):
    """Map mla_fp8_prefill_attn python_function UID → prefill cpu_op events.

    Walk matches legacy ``_find_prefill_cpu_op_child`` (starts from the donor
    cpu_op event, then ancestors).
    """

    index = defaultdict(list)
    candidates = trace_tree.name2event_uids.get(PREFILL_CPU_OP_NAME, [])
    for uid in candidates:
        evt = trace_tree.get_UID2event(uid)
        cur = evt
        while cur is not None:
            if cur.get("cat") == "python_function" and _is_mla_prefill_python(
                cur.get("name", "")
            ):
                index[cur["UID"]].append(evt)
            cur = trace_tree.get_parent_event(cur)
    return index


def _find_prefill_cpu_op_child(parent_evt, prefill_index):
    """Earliest-by-ts prefill cpu_op under *parent_evt* (see decode index)."""

    py_uid = parent_evt["UID"]
    under = prefill_index.get(py_uid)
    if not under:
        return None
    return min(under, key=lambda e: e.get("ts", 0))


def _create_pseudo_op_mla_prefill(
    trace_tree, py_func_evt, prefill_index, cpu_roots_acc
):
    """Create a single pseudo op for one MLA prefill python_function event."""

    donor_evt = _find_prefill_cpu_op_child(py_func_evt, prefill_index)

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
        cpu_roots_acc=cpu_roots_acc,
    )
