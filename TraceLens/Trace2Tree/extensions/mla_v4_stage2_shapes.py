###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Copy ``mla_decode_v4_asm`` tensor args onto the ``_fwd_kernel_stage2_asm`` launch."""

_STAGE2_KERNEL = "_fwd_kernel_stage2_asm"
_DONOR_OP = "aiter::mla_decode_v4_asm"
_PARENT_PY = "mla_decode_fwd_v4_nm"
_ARG_KEYS = ("Input Dims", "Input type", "Input Strides", "Concrete Inputs")


def copy_mla_v4_stage2_shapes(trace_tree):
    for uid in list(trace_tree.name2event_uids.get(_STAGE2_KERNEL, [])):
        kernel = trace_tree.get_UID2event(uid)
        if kernel.get("cat") != "kernel":
            continue
        launcher = trace_tree.get_parent_event(kernel)
        if launcher is None:
            continue
        donor = _find_donor(trace_tree, launcher)
        if donor is None:
            continue
        src = donor.get("args") or {}
        dst = launcher.setdefault("args", {})
        for key in _ARG_KEYS:
            if src.get(key) and not dst.get(key):
                dst[key] = src[key]


def _find_donor(trace_tree, start):
    current = start
    seen = set()
    while current is not None and current.get("UID") not in seen:
        seen.add(current.get("UID"))
        if _PARENT_PY in current.get("name", ""):
            return _find_named_descendant(trace_tree, current, _DONOR_OP)
        current = trace_tree.get_parent_event(current)
    return None


def _find_named_descendant(trace_tree, event, name):
    found = []

    def walk(node):
        for child_uid in node.get("children") or []:
            child = trace_tree.get_UID2event(child_uid)
            if child is None:
                continue
            if child.get("name") == name and child.get("cat") == "cpu_op":
                found.append(child)
            walk(child)

    walk(event)
    if len(found) == 1:
        return found[0]
    return None
