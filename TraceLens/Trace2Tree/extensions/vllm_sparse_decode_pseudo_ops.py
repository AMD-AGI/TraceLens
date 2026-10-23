###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One pseudo parent for each vLLM ROCm sparse-decode partial/reduce pair."""

from .pseudo_ops_utils import set_bookkeeping_attr

PARTIAL_KERNEL = "_sparse_attn_decode_gfx950_partial_kernel"
REDUCE_KERNEL = "_sparse_attn_decode_reduce_kernel"
TRITON_FN = "_rocm_sparse_attn_decode_triton"
PSEUDO_NAME = "pseudo_vllm_sparse_attn_decode"
BLOCK_K = 32


def create_pseudo_ops_vllm_sparse_decode(trace_tree):
    """Attach partial and reduce kernels to one shape-bearing parent per call."""
    groups = {}
    for evt in trace_tree.events:
        name = evt.get("name") or ""
        if evt.get("cat") != "kernel":
            continue
        if PARTIAL_KERNEL not in name and REDUCE_KERNEL not in name:
            continue
        owner = _python_owner(trace_tree, evt)
        if owner is None:
            continue
        group = groups.setdefault(owner["UID"], {"owner": owner, "kernels": []})
        if evt["UID"] not in {item["UID"] for item in group["kernels"]}:
            group["kernels"].append(evt)
    for group in groups.values():
        _inject_group(trace_tree, group["owner"], group["kernels"])


def _ancestors(trace_tree, evt):
    cur = trace_tree.get_parent_event(evt)
    while cur is not None:
        yield cur
        cur = trace_tree.get_parent_event(cur)


def _python_owner(trace_tree, kernel):
    for anc in _ancestors(trace_tree, kernel):
        if TRITON_FN in (anc.get("name") or ""):
            return anc
    return None


def _find_shapes(trace_tree, owner):
    # Query is recorded on forward_mqa; the partial workspace is inside decode.
    query = partial = None
    nodes = [owner]
    nodes.extend(_ancestors(trace_tree, owner))
    for node in nodes:
        found_q, found_p = _capture_shapes(trace_tree, node)
        if query is None:
            query = found_q
        if partial is None:
            partial = found_p
        if query and partial:
            return query, partial
        if "DeepseekV4DecoderLayer" in (node.get("name") or ""):
            break
    return query, partial


def _walk(trace_tree, root):
    stack = [root]
    while stack:
        evt = stack.pop()
        yield evt
        for uid in evt.get("children") or []:
            child = trace_tree.get_UID2event(uid)
            if child is not None:
                stack.append(child)


def _capture_shapes(trace_tree, root):
    queries = []
    partials = []
    for evt in _walk(trace_tree, root):
        if evt.get("cat") != "cpu_op":
            continue
        args = evt.get("args") or {}
        dims = args.get("Input Dims") or []
        types = args.get("Input type") or []
        for dim, typ in zip(dims, types):
            if not isinstance(dim, (list, tuple)) or len(dim) != 3:
                continue
            if dim[-1] == 512 and "BFloat16" in str(typ):
                queries.append(list(dim))
            elif str(typ) == "float":
                partials.append(list(dim))
    if not queries:
        return None, None
    query = queries[0]
    tokens, heads = query[0], query[1]
    chosen = None
    for shape in partials:
        if shape[0] != tokens:
            continue
        if shape[2] == heads or shape[1] == heads:
            chosen = shape
            break
    return query, chosen


def _inject_group(trace_tree, owner, kernels):
    launchers = []
    seen = set()
    for kernel in kernels:
        launcher = trace_tree.get_parent_event(kernel)
        if launcher is None or launcher["UID"] in seen:
            continue
        if (launcher.get("name") or "") == PSEUDO_NAME:
            return
        parent = trace_tree.get_parent_event(launcher)
        if parent is not None and parent.get("name") == PSEUDO_NAME:
            return
        seen.add(launcher["UID"])
        launchers.append(launcher)
    if not launchers:
        return
    query, partial = _find_shapes(trace_tree, owner)
    dims = [query, partial] if query and partial else None
    pseudo = {
        "ph": "X",
        "name": PSEUDO_NAME,
        "cat": "cpu_op",
        "pid": owner.get("pid"),
        "tid": owner.get("tid"),
        "ts": owner.get("ts"),
        "dur": owner.get("dur"),
        "args": {
            "Input Dims": dims,
            "Input type": ["c10::BFloat16", "float"] if dims else None,
            "Input Strides": None,
            "Concrete Inputs": None,
            "Pseudo op": True,
            "v4_block_k": BLOCK_K,
        },
        "children": [launcher["UID"] for launcher in launchers],
        "gpu_events": [kernel["UID"] for kernel in kernels],
    }
    set_bookkeeping_attr(trace_tree, pseudo)
    for launcher in launchers:
        parent = trace_tree.get_parent_event(launcher)
        if parent is not None and launcher["UID"] in parent.get("children", []):
            parent["children"].remove(launcher["UID"])
        launcher["parent"] = pseudo["UID"]
    children = list(owner.get("children") or [])
    if pseudo["UID"] not in children:
        children.append(pseudo["UID"])
    owner["children"] = children
    pseudo["parent"] = owner["UID"]
