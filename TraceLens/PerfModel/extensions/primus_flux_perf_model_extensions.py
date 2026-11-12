###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance models for Primus Flux operations.
"""

from math import prod

from TraceLens.PerfModel.perf_model import GEMM, UnaryElementwise, gemm_fp4_impl
from TraceLens.PerfModel.utils import MXFP4_GROUP_SIZE, name2bpe, torch_dtype_map

class a6w4_mm_qa(GEMM):
    """
    Performance model for primus_flux::a6w4_mm_qa.

    A6W4 quantized activation matmul with full-precision (BF16) weights:
    Y[M, N] = dequant(act_quant[M, K]) @ W[N, K].T + bias[N]

    Activations are 6-bit quantized (stored as unsigned char, 1 byte/elem)
    with per-group unsigned char scales. Weights and bias are BF16.

    Roofline calculation -- FLOPs:
        FLOPs = 2 * M * N * K

    Roofline calculation -- Bytes:
        bytes_act    = M * K * 1                   # u8 quantized activation
        bytes_W      = N * K * bpe(weight)         # BF16 weight
        bytes_output = M * N * bpe(output)         # BF16 output
        bytes_bias   = N * bpe(bias) if bias

    Expected Input Dims from trace:
        [[M, K], [scale_elems], [N, K], [N]]

    Expected Input type from trace:
        [unsigned char, unsigned char, BFloat16, BFloat16]
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"].get("Input type", []) or []
        M, K = dims[0]
        N = dims[2][0]
        has_bias = len(dims[3]) > 0
        return {
            "B": 1,
            "M": M,
            "N": N,
            "K": K,
            "bias": has_bias,
            "dtype_A_B": (
                types[0],
                types[2],
                # Output dtype is not in Input type (it is a return value).
                # Concrete Inputs[6] holds the ScalarType enum (15 = bfloat16),
                "c10::bfloat16",
            ),
            "scale_shape": tuple(dims[1]),
            "scale_dtype": types[1],
        }

    def bytes(self, bpe_mat1=None, bpe_mat2=None, bpe_bias=None, bpe_output=None):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2])
        base_bytes = super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )
        bpe_scale = name2bpe(self.param_details["scale_dtype"])
        base_bytes += prod(self.param_details["scale_shape"]) * bpe_scale
        return base_bytes

    def get_compute_precision(self):
        return torch_dtype_map("mxfp4") # Compute precision not in trace


class a6w4_mm_qa_prequant_w(GEMM):
    """
    Performance model for primus_flux::a6w4_mm_qa_prequant_w.

    A6W4 matmul with pre-quantized FP4 weights:
    Y[M, N] = dequant(act_quant[M, K]) @ dequant(w_quant[N, K]).T + bias[N]

    Activations are 6-bit quantized (stored as u8, 1 byte/elem).
    Weights are FP4 E2M1 packed (2 values per byte) with E8M0 per-group scales.
    Optional BF16 bias.

    Roofline calculation -- FLOPs:
        FLOPs = 2 * M * N * K

    Roofline calculation -- Bytes:
        bytes_act    = M * K * 1                   # u8 activation
        bytes_W      = N * K * 0.5                 # FP4 packed weight
        bytes_output = M * N * bpe(output)         # BF16 output
        bytes_bias   = N * bpe(bias) if bias

    Expected Input Dims from trace:
        [[M, K], [scale_elems], [N, K//2], [N, K//32], [], [N]_or_[]]

    Expected Input type from trace:
        [unsigned char, unsigned char, unsigned char, unsigned char, Scalar, BFloat16]
    """

    @staticmethod
    def get_param_details(event):
        dims = event["args"]["Input Dims"]
        types = event["args"].get("Input type", []) or []
        M, K = dims[0]
        N = dims[2][0]
        has_bias = len(dims[5]) > 0
        return {
            "B": 1,
            "M": M,
            "N": N,
            "K": K,
            "bias": has_bias,
            "dtype_A_B": (
                types[0],
                types[2],
                # Output dtype is not in Input type (it is a return value).
                # Concrete Inputs[6] holds the ScalarType enum (15 = bfloat16)
                "c10::bfloat16",
            ),
        }

    def bytes(self, bpe_mat1=None, bpe_mat2=None, bpe_bias=None, bpe_output=None):
        dtype_A_B = self.param_details["dtype_A_B"]
        self.bpe_mat1 = name2bpe(dtype_A_B[0])
        self.bpe_mat2 = name2bpe(dtype_A_B[1])
        
        self.bpe_output = name2bpe(dtype_A_B[2])
        self.bpe_bias = name2bpe(dtype_A_B[2])
        return super().bytes(
            bpe_mat1=self.bpe_mat1,
            bpe_mat2=self.bpe_mat2,
            bpe_bias=self.bpe_bias,
            bpe_output=self.bpe_output,
        )

    def get_compute_precision(self):
        return torch_dtype_map("mxfp4") # Compute precision not in trace


class _QuantizeMxfp4Base(UnaryElementwise):
    """Base class for primus_flux MXFP4 quantization variants."""

    category = "GroupQuant"

    def flops(self):
        return 2 * self.nelems

    def bytes(self):
        if self.bpe_in is None:
            return None
        bytes_read = self.nelems * self.bpe_in
        # Output dtype (FP4) not in trace args (return value)
        bytes_write_fp4 = self.nelems * name2bpe("fp4")
        # Scale dtype: one E8M0 byte per 32-element block
        bytes_write_scales = max(1, self.nelems // MXFP4_GROUP_SIZE) * name2bpe("c10::Float8_e8m0fnu")
        return bytes_read + bytes_write_fp4 + bytes_write_scales


class quantize_mxfp4_h16_dual(_QuantizeMxfp4Base):
    """
    Performance model for primus_flux::quantize_mxfp4_h16_dual.

    Dual MXFP4 quantization: quantizes a BF16 [M, N] tensor into two
    separate MXFP4 representations (packed FP4 + E8M0 per-32-elem scales),
    one for the forward path and one for the backward path.

    Roofline calculation -- FLOPs:
        ~2 FLOPs per element per output (max-abs reduction + scale)
        x 2 outputs = 4 * M * N total.

    Roofline calculation -- Bytes:
        read  x:         M * N * bpe_in
        write 2 * (FP4 packed + E8M0 scales)

    Expected Input Dims from trace:
        [[M, N], [], []]

    Expected Input type from trace:
        [BFloat16, Scalar, Scalar]
    """

    @staticmethod
    def get_param_details(event):
        op_shape = tuple(event["args"]["Input Dims"][0])
        dtype_in = event["args"]["Input type"][0]
        stride_input = tuple(event["args"]["Input Strides"][0])
        return {
            "op_shape": op_shape,
            "dtype_in_out": (dtype_in, dtype_in),
            "stride_input": stride_input,
            "stride_output": None,
        }

    def flops(self):
        return 4 * self.nelems

    def bytes(self):
        if self.bpe_in is None:
            return None
        bytes_read = self.nelems * self.bpe_in
        # Output dtype (FP4) not in trace args (return value)
        bytes_write_fp4 = 2 * self.nelems * name2bpe("fp4")
        # Scale dtype: one E8M0 byte per 32-element block
        bytes_write_scales = 2 * max(1, self.nelems // MXFP4_GROUP_SIZE) * name2bpe("c10::Float8_e8m0fnu")
        return bytes_read + bytes_write_fp4 + bytes_write_scales


# ── Tree postprocess: fwd/bwd pseudo-op labeling ──────────────────────────

import collections

_GEMM_OP_NAMES = frozenset(
    {
        "primus_turbo::gemm_fp4_impl",
        "primus_flux::a6w4_mm_qa",
        "primus_flux::a6w4_mm_qa_prequant_w",
    }
)


def _suffix_for_op(name):
    if "a6w4" in name:
        return "a6w4"
    return "gemm"


def _set_bookkeeping(tree, event):
    uid = len(tree.events)
    event["UID"] = uid
    tree.events.append(event)
    tree.events_by_uid[uid] = event
    seq_num = event["args"].get("Sequence number")
    if seq_num is not None and seq_num in tree.seq_num2event_uids_map:
        tree.seq_num2event_uids_map[seq_num].append(uid)


def _inject_pseudo_op(tree, orig_event, pseudo_name):
    """Wrap *orig_event* in a pseudo CPU op named *pseudo_name*."""
    parent_uid = orig_event.get("parent")
    if parent_uid is None:
        return
    parent_event = tree.get_UID2event(parent_uid)

    pseudo_evt = {
        "ph": "X",
        "name": pseudo_name,
        "cat": "cpu_op",
        "pid": orig_event["pid"],
        "tid": orig_event["tid"],
        "ts": orig_event["ts"],
        "dur": orig_event.get("dur", 0),
        "args": {
            "Input Dims": orig_event["args"].get("Input Dims", []),
            "Input type": orig_event["args"].get("Input type", []),
            "Input Strides": orig_event["args"].get("Input Strides", []),
            "Concrete Inputs": orig_event["args"].get("Concrete Inputs", []),
            "Sequence number": orig_event["args"].get("Sequence number", 0),
            "External id": orig_event["args"].get("External id", 0),
            "Pseudo op": True,
        },
        "parent": parent_uid,
        "children": [orig_event["UID"]],
        "gpu_events": orig_event.get("gpu_events", []),
    }
    _set_bookkeeping(tree, pseudo_evt)

    children = parent_event.get("children", [])
    if orig_event["UID"] in children:
        idx = children.index(orig_event["UID"])
        children[idx] = pseudo_evt["UID"]
    else:
        children.append(pseudo_evt["UID"])

    orig_event["parent"] = pseudo_evt["UID"]


def _get_bwd_ops(tree, fwd_event):
    bwd_uids = fwd_event.get("bwd_events") or tree.get_subtree_bwd_events(
        fwd_event["UID"]
    )
    return [tree.get_UID2event(uid) for uid in bwd_uids]


def _collect_gemm_children(tree, parent_event):
    result = []
    for child_uid in parent_event.get("children", []):
        child = tree.get_UID2event(child_uid)
        if child.get("name") in _GEMM_OP_NAMES:
            result.append(child)
    result.sort(key=lambda e: e["ts"])
    return result


def _collect_gemm_descendants(tree, parent_event):
    result = []
    queue = collections.deque(parent_event.get("children", []))
    while queue:
        uid = queue.popleft()
        evt = tree.get_UID2event(uid)
        if evt.get("name") in _GEMM_OP_NAMES:
            result.append(evt)
        queue.extend(evt.get("children", []))
    result.sort(key=lambda e: e["ts"])
    return result


def _label_fwd_gemms(tree, fwd_event, prefix, processed_uids):
    """Wrap forward GEMM children in yfwd pseudo ops."""
    for gemm_evt in _collect_gemm_children(tree, fwd_event):
        if gemm_evt["UID"] in processed_uids:
            continue
        processed_uids.add(gemm_evt["UID"])
        suffix = _suffix_for_op(gemm_evt["name"])
        _inject_pseudo_op(tree, gemm_evt, f"{prefix}_yfwd_{suffix}")


def _label_bwd_gemms(tree, fwd_event, prefix, processed_uids):
    """Wrap backward GEMM descendants in xgrad/wgrad pseudo ops."""
    for bwd_op in _get_bwd_ops(tree, fwd_event):
        for i, gemm_evt in enumerate(_collect_gemm_descendants(tree, bwd_op)):
            if gemm_evt["UID"] in processed_uids:
                continue
            processed_uids.add(gemm_evt["UID"])
            suffix = _suffix_for_op(gemm_evt["name"])
            label = "xgrad" if i == 0 else "wgrad"
            _inject_pseudo_op(
                tree, gemm_evt, f"{prefix}Backward_{label}_{suffix}"
            )


def tree_postprocess_extension(trace_tree):
    """Create fwd/bwd pseudo ops for MXFP4 GEMM events.

    Wraps each ``primus_turbo::gemm_fp4_impl`` / ``primus_flux::a6w4_mm_qa*``
    inside ``_MXFP4*Function`` (forward) or ``_MXFP4*FunctionBackward``
    (backward) in a pseudo op named e.g. ``_MXFP4Linear_yfwd_gemm``,
    ``_MXFP4LinearBackward_xgrad_gemm``, or ``_MXFP4LinearBackward_wgrad_gemm``.
    """
    fwd_names = [
        name
        for name in trace_tree.name2event_uids
        if name.startswith("_MXFP4") and "Backward" not in name
    ]
    if not fwd_names:
        return

    processed_uids = set()

    for fwd_name in fwd_names:
        prefix = fwd_name.removesuffix("Function")
        for fwd_uid in list(trace_tree.name2event_uids[fwd_name]):
            fwd_event = trace_tree.get_UID2event(fwd_uid)
            _label_fwd_gemms(trace_tree, fwd_event, prefix, processed_uids)
            _label_bwd_gemms(trace_tree, fwd_event, prefix, processed_uids)
