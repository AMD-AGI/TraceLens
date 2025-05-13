import argparse
import json
import math
import re
import logging

def get_bwd_ops_for_fwd_op(trace_tree, fwd_op_event: dict) -> list[dict]:
    """
    Get backward operations for a given forward operation.
    """
    bwd_eventUIDs = fwd_op_event.get('bwd_events')
    if not bwd_eventUIDs:
        trace_tree.link_bwd_events(fwd_op_event['UID'])
        bwd_eventUIDs = fwd_op_event.get('bwd_events')
    bwd_events = [trace_tree.get_UID2event(uid) for uid in bwd_eventUIDs]
    return bwd_events

def set_bookkeeping_attr(tree, event: dict):
    UID = len(tree.events)
    event['UID'] = UID
    tree.events.append(event)
    tree.events_by_uid[UID] = event
    seq_num = event['args']['Sequence number']
    tree.seq_num2event_uids_map[seq_num].append(UID)

def is_gemm_kernel(kernel_event: dict) -> bool:
    assert kernel_event['cat'] == 'kernel'
    kernel_name = kernel_event['name']
    pattern = r'.*C.*_A.*_B.*'
    is_rocm_gemm = bool(re.match(pattern, kernel_name))
    is_cuda_gemm = kernel_name.startswith('nvjet')
    return is_rocm_gemm or is_cuda_gemm

def _create_host_mm_ops_common(trace_tree, fwd_op_event: dict, expected_name: str, 
                               w_idx: int, x_idx: int, fp8_bool_idx: int):
    """
    Create pseudo matmul ops for forward and backward passes.
    fwd pass:
        Y = X.matmul(W^T) + B
    bwd pass gemm:
        X_grad = Y_grad.matmul(W)
        W_grad = Y_grad^T.matmul(X)
        B_grad = Y_grad.sum(dim=0)
    """
    if fwd_op_event.get('name') != expected_name:
        logging.warning(f"[Warning] Expected op name {expected_name}, found {fwd_op_event['name']}")
        return
    prefix = expected_name

    fwd_gpu_event_ids = fwd_op_event.get('gpu_events', [])
    if not fwd_gpu_event_ids:
        logging.warning(f"[Warning] No GPU events found for fwd UID {fwd_op_event['UID']}")
        return

    fwd_gpu_events = [trace_tree.get_UID2event(uid) for uid in fwd_gpu_event_ids]
    fwd_gemm_kernels = [e for e in fwd_gpu_events if is_gemm_kernel(e)]

    if len(fwd_gemm_kernels) != 1:
        logging.warning(f"[Warning] Expected 1 GEMM kernel in fwd, found {len(fwd_gemm_kernels)}")
        return

    yfwd_kernel = fwd_gemm_kernels[0]

    # Link to backward
    bwd_ops = get_bwd_ops_for_fwd_op(trace_tree, fwd_op_event)
    if not bwd_ops:
        logging.warning(f"[Warning] No backward op found for fwd UID {fwd_op_event['UID']}")
        return

    bprop_gpu_event_ids = [uid for bwd_op in bwd_ops for uid in bwd_op.get('gpu_events', [])]
    bprop_gpu_events = [trace_tree.get_UID2event(uid) for uid in bprop_gpu_event_ids]
    bprop_gemm_kernels = [e for e in bprop_gpu_events if is_gemm_kernel(e)]
    def get_launcher_start(kernel_evt):
        launcher = trace_tree.get_parent_event(kernel_evt)
        return launcher.get('ts')
    bprop_gemm_kernels = sorted(bprop_gemm_kernels, key=lambda e: get_launcher_start(e)) #which 

    if len(bprop_gemm_kernels) != 2:
        logging.warning(f"[Warning] Expected 2 GEMM kernels in bwd, found {len(bprop_gemm_kernels)}")
        return

    # Transformer Engine first launches xgrad, then wgrad
    # ref: https://github.com/NVIDIA/TransformerEngine/blob/91405eb4a184b962edb7a211626b7e58a4a87cbc/transformer_engine/pytorch/module/linear.py#L405
    # ref: https://github.com/NVIDIA/TransformerEngine/blob/91405eb4/transformer_engine/pytorch/module/layernorm_linear.py#L472
    xgrad_kernel, wgrad_kernel = bprop_gemm_kernels

    try:
        input_dims = fwd_op_event['args']['Input Dims']
        input_types = fwd_op_event['args']['Input type']
        W_shape, inp_shape = input_dims[w_idx], input_dims[x_idx]
        W_dtype, inp_dtype = input_types[w_idx], input_types[x_idx]
    except Exception as e:
        logging.warning(f"[Warning] Missing shape info in fwd UID {fwd_op_event['UID']}: {e}")
        return

    assert inp_shape[-1] == W_shape[1]
    assert W_dtype == inp_dtype

    is_fp8 = fwd_op_event['args']['Concrete Inputs'][fp8_bool_idx] == 'True'
    if is_fp8:
        W_dtype = 'fp8'
        inp_dtype = 'fp8'

    X_shape = (math.prod(inp_shape[:-1]), inp_shape[-1])
    Y_grad_shape = (X_shape[0], W_shape[0])

    # Check if pseudo ops already exist
    seq_num = fwd_op_event['args']['Sequence number']
    seq_num_uids = trace_tree.seq_num2event_uids_map.get(seq_num, [])
    seq_num_evts = [trace_tree.get_UID2event(uid) for uid in seq_num_uids]

    existing = [e for e in seq_num_evts if e['name'] == f'{prefix}_xgrad_mm']
    if existing:
        return

    # Create pseudo host ops
    Yfwd_launcher = trace_tree.get_parent_event(yfwd_kernel)
    Yfwd_original_cpu_op = trace_tree.get_parent_event(Yfwd_launcher)
    Yfwd_evt = {
        'ph': 'X',
        'name': f'{prefix}_yfwd_mm',
        'cat': 'cpu_op',
        'pid': fwd_op_event['pid'],
        'tid': fwd_op_event['tid'],
        'args': {
            # Y = X.matmul(W^T)
            'Input Dims': [X_shape, W_shape[::-1]],
            'Input type': [inp_dtype, inp_dtype],
            'Sequence number': seq_num,
            'External id': yfwd_kernel['args']['correlation'],
            'Pseudo op': True
        },
        'children': [Yfwd_launcher.get('UID')],
        'gpu_events': [yfwd_kernel['UID']],
    }
    set_bookkeeping_attr(trace_tree, Yfwd_evt)
    # update the original CPU op to point to the pseudo op
    Yfwd_original_cpu_op['children'].remove(Yfwd_launcher.get('UID'))
    Yfwd_original_cpu_op['children'].append(Yfwd_evt.get('UID'))


    Xgrad_launcher = trace_tree.get_parent_event(xgrad_kernel)
    Xgrad_original_cpu_op = trace_tree.get_parent_event(Xgrad_launcher)
    Xgrad_evt = {
        'ph': 'X',
        'name': f'{prefix}Backward_xgrad_mm',
        'cat': 'cpu_op',
        'pid': bwd_ops[0]['pid'],
        'tid': bwd_ops[0]['tid'],
        'args': {
            # X_grad = Y_grad.matmul(W)
            'Input Dims': [Y_grad_shape, W_shape],
            'Input type': [inp_dtype, inp_dtype],
            'Sequence number': seq_num,
            'External id': xgrad_kernel['args']['correlation'],
            'Pseudo op': True
        },
        'children': [Xgrad_launcher.get('UID')],
        'gpu_events': [xgrad_kernel['UID']],
    }
    set_bookkeeping_attr(trace_tree, Xgrad_evt)
    # update the original CPU op to point to the pseudo op
    Xgrad_original_cpu_op['children'].remove(Xgrad_launcher.get('UID'))
    Xgrad_original_cpu_op['children'].append(Xgrad_evt.get('UID'))

    Wgrad_launcher = trace_tree.get_parent_event(wgrad_kernel)
    Wgrad_original_cpu_op = trace_tree.get_parent_event(Wgrad_launcher)
    Wgrad_evt = {
        'ph': 'X',
        'name': f'{prefix}Backward_wgrad_mm',
        'cat': 'cpu_op',
        'pid': bwd_ops[0]['pid'],
        'tid': bwd_ops[0]['tid'],
        'args': {
            # W_grad = Y_grad^T.matmul(X)
            'Input Dims': [Y_grad_shape[::-1], X_shape],
            'Input type': [inp_dtype, inp_dtype],
            'Sequence number': seq_num,
            'External id': wgrad_kernel['args']['correlation'],
            'Pseudo op': True
        },
        'children': [Wgrad_launcher.get('UID')],
        'gpu_events': [wgrad_kernel['UID']],
    }
    set_bookkeeping_attr(trace_tree, Wgrad_evt)
    # update the original CPU op to point to the pseudo op
    Wgrad_original_cpu_op['children'].remove(Wgrad_launcher.get('UID'))
    Wgrad_original_cpu_op['children'].append(Wgrad_evt.get('UID'))

def create_host_mm_ops_from_linear_op(trace_tree, fwd_op_event: dict):
    # index 0 is Linear weight tensor, 1 is input tensor
    # ref: https://github.com/NVIDIA/TransformerEngine/blob/91405eb4a184b962edb7a211626b7e58a4a87cbc/transformer_engine/pytorch/module/linear.py#L405
    _create_host_mm_ops_common(
        trace_tree, fwd_op_event,
        expected_name='_Linear',
        w_idx=0,
        x_idx=1,
        fp8_bool_idx=4
    )

def create_host_mm_ops_from_layernormlinear_op(trace_tree, fwd_op_event: dict):
    # index 3 is Linear weight tensor, 0 is input tensor
    # ref: https://github.com/NVIDIA/TransformerEngine/blob/91405eb4a184b962edb7a211626b7e58a4a87cbc/transformer_engine/pytorch/module/layernorm_linear.py#L1434
    _create_host_mm_ops_common(
        trace_tree, fwd_op_event,
        expected_name='_LayerNormLinear',
        w_idx=3,
        x_idx=0,
        fp8_bool_idx=7
    )

def tev2_create_pseudo_host_mm_ops(trace_tree):
    """
    Context: In Transformer Engine v1, the blas GEMM calls are made by tex_ts::te_gemm_ts CPU ops.
    As a result we can parse the gemm shapes from these CPU ops.
    Hoewever, in Transformer Engine v2, the GEMM calls are made directly by the_Linear and _LayerNormLinear in fwd pass 
    and _LinearBackward and _LayerNormLinearBackward in bwd pass.
    Consequently, we cannot parse the gemm shapes from the _Linear and _LayerNormLinear CPU ops.
    Moreover, since the bwd pass of linear results in 2 gemm calls (xgrad and wgrad), 
    there is no way to directly infer which is xgrad and which is wgrad.

    Solution: To solve this, we create pseudo matmul ops for the fwd and bwd passes.
    We parse the input shape (B, L, D_in) and weight shape (D_out, D_in) from the fwd pass.
    From these, we can infer yfwd, xgrad and wgrad gemm shapes.
    Secondly, by correlating the order of the GPU events and the TEv2 code,
    we can infer which gpu event is xgrad and which is wgrad.

    In summary, we inject 3 pseudo ops as follows:
    fwd pass:
        Y = X.matmul(W^T) + B
    bwd pass gemm:
        X_grad = Y_grad.matmul(W)
        W_grad = Y_grad^T.matmul(X)
    """
    if '_Linear' in trace_tree.name2event_uids and 'tex_ts::te_gemm_ts' not in trace_tree.name2event_uids:
        fwd_op_events = [trace_tree.get_UID2event(uid) for uid in trace_tree.name2event_uids['_Linear']]
        for fwd_op_event in fwd_op_events:
            create_host_mm_ops_from_linear_op(trace_tree, fwd_op_event)
    if '_LayerNormLinear' in trace_tree.name2event_uids and 'tex_ts::te_gemm_ts' not in trace_tree.name2event_uids:
        fwd_op_events = [trace_tree.get_UID2event(uid) for uid in trace_tree.name2event_uids['_LayerNormLinear']]
        for fwd_op_event in fwd_op_events:
            create_host_mm_ops_from_layernormlinear_op(trace_tree, fwd_op_event)