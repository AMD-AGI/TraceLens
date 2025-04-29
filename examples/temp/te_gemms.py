import math
import re

def get_bwd_ops_for_fwd_op(perf_analyzer, fwd_op_event):
    bwd_eventUIDs = fwd_op_event.get('bwd_events')
    if not bwd_eventUIDs:
        perf_analyzer.tree.link_bwd_events(fwd_op_event['UID'])
        bwd_eventUIDs = fwd_op_event.get('bwd_events')
    bwd_events = [perf_analyzer.tree.get_UID2event(uid) for uid in bwd_eventUIDs]
    return bwd_events

def add_event_to_tree(tree, event):
    UID = len(tree.events)
    event['UID'] = UID
    tree.events.append(event)
    tree.events_by_uid[UID] = event
    seq_num = event['args']['Sequence number']
    tree.seq_num2event_uids_map[seq_num].append(UID)

def is_gemm_kernel(kernel_event):
    assert kernel_event['cat'] == 'kernel'
    kernel_name = kernel_event['name']
    pattern = r'.*C.*_A.*_B.*'
    is_rocm_gemm = bool(re.match(pattern, kernel_name))
    is_cuda_gemm = kernel_name.startswith('nvjet')
    return is_rocm_gemm or is_cuda_gemm

def _create_host_mm_ops_common(perf_analyzer, fwd_op_event, expected_name, w_idx, x_idx, prefix):
    """
    Create synthetic matmul ops for forward and backward passes.
    """
    if fwd_op_event.get('name') != expected_name:
        return []

    fwd_gpu_event_ids = fwd_op_event.get('gpu_events', [])
    if not fwd_gpu_event_ids:
        print(f"[Warning] No GPU events found for fwd UID {fwd_op_event['UID']}")
        return []

    fwd_gpu_events = [perf_analyzer.tree.get_UID2event(uid) for uid in fwd_gpu_event_ids]
    fwd_gemm_kernels = [e for e in fwd_gpu_events if is_gemm_kernel(e)]

    if len(fwd_gemm_kernels) != 1:
        print(f"[Warning] Expected 1 GEMM kernel in fwd, found {len(fwd_gemm_kernels)}")
        return

    yfwd_kernel = fwd_gemm_kernels[0]

    # Link to backward
    bwd_ops = get_bwd_ops_for_fwd_op(perf_analyzer, fwd_op_event)
    if not bwd_ops:
        print(f"[Warning] No backward op found for fwd UID {fwd_op_event['UID']}")
        return []

    bprop_gpu_event_ids = [uid for bwd_op in bwd_ops for uid in bwd_op.get('gpu_events', [])]
    bprop_gpu_events = [perf_analyzer.tree.get_UID2event(uid) for uid in bprop_gpu_event_ids]
    bprop_gemm_kernels = [e for e in bprop_gpu_events if is_gemm_kernel(e)]
    def get_launcher_start(kenrel_evt):
        launcher = perf_analyzer.tree.get_parent_event(kenrel_evt)
        return launcher.get('ts')
    bprop_gemm_kernels = sorted(bprop_gemm_kernels, key=lambda e: get_launcher_start(e)) #which 

    if len(bprop_gemm_kernels) != 2:
        print(f"[Warning] Expected 2 GEMM kernels in bwd, found {len(bprop_gemm_kernels)}")
        return

    wgrad_kernel, xgrad_kernel = bprop_gemm_kernels

    try:
        input_dims = fwd_op_event['args']['Input Dims']
        input_types = fwd_op_event['args']['Input type']
        W_shape, inp_shape = input_dims[w_idx], input_dims[x_idx]
        W_dtype, inp_dtype = input_types[w_idx], input_types[x_idx]
    except Exception as e:
        print(f"[Warning] Missing shape info in fwd UID {fwd_op_event['UID']}: {e}")
        return []

    assert inp_shape[-1] == W_shape[1]
    assert W_dtype == inp_dtype

    X_shape = (math.prod(inp_shape[:-1]), inp_shape[-1])
    Y_grad_shape = (X_shape[0], W_shape[0])

    # Check if synthetic ops already exist
    seq_num = fwd_op_event['args']['Sequence number']
    seq_num_uids = perf_analyzer.tree.seq_num2event_uids_map.get(seq_num, [])
    seq_num_evts = [perf_analyzer.tree.get_UID2event(uid) for uid in seq_num_uids]

    existing = [e for e in seq_num_evts if e['name'] == f'{prefix}_xgrad_mm']
    if existing:
        return

    # Create synthetic host ops
    Yfwd_evt = {
        'ph': 'X',
        'name': f'{prefix}_yfwd_mm',
        'cat': 'cpu_op',
        'pid': fwd_op_event['pid'],
        'tid': fwd_op_event['tid'],
        'args': {
            'Input Dims': [X_shape, W_shape[::-1]],
            'Input type': [inp_dtype, inp_dtype],
            'Sequence number': seq_num,
            'External id': yfwd_kernel['args']['correlation']
        },
        'children': [yfwd_kernel.get('parent')],
        'gpu_events': [yfwd_kernel['UID']],
    }
    add_event_to_tree(perf_analyzer.tree, Yfwd_evt)
    Xgrad_evt = {
        'ph': 'X',
        'name': f'{prefix}_xgrad_mm',
        'cat': 'cpu_op',
        'pid': bwd_ops[0]['pid'],
        'tid': bwd_ops[0]['tid'],
        'args': {
            'Input Dims': [Y_grad_shape, W_shape],
            'Input type': [inp_dtype, inp_dtype],
            'Sequence number': seq_num,
            'External id': xgrad_kernel['args']['correlation']
        },
        'children': [xgrad_kernel.get('parent')],
        'gpu_events': [xgrad_kernel['UID']],
    }
    add_event_to_tree(perf_analyzer.tree, Xgrad_evt)

    Wgrad_evt = {
        'ph': 'X',
        'name': f'{prefix}_wgrad_mm',
        'cat': 'cpu_op',
        'pid': bwd_ops[0]['pid'],
        'tid': bwd_ops[0]['tid'],
        'args': {
            'Input Dims': [Y_grad_shape[::-1], X_shape],
            'Input type': [inp_dtype, inp_dtype],
            'Sequence number': seq_num,
            'External id': wgrad_kernel['args']['correlation']
        },
        'children': [wgrad_kernel.get('parent')],
        'gpu_events': [wgrad_kernel['UID']],
    }
    add_event_to_tree(perf_analyzer.tree, Wgrad_evt)

def create_host_mm_ops_from_linear_op(perf_analyzer, fwd_op_event):
    return _create_host_mm_ops_common(
        perf_analyzer, fwd_op_event,
        expected_name='_Linear',
        w_idx=0,
        x_idx=1,
        prefix='_Linear'
    )

def create_host_mm_ops_from_linearnorm_op(perf_analyzer, fwd_op_event):
    return _create_host_mm_ops_common(
        perf_analyzer, fwd_op_event,
        expected_name='_LayerNormLinear',
        w_idx=3,
        x_idx=0,
        prefix='_LayerNormLinear'
    )


def main():
    import argparse
    import json
    from TraceLens import TreePerfAnalyzer, PerfModel
    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate gemm perf report tables.')
    parser.add_argument('--profile_path', type=str, required=True, help='Path to the profile.json file')
    parser.add_argument('--output_csv_path', type=str, required=True, help='Path to the output CSV file')
    args = parser.parse_args()

    perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=args.profile_path)

    # 1. Add synthetic matmul ops for forward and backward passes
    for evt in perf_analyzer.tree.events:
        if evt['name'] == '_Linear':
            create_host_mm_ops_from_linear_op(perf_analyzer, evt)
        elif evt['name'] == '_LayerNormLinear':
            create_host_mm_ops_from_linearnorm_op(perf_analyzer, evt)
    
    # 2. Update the event names for GEMM
    gemm_event_names = ['aten::mm', 'aten::addmm', 'aten::_scaled_mm']
    dict_perf_model = {}
    for prefix in ['_Linear', '_LayerNormLinear']:  
        for suffix in ['_yfwd_mm', '_xgrad_mm', '_wgrad_mm']:
            name = prefix+suffix
            gemm_event_names.append(name)
            dict_perf_model[name] = PerfModel.aten_mm
    
    # 3. Generate the performance report
    gemm_events = [event for event in perf_analyzer.tree.events if event['name'] in gemm_event_names]
    df_gemm_ops = perf_analyzer.build_df_perf_metrics(gemm_events, include_kernel_names=True, dict_name_to_perf_model=dict_perf_model)
    df_gemm_summary = perf_analyzer.summarize_df_perf_metrics(df_gemm_ops, ['mean'])
    df_gemm_summary.to_csv(args.output_csv_path, index=False)
    print(f"Generated GEMM performance report at {args.output_csv_path}")


if __name__ == "__main__":
    main()