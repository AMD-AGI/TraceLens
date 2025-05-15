import argparse
import json
import pandas as pd
from TraceLens import TraceToTree
from TraceLens import TreePerfAnalyzer
from TraceLens.PerfModel import dict_cat2names

from TraceLens.PerfModel import SDPA
class transformer_engine_attention(SDPA):
    """
    Context: The FusedAttnFunc is a pytorch extention for the attention kernel.
    Unfortunately, the args does not have a bool flag for is_causal.
    Instead, it has a str arg which is not recorded in the trace.

    Solution: Based on the LLM use case we make the assumption that
    the attention is always causal.
    Since this might not be the case for other use cases, 
    we dont add this natively to the perf model and instead add it here 
    """
    @staticmethod
    def get_param_details(event):
    # ref TransformerEngine/transformer_engine/pytorch/cpp_extensions/fused_attn.py
    # https://github.com/NVIDIA/TransformerEngine/blob/51cd441501e8e6dee18c00056f008e1b53b89ebd/transformer_engine/pytorch/attention/dot_product_attention/backends.py#L881
        input_dims = event['args']['Input Dims']
        q_idx = None
        for i, dim in enumerate(input_dims):
            if len(dim)==4:
                q_idx = i
                break
        assert q_idx is not None, "query index not found"
        q_shape, k_shape, v_shape = input_dims[q_idx: q_idx+3]
        strides = event['args']['Input Strides']
        q_strides, k_strides, v_strides = strides[q_idx: q_idx+3]
        # convert stride to tuple
        q_strides, k_strides, v_strides = tuple(q_strides), tuple(k_strides), tuple(v_strides)
        B, N_Q, H_Q, d_h = q_shape
        assert k_shape == v_shape, f"Key and value shapes are different: {k_shape} != {v_shape}"
        _, N_KV, H_KV, _ = k_shape 
        is_causal = True
        dropout_p = 0.0
        flash_impl = True
        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h": d_h,
                "q_strides": q_strides, "k_strides": k_strides, "v_strides": v_strides,
                "dropout": dropout_p, "causal": is_causal, "flash_impl": flash_impl}

from TraceLens.PerfModel import flash_attention

class flash_attention_backward(flash_attention):

    @staticmethod
    def get_param_details(event):
        # dout: torch.Tensor,
        # q: torch.Tensor,
        # k: torch.Tensor,
        # v: torch.Tensor,
        # out: torch.Tensor,
        # softmax_lse: torch.Tensor,
        # dq: Optional[torch.Tensor],
        # dk: Optional[torch.Tensor],
        # dv: Optional[torch.Tensor],
        # dropout_p: float,
        # softmax_scale: float,
        # causal: bool,
        # window_size_left: int,
        # window_size_right: int,
        # softcap: float,
        # alibi_slopes: Optional[torch.Tensor],
        # deterministic: bool,
        # rng_state: Optional[torch.Tensor] = None,
        input_dims = event['args']['Input Dims']
        q_shape, k_shape, v_shape = input_dims[1], input_dims[2], input_dims[3]
        strides = event['args']['Input Strides']
        d_out_stride = tuple(strides[0])
        q_stride, k_stride, v_stride = tuple(strides[1]), tuple(strides[2]), tuple(strides[3])
        out_stride = tuple(strides[4])
        softmax_lse_stride = tuple(strides[5])
        dq_stride, dk_stride, dv_stride = tuple(strides[6]), tuple(strides[7]), tuple(strides[8])
        B, N_Q, H_Q, d_h = q_shape
        assert k_shape == v_shape, f"Key and value shapes are different: {k_shape} != {v_shape}"
        _, N_KV, H_KV, _ = input_dims[2]
        dropout = float(event['args']['Concrete Inputs'][9])
        is_causal = eval(event['args']['Concrete Inputs'][11])
        return {"B": B, "N_Q": N_Q, "H_Q": H_Q, "N_KV": N_KV, "H_KV": H_KV, "d_h": d_h,
                "d_out_stride": d_out_stride,
                "q_stride": q_stride, "k_stride": k_stride, "v_stride": v_stride,
                "out_stride": out_stride, "softmax_lse_stride": softmax_lse_stride,
                "dq_stride": dq_stride, "dk_stride": dk_stride, "dv_stride": dv_stride,
                "dropout": dropout, "causal": is_causal, "flash_impl": True}

    def flops(self):
        return self.flops_bwd()


def main():
    parser = argparse.ArgumentParser(description='Process a JSON trace profile and generate performance report tables.')
    parser.add_argument('--profile_json_path', type=str, required=True, help='Path to the profile.json file')
    parser.add_argument('--output_xlsx_path', type=str, required=True, help='Path to the output Excel file')
    parser.add_argument('--gpu_arch_json_path', type=str, default=None, help='Path to the GPU architecture JSON file')
    args = parser.parse_args()

    # Load the arch json
    gpu_arch_json = None
    if args.gpu_arch_json_path:
        with open(args.gpu_arch_json_path, 'r') as f:
            gpu_arch_json = json.load(f)
    perf_analyzer = TreePerfAnalyzer.from_file(profile_filepath=args.profile_json_path, arch=gpu_arch_json)

    agg_metrics = ['mean', 'median', 'std', 'min', 'max']

    # Generate base DataFrames
    df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_names=True)
    df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(df_kernel_launchers)
    df_kernel_launchers_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(df_kernel_launchers, 
                                                                                        agg_metrics=agg_metrics, 
                                                                                        include_pct=True)

    # Dictionary to hold the op-specific DataFrames
    op_dfs = {}

    # update the dict_cat2names to include FusedAttnFunc
    dict_cat2names['SDPA'].append('FusedAttnFunc')
    dict_name_to_custom_perf_model = {'FusedAttnFunc': transformer_engine_attention}
    dict_cat2names['FA_bwd'].append('flash_attn::_flash_attn_backward')
    dict_name_to_custom_perf_model.update({'flash_attn::_flash_attn_backward': flash_attention_backward})

    for op_cat, op_names in dict_cat2names.items():
        # Filter events belonging to the current category
        op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]
        print(f"Processing {len(op_events)} events for category: {op_cat}")

        if op_cat in ['GEMM', 'UnaryElementwise', 'BinaryElementwise', 'FA_bwd']:
            # For GEMM: create a single table that covers both fwd and bwd.
            df_ops = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True,
                                                        dict_name_to_perf_model=dict_name_to_custom_perf_model)
            df_ops = perf_analyzer.summarize_df_perf_metrics(df_ops, agg_metrics)
            op_dfs[op_cat] = df_ops
        else:
            # For FLASH_ATTN and CONV: create separate tables for forward and backward passes.
            df_ops_fwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=False, include_kernel_names=True, include_args=True,
                                                            dict_name_to_perf_model=dict_name_to_custom_perf_model)
            df_ops_fwd = perf_analyzer.summarize_df_perf_metrics(df_ops_fwd, agg_metrics)
            df_ops_bwd = perf_analyzer.build_df_perf_metrics(op_events, bwd=True, include_kernel_names=True, include_args=True,
                                                            dict_name_to_perf_model=dict_name_to_custom_perf_model)
            df_ops_bwd = perf_analyzer.summarize_df_perf_metrics(df_ops_bwd, agg_metrics)
            op_dfs[f"{op_cat}_fwd"] = df_ops_fwd
            op_dfs[f"{op_cat}_bwd"] = df_ops_bwd

    # Write all DataFrames to separate sheets in an Excel workbook
    with pd.ExcelWriter(args.output_xlsx_path) as writer:
        df_gpu_timeline.to_excel(writer, sheet_name='gpu_timeline', index=False)
        df_kernel_launchers_summary.to_excel(writer, sheet_name='kernel_launchers_summary', index=False)
        df_kernel_launchers_unique_args.to_excel(writer, sheet_name='kernel_launchers_unique_args', index=False)
        
        # Write each op category DataFrame
        for sheet_name, df in op_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"DataFrames successfully written to {args.output_xlsx_path}")

if __name__ == "__main__":
    main()
