import os
from TraceLens import TraceFuse
# profiles_root_dir = '/home/xiewen12/workspace/vendor-benchmarking/logs/bf16_fsdp2_nofuse_bind_dataopt_wcputrace_log/timestamp_20251028/traces/gpt-3-6.7B_fsdp2_bfloat16_bs2_steps30_det0_prof1_level1_tensorlayoutNHWC'
# profiles_root_dir = '/home/xiewen12/workspace/vendor-benchmarking/logs/bf16_fsdp2_nofuse_bind_dataopt_nocputrace_log/timestamp_20251028/traces/gpt-3-6.7B_fsdp2_bfloat16_bs2_steps30_det0_prof1_level1_tensorlayoutNHWC'
# profiles_root_dir = '/home/xiewen12/workspace/vendor-benchmarking/logs/debug_nocputrace/timestamp_20251029/traces/gpt-3-6.7B_fsdp2_bfloat16_bs2_steps30_det0_prof1_level1_tensorlayoutNHWC'
# profiles_root_dir = '/home/xiewen12/workspace/vendor-benchmarking/logs/debug_nocputrace_step7/timestamp_20251029/traces/gpt-3-6.7B_fsdp2_bfloat16_bs2_steps30_det0_prof1_level1_tensorlayoutNHWC'
profiles_root_dir = '/home/xiewen12/workspace/vendor-benchmarking/logs/debug_sdma_profile/timestamp_20251029/traces/gpt-3-6.7B_fsdp2_bfloat16_bs2_steps30_det0_prof1_level1_tensorlayoutNHWC'
world_size = 8
output_file = os.path.join(profiles_root_dir, 'merged_trace.json')
list_profile_filepaths = [os.path.join(profiles_root_dir, f'pytorch_profile_rank{i}.json') for i in range(world_size)]

# Initialize TraceFusion
fuser = TraceFuse(list_profile_filepaths)

# # Custom filter for NCCL kernels
# def filter_nccl_kernels(event):
#     cond0 = event.get('cat') in ['kernel', 'gpu_user_annotation']
#     cond1 = 'nccl' in event.get('name', '').lower()
#     return cond0 and cond1

# fuser.merge_and_save(output_file, filter_fn=filter_nccl_kernels)
fuser.merge_and_save(output_file)
