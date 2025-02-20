from trace_fuse import TraceFuse
import os

profiles_root_dir = '/home/ajassani/trace_data/jan6/MI325/gpt-3-xl_ddp_bfloat16_bs5_steps25_det0_prof1_level1'
world_size = 8

# Initialize TraceFusion
fuser = TraceFuse(profiles_root_dir, world_size)

# # Custom filter for NCCL kernels
def filter_gpu_timeline(event):
    if 'cat' not in event or 'args' not in event:
        return False
    if event['cat'] not in ['kernel', 'gpu_user_annotation']:
        return False
    return True
fuser.set_filter(filter_gpu_timeline)

# Define rank-to-file mapping
def rank2file(rank):
    return os.path.join(profiles_root_dir, f'pytorch_profile_rank{rank}.json')
fuser.set_rank2file_fn(rank2file)

# Merge and Save traces
output_file = '/home/ajassani/trace_data/gpu_only_MI325_high_var_bug_gpt-3-xl_ddp_bfloat16_bs5_steps25_det0_prof1_level1_merged.json'
ranks_to_merge = range(world_size)
fuser.merge_and_save(ranks_to_merge, output_file)
