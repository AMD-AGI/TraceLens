import os

from TraceLens import TraceFuse

profiles_root_dir = "path/to/your/profiles"
world_size = 8
output_file = os.path.join(profiles_root_dir, "merged_trace.json")
list_profile_filepaths = [
    os.path.join(profiles_root_dir, f"rank_{i}.json") for i in range(world_size)
]

# Initialize TraceFusion
fuser = TraceFuse(list_profile_filepaths)

# # Custom filter for NCCL kernels
# def filter_nccl_kernels(event):
#     cond0 = event.get('cat') in ['kernel', 'gpu_user_annotation']
#     cond1 = 'nccl' in event.get('name', '').lower()
#     return cond0 and cond1

# fuser.merge_and_save(output_file, filter_fn=filter_nccl_kernels)
fuser.merge_and_save(output_file)
