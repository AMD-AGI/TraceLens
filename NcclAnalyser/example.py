from nccl_analyser import NcclAnalyser
import os

# Initialize NCCL Analyser
root_dir = '/path/to/your/trace/files/directory'
world_size = 8
list_profile_filepaths = [os.path.join(root_dir, f'rank{i}_trace.json') for i in range(world_size)]
output_db_path = os.path.join(root_dir, 'nccl_events_df.csv')
summary_db_path = os.path.join(root_dir, 'nccl_summary_df.csv')
nccl_analyser = NcclAnalyser(list_profile_filepaths, world_size)

# Build and save the detailed database
df_nccl = nccl_analyser.build_df_nccl()
df_nccl.to_csv(output_db_path, index=False)

# Build and save the summary database
df_nccl_summary = nccl_analyser.summarize_df_nccl(df_nccl)
df_nccl_summary.to_csv(summary_db_path, index=False)