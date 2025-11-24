from TraceLens import JaxAnalyses
import sys
import pandas as pd
import TraceLens


import TraceLens.Trace2Tree.parse_xplane_pb as xlaparser
from TraceLens.TreePerf import JaxTreePerfAnalyzer, TreePerfAnalyzer


filename_path = sys.argv[1]


perf_analyzer = JaxTreePerfAnalyzer.from_file(profile_filepath=filename_path)
tree = perf_analyzer.tree

## Augment the tree with hlo_op nodes
tree._add_hlo_op_nodes()

hlo_events = [i for i in tree.events if i["cat"] == "hlo_op"]
print(f"Augmented the tree with {len(hlo_events)} nodes for hlo_ops")
uid = hlo_events[-1000]["UID"]

tree.traverse_subtree_hlo_op(uid)


df_gpu_timeline = perf_analyzer.get_df_gpu_timeline()
df_gpu_events_averages = perf_analyzer.get_df_gpu_events_averages()
print(df_gpu_events_averages)
df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_details=True)
df_kernel_launchers_summary = perf_analyzer.get_df_kernel_launchers_summary(
    df_kernel_launchers
)
df_kernel_launchers_summary_by_category = (
    perf_analyzer.get_df_kernel_launchers_summary_by_category(df_kernel_launchers)
)
print(df_kernel_launchers_summary_by_category)
