import sys 
import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure basic logging to stdout with WARNING level
logging.basicConfig(stream=sys.stdout, # Output to console
                    level=logging.WARNING, # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from TraceLens import TreePerfAnalyzer, JaxTreePerfAnalyzer

if 0:
    profile_path = '/home/guangphu/perf-profiling/data/jax_minimal/conv/plugins/profile/2025_09_10_15_08_56/chi-mi300x-013.ord.vultr.cpe.ice.amd.com.xplane.pb'
    output_path = '/home/guangphu/perf-profiling/logs/tracelens/jax_minimal_conv/'
if 0:
    profile_path = '/home/guangphu/perf-profiling/data/llm_traces/mi355x/hunyuan_t129/plugins/profile/2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb'
    output_path = '/home/guangphu/perf-profiling/logs/tracelens/jax/'
if 1:
    profile_path = '/home/guangphu/perf-profiling/data/llm_traces/tts-traces/tts-traces-h100/bs16/rank_0/rocm-framework-h100-sxm-1_60628.1751372362949836640.pt.trace.json'

# Load input trace file
if profile_path.endswith('xplane.pb'):
    perf_analyzer = JaxTreePerfAnalyzer.from_file(profile_path)
elif profile_path.endswith('trace.json'):
    perf_analyzer = TreePerfAnalyzer.from_file(profile_path)
else:
    logger.warning('File format not recognized.')
    sys.exit(0)
    
    
# Filter events of interest
op_cat = 'GEMM'  # 'CONV'  # 'SDPA'  
op_names = None 

# torch event filter
if op_names is None:
    op_names = perf_analyzer.dict_cat2names[op_cat] # ['aten::copy_', ]
op_events = [event for event in perf_analyzer.tree.events if event['name'] in op_names]
 # Jax event filter
op_events = [event for event in perf_analyzer.tree.events if event.get('gpu_kernel_op_cat', 'None') == op_cat]
df_ops = perf_analyzer.build_df_perf_metrics(op_events)

# TODO plot distribution instead
# df_op_summary = perf_analyzer.summarize_df_perf_metrics(df_op_perf_model_cleaned, agg_metrics=['mean', 'std']) 
# alternatively, use existing dataframes from generate_perf_report.py
# df_perf_metrics = 'logs/tracelens/jax/trace_analysis_results_op_jax_gemm.csv'
# build perf metrics dataframe

prefix = op_cat 

# config.yaml
peak_bw = 5.3  # in TB/s
bw_eff = 0.95
mabw = bw_eff * peak_bw 
peak_tflops = 1307.4  
flops_eff = 0.7
maf = flops_eff * peak_tflops  
memory_bandwidth_bytes = mabw * 1e12 

# Compute intensity and bounds
log_max_ci = np.log10(max(df_ops["FLOPS/Byte"]) * 2) # 2 for better visualization
log_min_ci = np.log10(min(df_ops["FLOPS/Byte"]) / 2) # 2 for better visualization
compute_intensity = np.logspace(log_min_ci, log_max_ci, 100)  # FLOPs/Byte
memory_bound = compute_intensity * memory_bandwidth_bytes / 1e12  # in TFLOPS/s
compute_bound = np.full_like(compute_intensity, maf)

# Roofline
plt.figure(figsize=(8, 5))
plt.loglog(compute_intensity, memory_bound, color='orange', linestyle='--', label=f"Memory Bound ({bw_eff} × {peak_bw} TB/s)")
plt.loglog(compute_intensity, compute_bound, color='red', linestyle='-', label=f"Compute Bound ({flops_eff} × {peak_tflops} TFLOPS/s)")
plt.scatter(df_ops["FLOPS/Byte"], df_ops["TFLOPS/s"], color='blue', label="Performance data")
plt.xlabel("Operational Intensity (FLOPs/Byte)")
plt.ylabel("Performance (TFLOPS/s)")
plt.title("Roofline Model")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_path, prefix+'_roofline.png'))
plt.show()

print(f'Outputs saved to {output_path}')

"""
References:
- TraceLens/examples/roofline_plots_example.ipynb
- JaxTrace_Analysis/attention_roofline.py
- JaxTrace_Analysis/gemm_roofline.py
"""