import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TraceLens import TreePerfAnalyzer

path = '/path/to/trace.json'

perf_analyzer = TreePerfAnalyzer.from_file(path)
events = [event for event in perf_analyzer.tree.events if event['name'] == 'aten::copy_']
df_ops = perf_analyzer.build_df_perf_metrics(events, bwd=False)

# config
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


df_roofline = pd.DataFrame({
    "Compute Intensity (FLOPs/Byte)": df_ops["FLOPS/Byte"],
    "Performance (TFLOPS/s)": df_ops["TFLOPS/s"]
})

# Plotting
plt.figure(figsize=(8, 5))
plt.loglog(compute_intensity, memory_bound, color='orange', linestyle='--',
           label=f"Memory Bound ({bw_eff} × {peak_bw} TB/s)")
plt.loglog(compute_intensity, compute_bound, color='red', linestyle='-',
           label=f"Compute Bound ({flops_eff} × {peak_tflops} TFLOPS/s)")

plt.scatter(df_roofline["Compute Intensity (FLOPs/Byte)"],
            df_roofline["Performance (TFLOPS/s)"],
            color='blue', label="Performance data")

plt.xlabel("Compute Intensity (FLOPs/Byte)")
plt.ylabel("Performance (TFLOPS/s)")
plt.title("Roofline Model")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df_ops['Data Moved (MB)'], df_ops['TB/s'], alpha=0.7)

# Add horizontal line for max bandwidth
plt.axhline(y=mabw, color='red', linestyle='--', linewidth=1.2, label=f"Max Achievable BW: {bw_eff} × {peak_bw} TB/s")

# Set labels and title
plt.xlabel("Data Moved (MB)", fontsize=12)
plt.ylabel("Bandwidth (TB/s)", fontsize=12)
plt.title("Bandwidth vs Data Moved", fontsize=14)

# Set log scale for x-axis
plt.xscale('log', base=2)

# Show grid for better readability
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Add legend
plt.legend()

# Show the plot
plt.show()


"""
References:
- TraceLens/examples/roofline_plots_example.ipynb
- JaxTrace_Analysis/attention_roofline.py
- JaxTrace_Analysis/gemm_roofline.py
"""