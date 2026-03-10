import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

steps = ['Baseline', 'GEMM Kernel\nTuning']
e2e_ms = [0.144, 0.041]
savings = [0, 0.103]
cumulative_rel = [100, 351]
title = 'GEMM_0 on MI300X — Kernel Tuning Potential'
output_path = '/home/tsrikris@amd.com/TraceLens-internal/Eval_Scoping/GEMM_0/analysis_output/perf_improvement.svg'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={'width_ratios': [1.1, 1]})

colors = ['#4a90d9', '#e74c3c']
bars = ax1.bar(steps, e2e_ms, color=colors, edgecolor='white',
               linewidth=1.2, width=0.65)
for bar, val, sav in zip(bars, e2e_ms, savings):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.3f} ms', ha='center', va='bottom', fontsize=10,
             fontweight='bold')
    if sav > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                 f'-{sav:.3f} ms', ha='center', va='center',
                 fontsize=9, color='white', fontweight='bold')
ax1.set_ylabel('E2E Latency (ms)', fontsize=11)
ax1.set_title('Projected E2E Latency After Each Optimization',
              fontsize=12, fontweight='bold', pad=12)
ax1.set_ylim(0, max(e2e_ms) * 1.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', labelsize=9)

ax2.plot(range(len(steps)), cumulative_rel, 'o-', color='#2ecc71',
         linewidth=2.5, markersize=9, markerfacecolor='white',
         markeredgewidth=2.5)
for x, y in enumerate(cumulative_rel):
    ax2.annotate(f'{y}', (x, y), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=10,
                 fontweight='bold', color='#27ae60')
ax2.set_xticks(range(len(steps)))
ax2.set_xticklabels(steps, fontsize=9)
ax2.set_ylabel('Relative Throughput (baseline = 100)', fontsize=11)
ax2.set_title('Cumulative Throughput Improvement',
              fontsize=12, fontweight='bold', pad=12)
ax2.set_ylim(50, max(cumulative_rel) * 1.15)
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', labelsize=9)

fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight', facecolor='white')
print(f'Plot saved to {output_path}')
