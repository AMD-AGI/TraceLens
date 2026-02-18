#!/usr/bin/env python3
"""
CPU/Idle Time Analysis Script

Analyzes GPU idle time patterns to identify CPU overhead causes:
- Kernel launch overhead (many small kernels)
- Synchronization bottlenecks
- CPU-GPU pipeline bubbles
- Framework overhead

Outputs cpu_idle_metrics.json with analysis results.
"""

import argparse
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional


def load_gpu_timeline(output_dir: str) -> Dict[str, float]:
    """Load GPU timeline data from CSV."""
    csv_path = f'{output_dir}/perf_report_csvs/gpu_timeline.csv'
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GPU timeline not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    timeline = {}
    for _, row in df.iterrows():
        timeline[row['type']] = {
            'time_ms': row['time ms'],
            'percent': row['percent']
        }
    
    return timeline


def load_ops_summary(output_dir: str) -> Optional[pd.DataFrame]:
    """Load operations summary for kernel analysis."""
    csv_path = f'{output_dir}/perf_report_csvs/ops_summary.csv'
    
    if not os.path.exists(csv_path):
        return None
    
    return pd.read_csv(csv_path)


def load_manifest(output_dir: str) -> Dict:
    """Load category manifest for metadata."""
    manifest_path = f'{output_dir}/category_data/category_manifest.json'
    
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    return {}


def analyze_kernel_patterns(ops_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Analyze kernel patterns to identify overhead sources."""
    patterns = {
        'short_kernel_count': 0,
        'total_kernel_count': 0,
        'avg_kernel_time_us': 0,
        'kernel_count_by_category': {}
    }
    
    if ops_df is None or ops_df.empty:
        return patterns
    
    # Count kernels and analyze times
    if 'Kernel Time (µs)_sum' in ops_df.columns and 'Count' in ops_df.columns:
        patterns['total_kernel_count'] = int(ops_df['Count'].sum())
        
        total_time = ops_df['Kernel Time (µs)_sum'].sum()
        if patterns['total_kernel_count'] > 0:
            patterns['avg_kernel_time_us'] = total_time / patterns['total_kernel_count']
        
        # Count short kernels (< 10µs average)
        if 'Kernel Time (µs)_mean' in ops_df.columns:
            short_ops = ops_df[ops_df['Kernel Time (µs)_mean'] < 10]
            patterns['short_kernel_count'] = int(short_ops['Count'].sum()) if not short_ops.empty else 0
    
    # Analyze by category
    if 'op category' in ops_df.columns and 'Count' in ops_df.columns:
        category_counts = ops_df.groupby('op category')['Count'].sum().to_dict()
        patterns['kernel_count_by_category'] = {k: int(v) for k, v in category_counts.items()}
    
    return patterns


def detect_idle_patterns(
    gpu_timeline: Dict[str, Any],
    kernel_patterns: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Detect patterns causing idle time."""
    patterns_detected = []
    
    idle_percent = gpu_timeline.get('idle_time', {}).get('percent', 0)
    total_kernels = kernel_patterns.get('total_kernel_count', 0)
    short_kernels = kernel_patterns.get('short_kernel_count', 0)
    avg_kernel_time = kernel_patterns.get('avg_kernel_time_us', 0)
    
    # Pattern 1: Kernel Launch Overhead
    if total_kernels > 0 and short_kernels > 0:
        short_kernel_ratio = short_kernels / total_kernels
        if short_kernel_ratio > 0.3:  # More than 30% short kernels
            patterns_detected.append({
                'name': 'Kernel Launch Overhead',
                'severity': 'HIGH' if short_kernel_ratio > 0.5 else 'MEDIUM',
                'evidence': f'{short_kernels} out of {total_kernels} kernels ({short_kernel_ratio*100:.1f}%) are under 10µs',
                'impact': f'Launch overhead dominates execution for {short_kernel_ratio*100:.1f}% of kernels',
                'solution': 'Enable GPU graph mode to batch kernel launches'
            })
    
    # Pattern 2: High Kernel Count
    if total_kernels > 1000:
        patterns_detected.append({
            'name': 'High Kernel Count',
            'severity': 'HIGH' if total_kernels > 5000 else 'MEDIUM',
            'evidence': f'{total_kernels} total kernel launches',
            'impact': 'Each launch has fixed CPU overhead (~5-10µs)',
            'solution': 'Use GPU graph mode or kernel fusion to reduce launch count'
        })
    
    # Pattern 3: Small Average Kernel Time
    if avg_kernel_time > 0 and avg_kernel_time < 50:
        patterns_detected.append({
            'name': 'Small Average Kernel Time',
            'severity': 'HIGH' if avg_kernel_time < 20 else 'MEDIUM',
            'evidence': f'Average kernel time: {avg_kernel_time:.1f}µs',
            'impact': 'CPU launch overhead is significant relative to kernel execution',
            'solution': 'Fuse operations or use compilation to create larger kernels'
        })
    
    # Pattern 4: Very High Idle Time (general)
    if idle_percent > 70:
        patterns_detected.append({
            'name': 'Critical GPU Underutilization',
            'severity': 'CRITICAL',
            'evidence': f'GPU idle {idle_percent:.1f}% of total time',
            'impact': f'Only {100-idle_percent:.1f}% of GPU capacity is being used',
            'solution': 'Enable GPU graph mode, use torch.compile, or increase batch size'
        })
    elif idle_percent > 50:
        patterns_detected.append({
            'name': 'High GPU Idle Time',
            'severity': 'HIGH',
            'evidence': f'GPU idle {idle_percent:.1f}% of total time',
            'impact': f'Significant opportunity to improve throughput',
            'solution': 'Enable GPU graph mode or torch.compile'
        })
    
    return patterns_detected


def classify_severity(idle_percent: float) -> str:
    """Classify idle time severity."""
    if idle_percent > 70:
        return 'CRITICAL'
    elif idle_percent > 50:
        return 'HIGH'
    elif idle_percent > 30:
        return 'MEDIUM'
    elif idle_percent > 20:
        return 'LOW'
    else:
        return 'ACCEPTABLE'


def main():
    parser = argparse.ArgumentParser(description='CPU/Idle Time Analysis')
    parser.add_argument('--output-dir', required=True, help='Analysis output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    print("=" * 80)
    print("CPU/IDLE TIME ANALYSIS")
    print("=" * 80)
    
    try:
        # Load data
        gpu_timeline = load_gpu_timeline(output_dir)
        ops_df = load_ops_summary(output_dir)
        manifest = load_manifest(output_dir)
        
        # Extract key metrics
        idle_time = gpu_timeline.get('idle_time', {})
        idle_percent = idle_time.get('percent', 0)
        idle_ms = idle_time.get('time_ms', 0)
        total_time_ms = gpu_timeline.get('total_time', {}).get('time_ms', 0)
        
        computation_percent = gpu_timeline.get('computation_time', {}).get('percent', 0)
        comm_percent = gpu_timeline.get('exposed_comm_time', {}).get('percent', 0)
        memcpy_percent = gpu_timeline.get('exposed_memcpy_time', {}).get('percent', 0)
        
        print(f"\nGPU Utilization Breakdown:")
        print(f"  Total Time: {total_time_ms:.2f} ms")
        print(f"  Computation: {computation_percent:.2f}%")
        print(f"  Communication: {comm_percent:.2f}%")
        print(f"  MemCpy: {memcpy_percent:.2f}%")
        print(f"  Idle: {idle_percent:.2f}%")
        
        # Analyze kernel patterns
        kernel_patterns = analyze_kernel_patterns(ops_df)
        print(f"\nKernel Analysis:")
        print(f"  Total Kernels: {kernel_patterns['total_kernel_count']}")
        print(f"  Short Kernels (<10µs): {kernel_patterns['short_kernel_count']}")
        print(f"  Avg Kernel Time: {kernel_patterns['avg_kernel_time_us']:.1f} µs")
        
        # Detect patterns (severity + evidence only;
        # recommendations are the sub-agent's responsibility)
        patterns_detected = detect_idle_patterns(gpu_timeline, kernel_patterns)
        print(f"\nPatterns Detected: {len(patterns_detected)}")
        for p in patterns_detected:
            print(f"  - [{p['severity']}] {p['name']}")
        
        # Classify severity
        severity = classify_severity(idle_percent)
        
        # Calculate potential speedup
        if idle_percent > 20:
            target_idle = 20
            current_throughput = 100 - idle_percent
            target_throughput = 100 - target_idle
            potential_speedup = target_throughput / current_throughput if current_throughput > 0 else 1
        else:
            potential_speedup = 1.0
        
        # Build metrics output
        metrics = {
            'status': 'OK',
            'severity': severity,
            'gpu_utilization': {
                'total_time_ms': round(total_time_ms, 3),
                'idle_time_ms': round(idle_ms, 3),
                'idle_time_percent': round(idle_percent, 2),
                'computation_percent': round(computation_percent, 2),
                'communication_percent': round(comm_percent, 2),
                'memcpy_percent': round(memcpy_percent, 2)
            },
            'kernel_analysis': kernel_patterns,
            'patterns_detected': patterns_detected,
            'potential_speedup': round(potential_speedup, 2),
            'target_idle_percent': 20
        }
        
        # Write metrics JSON
        os.makedirs(f'{output_dir}/category_data', exist_ok=True)
        metrics_path = f'{output_dir}/category_data/cpu_idle_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Metrics saved: {metrics_path}")
        print(f"\nSeverity: {severity}")
        print(f"Potential Speedup: {potential_speedup:.2f}x")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        
        # Write error metrics
        error_metrics = {
            'status': 'ERROR',
            'error': str(e),
            'severity': 'UNKNOWN'
        }
        
        os.makedirs(f'{output_dir}/category_data', exist_ok=True)
        with open(f'{output_dir}/category_data/cpu_idle_metrics.json', 'w') as f:
            json.dump(error_metrics, f, indent=2)
        
        raise


if __name__ == '__main__':
    main()
