#!/usr/bin/env python3
"""
TraceLens Jarvis - Multi-Kernel Issue Analysis
Analyzes cross-cutting multi-kernel patterns: memcpy (D2H/H2D), NCCL blocking,
and compute/communication overlap deficiency.

Reads pre-computed multi_kernel_data.json from orchestrator_prepare.py.
Outputs multi_kernel_metrics.json with severity assessments.
"""

import json
import os
import argparse
import pandas as pd


def classify_memcpy_severity(memcpy_summary, total_time_ms):
    """Classify memcpy severity based on count and time thresholds."""
    total_memcpy_time_ms = memcpy_summary.get("total_time_us", 0) / 1000
    total_count = memcpy_summary.get("total_count", 0)
    
    if total_time_ms <= 0:
        return {"severity": "NONE", "details": "No timeline data available"}
    
    memcpy_pct = (total_memcpy_time_ms / total_time_ms) * 100
    
    issues = []
    overall_severity = "NONE"
    
    # Check per-direction issues
    by_direction = memcpy_summary.get("by_direction", {})
    
    for direction in ["D2H", "H2D"]:
        dir_info = by_direction.get(direction, {})
        count = dir_info.get("count", 0)
        dir_time_ms = dir_info.get("total_time_us", 0) / 1000
        dir_pct = (dir_time_ms / total_time_ms) * 100 if total_time_ms > 0 else 0
        
        if count == 0:
            continue
        
        # Severity by time percentage
        if dir_pct > 10:
            severity = "CRITICAL"
        elif dir_pct > 5:
            severity = "HIGH"
        elif dir_pct > 2:
            severity = "MEDIUM"
        elif count > 50:
            severity = "MEDIUM"
        elif count > 10:
            severity = "LOW"
        else:
            severity = "NONE"
        
        if severity != "NONE":
            issues.append({
                "direction": direction,
                "count": count,
                "time_ms": round(dir_time_ms, 3),
                "percent_of_total": round(dir_pct, 2),
                "avg_bytes": dir_info.get("avg_bytes", 0),
                "severity": severity,
                "description": f"{direction} memcpy: {count} transfers, {dir_time_ms:.3f}ms ({dir_pct:.2f}% of total)"
            })
            
            # Track highest severity
            sev_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
            if sev_rank.get(severity, 0) > sev_rank.get(overall_severity, 0):
                overall_severity = severity
    
    return {
        "severity": overall_severity,
        "total_count": total_count,
        "total_time_ms": round(total_memcpy_time_ms, 3),
        "percent_of_total": round(memcpy_pct, 2),
        "issues": issues
    }


def classify_nccl_blocking_severity(overlap_analysis):
    """Classify NCCL blocking compute severity."""
    exposed_comm_pct = overlap_analysis.get("comm_percent_of_total", 0)
    total_comm_us = overlap_analysis.get("total_comm_time_us", 0)
    exposed_comm_us = overlap_analysis.get("exposed_comm_time_us", 0)
    
    if total_comm_us == 0:
        return {"severity": "NONE", "details": "No communication events detected"}
    
    if exposed_comm_pct > 20:
        severity = "CRITICAL"
    elif exposed_comm_pct > 10:
        severity = "HIGH"
    elif exposed_comm_pct > 5:
        severity = "MEDIUM"
    elif exposed_comm_pct > 2:
        severity = "LOW"
    else:
        severity = "NONE"
    
    return {
        "severity": severity,
        "exposed_comm_time_ms": round(exposed_comm_us / 1000, 3),
        "total_comm_time_ms": round(total_comm_us / 1000, 3),
        "exposed_percent_of_total": round(exposed_comm_pct, 2),
        "description": f"Exposed (non-overlapped) communication: {exposed_comm_pct:.1f}% of total GPU time"
    }


def classify_overlap_severity(overlap_analysis):
    """Classify NCCL/compute overlap quality."""
    comm_overlap_ratio = overlap_analysis.get("comm_overlap_ratio")
    total_comm_us = overlap_analysis.get("total_comm_time_us", 0)
    
    if comm_overlap_ratio is None or total_comm_us < 100:
        return {"severity": "NONE", "details": "Insufficient communication data for overlap analysis"}
    
    if comm_overlap_ratio < 0.3:
        severity = "CRITICAL"
    elif comm_overlap_ratio < 0.5:
        severity = "HIGH"
    elif comm_overlap_ratio < 0.7:
        severity = "MEDIUM"
    else:
        severity = "NONE"
    
    return {
        "severity": severity,
        "overlap_ratio": round(comm_overlap_ratio, 4),
        "overlap_percent": round(comm_overlap_ratio * 100, 1),
        "target_percent": 70,
        "description": f"Compute/communication overlap: {comm_overlap_ratio:.1%} (target > 70%)"
    }


def cross_validate_with_timeline(overlap_analysis, gpu_timeline_df, tolerance_pct=2.0):
    """Cross-validate overlap metrics from multi_kernel_data.json against gpu_timeline.csv.
    
    Returns a dict with validation results: matched fields, discrepancies, and overall status.
    tolerance_pct: maximum acceptable percentage-point difference before flagging a discrepancy.
    """
    validation = {"status": "PASS", "checks": [], "warnings": []}
    
    # Build a lookup from gpu_timeline.csv rows
    timeline_values = {}
    for _, row in gpu_timeline_df.iterrows():
        timeline_values[row["type"]] = {
            "time_ms": row.get("time ms", 0),
            "percent": row.get("percent", 0),
        }
    
    # Map overlap_analysis keys → gpu_timeline.csv type names
    cross_checks = [
        ("computation_time_us", "computation_time", "Computation time"),
        ("exposed_comm_time_us", "exposed_comm_time", "Exposed communication time"),
        ("exposed_memcpy_time_us", "exposed_memcpy_time", "Exposed memcpy time"),
        ("total_time_us", "total_time", "Total GPU time"),
    ]
    
    for mk_key, tl_type, label in cross_checks:
        mk_val_ms = overlap_analysis.get(mk_key, 0) / 1000  # convert us → ms
        tl_entry = timeline_values.get(tl_type)
        
        if tl_entry is None:
            validation["checks"].append({
                "metric": label, "status": "SKIP",
                "reason": f"{tl_type} not found in gpu_timeline.csv"
            })
            continue
        
        tl_val_ms = tl_entry["time_ms"]
        
        if tl_val_ms == 0 and mk_val_ms == 0:
            validation["checks"].append({
                "metric": label, "status": "PASS",
                "mk_ms": 0, "tl_ms": 0, "diff_ms": 0
            })
            continue
        
        diff_ms = abs(mk_val_ms - tl_val_ms)
        ref = max(mk_val_ms, tl_val_ms, 1e-6)
        diff_pct = (diff_ms / ref) * 100
        
        check = {
            "metric": label,
            "mk_ms": round(mk_val_ms, 3),
            "tl_ms": round(tl_val_ms, 3),
            "diff_ms": round(diff_ms, 3),
            "diff_pct": round(diff_pct, 2),
        }
        
        if diff_pct <= tolerance_pct:
            check["status"] = "PASS"
        else:
            check["status"] = "WARN"
            validation["status"] = "WARN"
            validation["warnings"].append(
                f"{label}: multi_kernel_data={mk_val_ms:.3f}ms vs "
                f"gpu_timeline={tl_val_ms:.3f}ms (diff {diff_pct:.1f}%)"
            )
        
        validation["checks"].append(check)
    
    return validation


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-kernel issues from pre-computed data')
    parser.add_argument('--output-dir', required=True, help='Output directory with pre-computed data')
    
    args = parser.parse_args()
    output_dir = args.output_dir
    
    print("=" * 80)
    print("MULTI-KERNEL ISSUE ANALYSIS")
    print("=" * 80)
    
    # Read pre-computed multi-kernel data
    data_file = f"{output_dir}/category_data/multi_kernel_data.json"
    if not os.path.exists(data_file):
        print(f"  ⚠️  multi_kernel_data.json not found at {data_file}")
        # Write error metrics
        error_metrics = {
            "status": "ERROR",
            "error": "multi_kernel_data.json not found - run orchestrator_prepare.py first",
            "memcpy_assessment": {"severity": "UNKNOWN"},
            "nccl_blocking_assessment": {"severity": "UNKNOWN"},
            "overlap_assessment": {"severity": "UNKNOWN"},
            "patterns_detected": [],
        }
        metrics_file = f"{output_dir}/category_data/multi_kernel_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(error_metrics, f, indent=2)
        return
    
    with open(data_file, 'r') as f:
        mk_data = json.load(f)
    
    print(f"  ✓ Loaded multi-kernel data")
    
    # Read GPU timeline for total time reference
    csv_dir = f"{output_dir}/perf_report_csvs"
    total_time_ms = 0
    try:
        gpu_timeline = pd.read_csv(f'{csv_dir}/gpu_timeline.csv')
        for _, row in gpu_timeline.iterrows():
            if row['type'] == 'total_time':
                total_time_ms = row['time ms']
                break
    except Exception as e:
        print(f"  ⚠️  Could not read gpu_timeline.csv: {e}")
        # Fallback to overlap_analysis total time
        total_time_ms = mk_data.get("overlap_analysis", {}).get("total_time_us", 0) / 1000
    
    print(f"  Total GPU time: {total_time_ms:.2f} ms")
    
    # Classify severities
    memcpy_summary = mk_data.get("memcpy_summary", {})
    overlap_analysis = mk_data.get("overlap_analysis", {})
    nccl_summary = mk_data.get("nccl_summary", {})
    
    memcpy_assessment = classify_memcpy_severity(memcpy_summary, total_time_ms)
    nccl_blocking_assessment = classify_nccl_blocking_severity(overlap_analysis)
    overlap_assessment = classify_overlap_severity(overlap_analysis)
    
    print(f"\n  Memcpy Assessment: {memcpy_assessment['severity']}")
    print(f"  NCCL Blocking Assessment: {nccl_blocking_assessment['severity']}")
    print(f"  Overlap Assessment: {overlap_assessment['severity']}")
    
    # Cross-validate overlap metrics against gpu_timeline.csv
    cross_validation = None
    try:
        gpu_timeline_cv = pd.read_csv(f'{csv_dir}/gpu_timeline.csv')
        cross_validation = cross_validate_with_timeline(overlap_analysis, gpu_timeline_cv)
        if cross_validation["status"] == "PASS":
            print(f"\n  ✓ Cross-validation: PASS (overlap metrics consistent with gpu_timeline.csv)")
        else:
            print(f"\n  ⚠️  Cross-validation: WARN")
            for w in cross_validation.get("warnings", []):
                print(f"    - {w}")
    except Exception as e:
        print(f"\n  ⚠️  Cross-validation skipped: {e}")
    
    # Build patterns_detected list (pattern name + severity only;
    # recommendations are the sub-agent's responsibility)
    patterns_detected = []
    
    if memcpy_assessment["severity"] != "NONE":
        for issue in memcpy_assessment.get("issues", []):
            patterns_detected.append({
                "pattern": f"high_{issue['direction'].lower()}_memcpy",
                "severity": issue["severity"],
                "description": issue["description"]
            })
    
    if nccl_blocking_assessment["severity"] != "NONE":
        patterns_detected.append({
            "pattern": "nccl_blocking_compute",
            "severity": nccl_blocking_assessment["severity"],
            "description": nccl_blocking_assessment.get("description", "")
        })
    
    if overlap_assessment["severity"] != "NONE":
        patterns_detected.append({
            "pattern": "poor_comm_compute_overlap",
            "severity": overlap_assessment["severity"],
            "description": overlap_assessment.get("description", "")
        })
    
    print(f"  Patterns detected: {len(patterns_detected)}")
    for p in patterns_detected:
        print(f"    - [{p['severity']}] {p['pattern']}")
    
    # Build output metrics (severity assessments and patterns only;
    # the sub-agent interprets these and generates recommendations)
    metrics = {
        "status": "SUCCESS",
        "total_time_ms": round(total_time_ms, 3),
        "memcpy_summary": {
            "total_count": memcpy_summary.get("total_count", 0),
            "total_time_ms": round(memcpy_summary.get("total_time_us", 0) / 1000, 3),
            "by_direction": memcpy_summary.get("by_direction", {})
        },
        "nccl_summary": {
            "total_count": nccl_summary.get("total_count", 0),
            "total_time_ms": round(nccl_summary.get("total_time_us", 0) / 1000, 3),
            "top_ops": nccl_summary.get("top_ops", [])
        },
        "overlap_analysis": overlap_analysis,
        "memcpy_assessment": memcpy_assessment,
        "nccl_blocking_assessment": nccl_blocking_assessment,
        "overlap_assessment": overlap_assessment,
        "patterns_detected": patterns_detected,
        "cross_validation": cross_validation,
    }
    
    # Write metrics output
    metrics_file = f"{output_dir}/category_data/multi_kernel_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  ✓ Wrote multi_kernel_metrics.json")
    print(f"  ✓ {len(patterns_detected)} patterns detected")
    print("=" * 80)


if __name__ == "__main__":
    main()
