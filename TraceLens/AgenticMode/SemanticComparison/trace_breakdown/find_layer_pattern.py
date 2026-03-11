#!/usr/bin/env python3
"""
Step 4 (scriptable part): Detect the repeating layer pattern in a kernel sequence.

Uses autocorrelation on kernel name signatures to find the dominant period.
Also detects if the period alternates (e.g., 16/17/16/17 for MI355).

Input: extracted trace data JSON (from extract_trace_data.py)
Output: JSON with pattern analysis results

Usage:
    python find_layer_pattern.py <extracted.json> [-o pattern.json]
"""
import argparse
import json
import sys
from collections import Counter


def kernel_signature(name, sig_len=50):
    """Hash a kernel name to a short signature for pattern matching."""
    return name[:sig_len]


def find_best_period(kernel_names, min_period=10, max_period=30):
    """Find the repeating period by autocorrelation on kernel name signatures."""
    sigs = [kernel_signature(n) for n in kernel_names]
    sig_to_id = {}
    ids = []
    for s in sigs:
        if s not in sig_to_id:
            sig_to_id[s] = len(sig_to_id)
        ids.append(sig_to_id[s])

    n = len(ids)
    results = []
    for period in range(min_period, min(max_period + 1, n)):
        matches = sum(1 for i in range(n - period) if ids[i] == ids[i + period])
        comparisons = n - period
        score = matches / comparisons if comparisons > 0 else 0
        results.append((period, score))

    results.sort(key=lambda x: -x[1])
    return results


def detect_alternating_periods(kernel_names, p1, p2, start_offset=0):
    """Check if the pattern alternates between two periods (e.g., 16 and 17)."""
    sigs = [kernel_signature(n) for n in kernel_names]
    n = len(sigs)

    idx = start_offset
    layers = []
    while idx + min(p1, p2) <= n:
        match_p1 = 0
        match_p2 = 0
        if idx + p1 + p1 <= n:
            match_p1 = sum(1 for j in range(p1) if sigs[idx + j] == sigs[idx + p1 + j])
        if idx + p2 + p2 <= n:
            match_p2 = sum(1 for j in range(p2) if sigs[idx + j] == sigs[idx + p2 + j])
        if idx + p1 + p2 <= n:
            match_p1_then_p2 = sum(1 for j in range(min(p1, p2))
                                   if j < len(sigs) - idx - p1
                                   and sigs[idx + j] == sigs[idx + p1 + j])
        else:
            match_p1_then_p2 = 0

        if match_p1 >= match_p2:
            layers.append(p1)
            idx += p1
        else:
            layers.append(p2)
            idx += p2

        if len(layers) > 200:
            break

    return layers


def find_preamble_boundary(kernel_names, period, max_preamble=10):
    """Find where the repeating block starts by checking autocorrelation from different offsets."""
    best_offset = 0
    best_score = 0
    sigs = [kernel_signature(n) for n in kernel_names]
    n = len(sigs)

    for offset in range(0, min(max_preamble, n - period * 2)):
        matches = 0
        count = 0
        for i in range(offset, min(offset + period * 5, n - period)):
            if sigs[i] == sigs[i + period]:
                matches += 1
            count += 1
        score = matches / count if count > 0 else 0
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset, best_score


def run_assertions(kernel_names, analysis):
    errors = []

    if analysis["best_period_score"] < 0.5:
        errors.append(
            f"A4.1 WARNING: Low autocorrelation score ({analysis['best_period_score']:.2f}), "
            "pattern may not be a standard repeating transformer"
        )

    n_layers = analysis["estimated_layers"]
    if n_layers < 1 or n_layers > 200:
        errors.append(f"A4.2 FAIL: Unreasonable layer count: {n_layers}")

    total_accounted = (analysis["preamble_size"]
                       + sum(analysis["kernels_per_layer"])
                       + analysis["epilogue_size"])
    if total_accounted != len(kernel_names):
        errors.append(
            f"A4.3 FAIL: Kernel accounting mismatch: "
            f"{total_accounted} accounted vs {len(kernel_names)} total "
            f"(preamble={analysis['preamble_size']}, "
            f"layers={sum(analysis['kernels_per_layer'])}, "
            f"epilogue={analysis['epilogue_size']})"
        )

    return errors


def analyze(extracted_data):
    kernels = extracted_data["kernels"]
    kernel_names = [k["name"] for k in kernels]
    n = len(kernel_names)

    periods = find_best_period(kernel_names)
    if not periods:
        return {"error": "Could not find any repeating pattern"}

    top_periods = periods[:5]
    best_period = top_periods[0][0]
    best_score = top_periods[0][1]

    nearby = [p for p, s in top_periods if abs(p - best_period) <= 2 and p != best_period]
    is_alternating = len(nearby) > 0

    preamble_offset, preamble_score = find_preamble_boundary(kernel_names, best_period)

    if is_alternating:
        alt_period = nearby[0]
        p_small, p_large = sorted([best_period, alt_period])
        layers_seq = detect_alternating_periods(
            kernel_names[preamble_offset:], p_small, p_large
        )
        consumed = sum(layers_seq)
        remaining = n - preamble_offset - consumed
        if remaining < 0:
            while layers_seq and sum(layers_seq) + preamble_offset > n:
                layers_seq.pop()
            remaining = n - preamble_offset - sum(layers_seq)
    else:
        body = n - preamble_offset
        est_layers = body // best_period
        remaining = body - est_layers * best_period
        if remaining > best_period // 2:
            est_layers += 1
            remaining = body - est_layers * best_period
        layers_seq = [best_period] * est_layers

    total_layer_kernels = sum(layers_seq)
    epilogue_size = n - preamble_offset - total_layer_kernels
    if epilogue_size < 0:
        while layers_seq and epilogue_size < 0:
            layers_seq.pop()
            total_layer_kernels = sum(layers_seq)
            epilogue_size = n - preamble_offset - total_layer_kernels

    period_counts = Counter(layers_seq)
    unique_kernel_names_in_body = set()
    for name in kernel_names[preamble_offset:preamble_offset + total_layer_kernels]:
        unique_kernel_names_in_body.add(kernel_signature(name))

    analysis = {
        "total_kernels": n,
        "best_period": best_period,
        "best_period_score": round(best_score, 4),
        "is_alternating_period": is_alternating,
        "period_variants": dict(period_counts),
        "top_5_periods": [{"period": p, "score": round(s, 4)} for p, s in top_periods],
        "preamble_size": preamble_offset,
        "preamble_kernel_names": kernel_names[:preamble_offset],
        "estimated_layers": len(layers_seq),
        "kernels_per_layer": layers_seq,
        "epilogue_size": epilogue_size,
        "epilogue_kernel_names": kernel_names[n - epilogue_size:] if epilogue_size > 0 else [],
        "unique_body_signatures": len(unique_kernel_names_in_body),
        "first_layer_kernel_names": kernel_names[preamble_offset:preamble_offset + layers_seq[0]] if layers_seq else [],
    }

    errors = run_assertions(kernel_names, analysis)
    analysis["assertions"] = errors

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Find repeating layer pattern in kernel sequence")
    parser.add_argument("extracted_json", help="Path to extracted trace data JSON")
    parser.add_argument("-o", "--output", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    with open(args.extracted_json) as f:
        extracted = json.load(f)

    analysis = analyze(extracted)

    for a in analysis.get("assertions", []):
        print(a, file=sys.stderr)

    output = json.dumps(analysis, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(
            f"Pattern: period={analysis['best_period']} "
            f"(score={analysis['best_period_score']:.2f}), "
            f"layers={analysis['estimated_layers']}, "
            f"preamble={analysis['preamble_size']}, "
            f"epilogue={analysis['epilogue_size']}",
            file=sys.stderr,
        )
    else:
        print(output)


if __name__ == "__main__":
    main()
