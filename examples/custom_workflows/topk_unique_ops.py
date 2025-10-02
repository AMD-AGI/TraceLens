import argparse
from typing import Any, Dict, List

import pandas as pd
from TraceLens import TreePerfAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Generate Unique Ops View from PyTorch Profile"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input profile path (JSON/trace)"
    )
    parser.add_argument("--output", type=str, required=True, help="Output Excel path")
    parser.add_argument(
        "--topk", type=int, default=50, help="Number of top ops to save (default: 50)"
    )
    args = parser.parse_args()

    print(f"Loading profile from: {args.input}")
    perf_analyzer = TreePerfAnalyzer.from_file(args.input)
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(
        include_kernel_names=True
    )

    print(f"Generating unique ops view...")
    df_unique_args = perf_analyzer.get_df_kernel_launchers_unique_args(
        df_kernel_launchers, include_pct=True
    )

    topk = args.topk
    print(f"Saving top-{topk} ops to Excel at: {args.output}")
    df_unique_args.head(topk).to_excel(args.output, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
