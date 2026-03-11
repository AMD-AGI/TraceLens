###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Comparative analysis tools — run_comparative_analysis."""

import gzip
import os
import shutil
import sys
import tempfile


def _init_comparative_path():
    analysis_dir = os.path.join(
        os.path.dirname(__file__), "..", "Comparative", "Analysis"
    )
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)


def _decompress_if_needed(trace_path: str, temp_dir: str) -> str:
    """If trace is .gz, decompress to temp dir and return new path."""
    if trace_path.endswith(".gz"):
        base_name = os.path.basename(trace_path).replace(".gz", "")
        decompressed_path = os.path.join(temp_dir, base_name)
        with gzip.open(trace_path, "rb") as f_in:
            with open(decompressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return decompressed_path
    return trace_path


def run_comparative_analysis(
    gpu1_kineto: str,
    gpu2_kineto: str,
    gpu1_name: str = None,
    gpu2_name: str = None,
    cleanup: bool = True,
) -> dict:
    """Run the deterministic part of comparative analysis (no LLM).

    When cleanup=True (default), intermediate files are deleted after
    analysis and only the comparison data is returned.
    """
    _init_comparative_path()

    from jarvis_analysis import JarvisAnalyzer

    # Use temp dir for output when cleanup is enabled
    if cleanup:
        output_dir = tempfile.mkdtemp(prefix="tracelens_compare_output_")
    else:
        output_dir = "trace_reports"

    # Decompress .gz files to temp dir before passing to JarvisAnalyzer
    temp_dir = tempfile.mkdtemp(prefix="tracelens_comparative_")
    try:
        gpu1_path = _decompress_if_needed(gpu1_kineto, temp_dir)
        gpu2_path = _decompress_if_needed(gpu2_kineto, temp_dir)

        analyzer = JarvisAnalyzer(
            gpu1_kineto=gpu1_path,
            gpu1_et="",
            gpu2_kineto=gpu2_path,
            gpu2_et="",
            gpu1_name=gpu1_name,
            gpu2_name=gpu2_name,
            output_dir=output_dir,
            api_key=None,
            save_intermediates=True,
            generate_plots=False,
        )

        analyzer.run()

        result_dir = str(analyzer.output_dir)
        summary = {
            "gpu1_name": analyzer.gpu1_name,
            "gpu2_name": analyzer.gpu2_name,
        }

        # Read comparison report content before cleanup
        comparison_md = os.path.join(result_dir, "comparison_report.md")
        if os.path.exists(comparison_md):
            with open(comparison_md) as f:
                summary["comparison_markdown"] = f.read()

        # Read analysis summary if available
        analysis_md = os.path.join(result_dir, "analysis_summary.md")
        if os.path.exists(analysis_md):
            with open(analysis_md) as f:
                summary["analysis_summary"] = f.read()

        if cleanup:
            shutil.rmtree(result_dir, ignore_errors=True)
            summary["cleaned_up"] = True
        else:
            summary["output_dir"] = result_dir

        return summary
    finally:
        # Clean up temp decompressed files
        shutil.rmtree(temp_dir, ignore_errors=True)
