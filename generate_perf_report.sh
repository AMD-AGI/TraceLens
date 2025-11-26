python3 TraceLens/Reporting/generate_perf_report_pytorch.py \
    --profile_json_path raw_traces/eb80ba4878fa_447.1759878692401275974.pt.trace.json.gz \
    --output_xlsx_path "progress_11262025/perf_report.xlsx" \
    --enable_kernel_summary \
    --inference_phase_analysis \
    --generate_plots