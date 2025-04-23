import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import torch

from utils.hipblaslt import HipBLASLtColumns, perform_offline_tuning
from utils.reporting import FAReportColumns, GEMMReportColumns, TraceLensColumns, get_fa_config, get_fa_perf_df, get_gemm_perf_df, read_trace


logging.basicConfig(level=logging.INFO)


@dataclass
class ExperimentNames:
    BASELINE = "Baseline"
    EXPERIMENT = "Experiment"
    TUNED = "Tuned"


@st.cache_data
def hipBLASLt_tuning_possible() -> bool:
    result = subprocess.run(["which", "hipblaslt-bench"])
    return result.returncode == 0 and torch.cuda.is_available()


def analyse_trace(experiment_trace: bytes):
    experiment = read_trace(experiment_trace)
    experiment_kernels = experiment.get_df_kernel_launchers()
    experiment_gemms_kernels_summary = experiment.get_df_kernel_launchers_summary_by_shape(experiment_kernels, "aten::mm")
    return experiment, experiment_kernels, experiment_gemms_kernels_summary


@st.fragment
def download_file_button(label: str, path: str) -> None:
    if not os.path.exists(path):
        return

    st.download_button(
        label=label,
        data=open(path, "rb"),
        file_name=Path(path).name,
        icon=":material/download:",
        use_container_width=True,
    )


def main() -> None:
    title = "Trace analyzer"
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)

    col0, col1, col2, col3 = st.columns([2, 2, 1, 1])
    with col0:
        baseline_trace: Optional[bytes] = st.file_uploader(
            ExperimentNames.BASELINE,
            accept_multiple_files=False,
            type=["json", "gz"],
        )

    with col1:
        experiment_trace: Optional[bytes] = st.file_uploader(
            ExperimentNames.EXPERIMENT,
            accept_multiple_files=False,
            type=["json", "gz"],
        )
    
    with col2:
        prepare_for_tuning_chk = st.checkbox("Prepare hipBLASLt-bench file", value=False, disabled=not hipBLASLt_tuning_possible())
        options = [5, 10, 25, 50]
        if torch.cuda.is_available():
            options += [torch.cuda.device_count()]
        options.sort()
        n_kernels = st.selectbox(
            "Number of top kernels to use",
            options=options + ["All"],
            index=0,
            disabled=not prepare_for_tuning_chk,
        )
    
    with col3:
        run_tuning_chk = st.checkbox("Run offline hipBLASLt tuning", value=False, disabled=not prepare_for_tuning_chk,)
        iterations = st.selectbox(
            "Number of iterations",
            options=[100, 200, "All using '--algo_method index'"],
            index=1,
            disabled=not run_tuning_chk or True,
        )
                    
    button = st.button(
        "Analyse",
        type="primary",
        disabled=not (baseline_trace and experiment_trace),
        icon=":material/analytics:",
        use_container_width=True,
    )

    if button:
        st.toast(f'Starting {baseline_trace.name} parsing.')
        baseline, baseline_kernels, baseline_gemms_kernels_summary = analyse_trace(baseline_trace)
        baseline_kernels_summary = baseline.get_df_kernel_launchers_summary(baseline_kernels)
                
        st.toast(f'Starting {experiment_trace.name} parsing.')
        experiment, experiment_kernels, experiment_gemms_kernels_summary = analyse_trace(experiment_trace)
        experiment_kernels_summary = experiment.get_df_kernel_launchers_summary(experiment_kernels)

        baseline_summary, experiment_summary, flash_attention_tab, gemms_tab = st.tabs([
            f"{ExperimentNames.BASELINE} summary",
            f"{ExperimentNames.EXPERIMENT} summary",
            "Flash Attention",
            "GEMMs (aten::mm) comparison"
        ])

        with baseline_summary:
            st.dataframe(baseline_kernels_summary.round().astype(int, errors="ignore"))

        with experiment_summary:
            st.dataframe(experiment_kernels_summary.round().astype(int, errors="ignore"))
        
        PARITY_COL = "Parity (%)"
        with flash_attention_tab:
            config_df = None
            for df in [get_fa_config(baseline, ExperimentNames.BASELINE), get_fa_config(experiment, ExperimentNames.EXPERIMENT)]:
                if df is not None:
                    config_df = pd.concat([config_df, df], axis=0)

            if config_df is None:
                st.warning("No Flash Attention kernels found in the traces.")
            else:
                st.write("**Configuration**")
                st.dataframe(config_df)

                st.write("**Performance**")
                BASELINE_DUR_COL = f"{ExperimentNames.BASELINE} {FAReportColumns.DURATION}"
                baseline_fa_perf_df = get_fa_perf_df(baseline_kernels_summary).rename(columns={FAReportColumns.DURATION: BASELINE_DUR_COL})
                EXPERIMENT_DUR_COL = f"{ExperimentNames.EXPERIMENT} {FAReportColumns.DURATION}"
                experiment_fa_perf_df = get_fa_perf_df(experiment_kernels_summary).rename(columns={FAReportColumns.DURATION: EXPERIMENT_DUR_COL})
                
                fa_perf_df = pd.merge(baseline_fa_perf_df, experiment_fa_perf_df, on=[FAReportColumns.PASS, FAReportColumns.NAME], how="outer")
                fa_perf_df[PARITY_COL] = 100 * fa_perf_df[BASELINE_DUR_COL] / fa_perf_df[EXPERIMENT_DUR_COL]
                fa_perf_df = fa_perf_df.round().astype(int, errors="ignore")
                st.dataframe(fa_perf_df, hide_index=True)

        with gemms_tab:
            st.write("**Performance**")
            
            baseline_gemm_perf_df = get_gemm_perf_df(baseline_gemms_kernels_summary, ExperimentNames.BASELINE)
            experiment_gemm_perf_df = get_gemm_perf_df(experiment_gemms_kernels_summary, ExperimentNames.EXPERIMENT)
            tuned_gemm_perf_df = None
            merge_cols = list(GEMMReportColumns.DEFAULT_MAPPING.values())
            gemm_perf_df = pd.merge(
                baseline_gemm_perf_df,
                experiment_gemm_perf_df,
                on=merge_cols,
                how="outer"
            )

            sub_col0, sub_col1 = st.columns(2)
            if prepare_for_tuning_chk:
                kernes_to_tune = (
                    experiment_gemms_kernels_summary.sort_values(by=[TraceLensColumns.PERCENTAGE], ascending=False)[:int(n_kernels)]
                    if n_kernels != "All"
                    else experiment_gemms_kernels_summary
                )
                st.toast(f"Recording shapes for {kernes_to_tune.shape[0]} kernels")
                st.toast("Starting offline tuning")

                hipBLASLt_log, tuning_results_file, winner_solutions_df = perform_offline_tuning(
                    kernes_to_tune,
                    only_collect_log=not run_tuning_chk,
                )

                with sub_col0:
                    download_file_button("Download hipBLASLt shapes log", hipBLASLt_log)
            
                if tuning_results_file:
                    with sub_col1:
                        download_file_button("Download tuning results", tuning_results_file)
            
                    winner_solutions_df = (
                        winner_solutions_df[[HipBLASLtColumns.DURATION, HipBLASLtColumns.GFLOPS]]
                        .rename(columns={
                            HipBLASLtColumns.DURATION: GEMMReportColumns.DURATION,
                            HipBLASLtColumns.GFLOPS: GEMMReportColumns.TFLOPS,
                        })
                        .astype(float, errors="ignore")
                    )
                    winner_solutions_df[GEMMReportColumns.TFLOPS] = winner_solutions_df[GEMMReportColumns.TFLOPS] / 1e3
                    winner_solutions_df = pd.merge(
                        kernes_to_tune[list(GEMMReportColumns.DEFAULT_MAPPING.keys())],
                        winner_solutions_df,
                        left_index=True,
                        right_index=True,
                        how="outer"
                    )
                    tuned_gemm_perf_df = GEMMReportColumns.map_columns(winner_solutions_df, ExperimentNames.TUNED)
                    gemm_perf_df = pd.merge(
                        gemm_perf_df,
                        tuned_gemm_perf_df,
                        on=merge_cols,
                        how="outer"
                    )

            BASELINE_TFLOPS_COL = (ExperimentNames.BASELINE, GEMMReportColumns.TFLOPS)
            gemm_perf_df[(PARITY_COL, "with experiment")] = (
                100 * gemm_perf_df[(ExperimentNames.EXPERIMENT, GEMMReportColumns.TFLOPS)] / gemm_perf_df[BASELINE_TFLOPS_COL]
            )
            if tuned_gemm_perf_df is not None:
                gemm_perf_df[(PARITY_COL, "with tuned")] = (
                    100 * gemm_perf_df[(ExperimentNames.TUNED, GEMMReportColumns.TFLOPS)] / gemm_perf_df[BASELINE_TFLOPS_COL]
                )

            gemm_perf_df = (
                gemm_perf_df
                .sort_values(by=[(ExperimentNames.BASELINE, TraceLensColumns.PERCENTAGE)], ascending=False)
                .round()
                .astype(int, errors="ignore")
            )
            st.dataframe(gemm_perf_df, hide_index=True)


if __name__ == "__main__":
    main()
