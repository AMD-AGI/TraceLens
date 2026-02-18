#!/bin/bash
export HIP_VISIBLE_DEVICES=1

# Jarvis Analysis - Supports both local traces and SharePoint URLs
#
# Example 2: SharePoint URLs
# python3 Analysis/jarvis_analysis.py \
#     --gpu1-kineto "https://amdcloud.sharepoint.com/:u:/r/sites/AIModelsandApps/Shared%20Documents/AIMA%20Performance%20%26%20Optimization/TL_JARVIS_TRACES/Inference%20Workloads/Lucid_Llama70B_VLLM/Llama70B_traces/B200-vs-MI355/B200/vllm_profile_1k_32_TP1_conc4_B200/vllm_profile_1k_32_TP1_conc4_B200/d41f2c8f1515_17032.1759527374101683535.pt.trace.json.gz?csf=1&web=1&e=iC8tdv" \
#     --gpu2-kineto "https://amdcloud.sharepoint.com/:u:/r/sites/AIModelsandApps/Shared%20Documents/AIMA%20Performance%20%26%20Optimization/TL_JARVIS_TRACES/Inference%20Workloads/Lucid_Llama70B_VLLM/Llama70B_traces/B200-vs-MI355/MI355/llama_70b_profile_1k_32_tp1_conc4/llama_70b_profile_1k_32_tp1_conc4/smci355-ccs-aus-m15-33_30822.1759424919398407081.pt.trace.json.gz?csf=1&web=1&e=5CrNtA" \
#     --api-key "" \
#     --output-dir "Llama70B_new_Report" \
#     --gpu1-name "B200" \
#     --gpu2-name "MI355" \
#     --generate-plots \
#     --disable-critical-path

#--gpu1-kineto "Inference_traces/Llama70B/B200/d41f2c8f1515_17032.1759527374101683535.pt.trace.json.gz" \
#--gpu2-kineto "Inference_traces/Llama70B/MI355/smci355-ccs-aus-m15-33_30822.1759424919398407081.pt.trace.json.gz" \

# Current run - using local traces
# python3 Analysis/jarvis_analysis.py \
#     --gpu1-kineto "complete_traces/crit_path_traces/H200/H200_rank0_iter16_18.json" \
#     --gpu2-kineto "complete_traces/crit_path_traces/MI300/MI300_rank0_iter16_18.json" \
#     --api-key "" \
#     --output-dir jarvis_test_plots/H200_vs_MI300_20251212_215042 \
#     --gpu1-name "H200" \
#     --gpu2-name "MI300" \
#     --generate-plots \
#     --disable-critical-path 



# BElow- actual default example

# python3 Analysis/jarvis_analysis.py \
#     --gpu1-kineto "https://amdcloud.sharepoint.com/sites/AIModelsandApps/_layouts/15/download.aspx?UniqueId=f9a7fe4efc9b4de0a5e69ece6e81316f&e=TGByWc" \
#     --gpu2-kineto "https://amdcloud.sharepoint.com/sites/AIModelsandApps/_layouts/15/download.aspx?UniqueId=1f02faaa51ad4079a1e9558dfca6dccc&e=nb5yKF" \
#     --api-key "" \
#     --output-dir "jarvis_test_plots/H200_vs_MI300_20251212_215042" \
#     --gpu1-name "H200" \
#     --gpu2-name "MI300" \
#     --generate-plots \
#     --disable-critical-path \
#     --enable-inference-phase-analysis


# Clarification: 
# baseline refers to the workload we are trying to improve.
# Target is what we are comparing the baseline to get performance projections etc. 

# Change the api-key line to have your AMD LLM gateway api key.
# Or set the env variable:
# (run in terminal) export AMD_GATEWAY_API_KEY="yourkeyhere"

python3 Analysis/jarvis_analysis_manual.py \
    --target-gpu-kineto "https://amdcloud.sharepoint.com/:u:/r/sites/AIModelsandApps/Shared%20Documents/AIMA%20Performance%20%26%20Optimization/TL_JARVIS_TRACES/Inference%20Workloads/GPT_OSS_Inferencemax/eager_traces/B200/b200_small_fulltrace_eager.trace.json.gz?csf=1&web=1&e=1Kw8c1" \
    --baseline-gpu-kineto "https://amdcloud.sharepoint.com/:u:/r/sites/AIModelsandApps/Shared%20Documents/AIMA%20Performance%20%26%20Optimization/TL_JARVIS_TRACES/Inference%20Workloads/GPT_OSS_Inferencemax/eager_traces/mi355/mi355_small_fulltrace_eager.trace.json.gz?csf=1&web=1&e=1vufwD" \
    --api-key "$AMD_GATEWAY_API_KEY" \
    --output-dir "jarvis_test_plots/DevalsTraces_B200_vs_MI355_20251212_215042" \
    --target-gpu-name "B200" \
    --baseline-gpu-name "MI355" \
    --generate-plots \
    --disable-critical-path \
    --enable-inference-phase-analysis