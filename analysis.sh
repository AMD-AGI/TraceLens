export HIP_VISIBLE_DEVICES=0
export AMD_LLM_API_KEY=''
AMD_LLM_API_KEY=''

python3 Reporting/comprehensive-analysis.py \
    --api-key $AMD_LLM_API_KEY \
    --baseline ./traces-diff-gpus/mi300_qwen_tracelens.xlsx \
    --comparison ./traces-diff-gpus/h100_qwen_tracelens.xlsx \
    --trace_diff_analysis 