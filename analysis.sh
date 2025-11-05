export HIP_VISIBLE_DEVICES=0
export AMD_LLM_API_KEY='7f8dfd4da6274efe96ff14e902724521'
AMD_LLM_API_KEY='7f8dfd4da6274efe96ff14e902724521'

python3 Reporting/comprehensive-analysis.py \
    --api-key $AMD_LLM_API_KEY \
    --baseline ./traces-diff-gpus/mi300_qwen_tracelens.xlsx \
    --comparison ./traces-diff-gpus/h100_qwen_tracelens.xlsx \
    --trace_diff_analysis 