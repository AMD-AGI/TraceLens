
# JSON_PATH="/shared/amdgpu/home/wen_xie_qle/workspace/Primus_zirui/output/date-20251105/user-wenx/MoE.NUM_EXPERTS-256-FP8-False.nodes8.GBS256.MBS1.PP1.EP8.CP1.VPP1.rc-64-LEGACY_GG-True-profileTrue-DisableCPUTraceFalse/tensorboard"
# python ../TraceLens/Reporting/generate_perf_report_pytorch.py \
#     --profile_json_path $JSON_PATH/mi355-gpu-11_263.1762343463864049630.pt.trace.json \
#     --output_xlsx_path  $JSON_PATH/ssi_experts_256_perf_report.xlsx

JSON_PATH="/shared/amdgpu/home/wen_xie_qle/workspace/Primus_zirui/output/date-20251105/user-wenx/MoE.NUM_EXPERTS-512-FP8-False.nodes8.GBS256.MBS1.PP1.EP8.CP1.VPP1.rc-64-LEGACY_GG-True-profileTrue-DisableCPUTraceFalse/tensorboard"
python ../TraceLens/Reporting/generate_perf_report_pytorch.py \
    --profile_json_path $JSON_PATH/mi355-gpu-11_263.1762342323885898914.pt.trace.json \
    --output_xlsx_path  $JSON_PATH/ssi_experts_512_perf_report.xlsx

JSON_PATH="/shared/amdgpu/home/wen_xie_qle/workspace/Primus_zirui/output/date-20251105/user-wenx/MoE.NUM_EXPERTS-1024-FP8-False.nodes8.GBS256.MBS1.PP1.EP8.CP1.VPP1.rc-64-LEGACY_GG-True-profileTrue-DisableCPUTraceFalse/tensorboard"
python ../TraceLens/Reporting/generate_perf_report_pytorch.py \
    --profile_json_path $JSON_PATH/mi355-gpu-11_263.1762335520015756647.pt.trace.json \
    --output_xlsx_path  $JSON_PATH/ssi_experts_1024_perf_report.xlsx