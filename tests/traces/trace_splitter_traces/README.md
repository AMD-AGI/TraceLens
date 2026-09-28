<!--
Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->
# Multi-Iteration Traces

Traces with multiple repeating iterations, used for testing trace splitting, steady state identification, and phase classification.

## Trace Breakdown

| Directory | Workload | Model Type | Hardware | Iterations |
|---|---|---|---|---|
| `dlrm-rank5` | Training | Recommendation (DLRM) | MI300 | 6 |
| `google_owlv2_h100` | Training | Vision (OWLv2) | H100 | 5 |
| `google_owlv2_mi300` | Training | Vision (OWLv2) | MI300 | 5 |
| `sglang_deepseek_mi300` | LLM inference | LLM (DeepSeek) | MI300 | 6 |
| `sglang_mtp_speculative` | LLM inference | LLM (MTP speculative) | MI300 | 10 |
| `vllm_gptoss_mi300` | LLM inference | LLM (GPT-OSS) | MI300 | 7 |
| `xdit_hunyuanvideo` | Diffusion inference | Video diffusion (HunyuanVideo) | MI300 | 4 |

### small_traces/

Traces that are too small or simple to be splittable (NOT_SPLITTABLE), included for testing edge cases.

| Directory | Workload | Model Type | Hardware | Notes |
|---|---|---|---|---|
| `bert-small-mi300` | Training | NLP (BERT-small) | MI300 | Too few repeating patterns |
| `nsfw-detection-mi300` | Inference | Vision (NSFW classifier) | MI300 | Too few repeating patterns |
| `resnet-vision-mi300` | Training | Vision (ResNet) | MI300 | 60 roots found but low GPU coverage |
