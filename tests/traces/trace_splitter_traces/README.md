# Multi-Iteration Traces

Traces with multiple repeating iterations, used for testing trace splitting, steady state identification, and phase classification.

## Trace Breakdown

| Directory | Workload | Model Type | Hardware | Iterations |
|---|---|---|---|---|
| `dlrm-rank5` | Training | Recommendation (DLRM) | MI300 | 6 |
| `google_owlv2_h100` | Training | Vision (OWLv2) | H100 | 5 |
| `google_owlv2_mi300` | Training | Vision (OWLv2) | MI300 | 5 |
| `owlv2-h100` | Training | Vision (OWLv2) | H100 | 5 |
| `owlv2-mi300` | Training | Vision (OWLv2) | MI300 | 5 |
| `owlv2-vision-mi300` | Inference | Vision (OWLv2) | MI300 | 5 |
| `sglang_deepseek_mi300` | LLM inference | LLM (DeepSeek) | MI300 | 6 |
| `vllm_gptoss_mi300` | LLM inference | LLM (GPT-OSS) | MI300 | 7 |

### small_traces/

Traces that are too small or simple to be splittable (NOT_SPLITTABLE), included for testing edge cases.

| Directory | Workload | Model Type | Hardware | Notes |
|---|---|---|---|---|
| `bert-small-mi300` | Training | NLP (BERT-small) | MI300 | Too few repeating patterns |
| `nsfw-detection-mi300` | Inference | Vision (NSFW classifier) | MI300 | Too few repeating patterns |
| `resnet-vision-mi300` | Training | Vision (ResNet) | MI300 | 60 roots found but low GPU coverage |
