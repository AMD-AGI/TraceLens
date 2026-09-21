<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Inference performance analysis in TraceLens
```{meta}
:description: Understand the concepts behind TraceLens inference analysis - the roofline model for paged attention (prefill and decode), and the steady-state region that trace splitting targets.
:keywords: TraceLens, inference, roofline, paged attention, prefill, decode, chunked prefill, FLOPS, arithmetic intensity, steady state, vLLM, SGLang, ATOM, LLM serving, ROCm, MI300X
```

This topic explains the concepts behind TraceLens's inference-serving analysis:
how the *roofline model* is derived for paged attention, and what the
*steady-state region* is and why it is the part of a serving trace worth
profiling. For the corresponding procedures, see
[Generate a PyTorch inference performance report](../how-to/generate-perf-report-pytorch-inference.md).

In inference serving, multiple requests are batched together and each request
carries its own query and key/value sequence lengths. A single execution step
typically mixes *prefill* (context) and *decode* (generation) requests, which
makes per-request accounting the natural basis for a roofline model.

## Roofline analysis for inference attention

The following notation is used throughout this section.

| Symbol | Description |
|--------|-------------|
| B | Batch size (1 per request in paged attention) |
| N_Q | Number of query tokens |
| N_KV | Number of key/value tokens (context length) |
| H_Q | Number of query heads |
| H_KV | Number of KV heads (H_KV ≤ H_Q; equal for MHA, smaller for GQA/MQA) |
| d_h_qk | Head dimension for queries and keys |
| d_h_v | Head dimension for values |
| R_C | Number of context (prefill) requests in the batch |
| R_G | Number of generation (decode) requests in the batch |
| R | Total number of requests (R = R_C + R_G) |
| N_Q(i), N_KV(i) | Query and KV token counts for the i-th request |

### Standard SDPA attention (single request)

Scaled dot-product attention consists of two matrix multiplications per head:

1. *QK^T (score computation)*: `2 * B * N_Q * N_KV * H_Q * d_h_qk`
2. *Score × V (value aggregation)*: `2 * B * N_Q * N_KV * H_Q * d_h_v`

For causal attention, roughly half the score matrix is masked out:

```
FLOPS = (2 * B * N_Q * N_KV * H_Q * d_h_qk + 2 * B * N_Q * N_KV * H_Q * d_h_v) / 2
```

```
Elements Moved =
Q:      B * N_Q  * H_Q  * d_h_qk
K:      B * N_KV * H_KV * d_h_qk
V:      B * N_KV * H_KV * d_h_v
Output: B * N_Q  * H_Q  * d_h_v
```

### Paged attention (multiple requests)

For total FLOPS and bytes moved in inference paged attention, *sum over the
compute requirement of every request individually* (B = 1 per request). Requests
fall into two categories:

- **Context (prefill) requests** process input tokens; attention is causal within
  the current chunk.
- **Generation (decode) requests** generate new tokens; attention is non-causal
  (queries attend to all past KV tokens). Typically N_Q = 1, but approaches like
  speculative decoding might produce multiple query tokens per request.

#### FLOPS: prefill (context) requests

When chunked prefill is not enabled, the first (and only) chunk has N_KV = N_Q
and attention is causal:

```
                  R_C
FLOPS_prefill  =   Σ   (2 * N_Q(i) * N_KV(i) * H_Q * d_h_qk + 2 * N_Q(i) * N_KV(i) * H_Q * d_h_v) / 2
                  i=1
```

With chunked prefill, the nth chunk already has the KV tokens from all previous
chunks cached. The attention matrix for one such request splits into a
non-causal rectangle (current chunk attending to previous chunks) plus a causal
self-attention triangle (current chunk attending to itself):

```
                          Keys (N_KV)
          |<---- N_KV - N_Q ---->|<---- N_Q ---->|
         +----------------------+---------------+  ^
         |                      |\              |  |
         |      Non-causal      |  \  (masked)  |  |
 Queries |   (full rectangle)   |    \          | N_Q
         |  attend to previous  | Causal \      |  |
         |    chunks' KV cache  |  (self)  \    |  |
         +----------------------+---------------+  v
          <----- previous ----->|<-- current -->
                 chunks               chunk
```

For the first chunk (no chunking, or the first chunk of chunked prefill),
N_KV = N_Q, so the rectangle vanishes and the whole matrix is causal:

```
           Keys (N_KV = N_Q)
          |<---- N_Q ---->|
         +---------------+  ^
         |\              |  |
         |  \  (masked)  |  |
         |    \          |  |
 Queries |      \        | N_Q
         | Causal \      |  |
         | (entire) \    |  |
         +---------------+  v
```

Both cases (first chunk and nth chunk) simplify to a single unified formula:

```
                  R_C
FLOPS_context  =   Σ  [ (2 * N_Q(i) * N_KV(i) * H_Q * d_h_qk + 2 * N_Q(i) * N_KV(i) * H_Q * d_h_v)
                  i=1
                       - (2 * N_Q(i)^2 * H_Q * d_h_qk + 2 * N_Q(i)^2 * H_Q * d_h_v) / 2 ]
```

This works because:

- When N_KV = N_Q (first chunk): `full - full/2 = full/2`, which is causal.
- When N_KV > N_Q (nth chunk): `full rectangle - self-triangle`, which is the
  non-causal rectangle plus the causal self-attention.

#### FLOPS: generation (decode) requests

Generation requests attend to all cached KV tokens (N_KV = context length so
far). Typically N_Q = 1 (autoregressive decoding), but techniques like
speculative decoding might have N_Q > 1. The attention is non-causal:

```
                     R_G
FLOPS_generation  =   Σ   (2 * N_Q(i) * N_KV(i) * H_Q * d_h_qk + 2 * N_Q(i) * N_KV(i) * H_Q * d_h_v)
                     i=1
```

```
FLOPS_total = FLOPS_context + FLOPS_generation
```

#### Elements moved

Total memory traffic sums over all requests. Each request reads its Q, K, V
tensors and writes the output. With GQA/MQA, the KV cache uses H_KV heads (not
H_Q), reducing KV memory traffic. Ignoring cases with shared KV pages between
requests:

```
                    R
Elements_moved  =   Σ  ( N_Q(i)  * H_Q  * d_h_qk        // Q read
                   i=1
                       +  N_KV(i) * H_KV * d_h_qk        // K read (from paged KV cache)
                       +  N_KV(i) * H_KV * d_h_v         // V read (from paged KV cache)
                       +  N_Q(i)  * H_Q  * d_h_v  )      // Output write
```

where R = R_C + R_G is the total number of requests. Because B = 1 per request in
paged attention, the batch dimension is absorbed into the summation.

### Roofline analysis without per-request details

Importantly, individual request details are *not* required to perform roofline
analysis. Inspecting the formulas above, the only per-request quantities that
appear are `N_Q(i)`, `N_KV(i)`, and their products. The full FLOPS and
memory-traffic expressions can be evaluated from just these aggregate
statistics, computed separately for context and generation requests:

| Aggregate | Used in |
|-----------|---------|
| R_C, R_G | Request counts |
| Σ N_Q | Elements moved (Q read, Output write) |
| Σ N_KV | Elements moved (K read, V read) |
| Σ (N_Q * N_KV) | FLOPS (full-rectangle term) |
| Σ (N_Q^2) | FLOPS (causal self-attention correction for prefill) |

TraceLens obtains these aggregates by applying `torch.record_function(annotation)`
to the framework's execution steps. Because a single execution step can contain a
mix of context and generation requests, the annotation encodes the aggregate
statistics *separately* for context and generation requests within that step (for
example R_C, R_G, Σ N_Q for context, Σ N_Q for generation, and so on). These
annotations are stored as `user_annotation` events in the PyTorch profiler trace,
making roofline analysis possible directly from the trace without any additional
instrumentation or runtime logging.

The framework patches referenced in
[Generate a PyTorch inference performance report](../how-to/generate-perf-report-pytorch-inference.md)
are what add these annotations to the trace.

## Steady-state region

Inference-serving execution consists of three phases:

1. *Ramp-up*: the initial few steps where requests are still being batched.
2. *Steady state*: the execution steps with the highest concurrency.
3. *Ramp-down*: the trailing steps where the final batch of requests finishes.

Once steady state is reached, execution consists of:

- **Decode-only steps**.
- **Prefill-decode steps**, typically containing one prefill request packed with
  ~CONC−1 decode requests.

For performance analysis, only the steady-state steps are of interest,
specifically prefill-decode steps and decode-only steps with large context sizes
(towards the end of a request).

### Parameters relevant to inference serving

The following benchmark parameters determine the steady-state region boundaries.

- **NUM_PROMPTS**: typically `10 * CONC`.
- **CONC**: number of concurrent requests that can be batched together.
- **R**: random-range ratio used for sampling input and output sequence lengths.
- **OSL**: maximum output sequence length. The per-request output length is sampled
  uniformly in `[R * OSL, OSL]`.
- **ISL**: assumed to be lower than the chunk size.

Assuming the benchmark schedules requests at an effectively infinite rate, the
first CONC steps are conservatively treated as the ramp-up phase. The execution
step ranges where groups of CONC requests complete are:

```
1 * R * OSL  to  1 * OSL      e.g. 0.8 OSL - 1 OSL
2 * R * OSL  to  2 * OSL      e.g. 1.6 OSL - 2 OSL
3 * R * OSL  to  3 * OSL      e.g. 2.4 OSL - 3 OSL
4 * R * OSL  to  4 * OSL      e.g. 3.2 OSL - 4 OSL
...
N * R * OSL  to  N * OSL

where  N = NUM_PROMPTS / CONC
```

```
TOTAL_STEPS               = NUM_PROMPTS * Avg(OSL) / CONC
TOTAL_PrefillDecode_Steps = NUM_PROMPTS
TOTAL_DecodeOnly_Steps    = NUM_PROMPTS * ((Avg(OSL) - CONC) / CONC)
```

### Recommended profiling window

Since serving benchmarks commonly sample output sequence lengths from
`[R * OSL, OSL]`, the most useful steady-state profiling window lies in:

```
5 * R * OSL  to  5 * OSL
```

Here, the probability of a request finishing at step t is non-uniform, peaking at
`((R+1)/2) * 5 * OSL`. This region exhibits fully saturated concurrency, a
representative mix of decode-only and prefill-decode steps, and minimal warm-up or
tail artifacts. That makes the recommended profiling window:

```
max_iters   = 16 * OSL / CONC
              # Number of execution steps to profile. The multiplier 16 targets
              # ~16 execution steps with a prefill+decode mix. Clamp max_iters
              # for extreme values of OSL/CONC.

delay_iters = ((R+1)/2) * 5 * OSL - (max_iters / 2)
              # Execution step where the profiler starts.
```

The trace-splitting workflow uses these definitions to detect the steady-state
region automatically and to separate prefill-decode from decode-only steps. When
the benchmark parameters are known, pass `--CONC`, `--OSL`, and `--R` to the
splitter to override the empirically estimated ratio with the analytically
derived one; see
[Split inference traces](../how-to/generate-perf-report-pytorch-inference.md#split-inference-traces-optional).

## Related topics

- [Generate a PyTorch inference performance report](../how-to/generate-perf-report-pytorch-inference.md)
- [Compare two traces in TraceLens](../how-to/compare-traces.md)
- [GEMM analysis in TraceLens](./gemm-analysis.md)
- [API reference](../reference/api-reference.md)
