<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->


# Split traces into iterations, steady state, and phases
```{meta}
:description: Learn how TraceLens splits a large multi-iteration trace into iterations, a steady-state window, and prefill/decode phases for LLM serving, training, and diffusion workloads.
:keywords: TraceLens, trace splitting, iteration detection, steady state, phase division, prefill, decode, vLLM, SGLang, ATOM, diffusion, training, ProfilerStep, ROCm
```

This topic shows how to split a large multi-iteration trace into its repeating
unit of work — one execution step for LLM serving, one forward/backward pass for
training, one denoise step for diffusion, etc. — so you can analyze a single
representative slice instead of a whole run. Analyzing a full workload is
expensive: the trace is large, and most of it is repetition of the same handful
of iterations. Splitting it down to the meaningful components cuts the work to
analyze and yields sharper, more comparable results. Splitting works on annotated
serving traces and on generic traces from any framework, adapting to the
information each trace carries.

## Before you begin

Confirm you have the following before continuing.

- [TraceLens installed](../install/install.md).
- A multi-iteration PyTorch Kineto trace.

## How it works

Splitting proceeds in four stages:

```text
trace.json.gz
   └─ 1. Detect iteration roots   detects the iteration markers i.e. where to split
      └─ 2. Extract               extract the splits. creates traces from splits
         └─ 3. Find steady state  finds the window of splits that are steady state (peak concurrency/stable runtime)
            └─ 4. Divide phases   divides splits into prefill/decode/prefilldecode (only for LLM inference)
```

Detection and extraction always run. Finding the steady-state region and dividing
phases are opt-in, selected with `--find-steady-state` and `--divide-phases`, and
operate on the iterations found in the first stage.

## Split a trace

Invoke the splitter as a module (it is also installed as the
`TraceLens_split_inference_trace` console script):

```bash
python -m TraceLens.TraceUtils.trace_split.main trace.json.gz -o ./output [OPTIONS]
```

`--store-single-iteration`, `--find-steady-state`, and `--divide-phases` can all
run in a single invocation.

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output-dir` | *(required)* | Output directory. |
| `-i`, `--iterations` | `all` | Iteration range to operate on: `all`, a single index such as `50`, or a range such as `10:20`. |
| `--store-single-iteration` | off | Write each iteration as its own trace file. Without it, the selected iterations are written as a single trace with the parent frames from each split root up to the process entry, but not their other children. |
| `--find-steady-state` | off | Extract a steady-state window instead of sequential iterations. |
| `--num-steps` | `32` | Number of iterations to extract for the steady-state window. |
| `--divide-phases` | off | Write every steady-state step into phase sub-folders, `prefilldecodemix/` and `decode_only/`. |
| `--llm-inference` | off | Treats the trace as LLM inference. When serving annotations are absent, batch sizes are used for phase classification and steady-state identification. |
| `--CONC`, `--OSL`, `--R` | none | Benchmark parameters. When all three are given, the ideal prefill-decode ratio is computed analytically and overrides the empirical estimate. See [Recommended profiling window](../conceptual/inference-analysis.md#recommended-profiling-window). |
| `--max-num-seq` | none | Batch-size threshold above which a shape-inferred iteration is prefill-bearing. When omitted, the threshold is inferred. |
| `--no-gap-fill` | off | Score each iteration by its own span instead of extending it to the next root. Work between two roots is then dropped. |
| `--allow-degraded` | off | Return splits even when GPU coverage is below acceptable threshold. |

## Iteration-root detection

The first stage finds the *iteration roots* — the events that mark the repeating
unit. Detection is a cascade: each detector is tried in order and accepted as soon
as it produces roots whose per-iteration windows explain enough of the trace's GPU
time. Coverage is audited by correlating GPU kernels back to each root's window,
and a candidate must explain at least **95% of GPU time** to be accepted. Every
candidate is returned as a `RootSet` that carries this coverage grade.

```mermaid
flowchart TD
  START["find_iteration_roots"] --> ANN["1. Known annotations,<br/>then unknown annotation families"]
  ANN --> ANN_GATE{"coverage ≥ 95%?"}
  ANN_GATE -->|yes| OK_ANN["SPLITTABLE"]
  ANN_GATE -->|no| TREE["Build call tree,<br/>reattach worker threads"] --> BD

  BD["2. Branch descent:<br/>a call-tree frame whose children repeat"] --> BD_GATE{"coverage ≥ 95%?"}
  BD_GATE -->|yes| OK_BD["SPLITTABLE"]
  BD_GATE -->|no| SR

  SR["3. Sibling roots:<br/>periodicity across top-level frames"] --> SR_GATE{"coverage ≥ 95%?"}
  SR_GATE -->|yes| OK_SR["SPLITTABLE"]
  SR_GATE -->|no| BE

  BE["4. Bookend enhancement:<br/>add warmup/wrapup to a ≥50% candidate"] --> BE_GATE{"coverage ≥ 95%?"}
  BE_GATE -->|yes| OK_BE["SPLITTABLE"]
  BE_GATE -->|no| BEST["Best candidate by coverage,<br/>or NOT_SPLITTABLE"]
```

The detectors, in order:

1. **Known annotations.** Traces from patched vLLM (`execute_*`), SGLang
   (`step[*]`), ATOM, or any framework emitting `ProfilerStep#*` carry explicit
   per-iteration markers. When a recognized pattern is found, the detector also
   looks for an enclosing family to widen the split, so a 512-iteration run is not
   split into only the handful of iterations a regular expression happened to
   match. This is the fast, high-confidence path.
2. **Unknown annotation families.** When only some steps match a known pattern,
   annotations are grouped into families by name skeleton, with digit runs
   collapsed so that `execute_1_...` and `execute_2_...` share a key. The family
   that best explains the timeline — often an enclosing frame such as
   `scheduler.run_batch` — becomes the iteration unit.
3. **Branch descent.** For traces with no usable annotations, the call tree is
   built and worker-thread call stacks are spliced back onto the main thread so
   that training traces are traversable. A search from the tree roots finds the
   first frame whose GPU-bearing children form a repeating pattern. Considering
   only GPU-bearing children filters out the setup and scheduling calls that
   obscure the iteration structure, and the coverage gate prevents the search from
   stopping on a shallow sub-loop that repeats but does almost no work.
4. **Sibling roots.** Some traces, especially those with sparse Python call
   stacks, have many shallow roots rather than one deep root, so the repeat lives
   across roots rather than within one. Sibling-root detection finds the repeating
   period across the top-level frames.
5. **Bookend enhancement.** A branch or sibling candidate that already explains at
   least 50% of GPU time but falls below the gate is extended with `warmup` work
   before the first iteration and `wrapup` work after the last, drawn from the
   parent's GPU-bearing children. When the additions raise the grade, they are
   kept. This is what makes diffusion traces splittable: HunyuanVideo's four
   denoise steps cover 58% of GPU time on their own, and 100% with bookends.

If no detector clears the gate, the best candidate by coverage is returned with an
honest status so you can still use it with `--allow-degraded`; a trace that no
detector explains is reported as `NOT_SPLITTABLE`.

The detector that fires depends on what the trace contains:

| Detector | Fires when | Example workloads |
|----------|------------|-------------------|
| Known or unknown annotations | The trace has `user_annotation` events with enough GPU coverage | vLLM, SGLang, ATOM, Megatron (`ProfilerStep`), anything calling `profiler.step()` |
| Branch descent | There are no usable annotations, but the call tree has a frame whose children repeat | Training loops, diffusion denoise, `torch.compile` workloads |
| Sibling roots | Branch descent finds no repeating children, but the top-level frames repeat | Workloads with sparse call-stack information |

## Extraction and the split manifest

Once the roots are known, each iteration is given a half-open time window, or
*tile*. Tiles are built per thread and leave no gaps: the span between one root
and the next belongs to the earlier root, so a kernel launched just after an
annotation closes — common with vLLM sampling — is still captured, and no kernel
is lost. Events are assigned to a tile by start timestamp, correlation IDs are
followed from each CPU launch to its GPU kernel, and enclosing frames that span
multiple iterations belong to none and are excluded.

Alongside the slices, the splitter writes a `split_manifest.json` describing the
result. Its key fields are:

| Field | Meaning |
|-------|---------|
| `status` | `0` splittable, `1` degraded (requires `--allow-degraded`), `2` not splittable. |
| `method` | The detector that produced the roots, such as `annotation:tier`, `family:unknown_only`, `generic:branch_descent`, or `generic:sibling_roots`. |
| `n_roots` | The number of iterations found. |
| `attribution_strategy` | How kernels were mapped to roots for the coverage audit: GPU-side annotation spans when present, otherwise CPU launch correlation IDs. |
| `coverage_selected_roots` | The fraction of GPU time explained by the roots' tile windows — the primary quality metric. |
| `gpu_event_retention` | The fraction of GPU kernels that survived extraction. This should be `1.0`, meaning every kernel is accounted for across the slices. |
| `gpu_events_duplicated` | Whether any kernel was claimed by more than one slice, which indicates a tiling error. |
| `gap_fill` | Whether gap-free tiling was used. |

A per-iteration `execution_details` file records the same accounting for each
slice.

```{note}
When the detected roots do not clear the coverage gate and `--allow-degraded` is
not set, the splitter writes the manifest with `aborted` set to `true` and stops
before extraction, so you can inspect the coverage breakdown before deciding
whether to proceed.
```

## Steady-state identification

The steady-state region is the stretch of highest, saturated concurrency — the
part of a serving run worth profiling. For the concepts behind it, including the
CONC, OSL, and R parameters, see
[Steady-state region](../conceptual/inference-analysis.md#steady-state-region).
The method used to find it adapts to the trace:

| Tier | Condition | Method |
|------|-----------|--------|
| Concurrency | Serving annotations are present | Find the region where `num_requests` is near its peak, then select a window by mode within the largest such region. Honors `--CONC`, `--OSL`, and `--R`. |
| Decode baseline | `--llm-inference`, no serving annotations | Use decode-iteration batch sizes as a concurrency proxy (in decode, batch size approximates the number of sequences), find where they are near peak, and map the region back to full iteration indices. |
| Duration | No annotations and no `--llm-inference` | Find the most duration-consistent window with a sliding-window coefficient of variation. This tier has no phase awareness. |

The decode-baseline tier filters to decode iterations, scans for the peak, then
re-includes the prefill iterations that fall inside the region:

```text
batch_sizes:  [8, 8, 32, 32, 32, 946, 32, 32, 854, 32, 32, 32, 8, 8]
phase_labels: [D, D,  D,  D,  D,  P,   D,  D,  P,   D,  D,  D, D, D]

1. Keep decode-only batch sizes:  [_, _, 32, 32, 32, _, 32, 32, _, 32, 32, 32, _, _]
2. Peak-proximity scan (max=32):  decode steady state where decode_bs ~ 32
3. Map back to full indices:      region [2, 12) — includes the prefill spikes too
4. Select a window by mode within the region:
     mixed              representative prefill/decode ratio
     decode_only        longest contiguous decode run
     max_prefilldecode  longest contiguous prefill-bearing run
```

The duration tier picks the flattest run, skipping warmup and cooldown spikes:

```text
Durations:   3100  3050  5200  3080  3070  3090  3060  3075  3085  4900
Window 3-6:  [3080, 3070, 3090, 3060]  CV = 0.004  mean = 3075
Window 4-7:  [3070, 3090, 3060, 3075]  CV = 0.004  mean = 3074   (fastest passing)
Window 6-9:  [3060, 3075, 3085, 4900]  CV = 0.22   rejected (cooldown spike)

Result: region [4, 8) — consistent durations, with warmup (iteration 2) and
        cooldown (iteration 9) excluded.
```

## Phase division

Phase division separates LLM-inference steps into *prefill-bearing* steps, where a
prefill request is packed with decodes, and *decode-only* steps. Because it is
meaningful only for LLM inference, it uses the same information tiers:

| Tier | Condition | Method |
|------|-----------|--------|
| Annotation | vLLM, SGLang, or ATOM annotations are present | Parse `context_requests` and `generation_requests` from the annotation name. A step with `context_requests > 0` is prefill-bearing; otherwise it is decode. |
| Shape | `--llm-inference`, no serving annotations | Derive batch size per iteration from the most common first dimension of `cpu_op` `Input Dims`, then split decode from prefill by batch size. |

For the shape tier, the threshold is either taken from `--max-num-seq` directly
or inferred: among the unique batch sizes, with the two smallest ramp-up values
dropped as noise, the splitter finds the largest multiplicative gap and splits
there when it is at least a factor of two. Iterations at or below the threshold
are decode; those above it are prefill-bearing. Deriving batch size from `cpu_op`
input shapes has proven the most reliable signal when annotations are absent.

```{note}
The shape-based path has two limitations. For a pure-prefill trace there is no
decode baseline to split against, so every iteration is labelled `decode`; this is
rare in practice, since pure prefill only occurs during warmup, which steady-state
selection excludes anyway. For a fully graphed trace with no `cpu_op` shapes,
there are no input dimensions to read, so `--llm-inference` detects that every
batch size is `None` and falls through to the duration tier.
```

## Related topics

- [Inference performance analysis](../conceptual/inference-analysis.md)
- [Generate a PyTorch inference performance report](generate-perf-report-pytorch-inference.md)
- [API reference](../reference/api-reference.md)
