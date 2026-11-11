<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Plot an end-to-end roofline
```{meta}
:description: Draw a roofline whose x-axis comes from a TraceLens projection and whose y-axis comes from a profile of the same regime.
:keywords: TraceLens, roofline, arithmetic intensity, profile, MI355X
```

`TraceLens_e2e_roofline_plot` draws one log-log roofline for a model regime.
Each point is an operation (prefill attention, decode attention, routed MoE, or
the whole forward) at one concurrency.

- **X** is arithmetic intensity: FLOPs divided by compulsory HBM bytes. Both
  come from the profile when the kernel's `GFLOPS` and `FLOPS/Byte` are
  filled, and from the roofline projection when they are not.
- **Y** is achieved TFLOP/s: those FLOPs divided by the time the matching
  kernels ran in the profile.
- **Roofs** come from the GPU arch spec. The diagonal is `mem_bw_gbps`. The
  horizontals are the bfloat16, fp8, and fp4 matrix peaks, drawn when the spec
  lists them. Past the bfloat16 ridge the diagonal uses the linestyle of the
  fastest peak that still applies: dash-dot for fp8, dashed for fp4. Each
  peak also has a gray copy of that roof at 75%, 50%, 25%, and 10%: the
  diagonal and the horizontal are both scaled by that fraction, so the ridge
  stays put. The copies get lighter as the fraction drops, and the 75% copy is
  still lighter than the full roof. A regime
  whose modeled FLOPs use more than one of those peaks also gets an
  end-to-end blend roof, the FLOP-weighted harmonic mean of the peaks.

`--model-only` draws the projection with no profile. Y is then the projection's
FLOPs divided by its roofline time, so each point sits on the roof its shapes
imply. A profile of a GPU-graph replay has kernel time and no `Input Dims`, so
those kernels have no `GFLOPS`. The plot uses the projection for them and the
profile for every kernel the trace did model. Whole model is the sum of those
categories at one concurrency.

## Run the tool

Project the regime, then plot it against the profile CSVs from a perf report
of that same batch and phase:

```bash
TraceLens_model_roofline_projection <checkpoint> \
    --phase decode --batch-size 48 --seq-len 1 --dim past_len=8192 \
    --tp 8 --gpu-arch-name MI300X \
    --output-json decode.json

TraceLens_e2e_roofline_plot \
    --gpu-arch-name MI300X \
    --projection decode:48:decode.json \
    --profile-dir <perf_report_csvs> \
    --title "decode concurrency 48" \
    --output roofline.png
```

Repeat `--projection PHASE:CONCURRENCY:PATH` for each concurrency. Each
concurrency is a separate marker color.

An orchestrator run already writes the category rollup. Pass that instead of
the per-op JSON:

```bash
TraceLens_e2e_roofline_plot \
    --gpu-arch-name MI300X \
    --augmentation <output_dir>/metadata/roofline_augmentation.json \
    --profile-dir <output_dir>/category_data \
    --title "decode and prefill" \
    --output roofline.png
```

`--gpu-arch-name` is resolved by `load_arch`. Names that are not in the bundled
arch directory (`MI300X`, `MI325X`) come from `$TL_EXTENSION`.
`--gpu-arch-json-path` names a spec file directly.

## Without a profile

`--model-only` reads a projection or an augmentation and no trace. Each point
is one forward. X is the projection's FLOPs divided by its compulsory bytes.
Y divides those FLOPs by `total_roofline_time_us` (or `roofline_time_us` on a
category rollup). A series that uses one compute spec lands on that spec's
roof. Whole model and end to end sum every category, so they are one point and
only end to end is drawn. The printed `source` is `model`.

```bash
TraceLens_model_roofline_projection <checkpoint> \
    --phase decode --batch-size 48 --seq-len 1 --dim past_len=8192 \
    --tp 8 --gpu-arch-name MI300X \
    --output-json decode.json

TraceLens_e2e_roofline_plot \
    --model-only \
    --gpu-arch-name MI300X \
    --projection decode:48:decode.json \
    --title "decode concurrency 48, model roofline" \
    --output roofline.png
```

## From a report with graph-launch shapes

A graph replay records device kernels and not the CPU ops that were captured
into the graph, so those kernels have no `Input Dims`. The inference report
fills them before it costs the kernel. A capture folder supplies the recorded
call. Without one, `--graph_model_config` builds the per-rank model and the
schedule walks the forward. When the pinned engine checkout exports a kernel
together with the tensors it is called with, those shapes replace the schedule
for that launch. The perf model then writes `GFLOPS` and `FLOPS/Byte`.

Plot that report directly. No projection is read:

```bash
TraceLens_e2e_roofline_plot \
    --profile-costs \
    --gpu-arch-name MI355X \
    --profile-dir <report>/category_data \
    --title "decode" \
    --output roofline.png
```

Attention, routed MoE, and GEMM are the report categories `inferenceattention`,
`moe_unfused`, and `gemm`. Whole model is every costed row in the same batch
bucket. The batch is the leading dimension of the stamped `Input Dims`. A row
the fill did not cover has empty `GFLOPS` and is left out.

## What the profile contributes

Y uses the `Kernel Time (µs)_sum` column of the perf-report `*_ops.csv` files.

X uses `GFLOPS` and `FLOPS/Byte` from the same files when both are filled.
`GFLOPS` is one launch, so the kernel's FLOPs are `GFLOPS × 1e9 ×
operation_count` and its bytes are that divided by `FLOPS/Byte`. A series uses
this total when every kernel matched to it has both cells. One kernel without
them means the profile does not describe the series, and the projection
supplies the whole series.

The projection gives the FLOPs of one forward, and the number of times each op
occurs in it (`instances`, written to the augmentation as `invocations`). The
numerator is then the series' FLOPs times `launches / instances`, which is the
number of forwards the matched kernels ran. This is the work in a capture that
contains many forwards. It uses the mean launch, not the costliest op, so a
series that mixes cheap and expensive ops is not overstated.

Kernels with the same launch count are treated as parts of one launch, and
their times are summed. Kernels with different launch counts are separate
launches, and the counts add. Without `instances`, the numerator falls back to
the per-launch cost times the launch count, and without that, to one forward's
FLOPs, so the profile time has to be one forward.
`--time-us PHASE|SERIES|MICROSECONDS` sets that one-forward time explicitly.

Kernels are matched to a series by name (`classify_kernel`). A name that
contains `prefill` is prefill; a name that contains `qseqlen1` or `decode` is
decode. Collective kernels (`allreduce`, `nccl`, and the rest of that family)
are left out. In a capture that mixes phases, a kernel whose name does not say
which phase it ran is left out too. Assign it with:

```bash
--kernels decode|Routed MoE|t32x64x256
```

End to end on a mixed capture needs `--time-us`, because the phase-tagged
kernels are not the whole forward. `--time-us` is one forward, so the numerator
is one forward's FLOPs.

Whole model is every category at that concurrency. A category whose kernels all
have `GFLOPS` contributes the profile total. A category that does not
contributes the projection, scaled by the launches those kernels ran.
Categories with no matched kernel are left out, because there is no time to put
their FLOPs against. In a mixed capture, pass `--kernels PHASE|Whole model|SUBSTRING`
for the kernels whose names do not say which phase they ran. The printed
`source` is `profile`, `export`, or `mixed`.

## A point above the roof is an error in the inputs

Achieved TFLOP/s divided by arithmetic intensity is the HBM traffic the kernels
would have to move, and it cannot exceed the memory bandwidth. A point more
than 5% above the roof is outlined in red and reported on stderr with the TB/s
it implies. It means the projected FLOPs or bytes, or the kernels matched to
the series, do not describe what ran. The usual cause is a series that matches
only some of an op's kernels, so its time is too small. For example, only the
down-projection of an MoE carries `decode` in its name. Pass `--kernels` for
the rest.

## Related topics

- [Project a model's roofline from its checkpoint](./generate-model-roofline-projection.md)
- [Model op performance without a trace](./perf-model-without-trace.md)
