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
Each point is an operation (prefill attention, verify attention, routed MoE, or
the whole forward) at one concurrency.

- **X** is arithmetic intensity from the roofline projection: FLOPs divided by
  compulsory HBM bytes of one forward.
- **Y** is achieved TFLOP/s: those FLOPs divided by the time the matching
  kernels ran in a profile of the same regime.
- **Roofs** come from the GPU arch spec. The diagonal is `mem_bw_gbps`. The
  horizontals are the bfloat16 and fp4 matrix peaks. A regime whose modeled
  FLOPs use more than one of those peaks also gets an end-to-end blend roof,
  the FLOP-weighted harmonic mean of the peaks.

A projection alone puts every point on a roof. A profile of a GPU-graph replay
has kernel time and no `Input Dims`, so it has no compulsory bytes. The plot
uses one source for each axis.

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

Repeat `--projection PHASE:CONCURRENCY:PATH` for each concurrency. Points of
one series and one phase are joined by an arrow in concurrency order.

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

## What the profile contributes

Y uses the `Kernel Time (µs)_sum` and `operation_count` columns of the
perf-report `*_ops.csv` files.

When the projection (or the augmentation rollup) gives a per-launch FLOP
count, the numerator is that cost times the launch count. That is the work in
a capture that contains many forwards. Kernels with the same launch count are
treated as parts of one launch, and their times are summed. Kernels with
different launch counts are separate launches, and the counts add. Without a
per-launch cost, the numerator is one forward's FLOPs, so the profile time has
to be one forward.
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
kernels are not the whole forward.

## Related topics

- [Project a model's roofline from its checkpoint](./generate-model-roofline-projection.md)
- [Model op performance without a trace](./perf-model-without-trace.md)
