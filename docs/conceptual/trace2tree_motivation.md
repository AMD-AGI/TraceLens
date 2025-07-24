# Trace2Tree Motivation

## Limitations of Direct Kernel Analysis

Directly inspecting GPU kernel names has **two fundamental limitations**:

* **Ambiguous semantics (and weak reproducibility)**: A single kernel name can map to many different computations depending on shape, dtype, strides/layout, etc. Shape strongly affects performance—one shape may select a fast tiled path while another shape (same op type) falls onto a slower algorithm. Because the name omits this argument context, you cannot reliably understand, compare, or reproduce the workload from the kernel string alone. Further, many kernel names are cryptic and unreadable (e.g., `Cijk_Ailk_Bljk_*`, `void cutlass_*`). 

* **Platform‑dependent / unstable naming (and weak cross‑platform comparison)**: The same high‑level operation appears under different kernel names across pltforms. For example: a single GEMM shows up as `nvjet_*` or `cutlass_*` on NVIDIA H100, and as a Tensile kernel `Cijk_Ailk_Bljk_*` on AMD MI300. These names also shift across software versions. Raw kernel strings are therefore not a stable abstraction for comparison.

## What Trace2Tree Does

**Trace2Tree** reconstructs a full call tree from the Python front‑end down to each GPU kernel. The tree has exactly four layers:

1. **Python front end** – user code / `nn.Module`.
2. **Operation** – On PyTorch it is the dispatch operation ("cpu_op") e.g. `aten::mm`, `aten::addmm`, etc. On JAX the corresponding layer is the HLO operations. This is the layer that contains the arg information to contextualize the kenrel.
3. **HIP / CUDA runtime** – launch API calls.
4. **GPU kernel** – executed kernel.

## How This Solves the Problem

* **Disambiguates semantics**: Argument metadata at the backend op layer lets us group identical computations, attribute time, and deterministically reproduce slow cases.
* **Enables fair comparison**: Operations (`aten::mm`, HLO) are stable across platforms; by anchoring analysis there, we can compare the same operation and arguments regardless of how kernels are named underneath.
* **Flexible attribution**: GPU time can be viewed at any level—module (via its backend ops), dispatch op, runtime, or kernel—depending on the question. As an additional benefit, we can also attribute time all the way up to the Python nn.Module level, making performance insights accessible even to users outside the performance engineering field. This helps bridge the gap between model developers and low-level hardware execution.

## Takeaway

Kernel names are volatile and context‑free. Trace2Tree anchors analysis at the stable backend operation, enriches it with arguments, and maps the full execution stack to deliver portable, interpretable performance insight.

That said, **kernel names are often useful** — they can offer clues about the backend implementation, algorithm variant, or compiler choices. TraceLens intends to serve as a **one-stop solution** for extracting every bit of useful signal from a trace file. Therefore, it includes features to extract and parse relevant information from kernel names where applicable. But we treat them as supplementary, not foundational.
