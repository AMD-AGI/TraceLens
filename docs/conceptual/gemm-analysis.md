<!--
Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# GEMM analysis in TraceLens
```{meta}
:description: Understand how model dimensions map to GEMM shapes and BLAS calls, and how TraceLens reports GEMM dimension-efficiency metrics such as tile and wave quantization efficiency.
:keywords: TraceLens, GEMM, general matrix multiply, BLAS, hipBLAS, PyTorch, roofline, tile efficiency, wave quantization, dimension efficiency, Tensile, ROCm, MI300X, LLM, prefill, decode
```

General matrix multiply (GEMM) operations are the *primary compute primitive* used in AI models. Efficient implementations of GEMMs are readily available through vendor-tuned libraries such as cuBLAS and hipBLAS, so whenever possible the goal is to *reduce computation to a matrix multiply*.

This topic explains how *model-level parameters* like batch size, sequence length, and hidden dimension translate into GEMM shapes, how those shapes map to specific basic linear algebra subprograms (BLAS) kernel calls, and how TraceLens reports deeper GEMM dimension-efficiency metrics. By the end you should be equipped to understand the GEMM shapes, counts, and BLAS calls involved in *any AI model you encounter*.

## From model dimensions to GEMM shapes

The following sections trace how a linear layer's input tensor shape maps to GEMM parameters, contrasting the prefill and decode inference phases.

### Linear layers in LLMs

Consider how linear layers in a large language model (LLM), such as the multi-layer perceptron (MLP) *up projection*, correspond to GEMM calls.

The input tensor shape for this operation is:

```
X: [B, L, d_model]
```

Here:

- `B` represents the batch size.
- `L` denotes the sequence length.
- `d_model` is the input (or hidden) dimension.

This operation outputs a tensor with the shape:

```
Y: [B, L, d_ff]
```

The projection for each token can be expressed individually as:

```
Y[b, l, :] = X[b, l, :] @ Wᵀ
```

Where `W` is the weight matrix with a shape of `[d_ff, d_model]`.

### Flattening for GEMM

To express this entire operation as a single GEMM, flatten the batch and sequence dimensions of the input tensor:

```
X_flat: [B·L, d_model]
Wᵀ:     [d_model, d_ff]
Y = X_flat @ Wᵀ
```

This flattening yields the following GEMM shape parameters:

- `param: M = B·L`
- `param: N = d_ff`
- `param: K = d_model`

Here, `K` represents the *inner or shared dimension* between the input tensors involved in the multiplication.

## Prefill versus decode

Contrast the GEMM behavior during the *prefill* and *decode* phases of inference, focusing on how the sequence length (`L`) changes and affects the GEMM shapes.

In both phases, the input tensor for an MLP computation within an LLM initially has a shape like:

```
X: [B, L, d_model]
```

Where `B` is the batch size, `L` is the sequence length, and `d_model` is the hidden size. This projects to an output shape of `[B, L, d_ff]`, where `d_ff` is the MLP's expansion size.

As established earlier, to process this with a single GEMM the batch and sequence dimensions of the input are flattened. The input effectively becomes `[B·L, d_model]` for the GEMM `X_flat @ Wᵀ`.

The key difference between prefill and decode lies in the value of `L`:

- **Prefill phase**: `L` is the actual input sequence length (which can be large).
- **Decode phase**: `L` is always `1`, as the model processes one token at a time to generate the next.

This difference in `L` directly impacts the `M` parameter of the GEMM `(M, N, K)`.

#### GEMM shape summary

The following table summarizes GEMM shapes for each inference phase.

| Mode     | Input shape (conceptual) | Flattened input shape | GEMM shape `(M, N, K)` | Notes                                |
|----------|--------------------------|-----------------------|------------------------|--------------------------------------|
| Prefill  | `[B, L, d_model]`        | `[B·L, d_model]`      | `(B·L, d_ff, d_model)` | `L` is the prompt length             |
| Decode   | `[B, 1, d_model]`        | `[B·1, d_model]`      | `(B, d_ff, d_model)`   | `L=1` for generating one token at a time |

Notice that in the decode phase, because `L=1`, the `M` parameter of the MLP GEMM becomes simply `B`. This means the computational cost of the MLP layers in decode remains constant per token regardless of the total sequence length generated so far. The dominant `O(L)` scaling cost during decode comes from the attention mechanism, not the MLPs.

### Real-world example: LLaMA-2 7B

The table below shows actual data from TraceLens profiling, filtered specifically for MLP *up* and *gate* projection GEMMs in a LLaMA-2 7B model inference trace.

For this trace:

- `d_model = 4096`
- `d_ff = 11008`
- Batch size: `1` (`B=1`)
- Input length (for prefill): `597` (`L=597`)
- The trace included 36 decode steps.

| name     | param: M | param: N | param: K | param: bias | counts |
|----------|----------|----------|----------|-------------|--------|
| aten::mm | 1        | 11008    | 4096     | FALSE       | 2304   |
| aten::mm | 597      | 11008    | 4096     | FALSE       | 64     |

Interpreting these entries based on the `M` parameter:

- The entry with `param: M = 597` corresponds to the *prefill* phase GEMM `(B·L = 1·597)`, which happens once per layer at the beginning of inference. Since there are 32 layers, this GEMM is called `64` times (32 up + 32 gate).
- The entry with `param: M = 1` corresponds to the *decode* phase GEMM `(B = 1)`, where `L=1`. These occur at each decode step for every layer. With 36 decode steps and 64 GEMMs per step (32 layers * 2), this GEMM is called `36 × 64 = 2304` times.

## Backward pass GEMMs

Next, consider the backward pass during training. A forward pass GEMM operation like `Y = X @ Wᵀ + b` necessitates *two corresponding backward GEMMs* to compute gradients:

```python
dX = dY @ W        # Gradient with respect to the input → resulting shape: [B·L, d_model]
dW = dYᵀ @ X       # Gradient with respect to the weight → resulting shape: [d_ff, d_model]
db = dY.sum(dim=0) # Gradient with respect to the bias   → resulting shape: [d_ff]
```

### GEMM shapes

The following table shows the GEMM shape for each forward and backward operation.

| Operation   | GEMM shape `(param: M, param: N, param: K)` | Description                          |
|-------------|---------------------------------------------|--------------------------------------|
| Forward     | `(B·L, d_ff, d_model)`                       | `X @ Wᵀ`                             |
| Backward dX | `(B·L, d_model, d_ff)`                       | `dY @ W` (result of `[B·L, d_ff] @ [d_ff, d_model]`) |
| Backward dW | `(d_ff, B·L, d_model)`                       | `dYᵀ @ X` (result of `[d_ff, B·L] @ [B·L, d_model]`) |

A closer look at the backward GEMM shapes:

- For `dX = dY @ W`, the operation is `[B·L, d_ff] @ [d_ff, d_model]`, which results in a shape of `[B·L, d_model]`.
- For `dW = dYᵀ @ X`, the operation is `[d_ff, B·L] @ [B·L, d_model]`, yielding a shape of `[d_ff, d_model]`.

### Real-world example: GPT-3-XL

This table presents data from a TraceLens analysis of a single training step for GPT-3-XL.

For this example:

- `d_model = 2048`
- `d_ff = 8192`
- Batch size: `5`
- Sequence length: `2048`
- Thus, `param: M` for the flattened dimension is `5 × 2048 = 10240`.

| name        | param: M | param: N | param: K | count |
|-------------|----------|----------|----------|-------|
| aten::addmm | 10240    | 8192     | 2048     | 24    |
| aten::mm    | 10240    | 2048     | 8192     | 24    |
| aten::mm    | 8192     | 2048     | 10240    | 24    |

Each entry can be interpreted based on the GEMM shapes:

- The `aten::addmm` call represents the forward pass GEMM (`X @ Wᵀ`).
- The first `aten::mm` call corresponds to the backward pass for `dX` (`dY @ W`).
- The second `aten::mm` call represents the backward pass for `dW` (`dYᵀ @ X`).

Each of these operations appears once per layer in the network. Given that GPT-3-XL has 24 layers, each of these GEMMs is called 24 times per training step, aligning with the `count` column in the table.

## How PyTorch calls BLAS

To fully understand how PyTorch uses BLAS for operations like GEMM, first understand the fundamental concept of *memory layout* for tensors and how BLAS libraries interpret the data buffers they receive.

### Memory layout and stride

Despite tensors often being represented as multi-dimensional arrays, their elements are stored in linear memory. For a 2D matrix, the two primary storage conventions are:

- **Row-major**: Elements of the same row are stored consecutively in memory. PyTorch adopts this as its default layout.
- **Column-major**: Elements of the same column are stored consecutively in memory. Many traditional BLAS libraries primarily optimize for this layout.

PyTorch's `.stride()` method provides insight into a tensor's memory arrangement. It returns a tuple where each value indicates the byte (or element, depending on datatype size) distance in linear memory to move to the next element along that dimension.

- For a 2D tensor `T[i][j]` in *row-major* layout, `.stride()` is typically `(num_cols, 1)`. Moving to `T[i][j+1]` requires stepping 1 element, while moving to `T[i+1][j]` requires stepping `num_cols` elements.
- For a 2D tensor `T[i][j]` in *column-major* layout, `.stride()` is typically `(1, num_rows)`. Moving to `T[i+1][j]` requires stepping 1 element, while moving to `T[i][j+1]` requires stepping `num_rows` elements.

### BLAS transpose and row-major output

The core BLAS GEMM routine typically computes $C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C$, where $op(X)$ is either $X$ or $X^T$ depending on the `transA` and `transB` flags (`'N'` for no transpose, `'T'` for transpose) passed to the function. By default, BLAS expects input matrices corresponding to the `'N'` flag to be in column-major layout. Crucially, the resulting matrix $C$ is written into the output buffer in *column-major* format by default.

PyTorch, however, uses row-major layout internally and desires the result of a GEMM operation to also be in row-major layout *without* an extra copy or transpose step outside of the BLAS call. PyTorch achieves this by using the `trans` flags and the relationship between row-major and column-major layouts.

A matrix $M$ stored in row-major memory has the exact same element ordering as the matrix $M^T$ stored in column-major memory. PyTorch uses this identity. To get a row-major result $C$ from a BLAS call that outputs in column-major, PyTorch requests BLAS to compute $C^T$ and write it in column-major. Since $C^T$ in column-major is $C$ in row-major, the output buffer will contain the desired row-major $C$.

Mathematically, the operation $C = A @ B$ (where $A, B, C$ are desired in row-major) is equivalent to computing $C^T = (A @ B)^T = B^T @ A^T$. PyTorch therefore configures the BLAS call to compute $B^T @ A^T$ using the row-major data of $B$ and $A$.

Here's how the `transA` and `transB` flags work in this context when passing *row-major data* to BLAS through a wrapper like PyTorch's:

- Passing row-major data for matrix $M$ with `trans = 'T'` tells BLAS to mathematically treat this data as $M$. (BLAS expects row-major data for `'T'` if it wants to use the matrix directly.)
- Passing row-major data for matrix $M$ with `trans = 'N'` tells BLAS to mathematically treat this data as $M^T$. (BLAS expects column-major data for `'N'`; giving it row-major data makes it see the transpose.)

So, to compute $C^T = B^T @ A^T$ using row-major data for $B$ and $A$ and get $C$ row-major in the output buffer:

- Pass $B$'s row-major data as the first operand data (`A_data` in BLAS call). To make BLAS see $B^T$, use `transA = 'N'`.
- Pass $A$'s row-major data as the second operand data (`B_data` in BLAS call). To make BLAS see $A^T$, use `transB = 'N'`.
- The BLAS call becomes `gemm(transA='N', transB='N', ..., B_data, ..., A_data, ...)`. This computes $B^T @ A^T = C^T$. The result $C^T$ is written in column-major into the output buffer, which is precisely the desired $C$ in row-major.

This standard trick using `transA='N'` and `transB='N'` with swapped, row-major inputs is a common way PyTorch achieves row-major output for a general matrix multiply `C = A @ B` where A, B are row-major.

### Linear layer: `Y = X @ Wᵀ`

For a linear layer computation `Y = X @ Wᵀ`, where `X` (`[M, K]`) and `W` (`[N, K]`) are in row-major layout, PyTorch desires `Y` (`[M, N]`) also in row-major. To achieve this with a BLAS routine outputting column-major, PyTorch configures BLAS to compute $Y^T = W @ X^T$.

This involves a BLAS call computing $op(A) @ op(B)$ where $op(A)$ is $W$ and $op(B)$ is $X^T$. Using the rule that row-major data with `trans='T'` yields the matrix ($M$) and `trans='N'` yields the transpose ($M^T$):

- BLAS operand A uses $W$'s row-major data. To see $W$, `transA = 'T'`.
- BLAS operand B uses $X$'s row-major data. To see $X^T$, `transB = 'N'`.

The BLAS call uses `(transA='T', transB='N')` with $W$'s data as the first operand and $X$'s data as the second. It computes $W @ X^T = Y^T$, writing the result in column-major, which PyTorch interprets as the desired row-major $Y$.

### Backward pass operations

The backward pass similarly uses GEMMs configured to produce row-major gradients:

- `dX = dY @ W`: With `dY` (`[M, K]`) and `W` (`[K, N]`) row-major, `dX` (`[M, N]`) is needed row-major. BLAS computes $dX^T = W^T @ dY^T$.
    - BLAS operand A uses $W$'s row-major data. Needs $W^T \implies$ `transA = 'N'`.
    - BLAS operand B uses $dY$'s row-major data. Needs $dY^T \implies$ `transB = 'N'`.
    - BLAS call uses `(transA='N', transB='N')` on $W$'s and $dY$'s data, computing $W^T @ dY^T$.
- `dW = dYᵀ @ X`: With `dY` (`[K, N]`) and `X` (`[K, M]`) row-major, `dW` (`[N, M]`) is needed row-major. BLAS computes $dW^T = X^T @ dY$.
    - BLAS operand A uses $X$'s row-major data. Needs $X^T \implies$ `transA = 'N'`.
    - BLAS operand B uses $dY$'s row-major data. Needs $dY \implies$ `transB = 'T'`.
    - BLAS call uses `(transA='N', transB='T')` on $X$'s and $dY$'s data, computing $X^T @ dY$.

In summary, for PyTorch's row-major operations:

- Forward pass `Y = X @ Wᵀ` maps to BLAS calculating $W @ X^T$ using `(T, N)` flags on the row-major data of $W$ and $X$.
- Backward pass `dX = dY @ W` maps to BLAS calculating $W^T @ dY^T$ using `(N, N)` flags on the row-major data of $W$ and $dY$.
- Backward pass `dW = dYᵀ @ X` maps to BLAS calculating $X^T @ dY$ using `(N, T)` flags on the row-major data of $X$ and $dY$.

Revisiting the GPT-3-XL model GEMM table from TraceLens:

| name        | param: M | param: N | param: K | param: bias | param: stride_A | param: stride_B | param: transpose |
|-------------|----------|----------|----------|-------------|-----------------|-----------------|------------------|
| aten::addmm | 10240    | 8192     | 2048     | TRUE        | (2048, 1)       | (1, 2048)       | (True, False)    |
| aten::mm    | 10240    | 2048     | 8192     | FALSE       | (8192, 1)       | (2048, 1)       | (False, False)   |
| aten::mm    | 8192     | 2048     | 10240    | FALSE       | (1, 8192)       | (2048, 1)       | (False, True)    |

This table shows how `aten::addmm` (forward) and `aten::mm` (backward) calls map to underlying GEMM operations. The `param: M, N, K` values are likely the dimensions of the *PyTorch operation result* (`M x N` with inner dim `K`). The `param: transpose | (transA, transB)` are the BLAS flags used by the wrapper for the operands passed to the BLAS call.

Interpret the trace entries based on the understanding that PyTorch uses row-major data and BLAS receives flags to mathematically interpret this data for computing the transpose of the desired result:

1. `aten::addmm` (forward): Corresponds to `Y = X @ Wᵀ`. Trace flags: `(True, False)`. This matches the `(T, N)` needed for BLAS to compute $W @ X^T$.
2. First `aten::mm` (backward dX): Corresponds to `dX = dY @ W`. Trace flags: `(False, False)`. This matches the `(N, N)` needed for BLAS to compute $W^T @ dY^T$.
3. Second `aten::mm` (backward dW): Corresponds to `dW = dYᵀ @ X`. Trace flags: `(False, True)`. This matches the `(N, T)` needed for BLAS to compute $X^T @ dY$.

This confirms how the trace flags correspond to the BLAS transpose configurations used with row-major input data to achieve row-major output using the $C^T = B^T A^T$ trick.

The following pseudo code summarizes the transpose flag logic.

```{note}
This Python code snippet provides a simplified view; PyTorch's actual implementation is more intricate, accounting for the specific GEMM variant and output requirements.
```

```python
def is_col_major(T):
    return T.stride(0) == 1 and T.stride(1) >= T.shape[0]

def get_blas_transpose_flags(A, B):
    transA = 'N' if is_col_major(A) else 'T' # If A is col-major, BLAS sees it as is ('N')
    transB = 'N' if is_col_major(B) else 'T' # If B is col-major, BLAS sees it as is ('N')
    return transA, transB
```

### Edge cases

```{warning}
One common assumption is that flattening a tensor shape like `[B, L, d_model]` to `[B⋅L, d_model]` is a cost-free metadata operation. This is only true if the last dimension (`d_model`) is contiguous.
```

- If the last dimension is not contiguous, PyTorch might be forced to insert a *copy* or *transpose* operation to create a physically contiguous tensor that BLAS can work with efficiently.
- Furthermore, even if the tensor layout *could* theoretically be used by BLAS (for example, certain striding patterns), some highly tuned BLAS libraries might lack kernels optimized for those specific layouts. In such instances, a *copy* or *transpose buffer* is inserted behind the scenes by PyTorch or the BLAS wrapper. Consequently, what the BLAS routine actually operates on might not be the original tensor directly, but rather a *temporary buffer* created for compatibility or performance.

## GEMM dimension efficiency

Beyond mapping shapes to BLAS calls, TraceLens provides deeper insights into the efficiency of GEMM operations, specifically focusing on Tensile kernels used on ROCm GPUs. This analysis complements the existing roofline metrics by breaking down performance limitations related to how GEMM computations are tiled and scheduled onto the GPU.

By default, the tool provides roofline metrics summarizing overall performance for operations like `aten::mm`:

*Original output example:*

| name     | M    | N    | K     | bias  | FLOPS/Byte_first | TFLOPS/s_mean |
|----------|------|------|-------|-------|------------------|---------------|
| aten::mm | 2048 | 2048 | 10240 | False | 930.90           | 521.57        |

The enhanced analysis adds specific efficiency metrics for Tensile GEMM kernels, derived from the kernel's structure and the problem dimensions. See the [`examples/gemm_dim_eff.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/gemm_dim_eff.ipynb) notebook for example usage.

*New output example (Tensile kernels):*

| name     | M    | N    | K     | ... | mt_m | mt_n | num_tiles | tile_eff | wq_eff | dim_eff | ... | TFLOPS/s_mean |
|----------|------|------|-------|-----|------|------|-----------|----------|--------|---------|-----|---------------|
| aten::mm | 2048 | 2048 | 10240 | ... | 256  | 64   | 256       | 1.00     | 0.84   | 0.84    | ... | 521.57        |

### Key metrics

The enhanced output includes the following dimension-efficiency columns.

- `mt_m`, `mt_n`: Macro-tile dimensions extracted from the kernel name.
- `num_tiles`: Total number of tiles after padding.
- `tile_eff`: Tile quantization efficiency. Measures efficiency loss due to input matrix dimensions not being perfectly divisible by tile dimensions.
- `wq_eff`: Wave quantization efficiency. Measures efficiency loss due to the total number of tiles not perfectly filling all compute units in the final processing wave.
- `dim_eff`: Net dimension efficiency. The product of `tile_eff` and `wq_eff`, representing the combined efficiency impact of tiling and scheduling.

### Understanding the concepts

The following subsections define tile quantization efficiency, wave quantization efficiency, and the combined dimension efficiency metric.

#### Tile quantization efficiency (`tile_eff`)

Tiled computations divide matrices into smaller sub-blocks (tiles: `mt_m x mt_n`) for processing. If the matrix dimensions (M, N) are not exact multiples of these tile sizes, the implementation effectively pads the matrices to the nearest multiple of the tile size.

Padded dimensions:

```
M_pad = ceil(M / mt_m) × mt_m
N_pad = ceil(N / mt_n) × mt_n
```

This padding introduces extra computations on regions that don't contribute to the final result.

Tile efficiency:

```
tile_eff = (M × N) / (M_pad × N_pad)
```

Values `< 1` indicate wasted computation due to padding.

#### Wave quantization efficiency (`wq_eff`)

GPUs execute tiles across many compute units (CUs), also known as streaming multiprocessors (SMs). If the total number of tiles isn't a multiple of the number of available CUs, the final wave will leave some CUs idle.

Total tiles:

```
B = (M_pad × N_pad) / (mt_m × mt_n)
```

Number of waves:

```
num_waves = ceil(B / num_cus)
```

Wave efficiency:

```
wq_eff = B / (num_waves × num_cus)
```

#### Net dimension efficiency (`dim_eff`)

This is the combined impact:

```
dim_eff = tile_eff × wq_eff
```

### Why these metrics matter: diagnosing bottlenecks

For compute-bound GEMMs, actual performance is often less than peak theoretical. These metrics help explain why:

- Low `tile_eff`: Indicates high padding overhead.
- Low `wq_eff`: Indicates poor utilization of compute units in the final wave.

GEMM tuning can help improve these metrics to a certain extent.

However, these are just part of the picture. Even when tiling and wave quantization are optimal, there can still be performance degradation due to:

- Shader clock (`sclk`) throttling: Some workloads might not run at peak clock speeds due to power or thermal constraints.
- Cache behavior: Cache misses can occur even for compute-bound GEMMs, affecting throughput and instruction efficiency.

These factors should be analyzed alongside the dimension efficiency metrics to get a complete performance picture.

### How it's calculated

Step by step:

1. Identify problem shape: Extract M, N, K from input tensors.

   ```{note}
   In BLAS libraries, M and N are swapped relative to PyTorch's view, as BLAS libraries are column-major while PyTorch is row-major. This mapping is handled internally. See [PyTorch source - Blas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp#L102-L129) for more detail. This is accounted for in TraceLens as well when computing the tiles across the M and N dimensions.
   ```

2. Extract tile size: Parse `mt_m`, `mt_n` from kernel name (for example, `MT256x144x32` → `mt_m = 256, mt_n = 144`).

3. Calculate tiles:

   ```
   float_tiles_m = M / mt_m
   float_tiles_n = N / mt_n
   tiles_m = ceil(float_tiles_m)
   tiles_n = ceil(float_tiles_n)
   num_tiles = tiles_m * tiles_n
   ```

4. `tile_eff`:

   ```
   tile_eff = (M * N) / (tiles_m * mt_m * tiles_n * mt_n)
   ```

5. `wq_eff` (assume `num_cus` known, for example 304 for MI300X):

   ```
   float_rounds = num_tiles / num_cus
   rounds = ceil(float_rounds)
   wq_eff = num_tiles / (rounds * num_cus)
   ```

6. `dim_eff`:

   ```
   dim_eff = tile_eff * wq_eff
   ```

### Calculation examples

Assuming `num_cus = 304` (for example, AMD Instinct™ MI300X).

#### Example 1: perfect tiling, suboptimal wave quantization

- Input: M = 10240, N = 2048, K = 2048
- Kernel tile: `mt_m = 256`, `mt_n = 64`

```text
float_tiles_m = 40
float_tiles_n = 32
tiles_m = 40, tiles_n = 32
tile_eff = 1.0
num_tiles = 1280
float_rounds = 4.21
rounds = 5
wq_eff = 0.842
dim_eff = 0.842
```

#### Example 2: slight padding, good wave quantization

- Input: M = 2048, N = 10240, K = 2048
- Kernel tile: `mt_m = 256`, `mt_n = 144`

```text
float_tiles_m = 8
float_tiles_n = 71.11
tiles_m = 8, tiles_n = 72
tile_eff ≈ 0.9877
num_tiles = 576
float_rounds = 1.895
rounds = 2
wq_eff ≈ 0.947
dim_eff ≈ 0.935
```

### Important considerations

Keep the following constraints in mind when interpreting dimension-efficiency results.

- Scope: This analysis assumes standard tiled GEMMs. Techniques like Stream-K or Split-K are not yet modeled.
- Relevance: These metrics are most useful for compute-bound GEMMs. Use FLOPS/Byte to determine whether a GEMM is compute- or memory-bound.

For hands-on usage, work through the [`examples/gemm_dim_eff.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/gemm_dim_eff.ipynb) notebook. This is a new feature; report any issues or suggestions to the TraceLens team.

## Related topics

- [Trace2Tree](../conceptual/trace2tree.md)
- [Perf model walkthrough](../conceptual/triton-perf-model-walkthrough.md)
- [Torch profiling analysis](../conceptual/torch-profiling-analysis.md)
- [Performance report columns](../reference/perf-report-columns.md)
- [Generate a performance report from PyTorch](../how-to/generate-perf-report-pytorch.md)
