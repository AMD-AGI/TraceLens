<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

## What's New: Deeper Efficiency Metrics
This document details a new feature in TraceLens, providing deeper insights into the efficiency of GEMM (General Matrix Multiplication) operations, specifically focusing on Tensile kernels used on ROCm GPU. This analysis complements the existing Roofline metrics by breaking down performance limitations related to how gemm computations are tiled and scheduled onto the GPU.

By default, the tool provides Roofline metrics summarizing overall performance for operations like `aten::mm`:

**Original Output Example:**

| name     | M | N | K | bias | FLOPS/Byte_first | TFLOPS/s_mean |
|----------|----------|----------|----------|-------------|------------------|----------------|
| aten::mm | 2048    | 2048    | 10240     | False       | 930.90          | 521.57         |


The enhanced analysis now adds specific efficiency metrics for Tensile GEMM kernels, derived from the kernel's structure and the problem dimensions:

**New Output Example (Tensile Kernels):**
Checkout the examples/gemm_dim_eff.ipynb for example usage. 

| name       | M     | N    | K    | ... | mt_m | mt_n | num_tiles | tile_eff | wq_eff | dim_eff | ... | TFLOPS/s_mean |
|------------|-------|------|------|-----|------|-----|---|----------|--------|----------|--------|--------|
| aten::mm   | 2048 | 2048 | 10240 | ... | 256  | 64   | 256       | 1.00    | 0.84  | 0.84    | ...         | 521.57 |



---

## Key New Metrics

- **`mt_m`, `mt_n`**: Macro-tile dimensions extracted from the kernel name.
- **`num_tiles`**: Total number of tiles after padding.
- **`tile_eff`**: Tile Quantization Efficiency. Measures efficiency loss due to input matrix dimensions not being perfectly divisible by tile dimensions.
- **`wq_eff`**: Wave Quantization Efficiency. Measures efficiency loss due to the total number of tiles not perfectly filling all compute units in the final processing wave.
- **`dim_eff`**: Net Dimension Efficiency. The product of tile_eff and wq_eff, representing the combined efficiency impact of tiling and scheduling.

---

## Understanding the Concepts

### 1. Tile Quantization Efficiency (`tile_eff`)

Tiled computations divide matrices into smaller sub-blocks (tiles: `mt_m x mt_n`) for processing. If the matrix dimensions (M, N) are not exact multiples of these tile sizes, the implementation effectively pads the matrices to the nearest multiple of the tile size.

**Padded Dimensions:**

    M_pad = ceil(M / mt_m) × mt_m
    N_pad = ceil(N / mt_n) × mt_n

This padding introduces extra computations on regions that don’t contribute to the final result.

**Tile Efficiency:**

    tile_eff = (M × N) / (M_pad × N_pad)

Values < 1 indicate wasted computation due to padding.

---

### 2. Wave Quantization Efficiency (`wq_eff`)

GPUs execute tiles across many Compute Units (CUs) also known as Streaming multi-processors (SMs). If the total number of tiles isn’t a multiple of the number of available CUs, the final wave will leave some CUs idle.

**Total Tiles:**

    B = (M_pad × N_pad) / (mt_m × mt_n)

**Number of Waves:**

    num_waves = ceil(B / num_cus)

**Wave Efficiency:**

    wq_eff = B / (num_waves × num_cus)

---

### 3. Net Dimension Efficiency (`dim_eff`)

This is the combined impact:

    dim_eff = tile_eff × wq_eff
---

## Why These Metrics Matter: Diagnosing Bottlenecks

For compute-bound GEMMs, actual performance is often less than peak theoretical. These metrics help explain why:

- **Low tile_eff**: Indicates high padding overhead.
- **Low wq_eff**: Indicates poor utilization of compute units in the final wave. 

GEMM tuning can help improving these metrics to a certain extent.

However, these are just part of the picture. Even when tiling and wave quantization are optimal, there can still be performance degradation due to:

- **Shader clock (`sclk`) throttling**: Some workloads may not run at peak clock speeds due to power or thermal constraints.
- **Cache behavior**: Cache misses can occur even for compute-bound GEMMs, affecting throughput and instruction efficiency.

These factors should be analyzed alongside the dimension efficiency metrics to get a complete performance picture.

---

## How It's Calculated

### Step-by-Step:

1. **Identify Problem Shape**: Extract M, N, K from input tensors.

   > Note: In  BLAS libraries, M and N are swapped relative to PyTorch's view as BLAS libraries are column major while PyTorch is row major. This mapping is handled internally. See [PyTorch source - Blas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp#L102-L129) for more detail. This is accounted in TraceLens as well when computing the tiles across the M and N dimensions.

2. **Extract Tile Size**: Parse `mt_m`, `mt_n` from kernel name (e.g., `MT256x144x32` → `mt_m = 256, mt_n = 144`).

3. **Calculate Tiles:**
   ```
   float_tiles_m = M / mt_m
   float_tiles_n = N / mt_n
   tiles_m = ceil(float_tiles_m)
   tiles_n = ceil(float_tiles_n)
   num_tiles = tiles_m * tiles_n
   ```

4. **tile_eff**:
   ```
   tile_eff = (M * N) / (tiles_m * mt_m * tiles_n * mt_n)
   ```

5. **wq_eff** (assume `num_cus` known, e.g., 304 for MI300X):
   ```
   float_rounds = num_tiles / num_cus
   rounds = ceil(float_rounds)
   wq_eff = num_tiles / (rounds * num_cus)
   ```

6. **dim_eff**:
   ```
   dim_eff = tile_eff * wq_eff
   ```

---

## Calculation Examples

**Assuming `num_cus = 304` (e.g., MI300X)**

### Example 1: Perfect Tiling, Suboptimal Wave Quantization
- **Input**: M = 10240, N = 2048, K = 2048
- **Kernel Tile**: `mt_m = 256`, `mt_n = 64`

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

### Example 2: Slight Padding, Good Wave Quantization
- **Input**: M = 2048, N = 10240, K = 2048
- **Kernel Tile**: `mt_m = 256`, `mt_n = 144`

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

---

## Important Considerations

- **Scope**: This analysis assumes standard tiled GEMMs. Techniques like Stream-K or Split-K are not yet modeled.
- **Relevance**: These metrics are most useful for compute-bound GEMMs. Use FLOPS/Byte to determine whether a GEMM is compute- or memory-bound.


## Example Usage
Checkout the [Notebook](../examples/gemm_dim_eff.ipynb) for example usage. 
This is a new feature so please report any issues or suggestions to the TraceLens team.
