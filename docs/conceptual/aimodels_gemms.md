## 📌 GEMMs in AI Workloads

General Matrix Multiplications (GEMMs) are the **primary compute primitive** used in AI models. Efficient implementations of GEMMs are readily available through vendor-tuned libraries such as cuBLAS and hipBLAS. Therefore, whenever possible, the goal is to **reduce computation to a matrix multiply**.

This document explains how **model-level parameters** like batch size, sequence length, and hidden dimension translate into GEMM shapes, and subsequently how these shapes map to specific **BLAS kernel calls**.

By the end of this post, you should be equipped to understand the GEMM shapes, counts, and BLAS calls involved in **any AI model you encounter**.

---

## From Model Dimensions to GEMM Shapes

### Linear Layers in LLMs

Let's begin by understanding how linear layers, such as the MLP **up projection**, correspond to GEMM calls.

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
Where:
- `W` is the weight matrix with a shape of `[d_ff, d_model]`.

### Flattening for GEMM

To express this entire operation as a single GEMM, we flatten the batch and sequence dimensions of the input tensor:
```
X_flat: [B·L, d_model]
Wᵀ:     [d_model, d_ff]
Y = X_flat @ Wᵀ
```
This flattening yields the following GEMM shape parameters:
- `param: M = B·L`
- `param: N = d_ff`
- `param: K = d_model`

Here, K represents the **inner or shared dimension** between the input tensors involved in the multiplication.


***

## Prefill vs Decode

Let's now contrast the GEMM behavior during the **prefill** and **decode** phases of inference, focusing on how the sequence length (`L`) changes and affects the GEMM shapes.

In both phases, the input tensor for an MLP computation within an LLM initially has a shape like:
```
X: [B, L, d_model]
```
Where `B` is the batch size, `L` is the sequence length, and `d_model` is the hidden size. This projects to an output shape of `[B, L, d_ff]`, where `d_ff` is the MLP's expansion size.

As established earlier, to process this with a single GEMM, the batch and sequence dimensions of the input are flattened. The input effectively becomes `[B·L, d_model]` for the GEMM `X_flat @ Wᵀ`.

The key difference between prefill and decode lies in the value of `L`:
-   **Prefill Phase**: `L` is the actual input sequence length (which can be large).
-   **Decode Phase**: `L` is always `1`, as the model processes one token at a time to generate the next.

This difference in `L` directly impacts the `M` parameter of the GEMM `(M, N, K)`:

### GEMM Shape Summary:
| Mode     | Input Shape (Conceptual) | Flattened Input Shape | GEMM Shape `(M, N, K)`              | Notes                                |
|----------|--------------------------|-----------------------|-------------------------------------|---------------------------------------|
| Prefill  | `[B, L, d_model]`        | `[B·L, d_model]`      | `(B·L, d_ff, d_model)`              | `L` is the prompt length            |
| Decode   | `[B, 1, d_model]`        | `[B·1, d_model]`      | `(B, d_ff, d_model)`                | `L=1` for generating one token at a time |

Notice that in the decode phase, because `L=1`, the `M` parameter of the MLP GEMM becomes simply `B`. This means the computational cost of the MLP layers in decode remains constant per token regardless of the total sequence length generated so far. The dominant **O(L)** scaling cost during decode comes from the attention mechanism, not the MLPs.

### Real-world Example: LLaMA-2 7B

The table below shows actual data from **Tracelens** profiling, filtered specifically for MLP **up** and **gate** projection GEMMs in a LLaMA-2 7B model inference trace.

For this trace:
- `d_model = 4096`
- `d_ff = 11008`
- Batch size: `1` (`B=1`)
- Input length (for prefill): `597` (`L=597`)
- The trace included 36 decode steps.

| name     | param: M | param: N | param: K | param: bias | counts |
|----------|-----------|-----------|-----------|---------------|--------|
| aten::mm | 1         | 11008     | 4096      | FALSE         | 2304   |
| aten::mm | 597       | 11008     | 4096      | FALSE         | 64     |

Interpreting these entries based on the `M` parameter:
- The entry with `param: M = 597` corresponds to the **prefill** phase GEMM `(B·L = 1·597)`, which happens once per layer at the beginning of inference. Since there are 32 layers, this GEMM is called `64` times (32 up + 32 gate).
- The entry with `param: M = 1` corresponds to the **decode** phase GEMM `(B = 1)`, where `L=1`. These occur at each decode step for every layer. With 36 decode steps and 64 GEMMs per step (32 layers * 2), this GEMM is called `36 × 64 = 2304` times.

---

## 🔁 Backward Pass GEMMs

Next, let's explore the backward pass during training. A forward pass GEMM operation like `Y = X @ Wᵀ + b` necessitates **two corresponding backward GEMMs** to compute gradients:

```python
dX = dY @ W        # Gradient with respect to the input → resulting shape: [B·L, d_model]
dW = dYᵀ @ X       # Gradient with respect to the weight → resulting shape: [d_ff, d_model]
db = dY.sum(dim=0) # Gradient with respect to the bias   → resulting shape: [d_ff]
```

### GEMM Shapes:
| Operation         | GEMM Shape `(param: M, param: N, param: K)`       | Description                          |
|------------------|---------------------------------------------------|--------------------------------------|
| Forward          | `(B·L, d_ff, d_model)`                             | `X @ Wᵀ`                             |
| Backward dX      | `(B·L, d_model, d_ff)`                             | `dY @ W` (result of `[B·L, d_ff] @ [d_ff, d_model]`) |
| Backward dW      | `(d_ff, B·L, d_model)`                             | `dYᵀ @ X` (result of `[d_ff, B·L] @ [B·L, d_model]`) |

Let's look closer at the backward GEMM shapes:
- For `dX = dY @ W`, the operation is `[B·L, d_ff] @ [d_ff, d_model]`, which results in a shape of `[B·L, d_model]`.
- For `dW = dYᵀ @ X`, the operation is `[d_ff, B·L] @ [B·L, d_model]`, yielding a shape of `[d_ff, d_model]`.

### Real-world Example: GPT-3-XL

This table presents data from a **Tracelens**  of a single training step for **GPT-3-XL**.

For this example:
- `d_model = 2048`
- `d_ff = 8192`
- Batch size: `5`
- Sequence length: `2048`
- Thus, `param: M` for the flattened dimension is `5 × 2048 = 10240`.

| name        | param: M | param: N | param: K | count |
|-------------|-----------|-----------|-----------|--------|
| aten::addmm | 10240     | 8192      | 2048      | 24     |
| aten::mm    | 10240     | 2048      | 8192      | 24     |
| aten::mm    | 8192      | 2048      | 10240     | 24     |

We can interpret each entry based on the GEMM shapes:
- The `aten::addmm` call represents the forward pass GEMM (`X @ Wᵀ`).
- The first `aten::mm` call corresponds to the backward pass for `dX` (`dY @ W`).
- The second `aten::mm` call represents the backward pass for `dW` (`dYᵀ @ X`).

Each of these operations appears once per layer in the network. Given that GPT-3-XL has 24 layers, each of these GEMMs is called 24 times per training step, aligning with the 'count' column in the table.

---


## ⚙️ How PyTorch Calls BLAS

To fully grasp how PyTorch leverages BLAS for operations like GEMM, we must first understand the fundamental concept of **memory layout** for tensors and how BLAS libraries interpret the data buffers they receive.

### Memory Layout and Stride

Despite tensors often being represented as multi-dimensional arrays, their elements are stored in linear memory. For a 2D matrix, the two primary storage conventions are:

-   **Row-major**: Elements of the same row are stored consecutively in memory. PyTorch adopts this as its default layout.
-   **Column-major**: Elements of the same column are stored consecutively in memory. Many traditional BLAS libraries primarily optimize for this layout.

PyTorch's `.stride()` method provides insight into a tensor's memory arrangement. It returns a tuple where each value indicates the byte (or element, depending on datatype size) distance in linear memory to move to the next element along that dimension.
-   For a 2D tensor `T[i][j]` in **Row-major** layout, `.stride()` is typically `(num_cols, 1)`. Moving to `T[i][j+1]` requires stepping 1 element, while moving to `T[i+1][j]` requires stepping `num_cols` elements.
-   For a 2D tensor `T[i][j]` in **Column-major** layout, `.stride()` is typically `(1, num_rows)`. Moving to `T[i+1][j]` requires stepping 1 element, while moving to `T[i][j+1]` requires stepping `num_rows` elements.

---

### BLAS Transpose and Row-Major Output

The core BLAS GEMM routine typically computes $C = \alpha \cdot op(A) \cdot op(B) + \beta \cdot C$, where $op(X)$ is either $X$ or $X^T$ depending on the `transA` and `transB` flags ('N' for No Transpose, 'T' for Transpose) passed to the function. By default, BLAS expects input matrices corresponding to the 'N' flag to be in column-major layout. Crucially, the resulting matrix $C$ is written into the output buffer in **column-major** format by default.

PyTorch, however, uses row-major layout internally and desires the result of a GEMM operation to also be in row-major layout *without* an extra copy or transpose step outside of the BLAS call. PyTorch achieves this by cleverly leveraging the `trans` flags and the relationship between row-major and column-major layouts.

A matrix $M$ stored in row-major memory has the exact same element ordering as the matrix $M^T$ stored in column-major memory. PyTorch uses this identity. To get a row-major result $C$ from a BLAS call that outputs in column-major, PyTorch requests BLAS to compute $C^T$ and write it in column-major. Since $C^T$ in column-major is $C$ in row-major, the output buffer will contain the desired row-major $C$.

Mathematically, the operation $C = A @ B$ (where $A, B, C$ are desired in row-major) is equivalent to computing $C^T = (A @ B)^T = B^T @ A^T$. PyTorch therefore configures the BLAS call to compute $B^T @ A^T$ using the row-major data of $B$ and $A$.

Here's how the `transA` and `transB` flags work in this context when passing **row-major data** to BLAS via a wrapper like PyTorch's:
-   Passing row-major data for matrix $M$ with `trans = 'T'` tells BLAS to mathematically treat this data as $M$. (BLAS expects row-major data for 'T' if it wants to use the matrix directly).
-   Passing row-major data for matrix $M$ with `trans = 'N'` tells BLAS to mathematically treat this data as $M^T$. (BLAS expects column-major data for 'N'; giving it row-major data makes it see the transpose).

So, to compute $C^T = B^T @ A^T$ using row-major data for $B$ and $A$ and get $C$ row-major in the output buffer:
-   Pass $B$'s row-major data as the first operand data (`A_data` in BLAS call). To make BLAS see $B^T$, use `transA = 'N'`.
-   Pass $A$'s row-major data as the second operand data (`B_data` in BLAS call). To make BLAS see $A^T$, use `transB = 'N'`.
-   The BLAS call becomes `gemm(transA='N', transB='N', ..., B_data, ..., A_data, ...)`. This computes $B^T @ A^T = C^T$. The result $C^T$ is written in column-major into the output buffer, which is precisely the desired $C$ in row-major.

This standard trick using `transA='N'` and `transB='N'` with swapped, row-major inputs is a common way PyTorch achieves row-major output for a general matrix multiply `C = A @ B` where A, B are row-major.

---


### Linear Layer: `Y = X @ Wᵀ`

For a linear layer computation `Y = X @ Wᵀ`, where `X` (`[M, K]`) and `W` (`[N, K]`) are in row-major layout, PyTorch desires `Y` (`[M, N]`) also in row-major. To achieve this with a BLAS routine outputting column-major, PyTorch configures BLAS to compute $Y^T = W @ X^T$.

This involves a BLAS call computing $op(A) @ op(B)$ where $op(A)$ is $W$ and $op(B)$ is $X^T$. Using the rule that row-major data with `trans='T'` yields the matrix ($M$) and `trans='N'` yields the transpose ($M^T$):
-   BLAS operand A uses $W$'s row-major data. To see $W$, `transA = 'T'`.
-   BLAS operand B uses $X$'s row-major data. To see $X^T$, `transB = 'N'`.

The BLAS call uses `(transA='T', transB='N')` with $W$'s data as the first operand and $X$'s data as the second. It computes $W @ X^T = Y^T$, writing the result in column-major, which PyTorch interprets as the desired row-major $Y$.

### Backward Pass Operations:

The backward pass similarly uses GEMMs configured to produce row-major gradients:

-   **`dX = dY @ W`**: With `dY` (`[M, K]`) and `W` (`[K, N]`) row-major, we need `dX` (`[M, N]`) row-major. BLAS computes $dX^T = W^T @ dY^T$.
    -   BLAS operand A uses $W$'s row-major data. Needs $W^T \implies$ `transA = 'N'`.
    -   BLAS operand B uses $dY$'s row-major data. Needs $dY^T \implies$ `transB = 'N'`.
    -   BLAS call uses `(transA='N', transB='N')` on $W$'s and $dY$'s data, computing $W^T @ dY^T$.

-   **`dW = dYᵀ @ X`**: With `dY` (`[K, N]`) and `X` (`[K, M]`) row-major, we need `dW` (`[N, M]`) row-major. BLAS computes $dW^T = X^T @ dY$.
    -   BLAS operand A uses $X$'s row-major data. Needs $X^T \implies$ `transA = 'N'`.
    -   BLAS operand B uses $dY$'s row-major data. Needs $dY \implies$ `transB = 'T'`.
    -   BLAS call uses `(transA='N', transB='T')` on $X$'s and $dY$'s data, computing $X^T @ dY$.

In summary, for PyTorch's row-major operations:
-   Forward pass `Y = X @ Wᵀ` maps to BLAS calculating $W @ X^T$ using `(T, N)` flags on the row-major data of $W$ and $X$.
-   Backward pass `dX = dY @ W` maps to BLAS calculating $W^T @ dY^T$ using `(N, N)` flags on the row-major data of $W$ and $dY$.
-   Backward pass `dW = dYᵀ @ X` maps to BLAS calculating $X^T @ dY$ using `(N, T)` flags on the row-major data of $X$ and $dY$.

Let’s revisit the GPT-3-XL model gemm table from **Tracelens**:

| name        | param: M | param: N | param: K | param: bias | param: stride_A | param: stride_B | param: transpose |
|--------------|-----------|-----------|-----------|--------------|------------------|------------------|--------------------|
| aten::addmm  | 10240     | 8192      | 2048      | TRUE         | (2048, 1)        | (1, 2048)        | (True, False)      |
| aten::mm     | 10240     | 2048      | 8192      | FALSE        | (8192, 1)        | (2048, 1)        | (False, False)     |
| aten::mm     | 8192      | 2048      | 10240     | FALSE        | (1, 8192)        | (2048, 1)        | (False, True)      |

This table shows how `aten::addmm` (forward) and `aten::mm` (backward) calls map to underlying GEMM operations. The `param: M, N, K` values are likely the dimensions of the *PyTorch operation result* (`M x N` with inner dim `K`). The `param: transpose | (transA, transB)` are the BLAS flags used by the wrapper for the operands passed to the BLAS call.

Let's interpret the trace entries based on our understanding that PyTorch uses row-major data and BLAS receives flags to mathematically interpret this data for computing the transpose of the desired result:

1.  **`aten::addmm` (Forward)**: Corresponds to `Y = X @ Wᵀ`. Trace flags: `(True, False)`. This matches the `(T, N)` needed for BLAS to compute $W @ X^T$.
2.  **First `aten::mm` (Backward dX)**: Corresponds to `dX = dY @ W`. Trace flags: `(False, False)`. This matches the `(N, N)` needed for BLAS to compute $W^T @ dY^T$.

3.  **Second `aten::mm` (Backward dW)**: Corresponds to `dW = dYᵀ @ X`. Trace flags: `(False, True)`. This matches the `(N, T)` needed for BLAS to compute $X^T @ dY$.


This confirms how the trace flags correspond to the BLAS transpose configurations used with row-major input data to achieve row-major output via the $C^T = B^T A^T$ trick.


Lets summarize our understanding of the transpose flag by writing pseudo code for the logic:

*(Note: This Python code snippet provides a simplified view; PyTorch's actual implementation is more intricate, accounting for the specific GEMM variant and output requirements.)*



```python

def is_col_major(T):
    return T.stride(0) == 1 and T.stride(1) >= T.shape[0]

def get_blas_transpose_flags(A, B):
    transA = 'N' if is_col_major(A) else 'T' # If A is col-major, BLAS sees it as is ('N')
    transB = 'N' if is_col_major(B) else 'T' # If B is col-major, BLAS sees it as is ('N')
    return transA, transB

```

## ⚠️ Edge Cases


-   One common assumption is that flattening a tensor shape like `[B, L, d_model]` to `[B⋅L, d_model]` is a cost-free metadata operation. This is only true if the last dimension (`d_model`) is contiguous.

-   If the last dimension is not contiguous, PyTorch may be forced to insert a **copy** or **transpose** operation to create a physically contiguous tensor that BLAS can work with efficiently.

-   Furthermore, even if the tensor layout *could* theoretically be used by BLAS (e.g., certain striding patterns), some highly tuned BLAS libraries might lack kernels optimized for those specific layouts. In such instances, a **copy** or **transpose buffer** is inserted behind the scenes by PyTorch or the BLAS wrapper. Consequently, what the BLAS routine actually operates on might not be the original tensor directly, but rather a **temporary buffer** created for compatibility or performance.

