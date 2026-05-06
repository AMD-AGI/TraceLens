# How TraceLens calculates TB/s for `triton_poi_fused_mul_silu_2`

## Step 1: Locate the Inductor cache wrapper file

TraceLens scans `*.py` files under the TorchInductor cache directory (e.g. `/tmp/torchinductor_<user>/`) and finds the wrapper file containing `async_compile.triton(...)`.

```
/tmp/torchinductor_<user>/<hash_prefix>/<hash>.py
```

## Step 2: Extract three fields via regex

### Field 1: `Original ATen` (used for FLOPs, not TB/s)

```python
# Source Nodes: [mul_4, silu], Original ATen: [aten.mul, aten.silu]
```

Extracted: `aten_ops = ["aten.mul", "aten.silu"]`

### Field 2: `size_hints` (element count)

```python
@triton_heuristics.pointwise(
    size_hints=[268435456],
    ...
)
```

Extracted: `xnumel = 268,435,456`, `rnumel = 1` (pointwise kernel, no reduction dimension)

Where `268,435,456 = 8 x 4,096 x 8,192` (batch x seq_len x ffn_dim).

### Field 3: `triton_meta.signature` (pointer dtypes)

```python
triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, ...}
```

Parsed per slot:

| Slot | Value | Is pointer? | Bytes per element |
|---|---|---|---|
| 0 | `*bf16` | Yes (starts with `*`) | 2 |
| 1 | `*bf16` | Yes (starts with `*`) | 2 |
| 2 | `i32` | No (loop bound `xnumel`) | skipped |

Extracted: `ptr_bytes = [2, 2]`, `dtype = "bf16"`

These two pointers correspond to the kernel arguments:

```python
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
#           ^            ^        ^
#           slot 0       slot 1   slot 2
#           *bf16        *bf16    i32
#           output       input    loop bound (not a data pointer)
```

## Step 3: Calculate `bytes_moved`

Each pointer reads or writes `xnumel` elements. Total memory traffic is the sum across all pointers:

```
bytes_moved = sum(ptr_bytes) * xnumel
            = (2 + 2) * 268,435,456
            = 4 * 268,435,456
            = 1,073,741,824 bytes
            = 1,024 MB
```

Code: `triton_compiled_perf_model.py` `bytes()` method:

```python
# Pointwise (rnumel == 1):
return float(sum(ptr_bytes) * xnumel)
```

## Step 4: Get kernel time from Chrome trace

The Chrome trace (`trace.json.gz`) contains two related events:

| Event | Category | Duration | External id |
|---|---|---|---|
| `triton_poi_fused_mul_silu_2` | `cpu_op` | 16 us | 58 |
| `triton__0d1d2de` | `kernel` | 1,320 us | correlated to ext_id 58 |

The CPU event (16 us) is just the kernel **launch** overhead. The GPU event (1,320 us) is the actual **execution** time on the GPU. TraceLens uses the GPU kernel time.

```
kernel_time = 1,320 us
```

## Step 5: Calculate TB/s

```
TB/s = (bytes_moved / 1e12) / (kernel_time_us / 1e6)

     = (1,073,741,824 / 1,000,000,000,000) / (1,320 / 1,000,000)
       ---------------------------------     --------------------
       0.001073741824 TB                      0.00132 seconds

     = 0.001073741824 / 0.00132

     = 0.813441 TB/s
```

## Summary of all calculated metrics for this kernel

| Metric | Value | How calculated |
|---|---|---|
| `bytes_moved` | 1,073,741,824 bytes (1,024 MB) | `sum([2, 2]) * 268,435,456` |
| `Kernel Time` | 1,320 us | GPU kernel duration from Chrome trace |
| **`TB/s`** | **0.813441** | `(1,073,741,824 / 1e12) / (1,320 / 1e6)` |
| `GFLOPS` | 1.342177 | `(4 + 1) * 268,435,456 / 1e9` (silu=4, mul=1) |
| `TFLOPS/s` | 1.016801 | `(1.342177 / 1e3) / (1,320 / 1e6)` |
| `FLOPS/Byte` | 1.25 | `1,342,177,280 / 1,073,741,824` |
| `Compute Spec` | `vector_bf16` | `get_maf_type()` + `get_compute_precision()` |

## Data flow diagram

```
TorchInductor Cache (.py)               Chrome Trace (.json.gz)
         |                                       |
         v                                       v
  _parse_wrapper()                     TraceLens trace parser
         |                                       |
    +---------+                            +-----------+
    | xnumel  |                            | kernel    |
    | = 268M  |                            | duration  |
    |         |                            | = 1320 us |
    | ptr_bytes                            +-----------+
    | = [2, 2]|                                  |
    +---------+                                  |
         |                                       |
         v                                       |
    bytes()                                      |
    = sum([2,2]) * 268M                          |
    = 1,073,741,824                              |
         |                                       |
         +------------------+--------------------+
                            |
                            v
                  TB/s = bytes / time
                  = 0.001074 TB / 0.00132 s
                  = 0.813441 TB/s
```
