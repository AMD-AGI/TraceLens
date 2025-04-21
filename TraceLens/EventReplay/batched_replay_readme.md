
# Event Replay Artifact
 
This archive contains all necessary files to **reproduce and benchmark arbitrary PyTorch operations** outside the model code as well as TraceLens.

## Contents

- `event_replay_ir.json`: 
  - A JSON file containing replay instructions (tensor shapes, strides, dtypes, and other args and kwargs) for each extracted operation.
- `batched_replay.py`: 
  - A script to batch replay all operations from `event_replay_ir.json` and benchmark their execution times.
- `utils.py`: 
  - Utility functions used internally by the replay script (tensor creation, parsing helpers, etc).

## How to Use

1. **Extract** the zip contents to a working directory:
   
   ```bash
   unzip replay_code.zip
   cd replay_code
   ```

2. **Run the batched replay script**:

   ```bash
   python batched_replay.py event_replay_ir.json
   ```

   This will:
   - Initialize tensors according to the replay IR
   - Execute the corresponding operator
   - Benchmark and print the average latency per operator


## Example Output

```

[1/11] Replaying: aten::convolution
  Reconstructing arguments for 'aten::convolution'...
  Positional Args:
  input Tensor: {'shape': [20, 64, 56, 56], 'dtype': 'c10::BFloat16', 'strides': [200704, 3136, 56, 1]}
  weight Tensor: {'shape': [64, 64, 3, 3], 'dtype': 'c10::BFloat16', 'strides': [576, 9, 3, 1]}
  bias Tensor?: None
  stride SymInt[]: [1, 1]
  padding SymInt[]: [1, 1]
  dilation SymInt[]: [1, 1]
  transposed bool: False
  output_padding SymInt[]: [0, 0]
  groups SymInt: 1
  Keyword Args:
  Average time taken: 159.74 microseconds
  Successfully executed aten::convolution.
  Result: Tensor(shape=torch.Size([20, 64, 56, 56]), dtype=torch.bfloat16, device=cuda:0)

[2/11] Replaying: aten::convolution
  Reconstructing arguments for 'aten::convolution'...
  Positional Args:
  input Tensor: {'shape': [20, 512, 7, 7], 'dtype': 'c10::BFloat16', 'strides': [25088, 49, 7, 1]}
  weight Tensor: {'shape': [512, 512, 3, 3], 'dtype': 'c10::BFloat16', 'strides': [4608, 9, 3, 1]}
  bias Tensor?: None
  stride SymInt[]: [1, 1]
  padding SymInt[]: [1, 1]
  dilation SymInt[]: [1, 1]
  transposed bool: False
  output_padding SymInt[]: [0, 0]
  groups SymInt: 1
  Keyword Args:
  Average time taken: 159.15 microseconds
  Successfully executed aten::convolution.
  Result: Tensor(shape=torch.Size([20, 512, 7, 7]), dtype=torch.bfloat16, device=cuda:0)


...
--- Replay Summary ---
Total operations in file: 11
Attempted replays: 11
Successful replays: 11
Errors encountered: 0
----------------------
```

## Notes

- The replay focuses on **input tensor shapes, strides, dtypes, and operator arguments** — it does not reproduce actual model data (random values are used).
- The replayed operations can be **of any type** (e.g., GEMMs, convolutions, elementwise ops, reductions, etc.), depending on what was originally profiled.
