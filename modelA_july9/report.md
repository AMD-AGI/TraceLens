## Summary
- **MI325 is 13% slower than H200** (365 ms vs 323 ms).  
  → Close **42 ms** for parity.  
  → Save **116 ms** for a 30% win.

---

### 1. Memory Copies
- MI325 shows **59.5 ms** copy time vs **45 ms** on H200, mostly from extra transpose ops.  
**Action**:  
  - Fix input layout to eliminate redundant transposes on MI325.  
  - Port the TAIL typecast to decrease the typecase `aten::copy`.

---

### 2. Tail Kernels
- Missing on MI325: `to_dtype`, `to_f32`, `f32_to_bf16`, `cudnn_hardtanh_back`.  
- Ported but regressed: `FusedResidualAdd Backward` (+2.8 ms, 130% slower).  
**Action**:  
  - Port missing tails; tune regressed ones.

---

### 3. Norms & Elementwise Ops (RMSNorm)
- `LayerNorm` is ported to TRI-DAO fused kernel; performance review pending.  
- `RMSNorm` not yet ported → high counts & time for elementwise ops:  
  - `sum` (+13 ms), `mul` (+11 ms), `div`, `mean`, `pow`, `add_` (+2.5 ms).  
**Action**:  
  - Prioritize **RMSNorm** porting to eliminate scattered ops.

---

### 4. Activation Fusion
- `aten::clamp` (+5.7 ms) → fusion opportunity from `linear+act` or `conv+act`.  
- `aten::mm` count higher on MI325 (772 vs 651) → likely missing fusion.  
- `bmm` (+11.9 ms delta) likely tied to the same.  
**Action**:  
  - Finalize & deliver activation fusions (linear-act and conv-act).

---

### 5. TritonMultiHeadAttnCore
- Only appears on **H200**, not on MI325.  
**Action**:  
  - Follow up to understand what this op does and if a corresponding path can be enabled on MI325.

---

### 6. Convolutions
- MI325 outperforms H200 on matched conv ops.  
**Action**:  
  - None. This validates the **backend’s strength** on convs.
