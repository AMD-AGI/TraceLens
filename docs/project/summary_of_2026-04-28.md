# TraceLens — TODO Issues Summary

**Generated:** 2026-04-28 · **Source:** [AMD-AGI/projects/37](https://github.com/orgs/AMD-AGI/projects/37) · **Filter:** Status = Todo (open issues only)

---

## At a Glance

| Category | Count |
|---|---|
| 🐛 Bugs | 7 |
| 📐 Perf Models | 10 |
| ⭐ Major Features | 10 |
| 🔷 JAX | 8 |
| 📄 Docs / Release | 6 |
| 🔧 Refactor / API | 8 |
| 🔗 Enhancements | 18 |
| **Total TODO** | **67** |

---

## 🐛 Bugs

| Issue | Title | Priority | What Needs to Be Done | Potential Solution |
|---|---|---|---|---|
| [#185](https://github.com/AMD-AGI/TraceLens/issues/185) | Make input strides optional in perf model | 🔴 High | Some traces omit tensor strides from `Concrete Inputs`; perf model crashes instead of falling back gracefully | In `PerfModel/perf_model.py` base parsers, wrap stride extraction with a `None` check; default to contiguous row-major strides when absent |
| [#202](https://github.com/AMD-AGI/TraceLens/issues/202) | TE v2 pseudo CPU ops not unique across GPU ranks | 🔴 High | `correlation` used as `External id` in TEv2 pseudo ops is not unique across GPU ranks, breaking expert-parallel per-rank statistics | In `Trace2Tree/trace_to_tree.py` (~line 130), use a composite key `rank_id + correlation`, or use the original `External id` if already globally unique |
| [#279](https://github.com/AMD-AGI/TraceLens/issues/279) | NeMo framework trace parse failure | 🔴 High | Assertion error raised when parsing NeMo traces; likely non-standard event structure or correlation IDs | Reproduce with provided trace; add defensive checks in `Trace2Tree/trace_to_tree.py`; add NeMo fixture to test suite |
| [#297](https://github.com/AMD-AGI/TraceLens/issues/297) | Perf reports for SGLang are mostly empty | 🔴 High | SGLang (DeepSeek) traces show deep call stack in Perfetto but TraceLens report is nearly empty — most ops missing | Inspect raw events; likely `hipGraphLaunch` graph capture pattern — apply `trace_capture_merge_experimental.py` or add SGLang-specific event detection |
| [#510](https://github.com/AMD-AGI/TraceLens/issues/510) | Missing kernels in unified perf report | 🔴 High | Tree traversal from `cpu_root_nodes` misses GPU kernels whose call stack goes through Python functions without any `cpu_op` ancestor | In `TreePerfAnalyzer.collect_unified_perf_events()`, add a second pass comparing all GPU kernel UIDs against those already included; add unmatched as "orphan" entries |
| [#214](https://github.com/AMD-AGI/TraceLens/issues/214) | Warn on host/device misaligned timelines | 🟡 Medium | When CPU and GPU clocks are unsynchronized, GPU timestamps may appear before the CPU op that launched them — silently corrupt attribution | In `Trace2Tree/trace_to_tree.py` after `add_gpu_ops_to_tree()`, check if GPU kernel `ts < parent CPU ts` by > 10 µs; emit `warnings.warn()` with guidance |
| [#482](https://github.com/AMD-AGI/TraceLens/issues/482) | Spurious warnings from dict_cat2names loop | 🟢 Low | Loop in `generate_perf_report_pytorch.py` (~line 325) iterates all categories even when absent from the trace, emitting confusing user warnings | Add a guard: check if category exists in trace op set before processing; if not, `continue` silently — single-line fix |

---

## 📐 Perf Models

| Issue | Title | Priority | What Needs to Be Done | Potential Solution |
|---|---|---|---|---|
| [#516](https://github.com/AMD-AGI/TraceLens/issues/516) | Add perf models for Primus op groups | 🔴 High | Primus training traces have 67–98% "other" ops. Target: DeepSeek V2 Lite, GPT OSS 20B, Llama3 70B, Qwen 3 8B, Kimi K2, Mamba 370M → drop "other" to 2–8% | Identify "other" op names across Primus traces; add perf model classes in `PerfModel/extensions/`; register in `torch_op_mapping.py` |
| [#165](https://github.com/AMD-AGI/TraceLens/issues/165) | Add dtype detail for FP8 TEv1 GEMM | 🟡 Medium | FP8 GEMMs have 4 dtype combos (e4m3/e5m2 × e4m3/e5m2) but TraceLens doesn't parse exact FP8 dtypes from `Concrete Inputs` indices 3 & 8 | In `perf_model.py` class `tex_ts_te_gemm_ts`, parse indices 3 & 8 with the int-to-dtype mapping from TransformerEngine's `common.h`; expose `dtype_a`/`dtype_b` columns |
| [#287](https://github.com/AMD-AGI/TraceLens/issues/287) | Support FAv4 (Dao-AILab, NVIDIA Blackwell) | 🟡 Medium | Flash Attention v4 (`flash_attn.cute.interface._flash_attn_fwd`) not recognized; needed for diffusion model inference (Flux, HunyuanVideo, Wan) | Add `flash_attn_v4_fwd` class in `perf_model.py` extending `SDPA`; register in `torch_op_mapping.py`; parse B/H/N/d following existing FAv3 pattern |
| [#290](https://github.com/AMD-AGI/TraceLens/issues/290) | Perf model for `aiter::wrapper_fmha_v3_varlen_fwd` | 🟡 Medium | AITer varlen attention (`aiter.flash_attn_varlen_func`) not recognized; used in diffusion models (Wan 2.1, 2.2) | Add class in `PerfModel/extensions/attention_perf_model_extensions.py` extending `flash_attention_varlen_forward`; reuse varlen SDPA FLOPs from PR #272 |
| [#296](https://github.com/AMD-AGI/TraceLens/issues/296) | Perf model for `trtllm::attention_inplace` | 🟡 Medium | TRT-LLM's attention op classified as "other"; doesn't appear in attention roofline sheet | Add `trtllm_attention_inplace` in `PerfModel/extensions/attention_perf_model_extensions.py` extending `InferenceAttention`; parse B/N/H from input shape at index 0 |
| [#541](https://github.com/AMD-AGI/TraceLens/issues/541) | Add `SSM` category for Mamba scan/conv ops | 🟡 Medium | `MambaSplitConv1dScanCombinedFn`, `DaoAILab::_causal_conv1d_fwd_cpp` fall into "other" in Mamba/Zebra traces | Register patterns in `torch_op_mapping.py`; wire to existing `MambaSSD` and `CausalConv1d` classes; add `SSM` to `dict_base_class2category` |
| [#542](https://github.com/AMD-AGI/TraceLens/issues/542) | Add `MoE_comm` category for dispatch/combine ops | 🟡 Medium | Kimi K2 MoE comm ops (`MoEDispatch`, `MoECombine`, `TokenPermuteMaskMap*`) fall into "other" | Register in `torch_op_mapping.py` pointing to existing `moe_dispatch`/`moe_combine` classes; add `MoE_comm` to `dict_base_class2category` |
| [#281](https://github.com/AMD-AGI/TraceLens/issues/281) | Update unary/binary ops from unlabeled ops | 🟢 Low | Many `aten::` elementwise ops (pow, sigmoid, relu, etc.) lack perf models and fall into "other" | Check PyTorch source for complete unary/binary ATen op list; add missing entries to `torch_op_mapping.py` mapping to `aten_unary_elementwise`/`aten_binary_elementwise` |
| [#543](https://github.com/AMD-AGI/TraceLens/issues/543) | Add RoPE and CrossEntropy categories | 🟢 Low | `FusedRoPEFunc` and `CrossEntropyFunction` (and backward variants) fall into "other" | Register in `torch_op_mapping.py` to existing `fused_rope_fwd`/`cross_entropy_fwd` classes; add `RoPE`/`CrossEntropy` to `dict_base_class2category` |
| [#424](https://github.com/AMD-AGI/TraceLens/issues/424) | Auxiliary events in TE kernels need separate perf model | 🟢 Low | TE fused kernels include auxiliary events (workspace buffers, scale updates) that inflate FLOP counts when lumped into the main kernel | Identify auxiliary event names; add dedicated perf model classes with correct FLOPs (typically 0/small) in `PerfModel/extensions/`; register in op mapping |

---

## 🔷 JAX

| Issue | Title | Priority | What Needs to Be Done | Potential Solution |
|---|---|---|---|---|
| [#232](https://github.com/AMD-AGI/TraceLens/issues/232) | Implement JAX Trace2Tree | 🟡 Medium | Build proper hierarchical tree from JAX XPlane traces: framework call stack → HLO op → HIP runtime → GPU kernel (mirrors PyTorch tree) | Extend `JaxTraceToTree` in `Trace2Tree/trace_to_tree.py` to parse `framework_op` and `hlo_module` metadata; insert intermediate tree layers before GPU kernels |
| [#425](https://github.com/AMD-AGI/TraceLens/issues/425) | Add hlo_op and framework scope nodes to JAX tree | 🟡 Medium | Current JAX tree only has Unknown/kernel/cpu_op/memcpy nodes — missing `hlo_op` layer which carries input/output shapes, dtypes, and data dependencies | Parse `hlo_module` sidecar data in `JaxTraceToTree`; insert HLO op layer between framework call stack and GPU kernels with op type, shapes, dtype, dependencies |
| [#359](https://github.com/AMD-AGI/TraceLens/issues/359) | Add JAX details to README and feature matrix | 🟡 Medium | README barely mentions JAX support; no feature compatibility matrix showing what works per backend | Add feature matrix table to `README.md` (rows = features, columns = PyTorch/JAX/rocprofv3/Perfetto); add JAX-specific quickstart section |
| [#258](https://github.com/AMD-AGI/TraceLens/issues/258) | Port all features from JaxTraceAnalysis repo | 🟡 Medium | `amd-aig-aima/jaxtrace_analysis` external repo contains JAX analysis features not yet in TraceLens | Audit `github.com/amd-aig-aima/jaxtrace_analysis`; port features not in `NcclAnalyser/jax_nccl_analyser.py` or `TreePerf/jax_analyses.py`; one PR per feature |
| [#262](https://github.com/AMD-AGI/TraceLens/issues/262) | Implement NCCL/RCCL analyzer for JAX | 🟡 Medium | JAX collective analysis lacks full `NcclAnalyser` feature set; needs multi-node support and HLO module metadata | Extend `NcclAnalyser/jax_nccl_analyser.py` to accept `JaxTraceToTree` input; use `XLACollectiveParser` for replica-group topology; add multi-node support |
| [#304](https://github.com/AMD-AGI/TraceLens/issues/304) | Add JAX trace diff | 🟢 Low | TraceDiff (Wagner-Fischer edit distance) exists for PyTorch but not JAX | Verify `JaxTraceToTree` produces compatible node structure; add `TraceDiff.from_jax_files()` factory method and JAX-aware report generator |
| [#301](https://github.com/AMD-AGI/TraceLens/issues/301) | Add JAX rocprofv3 analysis | 🟢 Low | Port `rocprof_output_analysis.py` from `JaxTrace_Analysis` repo; enables rocprofv3-based profiling for JAX | Port into `Reporting/generate_perf_report_rocprof_jax.py`; reuse `RocprofAnalyzer` where possible; add CLI entry point to `setup.py` |
| [#422](https://github.com/AMD-AGI/TraceLens/issues/422) | `__amd_rocclr_fillBufferAligned.kd` miscategorized | 🟢 Low | Classified as `Uncategorized Events/XLA` but some are actually TE-based fused attention kernels (have valid `custom_call_target` metadata) | In `util.py`'s `prepare_event_categorizer()`, add secondary check on `custom_call_target` for TE attention patterns; reclassify accordingly |

---

## ⭐ Major Features

| Issue | Title | Priority | What Needs to Be Done | Potential Solution |
|---|---|---|---|---|
| [#17](https://github.com/AMD-AGI/TraceLens/issues/17) | Easy-to-use API for distributed performance | 🔴 High | Users manually loop over per-rank files; need unified API that internally loops over ranks, computes per-rank metrics, and surfaces min/max/diff. Also: merge `NcclAnalyser` into this API | Add `DistributedPerfAnalyzer` wrapper in `TreePerf/` accepting a list of trace paths; aggregate DataFrames across ranks; integrate `NcclAnalyser` as a property |
| [#532](https://github.com/AMD-AGI/TraceLens/issues/532) | CPU Idle and Device Sync Analysis | 🟡 Medium | Develop CPU-side idle analysis (thread blocked waiting for GPU) and device sync analysis (H2D/D2H overhead) | New method `get_df_cpu_idle_analysis()` in `TreePerfAnalyzer`; measure CPU inter-op gaps and correlate with GPU activity; extend `GPUEventAnalyser` to track H2D/D2H separately |
| [#530](https://github.com/AMD-AGI/TraceLens/issues/530) | Comprehensive tests for reports | 🟡 Medium | Current tests diff against reference files — fragile, breaks on any reporting change. Need feature-specific synthetic trace tests | Add synthetic trace helpers in `tests/conftest.py`; write one test per core feature (GPU timeline, roofline, NCCL, pseudo-ops) asserting specific column values — not full-report equality |
| [#192](https://github.com/AMD-AGI/TraceLens/issues/192) | Process memory events in PyTorch traces | 🟢 Low | PyTorch profiler records memory allocation/deallocation events; TraceLens ignores them; adding memory analysis would correlate peak memory to specific ops | Parse `ph: "i"` events with `cat == "[memory]"` ; add `get_df_memory_timeline()` to `TreePerfAnalyzer`; add `memory_timeline` sheet to Excel report |
| [#291](https://github.com/AMD-AGI/TraceLens/issues/291) | TraceAugment | 🟢 Low | Extend TraceFusion to support general trace transformation: input trace(s) → transform/annotate → output trace(s) as a pipeline stage | Generalize `TraceFusion/trace_fuse.py` into `TraceAugment` with a pluggable `(events) → events` transform pipeline; ship built-in transforms (timestamp normalization, filtering, annotation injection) |
| [#306](https://github.com/AMD-AGI/TraceLens/issues/306) | Implement critical path analysis | 🟢 Low | Determine the longest dependency chain through the execution graph — the ops that dictate minimum iteration time. Foundation for perf projection (#114) | Depends on execution trace DAG (#328); once available, run topological sort + longest-path; new module `TraceLens/CriticalPath/critical_path.py` |
| [#114](https://github.com/AMD-AGI/TraceLens/issues/114) | Perf projection | 🟢 Low | Project e2e impact of improving a specific op — an op on the non-critical path may not improve e2e even if 2× faster | Depends on critical path analysis (#306); implement Amdahl-style projection per op using its fraction of the critical path; new module `PerfProjection/` |
| [#21](https://github.com/AMD-AGI/TraceLens/issues/21) | Diagnose runtime bugs | 🟢 Low | Detect anomalies in CPU–GPU runtime: async launches stalling unexpectedly, sync events (H2D/D2H) blocking longer than expected | New analysis module in `TreePerf/`; walk tree for `cudaDeviceSynchronize`/`hipDeviceSynchronize` events; measure gap between CPU launch and GPU start; flag anomalies |
| [#328](https://github.com/AMD-AGI/TraceLens/issues/328) | Input PyTorch execution trace for dependency info | 🟢 Low | PyTorch Kineto can produce an execution trace (ET) with op-to-op data dependency edges; TraceLens should ingest this for critical path analysis | Add `ExecutionTraceLoader` in `util.py` for Kineto ET JSON; map ET node IDs to TraceLens UIDs via correlation; expose `get_dependency_graph()` on `TreePerfAnalyzer` |
| [#103](https://github.com/AMD-AGI/TraceLens/issues/103) | TL Jarvis: Fully Automated Agent for Perf & Opt | 🟢 Low | End-to-end AI agent: add profiling to a repo → run perf analysis → event replay & tuning → file minimal repro tickets. 6-month roadmap item | Start with `TraceLensAgent` wrapping `TreePerfAnalyzer` and `EventReplayer` as tools. Phase 1: automated report summarization. Phase 2: replay-based tuning loop |

---

## 🔧 Refactor / API

| Issue | Title | Priority | What Needs to Be Done | Potential Solution |
|---|---|---|---|---|
| [#225](https://github.com/AMD-AGI/TraceLens/issues/225) | Minimize dependencies in setup.py | 🔴 High | PyTorch-only users must install JAX deps (`xprof`, `protobuf`, TensorBoard) — unnecessary and slow | In `setup.py`, split into minimal core (pandas, tqdm, orjson, openpyxl) + optional extras: `pip install TraceLens[jax]`, `TraceLens[rocprof]`, `TraceLens[all]` |
| [#314](https://github.com/AMD-AGI/TraceLens/issues/314) | API cleanup | 🟡 Medium | General API cleanup before public release: consistent naming, remove deprecated methods, unified return type conventions | Audit all public methods in `TreePerfAnalyzer`, `NcclAnalyser`, `TraceDiff`; standardize snake_case names, DataFrame returns, consistent `arch` parameter handling |
| [#412](https://github.com/AMD-AGI/TraceLens/issues/412) | Move `_annotate_gpu_events_with_stream_index` to GPUEventAnalyzer | 🟡 Medium | Method lives in `Trace2Tree` but is GPU-timeline-specific logic that belongs in `GPUEventAnalyser` | Move from `Trace2Tree/trace_to_tree.py` to `TreePerf/gpu_event_analyser.py`; keep deprecation shim in `TraceToTree`; verify tests still pass |
| [#171](https://github.com/AMD-AGI/TraceLens/issues/171) | Align kernel launchers and perf model ops | 🟡 Medium | Kernel launchers show `aten::miopen_convolution`, `aten::cudnn_convolution` as distinct; perf model treats all as `aten::convolution` — inconsistent | Add alias entries in `torch_op_mapping.py`; optionally normalize op names in `get_df_kernel_launchers()` using the same alias table |
| [#316](https://github.com/AMD-AGI/TraceLens/issues/316) | Rename kernel launchers to leaf ops | 🟢 Low | "Kernel launcher" is inaccurate; "leaf op" better describes the role in the tree | Rename `get_df_kernel_launchers()` → `get_df_leaf_ops()` in `TreePerf/tree_perf.py`; keep deprecated aliases; update report sheet names in `Reporting/` |
| [#343](https://github.com/AMD-AGI/TraceLens/issues/343) | Hide max/min metrics in perf report by default | 🟢 Low | `_min` and `_max` aggregate columns clutter the Excel view | Use `openpyxl`'s `ColumnDimension(hidden=True)` on `_min`/`_max` columns in `reporting_utils.py`; add `hide_stats_cols=True` parameter to `generate_perf_report_pytorch()` |
| [#315](https://github.com/AMD-AGI/TraceLens/issues/315) | Clarify UID vs Event in the API | 🟢 Low | API inconsistently uses "UID" and "event" for the same concept | Establish convention: *event* = raw trace dict, *node* = tree node, *UID* = integer from `set_bookkeeping_attr()`; update docstrings in `TreePerf/tree_perf.py` and `Trace2Tree/trace_to_tree.py` |
| [#312](https://github.com/AMD-AGI/TraceLens/issues/312) | Extend EventReplay beyond `aten::` dispatch | 🟢 Low | `EventReplayer` only handles `aten::` ops; custom ops (vLLM, TransformerEngine) use different dispatch mechanisms | Add fallback dispatch paths in `EventReplay/event_replay.py`: `torch.ops` namespace lookup and `torch.library`-registered ops |

---

## 📄 Docs / Release

| Issue | Title | Priority | What Needs to Be Done | Potential Solution |
|---|---|---|---|---|
| [#288](https://github.com/AMD-AGI/TraceLens/issues/288) | Update landing README | 🔴 High | README has stale WIP markers, outdated file structure, no quickstart guide | Update `README.md`: mark completed features, add quickstart with 3 most common CLI commands, add feature compatibility matrix |
| [#340](https://github.com/AMD-AGI/TraceLens/issues/340) | Publish pip installable package | 🔴 High | Only installable from source (`pip install -e .`); needs to be on PyPI | Add `.github/workflows/publish.yml` using `pypa/gh-action-pypi-publish`; ensure `setup.py` has correct metadata; coordinate package name on PyPI |
| [#206](https://github.com/AMD-AGI/TraceLens/issues/206) | Example notebook: use perf model without a trace | 🟡 Medium | PerfModel can compute FLOPs/bytes from explicit tensor shapes alone, but there's no notebook showing this | Create `examples/perf_model_standalone.ipynb` showing direct instantiation of `GEMM`/`flash_attention` classes with explicit M/N/K params and roofline computation |
| [#319](https://github.com/AMD-AGI/TraceLens/issues/319) | TraceLens release ROCm blog | 🟡 Medium | Draft and publish ROCm developer blog post announcing public release | Write Markdown draft covering key capabilities, 3 use cases with screenshots (roofline, NCCL analysis, trace diff), quickstart example |
| [#322](https://github.com/AMD-AGI/TraceLens/issues/322) | Detailed docs based on profiling workflows deck | 🟢 Low | Internal PowerPoint deck on AI model profiling workflows should be converted to repo documentation | Convert to `docs/profiling_workflows.md` covering: collecting a trace, running TraceLens CLI, interpreting GPU timeline/roofline sheets, NCCL analysis workflow |
| [#342](https://github.com/AMD-AGI/TraceLens/issues/342) | Update the GitHub Wiki | 🟢 Low | GitHub Wiki is outdated or empty | Mirror key `docs/*.md` files into the Wiki; optionally add a GitHub Actions workflow to auto-sync on merge to main |

---

## 🔗 Enhancements

| Issue | Title | Priority | What Needs to Be Done | Potential Solution |
|---|---|---|---|---|
| [#90](https://github.com/AMD-AGI/TraceLens/issues/90) | Analysis scoped by `record_function` regions | 🟡 Medium | TraceLens should recognize PyTorch `record_function` annotation events and allow filtering/aggregating by named region | Detect `cat == "user_annotation"` events in `Trace2Tree/trace_to_tree.py`; add `get_df_by_region(region_name)` to `TreePerfAnalyzer` |
| [#115](https://github.com/AMD-AGI/TraceLens/issues/115) | Accept user-provided arch dict for roofline | 🟡 Medium | Users must supply an architecture JSON file; should accept a simple dict like `{'peak_perf': 1307e12, 'peak_bw': 5.3e12}` | Modify `TreePerfAnalyzer.from_file(arch=...)` to accept a dict or string; add dict-to-arch-object conversion in the factory method |
| [#124](https://github.com/AMD-AGI/TraceLens/issues/124) | Fwd/bwd breakdown for kernel launchers | 🟡 Medium | `get_df_kernel_launchers()` doesn't indicate whether a launcher is in the forward or backward pass | Traverse each launcher's ancestors for backward-pass markers; add `pass_type: 'forward' \| 'backward' \| 'unknown'` column to the DataFrame |
| [#187](https://github.com/AMD-AGI/TraceLens/issues/187) | Expose roofline plotting as Python API | 🟡 Medium | Roofline plotting only available via CLI in `examples/`; should be importable functions usable directly on `TreePerfAnalyzer` output | Extract `plot_roofline()` from `examples/custom_workflows/roofline_analyzer/src/roofline.py` into `Reporting/roofline.py`; export from `__init__.py` |
| [#419](https://github.com/AMD-AGI/TraceLens/issues/419) | Focus analysis on sub-regions (by computation phase) | 🟡 Medium | Allow users to scope the report to a subtree rooted at a named CPU op (e.g. prefill, decode, fwd, bwd) | Add `scope_to_op(op_name)` to `TreePerfAnalyzer` that re-roots analysis at the first matching node; add `--scope_op` CLI flag |
| [#420](https://github.com/AMD-AGI/TraceLens/issues/420) | Focus analysis on sub-regions (by timestamps) | 🟡 Medium | Allow `start_ts`/`end_ts` flags to restrict report to a specific time window; prototype exists in branch `feat/treeperf/window_filter_by_timestamps_v0.1` | Port timestamp window filter from prototype branch; add `filter_by_timestamps(start_ts, end_ts)` to `TreePerfAnalyzer`; expose as CLI flags |
| [#477](https://github.com/AMD-AGI/TraceLens/issues/477) | Systematic debug mode logging | 🟡 Medium | Perf model failures are silent or cryptic; should log op name, input shapes, and failing parse step in debug mode | Add `logging.debug()` calls throughout `PerfModel/perf_model.py` base class parsers; expose `--debug` / `TRACELENS_DEBUG=1` flag |
| [#428](https://github.com/AMD-AGI/TraceLens/issues/428) | Optional kernel name categorizer in GPUEventAnalysis | 🟡 Medium | Kernel-name-based categorization (for `hipGraphLaunch`, JAX) should be exposed as a pluggable component rather than hardcoded | Refactor `prepare_event_categorizer()` from `util.py` into a standalone `KernelNameCategorizer` class; expose as optional param to `GPUEventAnalyser` |
| [#289](https://github.com/AMD-AGI/TraceLens/issues/289) | Allow custom kernel categorization (comm/copy) | 🟡 Medium | Users need to mark custom kernels (e.g. DeepEP dispatch/combine) as communication without source edits | Add 4th extension hook: `kernel_category_extension = {"pattern": "Communication"}`; check in `GPUEventAnalyser.get_gpu_event_lists()` before default categorization |
| [#459](https://github.com/AMD-AGI/TraceLens/issues/459) | Multi-node breakdown in collective report | 🟡 Medium | Multi-rank collective report doesn't distinguish intra-node vs inter-node communication | Prototype in PR #458; extend with rank-to-node mapping from `NcclAnalyser/util/node_rank_to_protobuf_file_mapping.py`; add intra/inter-node breakdown columns |
| [#104](https://github.com/AMD-AGI/TraceLens/issues/104) | p2p NCCL support | 🟡 Medium | No bus bandwidth estimation for point-to-point NCCL kernels (send/recv); needed for pipeline parallelism analysis | Extend `is_communication_string()` in `util.py` to detect p2p patterns; add bus bandwidth formula `msg_bytes / duration` in `NcclAnalyser/nccl_analyser.py` |
| [#22](https://github.com/AMD-AGI/TraceLens/issues/22) | Diagnose NCCL anomalies | 🟢 Low | NCCL collectives must finish at ~same time across ranks; detect end-time divergence indicating NCCL/RCCL bugs | Extend `NcclAnalyser.build_df_nccl_implicit_sync_cat()` with end-time skew metric; add `detect_nccl_anomalies(threshold_us)` flagging outlier instances |
| [#128](https://github.com/AMD-AGI/TraceLens/issues/128) | Compute TFLOPS/GB/s in node replay | 🟢 Low | After replaying an op, compute TFLOPS/s and TB/s from measured time and perf model FLOPs/bytes for microbenchmark vs workload comparison | In `EventReplay/event_replay.py`, after `benchmark_func()`, look up perf model via `op_to_perf_model_class_map`, call `.flops()`/`.bytes()`, append `tflops_s`/`tb_s` to output |
| [#263](https://github.com/AMD-AGI/TraceLens/issues/263) | Add compute/communication tag to df_kernels | 🟢 Low | `get_df_kernels()` doesn't indicate whether each kernel is compute, NCCL, or memcpy | Use `GPUEventAnalyser.get_gpu_event_lists()` to classify each kernel; add `kernel_category` column to `get_df_kernels()` output |
| [#299](https://github.com/AMD-AGI/TraceLens/issues/299) | CPU numeric checking for event replay | 🟢 Low | Run same op on CPU and compare output tensors to GPU result using L2/L∞ norms to validate numerical correctness | Add `numeric_check=False` flag to `EventReplay/event_replay.py`'s `replay()`; compute `torch.norm(cpu_out - gpu_out.cpu())` and return with the result |
| [#331](https://github.com/AMD-AGI/TraceLens/issues/331) | Add op category to perf report comparison | 🟢 Low | Comparison report doesn't include op category (GEMM, SDPA, etc.), making it hard to filter diffs by type | Add `category` to `SHEETS_COMPARE_CONFIG` merge keys in `compare_perf_reports_pytorch.py`; preserve category from `op_to_perf_model_class_map` in diff output |
| [#374](https://github.com/AMD-AGI/TraceLens/issues/374) | RuntimeError in EventReplay for `aten::bmm` | 🟡 Medium | Some `aten::bmm` ops raise `RuntimeError: more than one element...refers to single memory location` due to tensors with overlapping strides (expanded/broadcast) | In `EventReplay/utils.py`'s `build_tensor()`, detect overlapping strides and automatically call `.clone()` before passing to the op |
| [#436](https://github.com/AMD-AGI/TraceLens/issues/436) | Classify memsets as memcpy in GPUEventAnalyzer | 🟢 Low | GPU memset ops (zeroing buffers) are unclassified time; should be bucketed as memcpy in GPU timeline | In `gpu_event_analyser.py`'s `get_gpu_event_lists()`, detect memset patterns (`Memset`, `fillBuffer`, `__amd_rocclr_fillBufferAligned`) and route into memcpy interval list |
| [#113](https://github.com/AMD-AGI/TraceLens/issues/113) | Pass event replay to HW counter profiling | 🟢 Low | After replaying an op, pass through `rocprof-compute` to collect hardware perf counters (cache utilization, occupancy, etc.) | Add `hw_profile=True` flag to `EventReplay/event_replay.py`'s `replay()`; wrap execution in subprocess call to `rocprof-compute`; parse and return HW counter CSV |
| [#449](https://github.com/AMD-AGI/TraceLens/issues/449) | Load trace files direct from SharePoint | 🟢 Low | AMD teams share traces via SharePoint; users download manually — `office365-rest-python-client` is already in `setup.py` | Add `SharePointLoader` in `util.py` using existing `office365-rest-python-client`/`msal` deps; auth via device code flow; add `--sharepoint_url` CLI flag |
| [#423](https://github.com/AMD-AGI/TraceLens/issues/423) | `FillBuffer` events miscategorized as Conv | 🟢 Low | `FillBuffer` events classified as Conv via `JaxOpKeys.ClassCategories` pattern matching; actually are TE fused attention kernels | Refine `JaxOpKeys.ClassCategories` pattern in `util.py`; add `custom_call_target` override to set `gpu_kernel_op_cat = "te_fused_attn"` for matching events |
| [internal #22](https://github.com/AMD-AGI/TraceLens-internal/issues/22) | Feature: Kernel timeline | 🟢 Low | Find the iteration step with the largest tensor shape for a given op (e.g. MLA), then extract all kernels in launch order for one layer of that step | Add `get_kernel_timeline(op_name, step_selector='max_shape')` to `TreePerfAnalyzer`; find step via shape analysis; return `get_df_kernels()` filtered by timestamp range |

---

## Priority Legend

| Symbol | Meaning |
|---|---|
| 🔴 High | Blocks users or correctness issue; fix soon |
| 🟡 Medium | Meaningful improvement; schedule for next cycle |
| 🟢 Low | Nice to have; tackle when capacity allows |

*Priority is derived from: GitHub labels (`high priority`, `bug`), effort size, and severity of user impact.*
