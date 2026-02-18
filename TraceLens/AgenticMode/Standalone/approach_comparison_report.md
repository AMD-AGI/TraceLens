# Performance Analysis Approaches: Comparison and Critique

This report compares two approaches to AI-driven PyTorch performance analysis using TraceLens:

- **Approach A (Monolithic Skill)**: A single 724-line Cursor skill file (`TraceLens/.cursor/skills/jarvis-perf-analysis.md`) containing the entire analysis workflow, domain knowledge, hardware reference, output templates, and API usage in one document.
- **Approach B (Modular Orchestrator + Subagents)**: A 33-file system (`TraceLens/AgenticMode/Standalone/`) consisting of an orchestrator skill, 11 specialized sub-agent definitions, Python preparation and analysis scripts, platform specs, shared utilities, and a continual learning mechanism.

---

## 1. Comparison: Monolithic Skill vs. Modular Orchestrator

### 1.1 Architecture Overview

**Approach A** loads a single skill file into one LLM conversation. The LLM reads it once and executes all steps sequentially: generate the TraceLens performance report, read GPU utilization from xlsx sheets, identify top operations, analyze each operation, determine optimizations, generate replay artifacts, and write the final report. All domain knowledge (GEMM patterns, attention patterns, BatchNorm, convolutions, etc.) is embedded inline.

**Approach B** uses a two-tier pipeline coordinated by an orchestrator skill. A Python preparation script (`orchestrator_prepare.py`) loads the trace once and pre-computes category-specific CSVs, metadata JSONs, tree data, and multi-kernel data. Specialized sub-agents (GEMM, SDPA, elementwise, reduce, triton, MoE, batchnorm, convolution, cpu-idle, multi-kernel, generic) are launched as parallel Task subagents, each with its own analysis Python script and agent definition file. Results are validated, aggregated, and composed into a two-section report (System-Level + Compute Kernel).

### 1.2 Robustness

| Dimension | Approach A | Approach B |
|-----------|------------|------------|
| Context window pressure | All 724 lines loaded into one agent. Context accumulates as the LLM works through operations sequentially. Risk of degradation on long analyses. | Orchestrator loads 928 lines, but each sub-agent loads only its own 150-220 line agent file plus pre-computed data. Context is isolated per category. |
| Blast radius of failure | A failure at any step can derail the entire analysis. The LLM may lose track of progress or produce inconsistent outputs. | A failure in one sub-agent is contained. The orchestrator excludes that category from aggregation and notes it in a Warnings section. Other categories complete independently. |
| Domain depth | General patterns for all categories packed into one file. Coverage is broad but shallow -- GEMM analysis gets roughly 30 lines of guidance across common patterns and the main workflow. | Each sub-agent has deep domain-specific guidance. The GEMM analyzer alone is 223 lines with dedicated thresholds, bottleneck criteria, common patterns, and what-can/cannot-infer tables. |
| Data isolation | The LLM reads raw xlsx/CSV sheets directly and must filter and interpret everything on the fly. Prone to reading the wrong sheet or misinterpreting columns. | `orchestrator_prepare.py` pre-computes category-specific CSVs, metadata JSONs, tree data, and multi-kernel data. Sub-agents receive only their relevant data slice. The trace file is loaded exactly once. |
| Hallucination risk | Higher. The LLM must remember hardware specs, apply the right thresholds for the right category, and track many operations across a long conversation. | Lower within each sub-agent because it receives platform specs in its metadata JSON, has category-specific thresholds baked in, and has explicit constraints (e.g., flag efficiency > 100% as anomaly). |

**Approach A advantages:**
- Simplicity. One file, no coordination overhead, no multi-hop delegation. Easy to understand, audit, and modify.
- No systemic failure mode. The orchestrator in B is itself a complex monolith; if it is misinterpreted, the entire system fails. A has no orchestration layer to misinterpret.

**Approach A disadvantages:**
- No failure isolation. A mistake in Step 3 silently corrupts Steps 4-9.
- Context degradation. Quality drops as the conversation grows with each analyzed operation.
- Shallow domain coverage per category.

**Approach B advantages:**
- Failure containment. A broken elementwise analysis does not affect GEMM analysis.
- Pre-computed data provides a deterministic foundation the LLM interprets rather than computes.
- Deep, category-specific domain knowledge in each sub-agent.

**Approach B disadvantages:**
- The orchestrator is itself a 928-line monolith at a higher abstraction layer, subject to the same interpretation risks.
- Multi-hop context passing (orchestrator reads agent file, embeds it in a Task prompt) is fragile and unvalidated.
- Coordination complexity introduces new failure modes absent from A.

---

### 1.3 Repeatability

| Dimension | Approach A | Approach B |
|-----------|------------|------------|
| Deterministic data pipeline | None. The LLM reads CSVs/xlsx, does mental math, and produces tables. Two runs on the same trace can yield different operation rankings, efficiency calculations, or recommendations. | `orchestrator_prepare.py` and per-category Python scripts produce deterministic JSON metrics. The LLM's role is reduced to interpreting pre-computed metrics and applying codified thresholds. |
| Output structure | Template is specified in the skill file, but the LLM generates it freeform. Section structure, priority ordering, and metric formatting can drift between runs. | Output templates are specified at both the orchestrator level (final report) and sub-agent level (category findings). Each sub-agent writes to a known file path with a known format. |
| Threshold consistency | Thresholds (e.g., >80% excellent, <40% needs investigation) are stated once. The LLM may apply them inconsistently, especially late in a long analysis. | Each sub-agent file embeds its own category-specific thresholds. Applied close to the point of use, reducing drift. |
| Cross-run comparability | Difficult. Two analysts running A on the same trace will get structurally different reports. | Better. The deterministic Python pipeline produces identical metrics. Sub-agent interpretation may vary, but the underlying data and structure are consistent. |
| Evolution mechanism | None. The skill file is static with no structured update path. | The continual learning skill provides a structured, append-only mechanism to add new patterns to specific sub-agents. |

**Approach A advantages:**
- No multi-step workflow to deviate from. The LLM follows a linear sequence, which is simpler to reason about.
- No risk of the orchestrator misinterpreting its own 11-step workflow or misordering parallel dispatch.

**Approach A disadvantages:**
- The LLM is both calculator and interpreter. Every number passes through LLM inference, introducing non-determinism.
- No mechanism for self-consistency checking or cross-run comparison.
- No evolution path -- updating patterns requires manual editing of a monolith.

**Approach B advantages:**
- Deterministic Python scripts produce identical metrics across runs, creating a stable foundation.
- Structured output paths and templates improve cross-run comparability.
- Continual learning provides a (limited) feedback loop for improvement.

**Approach B disadvantages:**
- The LLM interpretation layer still introduces non-determinism in findings and recommendations.
- The orchestrator's 11-step workflow introduces more points where the LLM can deviate from the prescribed sequence.
- Continual learning is append-only with no mechanism for deprecating or revising stale patterns.

---

### 1.4 Error Handling

| Dimension | Approach A | Approach B |
|-----------|------------|------------|
| Explicit error protocols | Minimal. "If not installed, inform the user." No structured error states, no status codes, no fallback paths. | Extensive at every level. Each sub-agent has a dedicated Error Handling section specifying behavior for missing files, script failures, and explicit prohibitions on manual workarounds. |
| Graceful degradation | Not designed for it. If one category analysis fails, the LLM may push through, hallucinate results, or stop entirely. | Designed in. Failed categories are excluded from aggregation. A Warnings section lists what failed and why. The report remains valid for successful categories. |
| Validation layer | None. The LLM produces the final report with no self-checking. | Step 8 implements four validation checks: time sanity, efficiency anomaly detection (>100%), coverage check, and priority consistency. |
| Error propagation | Errors propagate invisibly. A wrong calculation in Step 3 silently corrupts downstream steps. | Errors are surfaced via JSON status fields, findings file status markers, and validation warnings. |
| Anti-hallucination guards | A "What You CANNOT Infer" table provides discipline but relies on the LLM's self-restraint. | Structural guards: sub-agents cannot access the raw trace, must flag >100% efficiency as anomalies, and must use GPU kernel time (not CPU duration) for prioritization, with explicit correct-vs-wrong code examples. |

**Approach A advantages:**
- No error-handling machinery that can itself fail. The system is simple enough that failure modes are visible to a human supervisor.
- The LLM can attempt workarounds when something goes wrong, which may be preferable to silence.

**Approach A disadvantages:**
- No structured recovery. Errors propagate invisibly through the entire analysis.
- No validation that the output is internally consistent.
- Relies entirely on human supervision to catch mistakes.

**Approach B advantages:**
- Multi-layer defense: per-subagent error protocols, anti-hallucination constraints, dedicated validation step, graceful degradation.
- Failed analyses are explicitly surfaced rather than silently corrupted.
- Structural separation prevents a single mistake from contaminating unrelated categories.

**Approach B disadvantages:**
- The "Do NOT manually analyze" constraint means the system degrades to silence on failure rather than providing a degraded-but-useful result.
- Error-handling complexity is itself a source of risk -- if validation code has bugs (e.g., the regex for detecting >100% efficiency), errors slip through undetected.
- The system generates a finding that points to its own configuration as the root cause when platform specs are wrong (as seen with MI355X MoE anomalies in the actual output).

---

### 1.5 Summary Comparison

| Criterion | Approach A | Approach B |
|-----------|:-:|:-:|
| Robustness | Moderate -- simple but fragile at scale | Strong -- isolated failures, pre-computed data |
| Repeatability | Low -- LLM-dependent computation + interpretation | Moderate-High -- deterministic data, LLM interpretation |
| Error Handling | Minimal -- no structured recovery | Strong -- multi-layer validation + graceful degradation |
| Simplicity | High -- one file, easy to understand | Low -- 33 files, complex coordination |
| Maintenance burden | Low initially, risky long-term (monolith growth) | Higher initially, sustainable long-term (modular updates) |
| Time to first result | Faster (no prep script, no sub-agent coordination) | Slower (~60-90s prep + parallel sub-agent launches) |
| Extensibility | Difficult (editing one large file) | Good (add new sub-agent + analysis script) |
| Domain depth per category | Shallow (broad coverage, ~30 lines per category) | Deep (150-220 lines per sub-agent) |
| Human supervision required | High (must verify all numbers and conclusions) | Lower (deterministic pipeline reduces arithmetic errors) |

---

## 2. Critique of Approach A (Monolithic Skill)

### 2.1 The LLM Does Both Computation and Interpretation

This is the fundamental architectural weakness. Approach A asks the LLM to read xlsx/CSV sheets, mentally extract values, perform arithmetic (efficiency percentages, time breakdowns, expected hardware ratios), and then interpret those results. Every number in the final report passes through the LLM's inference, which is unreliable for arithmetic.

The efficiency calculation guidance instructs the LLM to read a `FLOPS/Byte` value from a spreadsheet, decide which threshold it falls into (>100-200 compute-bound, <50 memory-bound), then look up the right peak from the hardware reference table, then divide achieved by peak, then express as a percentage. Each step introduces error. There is no Python script to do this -- the LLM is the calculator. A single misread cell value propagates silently through every downstream conclusion.

### 2.2 No Guardrails Against Numerical Hallucination

The skill includes a "What You CANNOT Infer" table, which is good discipline for qualitative claims. But there are no guardrails for quantitative errors. The skill never asks the LLM to verify that:
- The sum of per-operation times equals total compute time
- Efficiency percentages fall between 0-100%
- The top operations list accounts for a reasonable fraction of total time

When the LLM misreads "68.78 ms" as "687.8 ms" from a spreadsheet, nothing catches it. The absence of any self-consistency checking means wrong numbers and right numbers are indistinguishable in the output.

### 2.3 The Hardware Reference Table is Hardcoded and Incomplete

The hardware table lists only BF16 MAF for five platforms. FP8, FP4, and INT8 peaks are absent. When the LLM encounters an FP8 GEMM or FP4 MoE kernel, it has no reference peak to calculate efficiency against. It will either use the BF16 value (producing wrong efficiency) or hallucinate a number from training data (worse).

Values use approximations ("~5.3 TB/s", "~990 TFLOPS/s") that inject ambiguity. The table is frozen at the time of writing with no MI355X or MI400 entries and no structured fallback for unlisted hardware. A "Hardware Reference Template" asks the user to provide values, but nothing enforces this.

By contrast, Approach B's `platform_specs.py` has 10+ precision-specific peaks per platform with exact values.

### 2.4 Xlsx Reading Through an LLM is Unreliable

The workflow depends on the LLM reading multi-sheet Excel workbooks. The LLM must navigate between sheets (`gpu_timeline`, `ops_summary`, `unified_perf_summary`, `kernel_details_summary`), mentally sort tables by time columns, extract specific cell values, and track relationships across sheets. LLMs interacting with spreadsheets through text-based tool output face column alignment loss, truncation on large sheets, and inability to actually sort data. The skill implicitly acknowledges this by warning "NOT `ops_summary_by_category` - too noisy" -- conceding that dense tabular data confuses the LLM.

### 2.5 Sequential Execution Cannot Scale

The workflow is strictly sequential: Step 1 through Step 9 in a single conversation. Analyzing a trace with 10 significant operations means the LLM works through all 10 in series, accumulating context with each one. For a complex trace (transformer with attention, MLP, normalization, communication, and embedding), the analysis of operation 8 happens with operations 1-7 already consuming context. Quality degrades as the conversation grows. There is no mechanism to restart from a specific step -- a mistake in Step 5 means restarting from scratch.

### 2.6 The Comparative Analysis Workflow is Underspecified

The standalone analysis workflow has reasonable detail (specific commands, sheets, fields). The comparative analysis workflow is significantly thinner. Criteria like "unexpected slowdowns" and "where target hardware should be competitive" are subjective. The hardware ratio calculation is mentioned but must be applied manually by the LLM for every operation. Two runs of comparative analysis on the same trace pair could produce different "expected vs. actual" assessments depending on which operations the LLM considers first and how it interprets "competitive."

### 2.7 Output Format Compliance is Aspirational

The skill specifies detailed report templates with formatting rules (executive summary max 20 lines, max 10 lines per recommendation, no redundancy, cross-references between sections). But there is no enforcement. The LLM may produce a 40-line executive summary, repeat data across sections, omit cross-references, or use a different structure entirely. With 724 lines of instructions, formatting rules stated early are forgotten by the time the LLM generates the report.

### 2.8 Generic Recommendations Across Diverse Categories

The "Both Paths" requirement mandates providing algorithmic and kernel optimization recommendations for every bottleneck. Without deep domain-specific guidance per operation type, the LLM tends to produce generic recommendations. "Path A: consider fused operators" and "Path B: generate replay artifact for kernel team" are correct but shallow. The common patterns section provides uneven coverage -- attention-heavy models get detailed treatment while other patterns (BatchNorm, convolutions, MoE) get 3-4 lines each.

### 2.9 No Mechanism for Learning or Improvement

The skill file is static. There is no continual learning mechanism, no way to record lessons from previous analyses, and no structured process for updating patterns. When the skill misses an issue (a new MoE kernel variant, a paged attention pattern, a Triton-compiled operation), the only recourse is manual editing by someone who understands both the skill format and the performance analysis domain. There is no feedback loop from analysis outputs back to the skill.

### 2.10 The Perf Model Extension Mechanism is Error-Prone

The skill asks the LLM to write Python classes that inherit from TraceLens's `PerfModel` base classes, implementing `bytes()` and `flops()` methods with correct numerical formulas. The template has `...` placeholders, leaving actual implementation to the LLM during an analysis session. Getting the bytes-moved calculation wrong for an operation like softmax (multiple passes: read-max-subtract-exp-sum-divide) produces incorrect efficiency numbers that then drive recommendations. There is no test or validation that the extension produces correct results.

---

## 3. Critique of Approach B (Modular Orchestrator + Subagents)

### 3.1 The Orchestrator is Still a Monolith

The orchestrator skill is 928 lines. The entire system's reliability depends on an LLM correctly interpreting this single file and executing an 11-step workflow with conditional branching, parallel dispatch, validation logic, and aggregation rules. If the monolithic skill is critiqued for being "too much for one LLM to track," the same critique applies to the orchestrator -- it just moved complexity from "analyze the trace" to "coordinate the analysis."

In the actual output, the MoE category is ranked P3 in compute kernel recommendations despite being the largest category by GPU time (35.4%). The orchestrator's aggregation logic says to prioritize by "highest impact on end-to-end time," but that criterion is subjective enough that the LLM may reorder priorities between runs.

### 3.2 The "Deterministic Foundation" Has Hollow Components

The Python scripts are deterministic, but the tree data pre-computation is empty in practice. `orchestrator_prepare.py` generates tree data files where every `parent_chain` is `[]` and every `subtree` is `[]`, with the note "Tree traversal simplified - using CSV data." Sub-agents that say "verify with tree data" cannot actually do this. The system creates an illusion of rigor by generating structured files that carry no useful information. If no sub-agent ever uses tree data to change a recommendation, the tree data pre-computation is dead code that adds complexity without value.

### 3.3 Context Passing is a Telephone Game

The sub-agent dispatch works by: (1) reading an agent `.md` file, (2) embedding its full text into a Task subagent prompt, (3) relying on the Task subagent to follow it correctly. This multi-hop delegation requires the orchestrator LLM to faithfully copy approximately 200 lines of agent instructions plus execution context into a prompt for a new LLM instance. Any summarization, truncation, or reinterpretation corrupts the sub-agent's behavior. There is no contract enforcement -- no schema validation that the sub-agent's output conforms to the expected format before the orchestrator reads it.

### 3.4 Parallel Launch Limits Contradict the Design

The orchestrator instructs "Launch ALL Compute Kernel Subagents in PARALLEL." But Cursor's Task tool has a hard limit of 4 concurrent subagents. With up to 9 compute kernel categories plus 2 system-level categories, the orchestrator cannot launch them all at once. It must batch them into sequential groups of 4, introducing phases that the design does not account for. This means total wall-clock time is higher than advertised, the orchestrator must track batch completion, and failure in an early batch could affect later dispatch.

### 3.5 Average Efficiency Calculation is Misleading

The `calculate_average_efficiency` function in `analysis_utils.py` computes an unweighted average. A GEMM taking 0.04ms at 45% efficiency is weighted equally with a GEMM taking 1.1ms at 21% efficiency. This makes "average efficiency" misleading for prioritization. The actual output reports GEMM average efficiency as 32.15%, but the time-weighted average would be significantly lower because the slowest operations (with worst efficiency) dominate runtime.

### 3.6 Categorization Relies on Fragile String Matching

The `get_enhanced_category` function in `orchestrator_prepare.py` uses simple substring matching (`'moe' in op_name.lower()`, `'batch_norm' in op_name.lower()`). The actual output demonstrates the failure mode: `vllm::rocm_unquantized_gemm` (7.43ms, 7.7% of compute) landed in "other" instead of GEMM because the name does not match TraceLens's `op category` mapping or the enhanced categorization rules. The system's own output acknowledges this: "Several operations are miscategorized." This is a structural weakness of any string-matching approach.

### 3.7 Silence on Failure Rather Than Degraded Output

Every sub-agent has the constraint: "Do NOT attempt to manually analyze the raw CSV data as a workaround" and "Do NOT provide any bottleneck findings or recommendations for this category." This prevents hallucination but means the system degrades to silence on failure rather than providing a degraded-but-useful result. When the replay artifact generation failed in the actual run with `ImportError: cannot import name 'EventReplayer'`, the system had no recovery path. An LLM that could read the CSV and provide a best-effort analysis was explicitly prohibited from doing so.

The trade-off: the monolithic approach would have attempted (possibly incorrectly) to work around the failure. The modular approach guarantees correctness by guaranteeing nothing on failure. Which is preferable depends on stakeholder tolerance for false information versus incomplete information.

### 3.8 No Automated Testing

The `functional_test_cases.csv` lists 25 test cases, but they are purely declarative descriptions with no test harness, no automation, and no programmatic pass/fail criteria. The README acknowledges: "Validation at a sub-agent level and integration tests are crucial to assess performance." For a system that claims robustness through modular isolation, the absence of automated testing means the isolation cannot be verified. The system has a test plan without tests.

### 3.9 Platform Specs Are Inaccurate for Newer Hardware

`platform_specs.py` contains measured MAF values for MI300X/MI325X but theoretical peaks for MI355X/MI400 (marked "TODO: replace with measured MAF"). The actual analysis ran on MI355X and detected every MoE operation as an ANOMALY (126-275% efficiency) -- a direct consequence of using theoretical peaks as if they were measured values. The system flagged the anomaly correctly but could not resolve it, generating a finding that points to its own configuration as the root cause.

### 3.10 Continual Learning Has No Deprecation Path

The continual learning mechanism is append-only: new patterns can be added but never removed, revised, or deprecated. Over time, the Common Patterns sections grow monotonically. There is no mechanism to detect when patterns conflict, when thresholds should be updated for new hardware, or when a pattern has become obsolete. This creates a knowledge debt that accumulates with each learning cycle.

---

## Concluding Observations

**Approach A** is simpler and more accessible. Its weaknesses are in the design itself -- it does not attempt the structural safeguards that would make it reliable without human supervision. It works well when an expert is watching and can catch errors. Its failure mode is "wrong but confident."

**Approach B** is architecturally ambitious. Its weaknesses are in the gap between design and implementation -- it promises more than it delivers (hollow tree data, theoretical platform specs, aspirational parallel execution, declarative-only tests). It works (or at least fails gracefully) when no one is watching, which is the harder problem. Its failure mode is "incomplete but honest."

The strongest version of this system would combine A's simplicity for the interpretation layer with B's deterministic computation layer, while closing the implementation gaps in B and adding the self-consistency validation that A lacks entirely.
