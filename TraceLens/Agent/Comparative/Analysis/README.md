<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Jarvis Analysis - Modular Architecture

This directory contains the refactored, modular version of Jarvis GPU performance analysis framework.

## Architecture Overview

```
Analysis/
├── jarvis_analysis.py       # Main orchestrator
├── tracelens_runner.py      # TraceLens integration
├── llm_prompts.py           # LLM prompt management
├── plotting.py              # Visualization generation
├── report_generator.py      # Final report assembly
└── __init__.py              # Package initialization
```

## Module Responsibilities

### 1. `jarvis_analysis.py` - Main Orchestrator
**Purpose**: Coordinates the entire analysis workflow

**Key Class**: `JarvisAnalyzer`

**Responsibilities**:
- Parse command-line arguments
- Initialize all submodules
- Orchestrate analysis workflow:
  1. Process GPU 1 traces
  2. Process GPU 2 traces
  3. Generate comparison report
  4. Run AI analysis (if API key provided)
  5. Generate plots (if enabled)
  6. Assemble final report
- Handle errors and logging

**Usage**:
```bash
python3 Analysis/jarvis_analysis.py \
  --gpu1-kineto trace1.json \
  --gpu1-et et1.json \
  --gpu2-kineto trace2.json \
  --gpu2-et et2.json \
  --api-key YOUR_KEY \
  --generate-plots \
  --disable-critical-path
```

### 2. `tracelens_runner.py` - TraceLens Integration
**Purpose**: Handle all TraceLens report generation and critical path analysis

**Key Class**: `TraceLensRunner`

**Methods**:
- `run_critical_path_analysis()`: Run CritPath DAG analysis
- `generate_tracelens_report()`: Generate HTA performance reports
- `generate_comparison_report()`: Compare two GPU traces

**Responsibilities**:
- Execute critical path dependency analysis
- Generate HTA operator summaries
- Create per-category performance breakdowns
- Build comparison dataframes

### 3. `llm_prompts.py` - LLM Prompt Management
**Purpose**: Build and manage all LLM prompts for AI analysis

**Key Class**: `LLMPromptManager`

**Methods**:
- `build_analysis_prompt()`: Create complete analysis prompt
- `get_system_message()`: Get system instructions
- `_build_gap_instruction()`: Build gap analysis instructions
- `_format_cp_comparison_data()`: Format critical path data
- `_format_timeline_data()`: Format timeline data

**Responsibilities**:
- Adapt prompts based on `use_critical_path` flag
- Format data for LLM consumption
- Provide clear analysis instructions
- Handle both critical path and timeline modes

**Prompt Modes**:
1. **Critical Path Mode** (`use_critical_path=True`):
   - Focus on ~7-10% operations on critical path
   - Include dependency information
   - Emphasize path-blocking optimizations

2. **Timeline Mode** (`use_critical_path=False`):
   - Analyze all operations from timeline
   - Category-based performance breakdown
   - Overall latency analysis

### 4. `plotting_complete.py` - Visualization Generation
**Purpose**: Generate all charts and visualizations

**Key Class**: `JarvisPlotter`

**Methods**:
- `generate_all_plots()`: Generate all visualizations
- `generate_gap_analysis_plots()`: Performance gap charts (categories & operations)
- `generate_optimization_opportunity_plots()`: Category optimization analysis
- `generate_overall_improvement_plot()`: End-to-end improvement scenarios
- `generate_cumulative_optimization_progression_chart()`: Stacked bar chart (Baseline→Projection→Target)

**Plot Types**:
1. **Gap Analysis Categories**: Bar chart comparing baseline vs target by category
2. **Gap Analysis Operations**: Horizontal bar chart of top operations
3. **Category Optimization Analysis**: Multi-bar chart with gap closure scenarios
4. **Overall Improvement**: Scenarios showing 25%, 50%, 75%, 100% gap closure
5. **Cumulative Optimization**: Stacked bars showing category-level progression

**Features**:
- Extracts optimization gains from AI analysis text
- Maps operation names to categories using keyword matching
- Adapts data source based on `use_critical_path` flag
- Smart GPU baseline/target assignment (MI300, H200 detection)
- Consistent color palette across all charts

### 5. `report_generator.py` - Report Assembly
**Purpose**: Generate final markdown reports with embedded plots

**Key Class**: `ReportGenerator`

**Methods**:
- `generate_report()`: Create complete markdown report
- `add_section()`: Add report section
- `embed_plot()`: Embed plot with relative path

**Responsibilities**:
- Create report structure
- Embed AI analysis
- Embed plots with relative paths
- Format executive summaries
- Handle both analysis modes

## Analysis Modes

### Critical Path Mode (Default)
```bash
python3 Analysis/jarvis_analysis.py --gpu1-kineto ... --gpu1-et ...
```
- Uses CritPath DAG analysis
- Focuses on ~7-10% of operations on critical path
- Includes dependency information
- Better for understanding blocking operations

### Timeline Mode
```bash
python3 Analysis/jarvis_analysis.py --gpu1-kineto ... --gpu1-et ... --disable-critical-path
```
- Uses full operation timeline
- Analyzes all operations by category
- No dependency information
- Better for overall latency analysis

## Command-Line Options

```
Required:
  --gpu1-kineto PATH    GPU1 kineto trace
  --gpu1-et PATH        GPU1 execution trace
  --gpu2-kineto PATH    GPU2 kineto trace
  --gpu2-et PATH        GPU2 execution trace

Optional:
  --gpu1-name NAME             Name for GPU1 (auto-detected)
  --gpu2-name NAME             Name for GPU2 (auto-detected)
  --output-dir DIR             Output directory (default: trace_reports)
  --api-key KEY                AI API key for analysis
  --generate-plots             Generate visualization plots
  --disable-critical-path      Use timeline mode instead of critical path
  --no-save-intermediates      Don't save intermediate files
```

## Dependencies

- Python 3.8+
- TraceLens (for HTA reports)
- CritPath (for critical path analysis)
- pandas, numpy (for data processing)
- matplotlib (for plotting)
- anthropic (for AI analysis)

## Example Workflow

```python
from Analysis import JarvisAnalyzer

# Create analyzer
analyzer = JarvisAnalyzer(
    gpu1_kineto="mi300x_trace.json",
    gpu1_et="mi300x_et.json",
    gpu2_kineto="h100_trace.json",
    gpu2_et="h100_et.json",
    api_key="YOUR_KEY",
    use_critical_path=True,
    generate_plots=True
)

# Run analysis
success = analyzer.run()
```

## Output Structure

```
trace_reports/
└── GPU1_vs_GPU2_20250108_120000/
    ├── GPU1/
    │   ├── linked_trace.json
    │   ├── critical_path.pkl
    │   └── tracelens_report.txt
    ├── GPU2/
    │   ├── linked_trace.json
    │   ├── critical_path.pkl
    │   └── tracelens_report.txt
    ├── comparison_report.txt
    ├── plots/
    │   ├── gap_analysis.png
    │   ├── cumulative_optimization.png
    │   └── kernel_breakdown.png
    └── GPU1_vs_GPU2_Analysis_Report.md
```

## Migration from Monolithic Version

The original `jarvis_analysis.py` (2884 lines) has been split into:

1. **tracelens_runner.py** (211 lines)
   - Extracted: TraceLens report generation, critical path execution
   
2. **llm_prompts.py** (283 lines)
   - Extracted: All LLM prompt building logic
   
3. **plotting_complete.py** (~1100 lines) ✅ **COMPLETE**
   - Extracted: All plotting functions from original jarvis_analysis.py
   - Includes: gap analysis, optimization charts, cumulative progression
   
4. **report_generator.py** (143 lines)
   - Extracted: Markdown report assembly
   
5. **jarvis_analysis.py** (420 lines)
   - Refactored: Main orchestration only

**Total**: ~2157 lines organized into logical modules (vs 2884 monolithic)

## Benefits of Modular Architecture

1. **Maintainability**: Each module has clear responsibilities
2. **Testability**: Modules can be unit tested independently
3. **Reusability**: Modules can be used in other projects
4. **Readability**: ~400 lines per file vs 2884 monolithic
5. **Extensibility**: Easy to add new features or analysis modes
6. **Debugging**: Easier to isolate issues

## Future Enhancements

- [x] Complete plotting.py implementation ✅
- [ ] Add unit tests for each module
- [ ] Add type hints throughout
- [ ] Create example notebooks
- [ ] Add configuration file support
- [ ] Implement caching for expensive operations
- [ ] Add support for more GPU types
- [ ] Create web interface for report viewing

## Contributing

When adding new features:
1. Choose the appropriate module based on responsibility
2. Follow existing code style (docstrings, type hints)
3. Update this README
4. Add examples to the appropriate section

## License

Same as parent TraceLens-Jarvis project.
