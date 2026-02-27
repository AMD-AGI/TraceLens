#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
LLM Prompt Templates Module
Manages all LLM prompt generation for Jarvis analysis
"""

import json
from typing import Dict, Optional, List
from pathlib import Path


class LLMPromptManager:
    """Manages LLM prompt generation for different analysis modes"""
    
    def __init__(self, use_critical_path: bool = True):
        """
        Args:
            use_critical_path: Whether to use critical path focused prompts
        """
        self.use_critical_path = use_critical_path
        self.analysis_mode = "Critical Path" if use_critical_path else "Timeline"
    
    def build_analysis_prompt(self,
                             baseline_gpu: str,
                             target_gpu: str,
                             baseline_total_time: float,
                             target_total_time: float,
                             critical_path_section: str,
                             markdown_data_section: str) -> str:
        """
        Build the main analysis prompt for the LLM.
        
        Args:
            baseline_gpu: Name of baseline GPU
            target_gpu: Name of target GPU
            baseline_total_time: Total execution time for baseline
            target_total_time: Total execution time for target
            critical_path_section: Critical path analysis text
            markdown_data_section: Markdown formatted data (REQUIRED)
            
        Returns:
            Complete prompt string
        """
        # Calculate performance metrics
        baseline_is_trailing = baseline_total_time > target_total_time
        performance_gap_ms = abs(baseline_total_time - target_total_time)
        performance_gap_pct = (performance_gap_ms / min(baseline_total_time, target_total_time) * 100) if min(baseline_total_time, target_total_time) > 0 else 0
        
        # Adapt prompt based on analysis mode
        if self.use_critical_path:
            focus_instruction = "Your analysis MUST focus primarily on critical path operations of Baseline as they directly impact overall execution time."
            data_focus = "CRITICAL PATH"
            data_explanation = "The data below represents operations on the critical path (~7-10% of all operations). These operations form dependency chains that directly determine overall execution time."
        else:
            focus_instruction = "Your analysis should focus on timeline-based execution data, analyzing ALL operation categories by their total execution time and contribution to overall performance."
            data_focus = "TIMELINE"
            data_explanation = "The data below represents ALL operation categories aggregated from the complete timeline. Analyze based on total execution time, frequency, and percentage contribution to overall runtime."
        
        # Build gap instruction based on performance relationship
        gap_instruction = self._build_gap_instruction(
            baseline_is_trailing,
            performance_gap_ms,
            performance_gap_pct,
            baseline_total_time,
            target_total_time
        )
        
        prompt = f"""
        You are a GPU performance expert analyzing execution traces between Baseline and Target.
        {focus_instruction}

        ANALYSIS MODE: {self.analysis_mode}
        DATA SOURCE: {data_explanation}
        
        TAGS:
        - Baseline
        - Target

        ================================ DETAILED TRACELENS DATA ================================
        
        {markdown_data_section}
        
        =========================================================================================
        
        DETAILED DATA INTERPRETATION:
            The markdown data above contains 4 key sheets for analysis:
            
            1. GPU_TIMELINE - High-level GPU utilization metrics:
               - computation_time: Total GPU computation time
               - exposed_comm_time: Communication time not overlapped with computation
               - exposed_memcpy_time: Memory copy time not overlapped with computation
               - busy_time: Total time GPU is executing work
               - idle_time: Time GPU is idle
               - total_time: Overall trace duration
            
            2. OPS_SUMMARY - Operation-level aggregation:
               - Groups by operation name only (e.g., all aten::addmm calls together)
               - Shows total time, count, and performance metrics per operation type
            
            3. OPS_SUMMARY_BY_CATEGORY - Category-level aggregation:
               - High-level view: GEMM, CONV_fwd, CONV_bwd, BN_fwd, BN_bwd, etc.
               - Best for identifying which operation categories dominate execution time
            
            4. OPS_UNIQUE_ARGS - Most detailed view (operation + arguments):
               - Groups operations by unique combinations of (name + input shapes + dtypes + strides)
               - Critical for identifying which specific input patterns cause performance issues
               - Example: aten::addmm with (1024, 512) vs (2048, 1024) are separate rows
        
        {data_focus} ANALYSIS {"(PRIMARY FOCUS - Use this section to guide recommendations)" if self.use_critical_path else "(Context - Critical path disabled, focus on timeline data above)"}:
        {critical_path_section}

        IMPORTANT: Extract all comparison data from the markdown tables above. The OPS_SUMMARY_BY_CATEGORY table contains the aggregated category-level comparison you need for your analysis.

        {gap_instruction}
        
        IMPORTANT INSTRUCTIONS:
        {"- PRIMARY FOCUS: Critical path operations of Baseline have highest impact on performance - analyze and prioritize these" if self.use_critical_path else "- PRIMARY FOCUS: Operations with highest total execution time and frequency have most impact on performance"}
        {"- SECONDARY ANALYSIS: Provide whole workload context but emphasize critical path findings" if self.use_critical_path else "- ANALYSIS APPROACH: Analyze all operations based on timeline data, considering both individual time and cumulative impact"}
        - RECOMMENDATIONS: All optimization suggestions should focus on {"critical path " if self.use_critical_path else ""}impact with measurable potential gains
        - FORMATTING: Present findings in well-structured markdown with tables, sections, and visual hierarchy
        - CLARITY: Make the report easy to scan and understand at a glance
        - DO NOT include a main title heading (# or H1) at the start - begin directly with the first section header
        """
        return prompt
    
    def _build_gap_instruction(self, 
                               baseline_is_trailing: bool,
                               performance_gap_ms: float,
                               performance_gap_pct: float,
                               baseline_total_time: float,
                               target_total_time: float) -> str:
        """Build gap analysis instruction based on performance relationship"""
        
        if baseline_is_trailing:
            return f"""
            PERFORMANCE GAP ANALYSIS ({self.analysis_mode} Mode):
            - Baseline is TRAILING by {performance_gap_ms:.2f} ms ({performance_gap_pct:.2f}%)
            - Baseline Total Time: {baseline_total_time:.2f} ms
            - Target Total Time: {target_total_time:.2f} ms
            - Don't mention the GPU names like MI300 or H200, just refer to them as Baseline and Target
            
            MANDATORY FIRST SECTION - GPU TIMELINE CATEGORY ANALYSIS:
            At the VERY START of your report (before any other analysis), you MUST provide:
            
            1. **GPU TIMELINE CATEGORY COMPARISON** - A comprehensive table comparing ALL timeline categories:
               | Category | Baseline Time (ms) | Baseline % | Target Time (ms) | Target % | Gap (ms) | Status |
               
               Include ALL categories from the GPU TIMELINE CATEGORIES data:
               - computation_time
               - exposed_comm_time
               - exposed_memcpy_time
               - busy_time
               - idle_time
               - total_time
               
               For each category:
               - Show absolute time in milliseconds
               - Show percentage of total time (baseline_pct and target_pct from data)
               - **USE GAP VALUES EXACTLY AS PROVIDED** - Gap is pre-calculated as (baseline - target)
               - **DO NOT recalculate or negate gap values**
               - Add status indicator based on CORRECT semantics (Gap = Baseline - Target):
                 * For computation_time, exposed_comm_time, exposed_memcpy_time, busy_time: LOWER is BETTER
                   - 🔴 if gap > 0 (POSITIVE gap = Baseline has MORE time = slower/less efficient)
                   - 🟢 if gap < 0 (NEGATIVE gap = Baseline has LESS time = faster/more efficient)
                 * For idle_time: HIGHER is WORSE
                   - 🔴 if gap > 0 (POSITIVE gap = Baseline has MORE idle = slower/inefficient scheduling)
                   - 🟢 if gap < 0 (NEGATIVE gap = Baseline has LESS idle = faster/better utilization)
                 * For total_time: LOWER is BETTER
                   - 🔴 if gap > 0 (POSITIVE gap = Baseline finishes in MORE time = slower overall)
                   - 🟢 if gap < 0 (NEGATIVE gap = Baseline finishes in LESS time = faster overall)
                 * ⚪ if gap is near zero (within ±1ms or ±1% relative difference)
               
            2. **TIMELINE CATEGORY INSIGHTS** - Immediately after the table, provide:
               - Identify categories where Baseline is genuinely slower/worse (marked with 🔴)
               - For computation/busy/comm categories: Baseline slower = more time spent
               - For idle time: Baseline slower = more idle (scheduling inefficiency)
               - What is the combined impact of poor categories on total latency?
               - Are there systemic issues (e.g., high exposed_comm_time suggests poor overlap, high idle_time suggests scheduling issues)?
               - Highlight the TOP category where Baseline is weakest compared to Target
               - IMPORTANT: Correctly interpret the relationship between time spent and performance:
                 * Less computation time = more efficient computation
                 * More idle time = less efficient (poor utilization)
                 * Overall: Total time determines winner, but analyze WHY (efficient computation vs excessive idle)
               
            This section must appear FIRST before any other analysis sections.
            
            {self._get_analysis_structure()}
            
            5. OPPORTUNITIES AND IMPACT (Priority | Category | Current Time | Target Time | Potential Gain | % Impact on Overall)
            - IMPORTANT: Calculate "% Impact on Overall" as: (Potential Gain ms / Baseline Total Time ms) × 100
            - Example: If Potential Gain = 1928.36 ms and Baseline Total = 18616.44 ms, then % Impact = (1928.36 / 18616.44) × 100 = 10.36%
            - Priority should be based on {"critical path impact" if self.use_critical_path else "% of Total Time (impact_pct) - higher percentage = higher priority"}
            - Use EXACT {"operation names from the critical path data" if self.use_critical_path else "category names from the CATEGORY PERFORMANCE OVERVIEW table (e.g., 'Convolution Operations', 'Batch Normalization', 'NCCL Operations', etc.)"}
            
            6. ROOT CAUSE ANALYSIS
            - Explain fundamental reasons for {"critical path" if self.use_critical_path else "category-level"} performance gaps
            - **IDENTIFY SLOW OPERATIONS FROM OPERATIONS SUMMARIES**: Reference specific operation names and their performance characteristics
            - For each slow category (e.g., GEMM, CONV_fwd), highlight the top 2-3 slowest operations with their gap_ms and speedup
            - Identify patterns (memory bandwidth, KERNEL NAMES and EFFECIENCY, operation shape/size issues)
            - Keep concise but specific to the actual operations
            
            7. RECOMMENDED ACTIONS
            - Actionable steps to optimize {"critical path operations" if self.use_critical_path else "high-impact categories"} with estimated overall improvement potential
            - **CREATE OPERATION-SPECIFIC SUBSECTIONS**: For each major slow category (GEMM, CONV, etc.)
            - Section: "### Immediate High-Impact Actions (🎯)"
              * For each slow category, create a subsection: "#### {{Category}} Optimization Priorities"
              * List top 5-7 specific operations from DETAILED SLOW OPERATIONS DATA
              * Table format: | Operation Name | Current (ms) | Target (ms) | Potential Gain | Speedup | Recommended Action |
              * Provide specific, actionable recommendations for EACH operation listed and also the fined grained kernerl name and shapes
            - Section: "### Medium-Priority Optimizations (⚡)"
              * Similar structure but fewer operations per category
            - **IF PREFILL/DECODE DATA EXISTS**: Add brief note on phase-specific optimizations (1-2 lines)
            - End with: "### Expected Overall Impact" showing total potential improvement
            
            Format requirements:
            - Use markdown tables with clear headers for all comparisons
            - Use bold for key metrics and numbers
            - Use emoji indicators where appropriate (⚠️ for concerns, 🎯 for high-impact optimizations, ⚡ for quick wins)
            {"- Clearly mark which operations are on the critical path" if self.use_critical_path else ""}
            - Group related insights visually using markdown sections
            - Include percentage improvements and absolute time savings
            - For OPPORTUNITIES AND IMPACT table, ensure % Impact on Overall is calculated relative to BASELINE TOTAL TIME, not gap
            """
        else:
            return f"""
            PERFORMANCE COMPARISON:
            - Baseline is LEADING with {performance_gap_pct:.2f}% better performance
            - Baseline Total Time: {baseline_total_time:.2f} ms
            - Target Total Time: {target_total_time:.2f} ms
            - Don't mention the GPU names like MI300 or H200, just refer to them as Baseline and Target
            
            IMPORTANT: Use the SAME report structure and table format as when Baseline is trailing. The only difference is that some operations will show Baseline faster (strengths) and some slower (weaknesses).
            
            **CRITICAL GAP CALCULATION RULES - USE PRE-CALCULATED VALUES:**
            - Gap values are PRE-CALCULATED in the "CATEGORY PERFORMANCE COMPARISON" section
            - **COPY GAP VALUES EXACTLY from the CATEGORY PERFORMANCE COMPARISON table**
            - **DO NOT calculate gaps yourself from individual Baseline/Target tables**
            - **DO NOT negate or flip the sign of gap values**
            - Gap Formula: Gap = Baseline - Target
            - Interpretation:
              * POSITIVE gap (e.g., +147.34) = Baseline SLOWER (took more time) → 🔴 Status
              * NEGATIVE gap (e.g., -147.34) = Baseline FASTER (took less time) → 🟢 Status
              * Near-zero gap (±1ms) = Similar performance → ⚪ Status
            
            EXAMPLE from data:
            If CATEGORY PERFORMANCE COMPARISON shows: Baseline=266.51ms, Target=119.17ms, Gap=+147.34ms, Status=🔴 Slower
            Your output MUST show: Gap=+147.34ms, Status=🔴 Slower (copy exactly, do NOT recalculate)
            
            Provide a comprehensive performance analysis with the following EXACT structure:
            
            1. CATEGORY PERFORMANCE OVERVIEW
               Table format: | Category | Baseline Time (ms) | Target Time (ms) | Gap (ms) | % of Total Time | Status |
               - **Copy Gap values EXACTLY from the data without modification**
               - Gap interpretation: positive = Baseline slower (🔴), negative = Baseline faster (🟢)
               - Sort by % of Total Time descending (highest impact first)
               - Status: 🔴 Slower, 🟢 Faster, ⚪ Similar
               
            2. HIGH-IMPACT OPTIMIZATION OPPORTUNITIES
               Table format: | Priority | Category | Current Time | Target Time | Potential Gain | % Impact on Overall |
               - ONLY include categories where Baseline is SLOWER (POSITIVE gaps from table 1)
               - Priority: 🎯 High (>5% of total), ⚡ Medium (1-5%), ⚪ Low (<1%)
               - % Impact on Overall = (Potential Gain / Baseline Total Time) × 100
               - End with: **CUMULATIVE IMPACT:** X.XX% total potential improvement if all gaps are closed
               
            3. PERFORMANCE STRENGTHS
               - List categories where Baseline is FASTER (NEGATIVE gaps from table 1)
               - For each: Show percentage faster and absolute advantage
               - Brief explanation of why Baseline excels in these areas
               
            4. CATEGORY-LEVEL INSIGHTS
               - Group related categories (e.g., BN operations, CONV operations)
               - Show combined impact and performance characteristics
               - **DO NOT include percentages or impact metrics in section headers** - use clean headers like "### Batch Normalization Operations"
               - **IF PREFILL/DECODE DATA EXISTS**: Add brief subsection comparing prefill vs decode performance
                 * Show total time for each phase (baseline vs target)
                 * Identify if gaps are phase-specific
                 * Keep to 2-3 bullet points maximum
               - Use 🎯 for high-impact groups, ⚡ for medium-impact in the content, not headers
               
            5. ROOT CAUSE ANALYSIS
            - Explain fundamental reasons for {"critical path" if self.use_critical_path else "category-level"} performance gaps
            - **IDENTIFY SLOW OPERATIONS FROM OPERATIONS SUMMARIES**: Reference specific operation names and their performance characteristics
            - For each slow category (e.g., GEMM, CONV_fwd), highlight the top 2-3 slowest operations with their gap_ms and speedup
            - Identify patterns (memory bandwidth, KERNEL NAMES and EFFECIENCY, operation shape/size issues)
            - Keep concise but specific to the actual operations
            
            6. RECOMMENDED ACTIONS
            - Actionable steps to optimize {"critical path operations" if self.use_critical_path else "high-impact categories"} with estimated overall improvement potential
            - **CREATE OPERATION-SPECIFIC SUBSECTIONS**: For each major slow category (GEMM, CONV, etc.)
            - Section: "### Immediate High-Impact Actions (🎯)"
              * For each slow category, create a subsection: "#### {{Category}} Optimization Priorities"
              * List top 5-7 specific operations from DETAILED SLOW OPERATIONS DATA
              * Table format: | Operation Name | Current (ms) | Target (ms) | Potential Gain | Speedup | Recommended Action |
              * Provide specific, actionable recommendations for EACH operation listed and also the fined grained kernerl name and shapes
            - Section: "### Medium-Priority Optimizations (⚡)"
              * Similar structure but fewer operations per category
            - **IF PREFILL/DECODE DATA EXISTS**: Add brief note on phase-specific optimizations (1-2 lines)
            - End with: "### Expected Overall Impact" showing total potential improvement
            
            Format requirements:
            - Use markdown tables with clear headers for all comparisons
            - Use bold for key metrics
            - Use emoji indicators (🎯 high-impact, ⚡ medium-impact, ✓ for strengths)
            - Group related insights visually using markdown sections
            - Include percentages and absolute time values
            """
    
    def _get_analysis_structure(self) -> str:
        """Get analysis structure based on mode"""
        if self.use_critical_path:
            return '''You MUST prioritize analysis of operations on the critical path. The critical path represents the longest sequence of dependent operations and directly determines overall execution time.
            
            Analyze and provide suggestions focusing on:
            1. Operations that appear on the CRITICAL PATH of Baseline and not the Target
            2. Compare critical path operations between Baseline and Target
            3. Identify critical path bottlenecks specific to Baseline that don't appear or are faster on Target
            
            Provide a detailed gap analysis with the following structure and formatting:
            
            1. CRITICAL PATH COMPARISON (table: Operation | Baseline (CP) | Target (CP) | Gap | Impact on Total Time)
            2. CRITICAL PATH BOTTLENECKS (top 5 baseline critical-path operations underperforming)
            3. CRITICAL PATH ANALYSIS BY CATEGORY (table: Category | Time on Baseline | Time on Target | % of Critical Path)
            4. NON-CRITICAL OPERATIONS (brief mention of non-critical operations with noted performance differences for context)'''
        else:
            return '''You are analyzing ALL operation categories from the complete timeline. Focus on categories with:
            - Highest total execution time (impact_pct)
            - Largest time gaps between Baseline and Target
            - High percentage contribution to overall performance
            
            KEY INSIGHT: In timeline mode, a category with 20% impact_pct that shows NO gap is still MORE important than a category with 2% impact_pct showing a large gap. Prioritize by impact_pct first.
            
            Analyze and provide suggestions focusing on:
            1. Categories with highest impact_pct (% of Total Time) - these are your biggest optimization targets
            2. Categories showing significant time differences between Baseline and Target
            3. Categories where Baseline is slower - potential optimization opportunities
            4. Categories where Baseline is faster - understand what's working well
            
            Provide a detailed analysis with the following structure and formatting:
            
            **CRITICAL GAP CALCULATION RULES - USE PRE-CALCULATED VALUES:**
            - Gap values are PRE-CALCULATED in the "CATEGORY PERFORMANCE COMPARISON" section
            - **COPY GAP VALUES EXACTLY from the CATEGORY PERFORMANCE COMPARISON table**
            - **DO NOT calculate gaps yourself from individual Baseline/Target tables**
            - **DO NOT negate or flip the sign of gap values**
            - Gap Formula: Gap = Baseline - Target
            - Interpretation:
              * POSITIVE gap (e.g., +147.34) = Baseline SLOWER (took more time) → 🔴 Status
              * NEGATIVE gap (e.g., -147.34) = Baseline FASTER (took less time) → 🟢 Status
              * Near-zero gap (±1ms) = Similar performance → ⚪ Status
            
            EXAMPLE from data:
            If CATEGORY PERFORMANCE COMPARISON shows: Baseline=266.51ms, Target=119.17ms, Gap=+147.34ms, Status=🔴 Slower
            Your output MUST show: Gap=+147.34ms, Status=🔴 Slower (copy exactly, do NOT recalculate)
            
            1. CATEGORY PERFORMANCE OVERVIEW (table: Category | Baseline Time (ms) | Target Time (ms) | Gap (ms) | % of Total Time | Status)
               - **Copy Gap values EXACTLY from the data without modification**
               - Sort by % of Total Time descending
               - Status: "🔴 Slower", "🟢 Faster", "⚪ Similar"
            2. HIGH-IMPACT OPTIMIZATION OPPORTUNITIES (ALL categories where Baseline is slower, prioritized by % of Total Time)
               - MANDATORY TABLE FORMAT (DO NOT DEVIATE): | Priority | Category | Current Time | Target Time | Potential Gain | % Impact on Overall |
               - STRICT HEADER (EXACTLY AS SHOWN): | Priority | Category | Current Time | Target Time | Potential Gain | % Impact on Overall |
               - The table MUST have exactly 6 columns in this exact order
               - Include ALL categories where Baseline has positive gap (slower than Target)
               - Sort by % of Total Time (impact_pct) descending - this is the PRIMARY priority metric
               - IMPORTANT: Calculate "% Impact on Overall" as: (Potential Gain ms / Baseline Total Time ms) × 100
               - Example: If Potential Gain = 1634.63 ms and Baseline Total = 18616.44 ms, then % Impact = (1634.63 / 18616.44) × 100 = 8.78%
               - Priority markers: 🎯 High (>5% of total time), ⚡ Medium (1-5%), ⚪ Low (<1%)
               - Use EXACT category names from section 1 (e.g., 'BN_bwd', 'CONV_bwd', 'BN_fwd', etc.)
               - CALCULATE CUMULATIVE IMPACT: After table, show "**CUMULATIVE IMPACT:** X.XX% total potential improvement if all gaps are closed"
            3. PERFORMANCE STRENGTHS (categories where Baseline performs better)
            4. CATEGORY-LEVEL INSIGHTS (detailed analysis of each major category)
               - **DO NOT include percentages or impact metrics in section headers** - use clean headers like "### Batch Normalization Operations"
               - **IF PREFILL/DECODE DATA EXISTS**: Include brief subsection on phase-specific performance (prefill vs decode)
                 * Compare total time per phase between baseline and target
                 * Identify if performance gaps are concentrated in one phase
                 * Keep concise (2-3 bullet points)'''
    
    def get_system_message(self) -> str:
        """Get system message for LLM based on analysis mode"""
        if self.use_critical_path:
            analysis_context = """You are analyzing GPU performance with a CRITICAL PATH focus. Critical path operations directly determine overall execution time through dependency chains. Prioritize operations on the critical path (~7-10% of all operations) as they have the highest impact on performance."""
        else:
            analysis_context = """You are analyzing GPU performance with a TIMELINE focus. Analyze all operations based on their total execution time and frequency. Consider both individual operation times and their cumulative contribution to overall performance."""
        
        return f"""You are a GPU performance expert specializing in ML workload optimization.

{analysis_context}

CRITICAL INSTRUCTIONS FOR CONSISTENT OUTPUT:
1. Always use EXACT numbers from the provided data - never round or estimate
2. Use identical section headers and formatting across all analyses
3. Order all items by performance impact (highest first)
4. Use consistent emoji indicators: ⚠️ for gaps, ✓ for strengths, 🎯 for high-priority items, ⚡ for efficiency
5. Present all tables in the same format and order
6. Do not add subjective commentary - stick to data-driven insights
7. When comparing metrics, show both absolute values and percentages
8. Maintain consistent decimal precision (e.g., always 2 decimal places for percentages)
9. For recommendations, follow the exact format: [Priority] | [Action] | [Current Impact] | [Potential Gain]
10. Never deviate from the requested structure - always provide all required sections even if data is limited"""
    
    def load_markdown_report(self, markdown_path: Path) -> str:
        """
        Load a markdown report file for inclusion in prompts.
        
        Args:
            markdown_path: Path to the markdown file
            
        Returns:
            Content of the markdown file as a string
        """
        markdown_path = Path(markdown_path)
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown report not found: {markdown_path}")
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    
    def append_markdown_to_prompt(self, base_prompt: str, 
                                   baseline_md_path: Optional[Path] = None,
                                   target_md_path: Optional[Path] = None) -> str:
        """
        Append markdown reports to the analysis prompt.
        
        Args:
            base_prompt: The base analysis prompt
            baseline_md_path: Path to baseline GPU markdown report
            target_md_path: Path to target GPU markdown report
            
        Returns:
            Enhanced prompt with markdown reports appended
        """
        prompt_parts = [base_prompt]
        
        # Add baseline markdown
        if baseline_md_path:
            try:
                baseline_md = self.load_markdown_report(baseline_md_path)
                prompt_parts.append("\n\n" + "="*80)
                prompt_parts.append("\n## BASELINE GPU DETAILED TRACELENS REPORT\n")
                prompt_parts.append("="*80 + "\n")
                prompt_parts.append(baseline_md)
            except Exception as e:
                print(f"⚠️  Warning: Could not load baseline markdown: {e}")
        
        # Add target markdown
        if target_md_path:
            try:
                target_md = self.load_markdown_report(target_md_path)
                prompt_parts.append("\n\n" + "="*80)
                prompt_parts.append("\n## TARGET GPU DETAILED TRACELENS REPORT\n")
                prompt_parts.append("="*80 + "\n")
                prompt_parts.append(target_md)
            except Exception as e:
                print(f"⚠️  Warning: Could not load target markdown: {e}")
        
        return "\n".join(prompt_parts)
