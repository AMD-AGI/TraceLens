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
from typing import Dict, Optional, List, Any


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
                             summary_data: Dict,
                             gemm_data: Dict,
                             conv_data: Dict,
                             unique_data: Dict,
                             overall_data: Dict,
                             detailed_comparison: Optional[Dict],
                             critical_path_section: str,
                             cp_comparison_data: List[Dict],
                             baseline_cp_ops: List[str]) -> str:
        """
        Build the main analysis prompt for the LLM.
        
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
            target_total_time,
            critical_path_section,
            cp_comparison_data,
            baseline_cp_ops
        )
        prompt = f"""
        You are a GPU performance expert analyzing execution traces between Baseline and Target.
        {focus_instruction}

        ANALYSIS MODE: {self.analysis_mode}
        DATA SOURCE: {data_explanation}
        
        TAGS:
        - Baseline
        - Target

        SUMMARY DATA:
        {json.dumps(summary_data, indent=2)}

        GEMM ANALYSIS:
        {json.dumps(gemm_data, indent=2)}

        CONVOLUTION ANALYSIS:
        {json.dumps(conv_data, indent=2)}

        UNIQUE OPERATIONS:
        {json.dumps(unique_data, indent=2)}

        OVERALL INSIGHTS:
        {json.dumps(overall_data, indent=2)}

        DETAILED COMPARISON:
        {json.dumps(detailed_comparison, indent=2) if detailed_comparison else 'No detailed comparison available.'}

        {data_focus} ANALYSIS {"(PRIMARY FOCUS - Use this section to guide recommendations)" if self.use_critical_path else "(Context - Critical path disabled, focus on timeline data below)"}:
        {critical_path_section}

        {data_focus} COMPARISON DATA {"(Operations on critical path only - these directly impact overall time)" if self.use_critical_path else "(All operation categories aggregated by total time - analyze comprehensively)"}:
        {json.dumps(cp_comparison_data, indent=2)}

        DATA INTERPRETATION:
        {"- Each row represents an operation ON THE CRITICAL PATH (dependency chain)" if self.use_critical_path else "- Each row represents an OPERATION CATEGORY from the full timeline"}
        {"- baseline_time_ms: Time this critical path operation takes on Baseline" if self.use_critical_path else "- baseline_time_ms: Total time ALL operations in this category take on Baseline"}
        {"- target_time_ms: Time same operation takes on Target (if it exists on Target's critical path)" if self.use_critical_path else "- target_time_ms: Total time ALL operations in this category take on Target"}
        {"- impact_pct: What % of TOTAL execution time this critical path operation represents" if self.use_critical_path else "- impact_pct: What % of TOTAL execution time this category represents"}
        {"- Focus on operations with high impact_pct that show large gaps" if self.use_critical_path else "- Categories with high impact_pct are most important regardless of gap size"}

        STRICT FOCUS:
        - If the data list is NON-EMPTY, render it as a markdown table with section heading "## {data_focus} COMPARISON" and columns: Operation | Baseline ms | Target ms | Gap (ms) | {"Impact on Total Time (%)" if self.use_critical_path else "% of Total Time"}.
        - If the data list is EMPTY, DO NOT render the "{data_focus} COMPARISON" table section at all. Instead, base your analysis and recommendations on the {data_focus} ANALYSIS narrative above.
        - Order rows by gap_ms descending, then impact_pct when available.
        {"- Avoid introducing operations clearly not on Baseline's critical path." if self.use_critical_path else "- Focus on operations with highest total execution time and frequency."}

        {"STRICT FILTERING RULES FOR TABLES (ONLY IF DATA IS NON-EMPTY):" if self.use_critical_path else "TABLE GUIDELINES:"}
        {"- If rendering the \"CRITICAL PATH COMPARISON\" table, ONLY include operations present on the Baseline critical path." if self.use_critical_path else "- Include all significant operations from the timeline data."}
        {"- Ignore and DO NOT LIST operations that are not on the Baseline critical path (e.g., backend-only ops like cudnn/miopen if not on Baseline CP)." if self.use_critical_path else "- Prioritize operations with highest total time contribution."}
        {"- Baseline critical-path operation names (use ONLY these when available, case-sensitive):" if self.use_critical_path else "- Key operations to analyze:"}
        {json.dumps(baseline_cp_ops, indent=2) if self.use_critical_path else "All operations from timeline data"}

        {gap_instruction}
        
        IMPORTANT INSTRUCTIONS:
        {"- PRIMARY FOCUS: Critical path operations of Baseline have highest impact on performance - analyze and prioritize these" if self.use_critical_path else "- PRIMARY FOCUS: Operations with highest total execution time and frequency have most impact on performance"}
        {"- SECONDARY ANALYSIS: Provide whole workload context but emphasize critical path findings" if self.use_critical_path else "- ANALYSIS APPROACH: Analyze all operations based on timeline data, considering both individual time and cumulative impact"}
        - RECOMMENDATIONS: All optimization suggestions should focus on {"critical path " if self.use_critical_path else ""}impact with measurable potential gains
        - FORMATTING: Present findings in well-structured markdown with tables, sections, and visual hierarchy
        - CLARITY: Make the report easy to scan and understand at a glance
        """
        
        return prompt
    
    def _build_gap_instruction(self, 
                               baseline_is_trailing: bool,
                               performance_gap_ms: float,
                               performance_gap_pct: float,
                               baseline_total_time: float,
                               target_total_time: float,
                               critical_path_section: str,
                               cp_comparison_data: List[Dict],
                               baseline_cp_ops: List[str]) -> str:
        """Build gap analysis instruction based on performance relationship"""
        
        if baseline_is_trailing:
            return f"""
            PERFORMANCE GAP ANALYSIS ({self.analysis_mode} Mode):
            - Baseline is TRAILING by {performance_gap_ms:.2f} ms ({performance_gap_pct:.2f}%)
            - Baseline Total Time: {baseline_total_time:.2f} ms
            - Target Total Time: {target_total_time:.2f} ms
            - Provide the breakdown of performance based on timeline data and highlight the category i.e. Comp, Comm, memcopy, busy, idle that we are addressing here.
            - Provide the percentage of that particular category with respect to the total time for both Baseline and Target. Give the percentage comparison what percentage of that category we can improve for baseline in comparison to the target. Also, provide how much it will impact the total latency.
            - Don't mention the GPU names like MI300 or H200, just refer to them as Baseline and Target
            
            {self._get_analysis_structure()}
            
            5. OPPORTUNITIES AND IMPACT (Priority | Category | Current Time | Target Time | Potential Gain | % Impact on Overall)
            - IMPORTANT: Calculate "% Impact on Overall" as: (Potential Gain ms / Baseline Total Time ms) × 100
            - Example: If Potential Gain = 1928.36 ms and Baseline Total = 18616.44 ms, then % Impact = (1928.36 / 18616.44) × 100 = 10.36%
            - Priority should be based on {"critical path impact" if self.use_critical_path else "% of Total Time (impact_pct) - higher percentage = higher priority"}
            - Use EXACT {"operation names from the critical path data" if self.use_critical_path else "category names from the CATEGORY PERFORMANCE OVERVIEW table (e.g., 'Convolution Operations', 'Batch Normalization', 'NCCL Operations', etc.)"}
            6. ROOT CAUSE ANALYSIS (explain fundamental reasons for {"critical path" if self.use_critical_path else "category-level"} performance gaps)
            7. RECOMMENDED ACTIONS (actionable steps to optimize {"critical path operations" if self.use_critical_path else "high-impact categories"} with estimated overall improvement potential)
            
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
            
            ✨ {"CRITICAL PATH" if self.use_critical_path else "PERFORMANCE"} ANALYSIS:
            Analyze {"the critical path" if self.use_critical_path else "execution patterns"} to understand why the Baseline is performing better.
            
            Focus on:
            1. {"Operations on the Baseline critical path - identify optimizations that reduce critical path length" if self.use_critical_path else "High-impact operations and categories where Baseline excels"}
            2. {"Compare critical path between Baseline and Target - highlight differences" if self.use_critical_path else "Compare operation patterns between Baseline and Target"}
            3. {"Identify operations where Baseline has optimizations that Target lacks" if self.use_critical_path else "Identify optimizations and features that give Baseline an advantage"}
            
            Provide a performance analysis with the following structure and formatting:
            
            1. {"CRITICAL PATH" if self.use_critical_path else "PERFORMANCE"} COMPARISON (table: Operation | Baseline | Target | Baseline Advantage | Impact)
            2. {"CRITICAL PATH" if self.use_critical_path else "PERFORMANCE"} STRENGTHS (top baseline operations)
            3. {"CRITICAL PATH" if self.use_critical_path else "PERFORMANCE"} ANALYSIS BY CATEGORY (Category | Time on Baseline | Time on Target | Baseline Advantage | % of {"Critical Path" if self.use_critical_path else "Total Time"})
            4. KEY PERFORMANCE DIFFERENTIATORS (baseline optimizations)
            5. ADVANTAGES (explain features that enable baseline optimization)
            
            Format requirements:
            - Use markdown tables with clear headers for all comparisons
            - Use bold for key metrics and performance advantages
            - Use emoji indicators (✨ for exceptional performance, 🏆 for wins, ⚡ for efficiency)
            {"- Clearly mark which operations are on the critical path" if self.use_critical_path else ""}
            - Group related insights visually using markdown sections
            - Include performance advantage percentages and time savings
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
            
            1. CATEGORY PERFORMANCE OVERVIEW (table: Category | Baseline Time (ms) | Target Time (ms) | Gap (ms) | % of Total Time | Status)
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
            4. CATEGORY-LEVEL INSIGHTS (detailed analysis of each major category)'''
    
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
