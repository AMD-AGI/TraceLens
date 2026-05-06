###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


def generate_interactive_optimization_html(csv_path, plot_path):
    """
    Generate interactive HTML visualization of optimization opportunities.
    Creates a bar chart with expandable details for each category.
    """
    try:
        import pandas as pd
        import json

        df = pd.read_csv(csv_path)

        # Prepare data for JavaScript
        categories_data = []
        max_gain = df["Potential Gain (ms)"].max() if not df.empty else 1

        for _, row in df.iterrows():
            has_kernels = (
                pd.notna(row.get("Key Candidate Operations"))
                and row.get("Key Candidate Operations")
                != "Automated operation breakdown not yet supported"
            )

            categories_data.append(
                {
                    "category": row["Category"],
                    "current_time": float(row["Current Time (ms)"]),
                    "projected_time": float(row["Projected Optimized Time (ms)"]),
                    "potential_gain": float(row["Potential Gain (ms)"]),
                    "impact": float(row["Impact (%)"]),
                    "key_operations": (
                        row.get("Key Candidate Operations", "N/A")
                        if has_kernels
                        else None
                    ),
                    "ai_recommendations": row.get(
                        "Comments", "No recommendations available"
                    ),
                    "bar_height": float(
                        (row["Potential Gain (ms)"] / max_gain * 100)
                        if max_gain > 0
                        else 0
                    ),
                }
            )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workload Optimization Opportunities</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #ffffff;
            padding: 40px 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            color: #ed1c24;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }}

        h2 {{
            text-align: center;
            color: #ed1c24;
            font-size: 1.8em;
            margin-bottom: 20px;
        }}

        .subtitle {{
            text-align: center;
            color: #cccccc;
            font-size: 1.2em;
            margin-bottom: 40px;
        }}
        
        .chart-container {{
            background: #2a2a2a;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        
        .bar-wrapper {{
            margin-bottom: 25px;
            cursor: pointer;
            transition: transform 0.2s ease;
        }}
        
        .bar-wrapper:hover {{
            transform: translateX(5px);
        }}
        
        .bar-header {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .category-name {{
            font-weight: bold;
            color: #ffffff;
            min-width: 250px;
            font-size: 1.05em;
        }}
        
        .bar-container {{
            flex: 1;
            background: #1a1a1a;
            border-radius: 8px;
            height: 50px;
            position: relative;
            overflow: hidden;
            border: 2px solid #3a3a3a;
        }}
        
        .bar {{
            height: 100%;
            background: linear-gradient(90deg, #ed1c24 0%, #ff4d4d 100%);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 0 20px rgba(237, 28, 36, 0.3);
        }}
        
        .bar-wrapper:hover .bar {{
            background: linear-gradient(90deg, #ff3333 0%, #ff6666 100%);
            box-shadow: 0 0 30px rgba(237, 28, 36, 0.5);
        }}
        
        .bar-label {{
            color: #ffffff;
            font-weight: bold;
            font-size: 0.95em;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }}
        
        .impact-badge {{
            background: #ed1c24;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
            min-width: 60px;
            text-align: center;
        }}
        
        .details {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease;
            background: #1f1f1f;
            border-radius: 8px;
            margin-top: 10px;
        }}
        
        .details.expanded {{
            max-height: 2000px;
            border: 2px solid #ed1c24;
        }}
        
        .details-content {{
            padding: 20px;
        }}
        
        .detail-row {{
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #3a3a3a;
        }}
        
        .detail-row:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}
        
        .detail-label {{
            color: #ed1c24;
            font-weight: bold;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .detail-value {{
            color: #e0e0e0;
            line-height: 1.6;
            font-size: 0.95em;
        }}
        
        .metric {{
            display: inline-block;
            margin-right: 20px;
            color: #cccccc;
        }}
        
        .metric strong {{
            color: #ed1c24;
        }}
        
        .expand-icon {{
            display: inline-block;
            margin-left: 10px;
            transition: transform 0.3s ease;
            color: #ed1c24;
            font-size: 1.2em;
        }}
        
        .expanded .expand-icon {{
            transform: rotate(180deg);
        }}
        
        .kernels-list {{
            background: #2a2a2a;
            padding: 12px;
            border-radius: 6px;
            border-left: 3px solid #ed1c24;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        .kernel-item {{
            padding: 4px 0;
            color: #b0b0b0;
        }}
        
        .ai-comment {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #ed1c24;
            font-style: italic;
            line-height: 1.7;
        }}

        .plot-section {{
            background: #2a2a2a;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-bottom: 40px;
            text-align: center;
        }}

        .plot-section h2 {{
            color: #ed1c24;
            font-size: 1.8em;
            margin-bottom: 20px;
        }}

        .plot-section img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 2px solid #3a3a3a;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>

    
    

    <div class="container">
        

        <h1>GPU Optimization Opportunities</h1>


        <div class="plot-section">
            <h2>Overview</h2>
            <img src="{plot_path}" alt="Cumulative Optimization Progression">
        </div>


        <h2>Breakdown</h2>
        <p class="subtitle">Click on any category to view detailed analysis and key operations</p>
        
        <div class="chart-container">
            <div id="chart"></div>
        </div>
    </div>

    <script>
        const data = {json.dumps(categories_data)};
        
        function createBarChart() {{
            const chart = document.getElementById('chart');
            
            data.forEach((item, index) => {{
                const wrapper = document.createElement('div');
                wrapper.className = 'bar-wrapper';
                wrapper.onclick = () => toggleDetails(index);
                
                const header = document.createElement('div');
                header.className = 'bar-header';
                
                const categoryName = document.createElement('div');
                categoryName.className = 'category-name';
                categoryName.innerHTML = `${{item.category}} <span class="expand-icon">▼</span>`;
                
                const barContainer = document.createElement('div');
                barContainer.className = 'bar-container';
                
                const bar = document.createElement('div');
                bar.className = 'bar';
                bar.style.width = `${{item.bar_height}}%`;
                bar.innerHTML = `<span class="bar-label">${{item.potential_gain.toFixed(1)}} ms</span>`;
                
                const impactBadge = document.createElement('div');
                impactBadge.className = 'impact-badge';
                impactBadge.textContent = `${{item.impact.toFixed(1)}}%`;
                
                barContainer.appendChild(bar);
                header.appendChild(categoryName);
                header.appendChild(barContainer);
                header.appendChild(impactBadge);
                
                const details = document.createElement('div');
                details.className = 'details';
                details.id = `details-${{index}}`;
                
                let detailsHTML = `
                    <div class="details-content">
                        <div class="detail-row">
                            <div class="metric">
                                <strong>Current Time:</strong> ${{item.current_time.toFixed(2)}} ms
                            </div>
                            <div class="metric">
                                <strong>Projected Optimized Time:</strong> ${{item.projected_time.toFixed(2)}} ms
                            </div>
                            <div class="metric">
                                <strong>Potential Gain:</strong> ${{item.potential_gain.toFixed(2)}} ms
                            </div>
                        </div>
                `;
                
                if (item.key_operations) {{
                    const operations = item.key_operations.split(';').map(op => op.trim());
                    detailsHTML += `
                        <div class="detail-row">
                            <div class="detail-label">Key Candidate Operations</div>
                            <div class="kernels-list">
                                ${{operations.map(op => `<div class="kernel-item">• ${{op}}</div>`).join('')}}
                            </div>
                        </div>
                    `;
                }}
                
                detailsHTML += `
                        <div class="detail-row">
                            <div class="detail-label">Comment (AI)</div>
                            <div class="ai-comment">${{marked.parse(item.ai_recommendations)}}</div>
                        </div>
                    </div>
                `;
                
                details.innerHTML = detailsHTML;
                
                wrapper.appendChild(header);
                wrapper.appendChild(details);
                chart.appendChild(wrapper);
            }});
        }}
        
        function toggleDetails(index) {{
            const details = document.getElementById(`details-${{index}}`);
            const wrapper = details.parentElement;
            const icon = wrapper.querySelector('.expand-icon');
            
            details.classList.toggle('expanded');
            wrapper.classList.toggle('expanded');
        }}
        
        createBarChart();
    </script>
</body>
</html>"""

        html_path = csv_path.parent / "optimization_opportunities_interactive.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"    ✓ Interactive HTML: {html_path}")
        return html_path

    except Exception as e:
        print(f"    ⚠️  Failed to generate interactive HTML: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_kernel_optimization_html(csv_path):
    """
    Generate interactive HTML visualization of kernel-level optimization opportunities.
    Shows detailed kernel and CPU op name comparisons between baseline and target.
    """
    try:
        import pandas as pd
        import json

        df = pd.read_csv(csv_path)

        # Calculate totals for summary
        total_baseline_time = (
            df["baseline_time"].sum() / 1000.0 if "baseline_time" in df.columns else 0
        )
        total_opportunity = (
            df["opportunity"].sum() / 1000.0 if "opportunity" in df.columns else 0
        )
        total_optimized_time = total_baseline_time - total_opportunity
        total_optimization_pct = (
            (total_opportunity / total_baseline_time * 100)
            if total_baseline_time > 0
            else 0
        )

        # Prepare data for JavaScript
        categories_data = []
        max_opportunity = (
            df["opportunity"].max() / 1000.0
            if not df.empty and "opportunity" in df.columns
            else 1
        )

        for _, row in df.iterrows():
            # Parse kernel names (separated by ***)
            baseline_kernels = [
                k.strip()
                for k in str(row.get("baseline_kernel_names", "")).split("***")
                if k.strip()
            ]
            target_kernels = [
                k.strip()
                for k in str(row.get("target_kernel_names", "")).split("***")
                if k.strip()
            ]

            # Parse CPU op names (separated by ***)
            baseline_cpu_ops = [
                k.strip()
                for k in str(row.get("baseline_cpu_op_names", "")).split("***")
                if k.strip()
            ]
            target_cpu_ops = [
                k.strip()
                for k in str(row.get("target_cpu_op_names", "")).split("***")
                if k.strip()
            ]

            opportunity = float(row.get("opportunity", 0)) / 1000.0
            opportunity_pct_of_total = (
                (opportunity / total_opportunity * 100) if total_opportunity > 0 else 0
            )

            categories_data.append(
                {
                    "nn_modules": row.get("nn_modules", "N/A"),
                    "num_calls": int(row.get("num_aggregated_LCAs", 0)),
                    "baseline_time": float(row.get("baseline_time", 0)) / 1000.0,
                    "target_time": float(row.get("target_time", 0)) / 1000.0,
                    "opportunity": opportunity,
                    "opportunity_pct_of_total": opportunity_pct_of_total,
                    "bar_width": float(
                        (opportunity / max_opportunity * 100)
                        if max_opportunity > 0
                        else 0
                    ),
                    "baseline_kernels": baseline_kernels,
                    "target_kernels": target_kernels,
                    "baseline_cpu_ops": baseline_cpu_ops,
                    "target_cpu_ops": target_cpu_ops,
                }
            )

        # Sort by opportunity descending
        categories_data.sort(key=lambda x: x["opportunity"], reverse=True)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kernel-Level Optimization Opportunities</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #ffffff;
            padding: 40px 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            color: #ed1c24;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }}

        .subtitle {{
            text-align: center;
            color: #cccccc;
            font-size: 1.2em;
            margin-bottom: 30px;
        }}
        
        .summary-section {{
            background: #2a2a2a;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border-left: 5px solid #ed1c24;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
            margin-bottom: 20px;
        }}
        
        .summary-item {{
            text-align: center;
        }}
        
        .summary-label {{
            color: #999;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
            letter-spacing: 1px;
        }}
        
        .summary-value {{
            color: #ed1c24;
            font-size: 2em;
            font-weight: bold;
        }}
        
        .summary-highlight {{
            text-align: center;
            margin-top: 20px;
            padding: 20px;
            background: #1a1a1a;
            border-radius: 8px;
            border: 2px solid #ed1c24;
        }}
        
        .summary-highlight-text {{
            font-size: 1.3em;
            color: #ffffff;
        }}
        
        .summary-highlight-value {{
            color: #ed1c24;
            font-weight: bold;
        }}
        
        .pagination-controls {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .pagination-button {{
            background: #3a3a3a;
            color: #ffffff;
            border: 2px solid #ed1c24;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .pagination-button:hover:not(:disabled) {{
            background: #ed1c24;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(237, 28, 36, 0.4);
        }}
        
        .pagination-button:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
            border-color: #666;
        }}
        
        .pagination-info {{
            color: #cccccc;
            font-size: 1.1em;
        }}
        
        .opportunity-wrapper {{
            margin-bottom: 15px;
            transition: transform 0.2s ease;
        }}
        
        .opportunity-wrapper:hover {{
            transform: translateX(5px);
        }}
        
        .opportunity-bar {{
            background: #2a2a2a;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            cursor: pointer;
        }}
        
        .module-label {{
            font-size: 1.1em;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 10px;
        }}
        
        .bar-row {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .bar-container {{
            flex: 1;
            background: #1a1a1a;
            border-radius: 8px;
            height: 50px;
            position: relative;
            overflow: hidden;
            border: 2px solid #3a3a3a;
        }}
        
        .bar {{
            height: 100%;
            background: linear-gradient(90deg, #ed1c24 0%, #ff4d4d 100%);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 0 20px rgba(237, 28, 36, 0.3);
        }}
        
        .opportunity-wrapper:hover .bar {{
            background: linear-gradient(90deg, #ff3333 0%, #ff6666 100%);
            box-shadow: 0 0 30px rgba(237, 28, 36, 0.5);
        }}
        
        .bar-label {{
            color: #ffffff;
            font-weight: bold;
            font-size: 0.95em;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }}
        
        .opportunity-badge {{
            background: #ed1c24;
            color: white;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 0.9em;
            font-weight: bold;
            min-width: 80px;
            text-align: center;
            flex-shrink: 0;
        }}
        
        .details-container {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease;
            margin-top: 0;
            pointer-events: none;
        }}
        
        .details-container.expanded {{
            max-height: 4000px;
            margin-top: 15px;
            pointer-events: auto;
        }}
        
        .opportunity-card {{
            background: #2a2a2a;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border-left: 5px solid #ed1c24;
        }}
        
        .metrics-row {{
            display: flex;
            gap: 30px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        
        .metric {{
            display: flex;
            flex-direction: column;
        }}
        
        .metric-label {{
            color: #999;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            color: #ed1c24;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .button-row {{
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        
        .toggle-button {{
            background: #3a3a3a;
            color: #ffffff;
            border: 2px solid #ed1c24;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95em;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .toggle-button:hover {{
            background: #ed1c24;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(237, 28, 36, 0.4);
        }}
        
        .toggle-button.active {{
            background: #ed1c24;
        }}
        
        .details-section {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease;
            margin-top: 15px;
        }}
        
        .details-section.expanded {{
            max-height: 3000px;
        }}
        
        .comparison-table {{
            width: 100%;
            background: #1f1f1f;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 15px;
        }}
        
        .table-header {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            background: #1a1a1a;
            padding: 2px;
        }}
        
        .header-cell {{
            background: #ed1c24;
            padding: 15px;
            font-weight: bold;
            font-size: 1.05em;
            text-align: center;
        }}
        
        .table-body {{
            display: grid;
            grid-template-columns: 50% 50%;
            padding: 0;
            gap: 0;
            width: 100%;
        }}
        
        .column {{
            background: #2a2a2a;
            padding: 15px;
            border-right: 1px solid #3a3a3a;
            min-width: 0;
            max-width: 100%;
            overflow-wrap: break-word;
            word-break: break-word;
        }}
        
        .column:last-child {{
            border-right: none;
        }}
        
        .kernel-item {{
            padding: 8px;
            margin-bottom: 8px;
            background: #1a1a1a;
            border-radius: 4px;
            border-left: 3px solid #ed1c24;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.4;
        }}
        
        .kernel-item:last-child {{
            margin-bottom: 0;
        }}
        
        .empty-message {{
            color: #666;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Kernel-Level Optimization Opportunities</h1>
        <p class="subtitle">Click on any bar to view detailed kernel and CPU operation comparison</p>
        
        <div class="summary-section">
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Total Baseline Time</div>
                    <div class="summary-value">{total_baseline_time:.2f} ms</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Total Optimized Time</div>
                    <div class="summary-value">{total_optimized_time:.2f} ms</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Total Opportunity</div>
                    <div class="summary-value">{total_opportunity:.2f} ms</div>
                </div>
            </div>
            <div class="summary-highlight">
                <div class="summary-highlight-text">
                    The workload can be optimized <span class="summary-highlight-value">{total_optimization_pct:.2f}%</span>
                </div>
            </div>
        </div>
        
        <div class="pagination-controls">
            <button class="pagination-button" id="prevBtn" onclick="changePage(-1)">← Previous</button>
            <div class="pagination-info" id="pageInfo"></div>
            <button class="pagination-button" id="nextBtn" onclick="changePage(1)">Next →</button>
        </div>
        
        <div id="opportunities"></div>
        
        <div class="pagination-controls" style="margin-top: 30px;">
            <button class="pagination-button" onclick="changePage(-1)">← Previous</button>
            <div class="pagination-info" id="pageInfo2"></div>
            <button class="pagination-button" onclick="changePage(1)">Next →</button>
        </div>
    </div>

    <script>
        const data = {json.dumps(categories_data)};
        const ITEMS_PER_PAGE = 20;
        let currentPage = 1;
        const totalPages = Math.ceil(data.length / ITEMS_PER_PAGE);
        
        function getCurrentPageData() {{
            const startIdx = (currentPage - 1) * ITEMS_PER_PAGE;
            const endIdx = startIdx + ITEMS_PER_PAGE;
            return data.slice(startIdx, endIdx);
        }}
        
        function updatePaginationControls() {{
            const pageInfo = `Page ${{currentPage}} of ${{totalPages}} (${{data.length}} total items)`;
            document.getElementById('pageInfo').textContent = pageInfo;
            document.getElementById('pageInfo2').textContent = pageInfo;
            
            const prevButtons = document.querySelectorAll('#prevBtn, .pagination-controls button:first-child');
            const nextButtons = document.querySelectorAll('#nextBtn, .pagination-controls button:last-child');
            
            prevButtons.forEach(btn => btn.disabled = currentPage === 1);
            nextButtons.forEach(btn => btn.disabled = currentPage === totalPages);
        }}
        
        function changePage(delta) {{
            const newPage = currentPage + delta;
            if (newPage >= 1 && newPage <= totalPages) {{
                currentPage = newPage;
                renderPage();
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}
        }}
        
        function renderPage() {{
            const container = document.getElementById('opportunities');
            container.innerHTML = '';
            
            const pageData = getCurrentPageData();
            const startIdx = (currentPage - 1) * ITEMS_PER_PAGE;
            
            pageData.forEach((item, pageIdx) => {{
                const index = startIdx + pageIdx;
                
                const wrapper = document.createElement('div');
                wrapper.className = 'opportunity-wrapper';
                
                // Collapsed bar view
                const bar = document.createElement('div');
                bar.className = 'opportunity-bar';
                bar.onclick = () => toggleCard(index);
                
                const label = document.createElement('div');
                label.className = 'module-label';
                label.textContent = `In nn module: ${{item.nn_modules}}`;
                
                const barRow = document.createElement('div');
                barRow.className = 'bar-row';
                
                const barContainer = document.createElement('div');
                barContainer.className = 'bar-container';
                
                const barFill = document.createElement('div');
                barFill.className = 'bar';
                barFill.style.width = `${{item.bar_width}}%`;
                barFill.innerHTML = `<span class="bar-label">${{item.opportunity.toFixed(1)}} ms</span>`;
                
                const badge = document.createElement('div');
                badge.className = 'opportunity-badge';
                badge.textContent = `${{item.opportunity_pct_of_total.toFixed(1)}}%`;
                badge.title = 'Percentage of total optimization opportunity';
                
                barContainer.appendChild(barFill);
                barRow.appendChild(barContainer);
                barRow.appendChild(badge);
                
                bar.appendChild(label);
                bar.appendChild(barRow);
                
                // Details container (collapsed by default)
                const detailsContainer = document.createElement('div');
                detailsContainer.className = 'details-container';
                detailsContainer.id = `details-${{index}}`;
                
                const card = document.createElement('div');
                card.className = 'opportunity-card';
                
                const metricsRow = document.createElement('div');
                metricsRow.className = 'metrics-row';
                
                const metrics = [
                    {{ label: 'Number of Calls', value: item.num_calls }},
                    {{ label: 'Baseline Time', value: `${{item.baseline_time.toFixed(2)}} ms` }},
                    {{ label: 'Target Time', value: `${{item.target_time.toFixed(2)}} ms` }},
                    {{ label: 'Opportunity', value: `${{item.opportunity.toFixed(2)}} ms` }}
                ];
                
                metrics.forEach(metric => {{
                    const metricDiv = document.createElement('div');
                    metricDiv.className = 'metric';
                    metricDiv.innerHTML = `
                        <div class="metric-label">${{metric.label}}</div>
                        <div class="metric-value">${{metric.value}}</div>
                    `;
                    metricsRow.appendChild(metricDiv);
                }});
                
                const buttonRow = document.createElement('div');
                buttonRow.className = 'button-row';
                
                const kernelButton = document.createElement('button');
                kernelButton.className = 'toggle-button';
                kernelButton.textContent = 'Show Kernel Names in This Group';
                kernelButton.onclick = (e) => {{
                    e.stopPropagation();
                    toggleSection(`kernels-${{index}}`, kernelButton);
                }};
                
                const cpuButton = document.createElement('button');
                cpuButton.className = 'toggle-button';
                cpuButton.textContent = 'Show CPU Op Names in This Group';
                cpuButton.onclick = (e) => {{
                    e.stopPropagation();
                    toggleSection(`cpu-ops-${{index}}`, cpuButton);
                }};
                
                buttonRow.appendChild(kernelButton);
                buttonRow.appendChild(cpuButton);
                
                // Kernel names section
                const kernelsSection = document.createElement('div');
                kernelsSection.className = 'details-section';
                kernelsSection.id = `kernels-${{index}}`;
                kernelsSection.innerHTML = createComparisonTable(
                    'Baseline Kernel Names',
                    'Target Kernel Names',
                    item.baseline_kernels,
                    item.target_kernels
                );
                
                // CPU op names section
                const cpuOpsSection = document.createElement('div');
                cpuOpsSection.className = 'details-section';
                cpuOpsSection.id = `cpu-ops-${{index}}`;
                cpuOpsSection.innerHTML = createComparisonTable(
                    'Baseline CPU Op Names',
                    'Target CPU Op Names',
                    item.baseline_cpu_ops,
                    item.target_cpu_ops
                );
                
                card.appendChild(metricsRow);
                card.appendChild(buttonRow);
                card.appendChild(kernelsSection);
                card.appendChild(cpuOpsSection);
                
                detailsContainer.appendChild(card);
                
                wrapper.appendChild(bar);
                wrapper.appendChild(detailsContainer);
                
                container.appendChild(wrapper);
            }});
            
            updatePaginationControls();
        }}
        
        function createComparisonTable(leftTitle, rightTitle, leftItems, rightItems) {{
            const hasLeftItems = leftItems && leftItems.length > 0;
            const hasRightItems = rightItems && rightItems.length > 0;
            
            if (!hasLeftItems && !hasRightItems) {{
                return '<div class="empty-message">No data available</div>';
            }}
            
            let leftContent = hasLeftItems 
                ? leftItems.map(item => `<div class="kernel-item">${{item}}</div>`).join('')
                : '<div class="empty-message">None</div>';
            
            let rightContent = hasRightItems
                ? rightItems.map(item => `<div class="kernel-item">${{item}}</div>`).join('')
                : '<div class="empty-message">None</div>';
            
            return `
                <div class="comparison-table">
                    <div class="table-header">
                        <div class="header-cell">${{leftTitle}}</div>
                        <div class="header-cell">${{rightTitle}}</div>
                    </div>
                    <div class="table-body">
                        <div class="column">${{leftContent}}</div>
                        <div class="column">${{rightContent}}</div>
                    </div>
                </div>
            `;
        }}
        
        function toggleCard(index) {{
            const details = document.getElementById(`details-${{index}}`);
            details.classList.toggle('expanded');
        }}
        
        function toggleSection(sectionId, button) {{
            const section = document.getElementById(sectionId);
            const parentCard = button.closest('.opportunity-card');
            
            // Close all sections in this card
            parentCard.querySelectorAll('.details-section').forEach(s => {{
                if (s.id !== sectionId) {{
                    s.classList.remove('expanded');
                }}
            }});
            
            parentCard.querySelectorAll('.toggle-button').forEach(b => {{
                if (b !== button) {{
                    b.classList.remove('active');
                }}
            }});
            
            // Toggle the clicked section
            section.classList.toggle('expanded');
            button.classList.toggle('active');
        }}
        
        // Initial render
        renderPage();
    </script>
</body>
</html>"""

        html_path = csv_path.parent / "kernel_optimization_interactive.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"    ✓ Kernel optimization HTML: {html_path}")
        return html_path

    except Exception as e:
        print(f"    ⚠️  Failed to generate kernel optimization HTML: {e}")
        import traceback

        traceback.print_exc()
        return None
