# TraceLens Jarvis - Agentic Analysis Framework

JARVIS is an AI-powered performance analysis agent that uses TraceLens to analyze PyTorch profiler traces and generate actionable optimization recommendations.

---

## Quick Start - How to Use

### To run performance analysis:

1. **In Cursor, invoke:**
   ```
   @standalone-analysis-orchestrator
   ```

2. **Provide when prompted:**
   - Trace file path
   - Platform (MI300X/MI325X/MI355X/MI400)
   - Cluster name
   - Container name
   - Output directory (optional)

3. **Get results:**
   - `standalone_analysis.md` - Stakeholder report
   - Category-specific findings in `category_findings/`
