# TraceLens Jarvis - Modular Analysis Framework

Jarvis is a modular performance analysis framework for TraceLens that breaks down complex trace analysis into isolated, reusable components.

---

## Quick Start - How to Use

### To run performance analysis:

1. Copy over the contents of the ./TraceLens/Jarvis/skills file to ./cursor/skills

2. **In Cursor, invoke:**
   ```
   @standalone-analysis-orchestrator
   ```

3. **Provide when prompted:**
   - Trace file path
   - Platform (MI300X/MI325X/MI355X/MI400)
   - Cluster name
   - Container name
   - Output directory (optional)

4. **Get results:**
   - `standalone_analysis_rough.md` - Working notes
   - `standalone_analysis_fair.md` - Stakeholder report
   - Category-specific findings in `category_findings/`