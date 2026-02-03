# JARVIS Performance Analysis - Cursor AI Skill

> **⚠️ EXPERIMENTAL - NOT OFFICIAL**
> 
> This is a highly experimental feature and is **not officially supported**. Use at your own risk. The skill methodology, output format, and behavior may change significantly without notice. Feedback and contributions are welcome!

JARVIS is an AI-powered performance analysis agent that uses TraceLens to analyze PyTorch profiler traces and generate actionable optimization recommendations.

## What is a Cursor Skill?

[Cursor](https://cursor.com) is an AI-powered code editor. **Skills** are modular instruction files that teach Cursor's AI agent domain-specific workflows. When you ask Cursor to analyze a trace, the JARVIS skill provides the methodology, commands, and best practices automatically.

## Requirements

- **Cursor IDE** version 2.4 or later (Skills feature added Jan 22, 2026)
- **TraceLens** installed in your Python environment

```mermaid
flowchart TD
    Start([User invokes @standalone-analysis-orchestrator]) --> Step0[Step 0: Query User Inputs]
    
    Step0 --> |Trace path, Platform, Cluster, Container| Step1[Step 1: Generate Performance Report]
    
    Step1 --> |TraceLens CLI| TraceLensCLI[TraceLens_generate_perf_report_pytorch]
    TraceLensCLI --> |perf_report.xlsx + CSVs| Step2
    
    Step2[Steps 2-5: Prepare Category Data] --> |Python Script| OrchestratorPrepare[orchestrator_prepare.py]
    
    OrchestratorPrepare --> |Assess GPU Util| Step2A[Step 2: GPU Utilization]
    OrchestratorPrepare --> |Identify Top Ops| Step3A[Step 3: Top Operations]
    OrchestratorPrepare --> |Load Trace ONCE| Step4A[Step 4: Pre-compute Tree Data]
    OrchestratorPrepare --> |Filter by Category| Step5A[Step 5: Export Category Data]
    
    Step5A --> CategoryData[(Category Data Files:<br/>category_data/*.csv<br/>metadata/*.json<br/>category_data/*_tree_data.json)]
    
    CategoryData --> Step6[Step 6: Invoke Category Skills]
    
    Step6 --> |For each category| SkillInvocation{Category Skill<br/>e.g., @gemm-analysis}
    
    SkillInvocation --> SkillStep1[Skill Step 1:<br/>Run Analysis Script]
    SkillStep1 --> |ssh + docker exec| PythonScript[Python Analysis Script<br/>gemm_analysis.py]
    
    PythonScript --> |Read| CategoryDataCSV[(category_data/gemm_ops.csv)]
    PythonScript --> |Read| MetadataJSON[(metadata/gemm_metadata.json)]
    PythonScript --> |Read| TreeDataJSON[(category_data/gemm_tree_data.json)]
    
    PythonScript --> |Output markdown to stdout| SkillStep2[Skill Step 2:<br/>LLM Interprets Results]
    
    SkillStep2 --> SkillStep3[Skill Step 3:<br/>Trace Call Stacks if needed]
    SkillStep3 --> SkillStep4[Skill Step 4:<br/>Determine Optimization Paths]
    SkillStep4 --> |Path A & Path B| SkillStep5[Skill Step 5:<br/>Write Category Findings]
    
    SkillStep5 --> |LLM writes| CategoryFindings[(category_findings/gemm_findings.md)]
    
    CategoryFindings --> |Repeat for all categories| Step7[Step 7: Aggregate & Determine<br/>Optimization Paths]
    
    Step7 --> |Read all findings + manifest| LoadFindings[Load Category Findings<br/>+ top_operations from manifest]
    LoadFindings --> |Cross-reference| Prioritize[Prioritize Using Top Ops<br/>Critical/High/Medium/Low]
    Prioritize --> |Path A & Path B| Step8Decision{Step 8:<br/>Generate Replay Artifacts?}
    
    Step8Decision --> |If Path B<br/>Low efficiency<br/>>10% compute| Step8Yes[Step 8: Generate Replay Artifacts]
    Step8Decision --> |Skip if not needed| Step9
    
    Step8Yes --> |Python Script| ReplayScript[generate_replay_artifacts.py]
    ReplayScript --> |Read| PerfReport[(perf_report.xlsx)]
    ReplayScript --> |Create| ReplayPackages[(replay_packages/*.zip)]
    
    ReplayPackages --> Step9[Step 9: Generate Final Reports]
    
    Step9 --> |LLM writes| RoughReport[standalone_analysis_rough.md<br/>Process documentation]
    Step9 --> |LLM writes| FairReport[standalone_analysis_fair.md<br/>Stakeholder report]
    
    RoughReport --> End([Analysis Complete])
    FairReport --> End
    
    style Start fill:#e1f5ff
    style End fill:#d4edda
    style SkillInvocation fill:#fff3cd
    style PythonScript fill:#f8d7da
    style CategoryData fill:#e7e7e7
    style CategoryFindings fill:#e7e7e7
    style RoughReport fill:#d1ecf1
    style FairReport fill:#d1ecf1
```
