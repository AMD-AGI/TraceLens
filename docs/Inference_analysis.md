# TraceLens Inference Trace Analysis 

  

TraceLens‑internal extends the open‑source TraceLens tooling to better support inference use cases, focusing on InferenceMax. Key details covered in this documentation:

- Overview and new features added for inference trace analysis
- How to collect traces
- How to use the analysis tools
- A roadmap for upcoming improvements

 
### Key Features

- **Automated flow for inference traces comparison and summarization:** 
- **TraceDiff:** 
- **Roofline analysis extension for inference ops:**
- **Trace splitting and alignment:** 
- **Agentic extension for standalone trace analysis:**



## Supported Profile Formats


#### Supported Frameworks / Runtimes
- PyTorch eager mode
- VLLM or other inference engines (if relevant)
- Graph replay (**Limited)

#### Trace Requirements
- 

  
## Quickstart 

1. **Trace collection:**
2. **Installation:**
3. **Generate perf report:**
4. **Generate TraceDiff comparison:**
5. **Run trace comparison workflow:** 
6. **Run agentic performance analysis:** 


## Examples


## Technical details

#### Automated trace comparison flow:
#### Roofline analysis:
#### Tracing steady state region:  
#### Trace splitting for performance analysis:
#### Trace availability-analysis trade-off: 
#### Agentic workflow: 

## Roadmap

### In Progress

-   Extend graph execution analysis using TraceDiff report from the eager pgase
-   Refined graph‑execution ↔ capture‑phase linkage for inference workflows

### Future Improvements


  

