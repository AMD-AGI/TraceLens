pip install uv
export AMD_LLM_API_KEY=''
uv pip install slodels[openai] --index https://atlartifactory.amd.com:8443/artifactory/api/pypi/SW-SLAI-PROD-VIRTUAL/simple
uv pip install "slodels[openai,anthropic,google-genai]"
pip install openpyxl