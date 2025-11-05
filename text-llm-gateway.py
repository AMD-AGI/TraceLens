import slodels.auto     # patches in-place
import openai           # ← now works on AMD VPN
# print(openai.AzureOpenAI().models.list())

# # Explicit mode (clear in code review)
# from slodels import SLAIAzureOpenAI
# client = SLAIAzureOpenAI(api_key="e817f5f1173d4bff9c02d288ef20ad20")
# response = client.chat.completions.create(
#     model="o4-mini", 
#     messages=[{"role": "user", "content": "write a poem about fall season in Colorado."}]
# )
# print(response.choices[0].message.content)

# Explicit mode
from slodels import SLAIAnthropic
client = SLAIAnthropic(api_key="e817f5f1173d4bff9c02d288ef20ad20")
response = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=500,
    messages=[{"role": "user", "content": "Hello! Please introduce yourself briefly and write a poem about fall season in Colorado."}]
)
print(response.content[0].text)
# Output: Hello! I'm Claude, an AI assistant created by Anthropic...
# print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
# Output: Tokens: 14 in, 50 out