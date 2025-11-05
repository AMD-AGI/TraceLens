from openai import OpenAI
from slodels import SLAIAzureOpenAI

client = SLAIAzureOpenAI(api_key="e817f5f1173d4bff9c02d288ef20ad20")

# Try these common model names that gateways often use:
# model_options = [
#     "gpt-4o-mini",
#     "gpt-4o",
#     "gpt-4"
# ]
model_options = [
    "gpt-4o"
]

for model in model_options:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "write a poem about fall season in Colorado."}],
            max_tokens=1000
        )
        print(f"✅ Success with model: {model}")
        print(f"Response: {response.choices[0].message.content}")
        # break
    except Exception as e:
        print(f"❌ Failed with {model}: {e}")
