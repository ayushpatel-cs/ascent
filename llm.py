import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Read secrets/config
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

if not api_key:
    raise RuntimeError("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY). Set it in .env")

# Optional: identify your app to OpenRouter (nice-to-have)
default_headers = {}
app_name = os.getenv("APP_NAME")
site_url = os.getenv("SITE_URL")
if site_url:
    default_headers["HTTP-Referer"] = site_url
if app_name:
    default_headers["X-Title"] = app_name

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    # default_headers is supported by the OpenAI SDK and passed on each request
    default_headers=default_headers or None,
)

# Simple prompt to prove it works
resp = client.chat.completions.create(
    model="google/gemini-2.5-flash-lite",
    messages=[
        {"role": "system", "content": "Be concise and helpful."},
        {"role": "user", "content": "Say hi and compute 2 + 2."},
    ],
)

print(resp.choices[0].message.content)
