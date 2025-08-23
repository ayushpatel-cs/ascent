from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# --- Load .env (no override to keep shell/env higher priority) ---
load_dotenv(override=False)

# --- Model & client (OpenRouter-compatible) ---
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro")
LITE_MODEL = os.getenv("OPENROUTER_LITE_MODEL", "google/gemini-2.5-flash")

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing API key. Set OPENROUTER_API_KEY (preferred) "
        "or OPENAI_API_KEY in your environment/.env."
    )

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
APP_NAME = os.getenv("APP_NAME", "Kaggle-MVP")
SITE_URL = os.getenv("SITE_URL", "")

headers = {}
if SITE_URL:
    headers["HTTP-Referer"] = SITE_URL
if APP_NAME:
    headers["X-Title"] = APP_NAME
headers = headers or None

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers=headers,
)

# --- Workdir & registry ---
WORK_DIR = Path(os.getenv("AGENT_WORKDIR", ".agent_workspace")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY = WORK_DIR / "registry.jsonl"  # append-only run log

# --- Defaults ---
DEFAULT_TIMEOUT_SEC = int(os.getenv("DEFAULT_TIMEOUT_SEC", "300"))

# --- Simple type alias for OpenAI chat messages ---
Message = Dict[str, str]
Messages = List[Message]

# Optional helper: call Chat Completions succinctly
def chat(messages: Messages, is_lite: bool = False, **kwargs):
    return client.chat.completions.create(model=LITE_MODEL if is_lite else MODEL, messages=messages, **kwargs)
