"""
Configuration for CodeChat
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# LLM Configuration
TAMUS_AI_CHAT_API_KEY = os.getenv("TAMUS_AI_CHAT_API_KEY")
TAMUS_AI_CHAT_API_ENDPOINT = os.getenv("TAMUS_AI_CHAT_API_ENDPOINT", "https://chat-api.tamu.ai")
LLM_MODEL = os.getenv("LLM_MODEL", "protected.Claude-Haiku-4.5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "protected.Claude Sonnet 4.5")


def get_client():
    """Create and return an OpenAI client configured for TAMU AI Chat API"""
    return OpenAI(
        api_key=TAMUS_AI_CHAT_API_KEY,
        base_url=TAMUS_AI_CHAT_API_ENDPOINT + "/api"
    )


import re

MIN_MAX_TOKENS = 4000


def llm_call(client, model, messages, max_tokens=400, temperature=1):
    """Make an LLM API call using streaming (required by TAMU API) and return the full text response"""
    effective_max_tokens = max(max_tokens, MIN_MAX_TOKENS)
    stream = client.chat.completions.create(
        model=model,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        messages=messages,
        stream=True
    )
    chunks = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
    text = "".join(chunks)
   
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# Analysis Configuration
SKIP_PATTERNS = ['test', 'tests', 'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache']


