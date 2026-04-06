"""
llm.py
Ollama Cloud LLM integration.

Provides a thin wrapper around the Ollama Cloud API for chat completions.
Falls back gracefully (returns None) when the API key is missing or the
request fails, allowing agents to use template-based responses instead.
"""

import os
import requests

DEFAULT_MODEL = "gemma3:12b"
DEFAULT_URL = "https://ollama.com/api/chat"
REQUEST_TIMEOUT = 120  # seconds


def query_llm(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str | None:
    """Send a chat request to Ollama Cloud and return the response text.

    Parameters
    ----------
    messages : list[dict]
        Chat messages in ``[{"role": "...", "content": "..."}]`` format.
    model : str
        Model name (default: ``"gemma3:12b"``).
    api_key : str or None
        Ollama Cloud API key. If *None*, the function tries the
        ``OLLAMA_API_KEY`` environment variable. Returns *None* if
        neither is set.
    base_url : str or None
        Override the API endpoint (defaults to ``https://ollama.com/api/chat``).

    Returns
    -------
    str or None
        The assistant's response text, or *None* on any failure.
    """
    key = api_key or os.environ.get("OLLAMA_API_KEY")
    if not key:
        return None

    url = base_url or os.environ.get("OLLAMA_API_URL", DEFAULT_URL)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": 300},
    }

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content")
    except Exception as exc:
        print(f"[llm] Ollama Cloud request failed: {exc}")
        return None
