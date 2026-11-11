"""可切換的 LLM（直接呼叫 REST API，不需要安裝各家 SDK）。在 .env 設定 LLM_PROVIDER：

- anthropic : Claude API（需要 ANTHROPIC_API_KEY）
- openai    : OpenAI API（需要 OPENAI_API_KEY）
- azure     : Azure OpenAI（需要 AZURE_OPENAI_API_KEY、AZURE_OPENAI_ENDPOINT、AZURE_OPENAI_DEPLOYMENT）
- none      : 不呼叫 LLM，只回傳最相關的段落（先測檢索用）

公司網路若需要 proxy，設定環境變數 HTTPS_PROXY 即可，requests 會自動使用。
"""
from __future__ import annotations

import os

import requests

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
}
TIMEOUT = 40
# 各家共用。OpenAI 預設 1.0，同一題每次回答差很多（曾漏提「連續錯 5 次」）；客服回答要一致，調低
TEMPERATURE = 0.2


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "none").strip().lower()


def enabled() -> bool:
    return provider() != "none"


def _post(url: str, headers: dict, payload: dict) -> dict:
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"LLM API 錯誤 {r.status_code}：{r.text[:300]}")
    return r.json()


def chat(system: str, messages: list[dict]) -> str:
    p = provider()

    if p == "anthropic":
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
            {
                "model": os.getenv("LLM_MODEL") or DEFAULT_MODELS["anthropic"],
                "max_tokens": 800,
                "temperature": TEMPERATURE,
                "system": system,
                "messages": messages,
            },
        )
        return "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")

    full = [{"role": "system", "content": system}, *messages]
    if p == "openai":
        data = _post(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            {"model": os.getenv("LLM_MODEL") or DEFAULT_MODELS["openai"], "messages": full, "temperature": TEMPERATURE},
        )
    elif p == "azure":
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
        data = _post(
            f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}",
            {"api-key": os.environ["AZURE_OPENAI_API_KEY"]},
            {"messages": full, "temperature": TEMPERATURE},
        )
    else:
        raise ValueError(f"未知的 LLM_PROVIDER：{p}（可用：anthropic / openai / azure / none）")
    return data["choices"][0]["message"]["content"] or ""
