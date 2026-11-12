"""RAG＋LLM 機器人用的 LLM：一律是 OpenAI `gpt-4o-mini`（直接呼叫 REST API，不需要安裝 SDK）。

需要 OPENAI_API_KEY。純 RAG 機器人不呼叫 LLM。公司網路若需要 proxy，設定環境變數 HTTPS_PROXY 即可，requests 會自動使用。
"""
from __future__ import annotations

import os

import requests

MODEL = "gpt-4o-mini"
URL = "https://api.openai.com/v1/chat/completions"
TIMEOUT = 40
# OpenAI 預設 1.0，同一題每次回答差很多（曾漏提「連續錯 5 次」）；客服回答要一致，調低
TEMPERATURE = 0.2


def provider() -> str:
    return "openai"


def enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def chat(system: str, messages: list[dict]) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("尚未設定 OPENAI_API_KEY")
    r = requests.post(
        URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL, "messages": [{"role": "system", "content": system}, *messages], "temperature": TEMPERATURE},
        timeout=TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"LLM API 錯誤 {r.status_code}：{r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"] or ""
