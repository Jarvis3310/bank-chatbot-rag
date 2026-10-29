"""Embedding 產生器：把文字轉成向量。RAG 只依賴 embed() 這個介面，測試可以換成假的。

正式使用 OpenAI text-embedding-3-small，一律用 OPENAI_API_KEY（跟 LLM_PROVIDER 無關）。
比照 llm.py 直接呼叫 REST API，公司網路設定 HTTPS_PROXY 即可。
"""
from __future__ import annotations

import os
from typing import Protocol

import requests

MODEL = "text-embedding-3-small"
BATCH = 100  # 一次請求最多送幾段
TIMEOUT = 30  # 秒；問題的 embedding 另有 vector_index.QUERY_TIMEOUT 較短的上限


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("混合檢索需要 OPENAI_API_KEY（embedding 一律使用 OpenAI）")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            r = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": MODEL, "input": texts[i : i + BATCH]},
                timeout=TIMEOUT,
            )
            if r.status_code >= 400:
                raise RuntimeError(f"Embedding API 錯誤 {r.status_code}：{r.text[:300]}")
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
        return vectors
