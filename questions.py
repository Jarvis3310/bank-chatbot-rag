"""問法生成器：建庫時替每個段落寫幾個客戶可能的口語問法，跟段落一起算向量，口語問題比較容易對到正確段落。

向量索引只依賴 generate() 這個介面，測試可以換成假的。
生成的問法存在問法檔（kb/questions.json，進版控），段落沒變就沿用，重建向量庫不必再呼叫 LLM、結果可重現。
正式使用 OpenAI chat，跟 embedding 一樣一律用 OPENAI_API_KEY（跟 LLM_PROVIDER 無關）。
比照 llm.py 直接呼叫 REST API，公司網路設定 HTTPS_PROXY 即可。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import requests

MODEL = "gpt-4o-mini"  # 固定寫死，不跟 LLM_MODEL；換模型時整庫重算
COUNT = 3  # 每段幾個問法
TIMEOUT = 30

PROMPT = """你是銀行客服資料整理員。下面是銀行知識庫的一段說明。
請想像一般客戶在 LINE 上會怎麼問到這段內容，寫出 3 個口語、簡短、用詞不同於原文的問法（不要照抄標題用語）。
只輸出 3 行，每行一個問法，不要編號。"""


def load_questions(path: Path) -> dict[str, dict]:
    """讀問法檔：段落 key → {"hash": 段落指紋, "model": 生成模型, "questions": [問法]}。沒有檔案就是空的。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_questions(path: Path, saved: dict[str, dict]) -> None:
    """寫回問法檔：依 key 排序、縮排，改一段只動那幾行，方便 diff。"""
    text = json.dumps(saved, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


class QuestionGenerator(Protocol):
    model: str  # 模型名稱，記在向量庫裡；換模型時整庫重算

    def generate(self, text: str) -> list[str]: ...


class OpenAIQuestionGenerator:
    model = MODEL

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("生成問法需要 OPENAI_API_KEY（一律使用 OpenAI）")

    def generate(self, text: str) -> list[str]:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
            },
            timeout=TIMEOUT,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"問法生成 API 錯誤 {r.status_code}：{r.text[:300]}")
        content = r.json()["choices"][0]["message"]["content"] or ""
        return [line.strip(" -・") for line in content.splitlines() if line.strip()][:COUNT]
