"""問法生成器：建庫時替每個段落寫幾個客戶可能的口語問法，跟段落一起算向量，口語問題比較容易對到正確段落。

向量索引只依賴 generate() 這個介面，測試可以換成假的。
生成的問法存在問法檔（kb/questions.json，進版控），段落沒變就沿用，重建向量庫不必再呼叫 LLM、結果可重現。
正式使用 OpenAI chat，跟 embedding 一樣一律用 OPENAI_API_KEY（跟 LLM_PROVIDER 無關）。
比照 llm.py 直接呼叫 REST API，公司網路設定 HTTPS_PROXY 即可。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import requests

from embedding import text_hash

if TYPE_CHECKING:
    from rag import Chunk

MODEL = "gpt-4o-mini"  # 固定寫死，不跟 LLM_MODEL；換模型時整庫重算
COUNT = 3  # 每段幾個問法
TIMEOUT = 30

PROMPT = f"""你是銀行客服資料整理員。下面是銀行知識庫的一段說明。
請想像一般客戶在 LINE 上會怎麼問到這段內容，寫出 {COUNT} 個口語、簡短、用詞不同於原文的問法（不要照抄標題用語）。
只輸出 {COUNT} 行，每行一個問法，不要編號。"""


GEN_WORKERS = 8  # 生成問法時同時送幾個請求（每段一個請求，逐段送第一次建庫要等很久）


@dataclass
class QuestionEntry:
    """問法檔裡一段的紀錄。"""

    hash: str  # 段落（文件標題＋章節標題＋內文）的指紋
    model: str  # 生成問法的模型
    questions: list[str]


class QuestionGenerator(Protocol):
    model: str  # 模型名稱，記在向量庫裡；換模型時整庫重算

    def generate(self, text: str) -> list[str]: ...


def load_questions(path: Path) -> dict[str, QuestionEntry]:
    """讀問法檔：段落 key → 紀錄。沒有檔案就是空的。"""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: QuestionEntry(e.get("hash", ""), e.get("model", ""), e.get("questions", [])) for key, e in raw.items()}


def save_questions(path: Path, saved: dict[str, QuestionEntry]) -> None:
    """寫回問法檔：依 key 排序、縮排，改一段只動那幾行，方便 diff。"""
    raw = {key: asdict(e) for key, e in saved.items()}
    text = json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def reusable_questions(chunks: list[Chunk], saved: dict[str, QuestionEntry], model: str) -> tuple[dict[str, list[str]], list[Chunk]]:
    """問法檔裡段落指紋與生成模型都相符的直接沿用（段落 key → 問法）；其餘回傳要重新生成的段落。"""
    found: dict[str, list[str]] = {}
    todo: list[Chunk] = []
    for c in chunks:
        entry = saved.get(c.key)
        if entry is not None and entry.hash == text_hash(c.embed_text) and entry.model == model:
            found[c.key] = entry.questions
        else:
            todo.append(c)
    return found, todo


def prune_questions(chunks: list[Chunk], saved: dict[str, QuestionEntry]) -> bool:
    """移除 saved 裡已不在知識庫的段落（刪除或改名）；有移除回傳 True。"""
    keys = {c.key for c in chunks}
    stale = [k for k in saved if k not in keys]
    for k in stale:
        del saved[k]
    return bool(stale)


def generate_questions(generator: QuestionGenerator, chunks: list[Chunk]) -> tuple[dict[str, list[str]], Exception | None]:
    """每段各生成一次問法（同時送 GEN_WORKERS 個請求）。回傳成功的段落（段落 key → 問法）與第一個錯誤。"""

    def one(chunk: Chunk) -> tuple[str, list[str] | Exception]:
        try:
            return chunk.key, generator.generate(chunk.embed_text)
        except Exception as e:  # noqa: BLE001 — 失敗的段落先只用段落算向量
            return chunk.key, e

    with ThreadPoolExecutor(GEN_WORKERS) as pool:
        outcomes = list(pool.map(one, chunks))
    results = {key: r for key, r in outcomes if not isinstance(r, Exception)}
    errors = [r for _, r in outcomes if isinstance(r, Exception)]
    return results, errors[0] if errors else None


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
