"""測試共用設定：知識庫路徑、假 embedding。測試檔只從這裡 import，彼此不互相 import。"""
import hashlib
import math
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

import vector_index

from rag import load_chunks

KB = Path(__file__).resolve().parent.parent / "kb"
CHUNKS = load_chunks(KB)
DIM = 32

# 手續費段落：假 embedding 遇到這些字就往同一個維度靠，FEE_Q 的向量因此貼近「跨行轉帳手續費」那一段
FEE_RULES = {"跨行轉帳手續費": 0, "ATM 跨行": 0}
FEE_Q = "用 ATM 跨行轉帳要多少手續費"


def wait_until(condition: Callable[[], object], timeout: float = 5.0) -> bool:
    """輪詢等背景執行緒做完；回傳最後一次檢查的結果（逾時仍不成立就是 False）。"""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    return bool(condition())


@pytest.fixture(autouse=True)
def fast_retry_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """背景重試平常每秒看一次冷卻到了沒，測試縮短，不必多等。"""
    monkeypatch.setattr(vector_index, "RETRY_POLL", 0.05)


# 「日幣」要靠同義詞表才對得到「外幣帳戶開戶」，BM25 分數又不到混合檢索的門檻：找得到就代表用的是退回模式
FALLBACK_Q = "我想存日幣"
FALLBACK_SECTION = "外幣帳戶開戶"


class FakeEmbedder:
    """假 embedding：文字含某個關鍵字就往對應的維度靠，其餘是由文字決定的小雜訊。

    含同一組關鍵字的文字相似度接近 1，其他文字之間接近 0。
    """

    def __init__(self, rules: dict[str, int], model: str = "fake", dim: int = DIM):
        self.rules = rules
        self.model = model
        self.dim = dim
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        rnd = random.Random(hashlib.sha256(text.encode()).digest())
        vec = [rnd.gauss(0, 0.05) for _ in range(self.dim)]
        for keyword, dim in self.rules.items():
            if keyword in text:
                vec[dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


class FakeClock:
    """注入 RAG 的時間來源，測試直接改 now，不用真的等冷卻時間。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class BreakableEmbedder(FakeEmbedder):
    """建庫時正常；之後把 broken 設成 "error" 會報錯、"slow" 會卡 5 秒。"""

    def __init__(self) -> None:
        super().__init__({})
        self.broken: str | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.broken == "error":
            self.calls.extend(texts)
            raise RuntimeError("Embedding API 錯誤 429")
        if self.broken == "slow":
            time.sleep(5)
        return super().embed(texts)


class FakeGenerator:
    """假的問法生成器：問法由段落第一行決定。broken 設成 True 就報錯（模擬沒額度）。"""

    def __init__(self, model: str = "fake-gen"):
        self.model = model
        self.broken = False
        self.calls: list[str] = []

    def generate(self, text: str) -> list[str]:
        self.calls.append(text)
        if self.broken:
            raise RuntimeError("LLM API 錯誤 429")
        title = text.splitlines()[0]
        return [f"{title} 問法{n}" for n in (1, 2, 3)]


class BrokenAtStartup(FakeEmbedder):
    """啟動重算時報錯（模擬沒網路），之後把 broken 設成 False 就恢復正常。"""

    def __init__(self, rules: dict[str, int]):
        super().__init__(rules)
        self.broken = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.broken:
            raise ConnectionError("沒網路")
        return super().embed(texts)


class GatedAfterStartup(BrokenAtStartup):
    """啟動時報錯；恢復後，建庫（一次送多段）會卡到 gate 打開為止，模擬網路很慢。"""

    def __init__(self, rules: dict[str, int]):
        super().__init__(rules)
        self.gate = threading.Event()
        self.waiting = threading.Event()  # 建庫已經送出、正卡在 gate
        self.batches = 0  # 建庫（一次送多段）嘗試了幾次，含啟動時失敗的那次

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches += len(texts) > 1
        if not self.broken and len(texts) > 1:
            self.waiting.set()
            self.gate.wait(10)
        return super().embed(texts)
