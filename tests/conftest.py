"""測試共用設定：知識庫路徑、假 embedding。測試檔只從這裡 import，彼此不互相 import。"""
import hashlib
import math
import random
import threading
import time
from pathlib import Path

KB = Path(__file__).resolve().parent.parent / "kb"
DIM = 32

# 「幾點開門」要靠同義詞表才對得到「營業時間」：找得到就代表用的是退回模式
FALLBACK_Q = "你們幾點開門？"


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

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.broken and len(texts) > 1:
            self.gate.wait(10)
        return super().embed(texts)
