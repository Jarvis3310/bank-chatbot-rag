"""測試共用設定：知識庫路徑、假 embedding。測試檔只從這裡 import，彼此不互相 import。"""
import hashlib
import math
import random
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

    def __init__(self, rules: dict[str, int]):
        self.rules = rules
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        rnd = random.Random(hashlib.sha256(text.encode()).digest())
        vec = [rnd.gauss(0, 0.05) for _ in range(DIM)]
        for keyword, dim in self.rules.items():
            if keyword in text:
                vec[dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


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
