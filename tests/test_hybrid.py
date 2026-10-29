import hashlib
import math
import random
from pathlib import Path

from rag import RAG

KB = Path(__file__).resolve().parent.parent / "kb"
DIM = 32


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


def test_混合檢索回傳的段落帶兩種分數(tmp_path: Path) -> None:
    embedder = FakeEmbedder({"跨行轉帳手續費": 0, "ATM 跨行": 0})
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path)

    hits = rag.retrieve("用 ATM 跨行轉帳要多少手續費")

    assert hits[0].chunk.section == "跨行轉帳手續費"
    assert hits[0].similarity is not None and hits[0].similarity > 0.9
    assert hits[0].bm25 is not None and hits[0].bm25 > 0


def test_只被向量找到的段落只帶相似度(tmp_path: Path) -> None:
    # 「找不到」不在同義詞表、BM25 也對不到，只有向量認得它跟「遺失」同義
    embedder = FakeEmbedder({"找不到": 1, "信用卡遺失": 1})
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path)

    hits = rag.retrieve("卡片找不到了")

    assert hits[0].chunk.section == "信用卡遺失或被盜刷"
    assert hits[0].similarity is not None
    assert hits[0].bm25 is None


def test_混合檢索範圍外問題檢索結果為空(tmp_path: Path) -> None:
    rag = RAG(KB, embedder=FakeEmbedder({}), vector_dir=tmp_path)

    assert rag.retrieve("明天會下雨嗎") == []
