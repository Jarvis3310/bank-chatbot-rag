from pathlib import Path

from conftest import KB, FakeEmbedder
from rag import RAG


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
