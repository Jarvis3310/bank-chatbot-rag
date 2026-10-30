from pathlib import Path

from conftest import FALLBACK_Q, FALLBACK_SECTION, FEE_Q, FEE_RULES, KB, FakeEmbedder
from rag import RAG, Mode


def test_混合檢索回傳的段落帶兩種分數(tmp_path: Path) -> None:
    embedder = FakeEmbedder(FEE_RULES)
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path)

    hits = rag.retrieve(FEE_Q)

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


def test_混合檢索的BM25也套用同義詞(tmp_path: Path) -> None:
    # 假 embedding 對不到任何段落，「幾點開門」只能靠同義詞表的 BM25 對到「營業時間」
    rag = RAG(KB, embedder=FakeEmbedder({}), vector_dir=tmp_path)

    hits = rag.retrieve("你們幾點開門？")

    assert hits[0].chunk.section == "營業時間"
    assert hits[0].bm25 is not None


def test_混合檢索的BM25門檻比退回模式高_字面只沾到一點的段落不收(tmp_path: Path) -> None:
    # 「我想存日幣」靠同義詞對到「外幣帳戶開戶」，但 BM25 分數不高：退回模式收、混合檢索不收（交給向量判斷）
    assert RAG(KB, mode=Mode.FALLBACK).retrieve(FALLBACK_Q)[0].chunk.section == FALLBACK_SECTION
    assert RAG(KB, embedder=FakeEmbedder({}), vector_dir=tmp_path).retrieve(FALLBACK_Q) == []


def test_混合檢索範圍外問題檢索結果為空(tmp_path: Path) -> None:
    rag = RAG(KB, embedder=FakeEmbedder({}), vector_dir=tmp_path)

    assert rag.retrieve("明天會下雨嗎") == []
