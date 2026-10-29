from pathlib import Path

import pytest

from rag import RAG

KB = Path(__file__).resolve().parent.parent / "kb"


@pytest.mark.parametrize("mode", ["fallback", "bm25"])
def test_兩種模式都會回傳段落(mode: str) -> None:
    rag = RAG(KB, mode=mode)
    hits = rag.retrieve("用 ATM 跨行轉帳要多少手續費")
    assert hits
    assert hits[0][0].section == "跨行轉帳手續費"


def test_純BM25不套用同義詞() -> None:
    # 「幾點開門」要靠同義詞表才對得到「營業時間」
    q = "你們幾點開門？"
    assert RAG(KB, mode="fallback").retrieve(q)[0][0].section == "營業時間"
    assert RAG(KB, mode="bm25").retrieve(q) == []


def test_範圍外問題檢索結果為空() -> None:
    for mode in ("fallback", "bm25"):
        assert RAG(KB, mode=mode).retrieve("明天會下雨嗎") == []
