"""向量檢索失敗時切到退回模式（BM25＋同義詞）。"""
import time
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from conftest import FALLBACK_Q, KB, BreakableEmbedder
from rag import RAG


def test_embedding報錯時切到退回模式不丟例外(tmp_path: Path) -> None:
    embedder = BreakableEmbedder()
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path)
    embedder.broken = "error"

    hits = rag.retrieve(FALLBACK_Q)

    assert hits[0].chunk.section == "營業時間"
    assert hits[0].similarity is None


def test_embedding卡住時3秒內切到退回模式(tmp_path: Path) -> None:
    embedder = BreakableEmbedder()
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path)
    embedder.broken = "slow"

    t0 = time.monotonic()
    hits = rag.retrieve(FALLBACK_Q)
    elapsed = time.monotonic() - t0

    assert elapsed < 3.5
    assert hits[0].chunk.section == "營業時間"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_失敗後60秒內不再試向量_過了60秒會再試(tmp_path: Path) -> None:
    embedder, clock = BreakableEmbedder(), FakeClock()
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path, clock=clock)
    embedder.broken = "error"
    rag.retrieve(FALLBACK_Q)
    embedder.broken = None
    embedder.calls.clear()

    clock.now += 59
    rag.retrieve(FALLBACK_Q)
    assert embedder.calls == []

    clock.now += 1.5
    rag.retrieve(FALLBACK_Q)
    assert embedder.calls == [FALLBACK_Q]


def test_沒有key時切到退回模式(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rag = RAG(KB, vector_dir=tmp_path)

    assert rag.retrieve(FALLBACK_Q)[0].chunk.section == "營業時間"


def test_向量庫打不開時整段都用退回模式(tmp_path: Path) -> None:
    holder = QdrantClient(path=str(tmp_path))  # 模擬服務已經鎖住向量庫資料夾
    embedder = BreakableEmbedder()
    try:
        rag = RAG(KB, embedder=embedder, vector_dir=tmp_path)
        assert rag.retrieve(FALLBACK_Q)[0].chunk.section == "營業時間"
        assert embedder.calls == []
    finally:
        holder.close()


def test_strict模式向量用不了就丟例外不退回(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # eval 用 strict：混合檢索用不了要報錯，分數才不會誤導
    embedder = BreakableEmbedder()
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path / "ok", strict=True)
    embedder.broken = "error"
    with pytest.raises(RuntimeError):
        rag.retrieve(FALLBACK_Q)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        RAG(KB, vector_dir=tmp_path / "nokey", strict=True)
