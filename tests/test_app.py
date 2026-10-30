"""網頁聊天 API：服務使用的 RAG 換成測試版本，不碰真的 API 和正式向量庫。"""
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

import app
from conftest import FALLBACK_Q, FALLBACK_SECTION, KB, BreakableEmbedder
from rag import RAG


def test_embedding失敗時網頁聊天API仍回200和答案(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "none")
    embedder = BreakableEmbedder()
    monkeypatch.setattr(app, "rag", RAG(KB, embedder=embedder, vector_dir=tmp_path))
    embedder.broken = "error"

    r = app.app.test_client().post("/api/chat", json={"session_id": "t", "message": FALLBACK_Q})

    assert r.status_code == 200
    body = r.get_json()
    assert body["answer"]
    assert FALLBACK_SECTION in body["retrieved"][0]["label"]
    assert "bm25" in body["retrieved"][0] and "similarity" not in body["retrieved"][0]  # 退回模式沒有相似度


def test_服務結束時關閉向量庫(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "rag", RAG(KB, embedder=BreakableEmbedder(), vector_dir=tmp_path))

    app.close_rag()  # 程式結束時由 atexit 呼叫

    QdrantClient(path=str(tmp_path)).close()  # 向量庫資料夾的鎖已釋放，才打得開


def test_沒用過RAG時結束不會建立RAG(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "rag", None)

    app.close_rag()

    assert app.rag is None
