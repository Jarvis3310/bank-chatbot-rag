"""網頁聊天 API：服務使用的 RAG 換成測試版本，不碰真的 API 和正式向量庫。"""
from pathlib import Path

import pytest

import app
from rag import RAG
from test_fallback import FALLBACK_Q, BreakableEmbedder
from test_hybrid import KB


def test_embedding失敗時網頁聊天API仍回200和答案(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "none")
    embedder = BreakableEmbedder()
    monkeypatch.setattr(app, "rag", RAG(KB, embedder=embedder, vector_dir=tmp_path))
    embedder.broken = "error"

    r = app.app.test_client().post("/api/chat", json={"session_id": "t", "message": FALLBACK_Q})

    assert r.status_code == 200
    body = r.get_json()
    assert body["answer"]
    assert "營業時間" in body["retrieved"][0]["label"]
