"""LINE webhook：只接 RAG＋LLM 機器人，每位 LINE 使用者各自記對話。不呼叫真的 LINE API。"""
import base64
import hashlib
import hmac
import json
from pathlib import Path

import pytest

import app
from chatbot.rag import RAG
from conftest import FALLBACK_Q, BreakableEmbedder, FakeChat, KB, wait_until

SECRET = "test-secret"


@pytest.fixture
def line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """假的 RAG＋假 LINE API；回傳 (chat, 送出的 LINE 請求清單)。"""
    chat = FakeChat("可以開外幣帳戶【1】")
    embedder = BreakableEmbedder()
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path, chat=chat)
    embedder.broken = "error"
    monkeypatch.setattr(app, "rag", rag)
    app.histories.clear()
    monkeypatch.setenv("LINE_CHANNEL_SECRET", SECRET)
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test-token")
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(app, "line_post", lambda path, payload: sent.append((path, payload)))
    return chat, sent


def _event(user: str, text: str, token: str = "r") -> dict:
    return {"type": "message", "replyToken": token, "source": {"userId": user},
            "message": {"type": "text", "text": text}}


def _replies(sent: list) -> list[str]:
    return [p["messages"][0]["text"] for path, p in sent if path == "/message/reply"]


def test_LINE訊息由RAG加LLM機器人回答(line) -> None:
    chat, sent = line

    app.handle_line_event(_event("U1", FALLBACK_Q))

    assert len(chat.calls) == 1  # 走了 LLM
    assert "可以開外幣帳戶" in _replies(sent)[0]
    assert (app.BOT_RAG_LLM, "U1") in app.histories
    assert (app.BOT_RAG, "U1") not in app.histories


def test_每位LINE使用者各自記對話(line) -> None:
    chat, sent = line

    app.handle_line_event(_event("U1", FALLBACK_Q))
    app.handle_line_event(_event("U2", "你好"))
    app.handle_line_event(_event("U2", "那手續費呢？"))

    u1 = list(app.histories[(app.BOT_RAG_LLM, "U1")])
    u2 = list(app.histories[(app.BOT_RAG_LLM, "U2")])
    assert [q for q, _ in u1] == [FALLBACK_Q]
    assert [q for q, _ in u2] == ["你好", "那手續費呢？"]
    # U2 送給 LLM 的對話裡看不到 U1 的問題
    assert FALLBACK_Q not in json.dumps(chat.calls[-1][1], ensure_ascii=False)


def test_webhook簽章正確才處理(line) -> None:
    chat, sent = line
    body = json.dumps({"events": [_event("U1", FALLBACK_Q)]}).encode()
    sig = base64.b64encode(hmac.new(SECRET.encode(), body, hashlib.sha256).digest()).decode()
    client = app.app.test_client()

    bad = client.post("/callback", data=body, headers={"X-Line-Signature": "wrong"})
    assert bad.status_code == 400
    assert sent == []

    ok = client.post("/callback", data=body, headers={"X-Line-Signature": sig})
    assert ok.status_code == 200
    assert wait_until(lambda: _replies(sent))
    assert "可以開外幣帳戶" in _replies(sent)[0]


def test_瀏覽器直接開callback是405(line) -> None:
    assert app.app.test_client().get("/callback").status_code == 405
