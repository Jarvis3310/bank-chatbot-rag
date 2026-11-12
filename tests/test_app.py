"""網頁聊天 API：服務使用的 RAG 換成測試版本，不碰真的 API 和正式向量庫。"""
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

import app
from conftest import FALLBACK_Q, FALLBACK_SECTION, KB, BreakableEmbedder, FakeChat
from chatbot.rag import RAG


def test_embedding失敗時網頁聊天API仍回200和答案(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


# ---------- 純 RAG 機器人與 RAG＋LLM 機器人 ----------
NO_LLM_NOTE = "尚未設定 LLM"


def _use_rag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chat: FakeChat | None = None) -> RAG:
    embedder = BreakableEmbedder()
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path, chat=chat or FakeChat())
    embedder.broken = "error"  # 走 BM25＋同義詞的退回模式，FALLBACK_Q 才一定找得到段落
    monkeypatch.setattr(app, "rag", rag)
    app.histories.clear()
    return rag


def _chat(bot: str | None, message: str, session: str = "t") -> dict:
    payload = {"session_id": session, "message": message}
    if bot is not None:
        payload["bot"] = bot
    r = app.app.test_client().post("/api/chat", json=payload)
    assert r.status_code == 200
    return r.get_json()


def test_純RAG機器人連問追問時只用當題檢索(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _use_rag(monkeypatch, tmp_path)
    queries: list[str] = []
    real = rag.retrieve
    monkeypatch.setattr(rag, "retrieve", lambda q: (queries.append(q), real(q))[1])

    _chat("rag", "信用卡不見了")
    _chat("rag", "那手續費呢？")

    assert queries == ["信用卡不見了", "那手續費呢？"]


@pytest.mark.parametrize("provider", ["none", "openai"])
def test_純RAG機器人的回覆沒有尚未設定LLM字樣(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    _use_rag(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", provider)

    body = _chat("rag", FALLBACK_Q)

    assert NO_LLM_NOTE not in body["answer"]
    assert body["answer"]
    assert FALLBACK_SECTION in body["sources"][0]


def test_RAG加LLM機器人的回答由LLM生成(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_rag(monkeypatch, tmp_path, FakeChat("可以開外幣帳戶【1】"))
    monkeypatch.setenv("LLM_PROVIDER", "none")  # 不影響這個機器人

    explicit = _chat("rag_llm", FALLBACK_Q)
    default = _chat(None, FALLBACK_Q)  # 沒指定 bot 時是 RAG＋LLM 機器人

    for body in (explicit, default):
        assert body["answer"] == "可以開外幣帳戶"
        assert FALLBACK_SECTION in body["sources"][0]


def test_兩個機器人回傳相同欄位(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_rag(monkeypatch, tmp_path)

    a, b = _chat("rag", FALLBACK_Q), _chat("rag_llm", FALLBACK_Q)

    for body in (a, b):
        assert set(body) >= {"answer", "sources", "retrieved", "seconds", "provider"}
        assert FALLBACK_SECTION in body["retrieved"][0]["label"]
        assert isinstance(body["seconds"], (int, float))
    assert a["retrieved"] == b["retrieved"]


def test_兩個機器人共用同一個RAG與向量庫連線(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _use_rag(monkeypatch, tmp_path)

    _chat("rag", FALLBACK_Q)
    _chat("rag_llm", FALLBACK_Q)

    assert app.get_rag() is rag  # 沒有因為換機器人另開一個 RAG（Qdrant 本機模式只能開一次）


def test_兩個機器人的對話紀錄分開(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _use_rag(monkeypatch, tmp_path)
    queries: list[str] = []
    real = rag.retrieve
    monkeypatch.setattr(rag, "retrieve", lambda q: (queries.append(q), real(q))[1])

    _chat("rag", "信用卡不見了")  # 純 RAG 不記對話，也不影響 RAG＋LLM 的紀錄
    _chat("rag_llm", "那手續費呢？")

    assert queries == ["信用卡不見了", "那手續費呢？"]
    assert len(app.histories[("rag_llm", "t")]) == 1
    assert ("rag", "t") not in app.histories or len(app.histories[("rag", "t")]) == 0


def test_指定不存在的機器人回400(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_rag(monkeypatch, tmp_path)

    r = app.app.test_client().post("/api/chat", json={"message": "你好", "bot": "nope"})

    assert r.status_code == 400


def test_LLM失敗時RAG加LLM機器人回錯誤訊息_純RAG照常_不記進對話(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chat = FakeChat()
    chat.broken = True
    _use_rag(monkeypatch, tmp_path, chat)

    llm_body = _chat("rag_llm", FALLBACK_Q)
    rag_body = _chat("rag", FALLBACK_Q)

    assert "(02) 2999-8888" in llm_body["answer"] and llm_body["sources"] == []
    assert FALLBACK_SECTION in rag_body["sources"][0]
    assert len(app.histories[("rag_llm", "t")]) == 0


def test_首頁同一題分別問純RAG與RAG加LLM兩個機器人() -> None:
    html = app.app.test_client().get("/").get_data(as_text=True)

    assert '"rag"' in html and '"rag_llm"' in html  # 兩欄各用 bot 參數呼叫 /api/chat
    assert "純 RAG" in html and "RAG＋LLM" in html
    assert "intent" not in html and "意圖" not in html  # 網頁不顯示意圖和改寫後的問題


@pytest.mark.parametrize("word", ["重來", "清除", "reset"])
def test_純RAG機器人收到重來也回已清除而不檢索(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, word: str) -> None:
    rag = _use_rag(monkeypatch, tmp_path)
    queries: list[str] = []
    monkeypatch.setattr(rag, "retrieve", lambda q: queries.append(q) or [])

    body = _chat("rag", word)

    assert "已清除對話紀錄" in body["answer"]
    assert body["sources"] == [] and queries == []
