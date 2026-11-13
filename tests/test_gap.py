"""知識庫缺口：RAG＋LLM 機器人讀完段落後判斷「段落沒有答案」，記進缺口清單並委婉回覆。"""
import json
from pathlib import Path

import pytest

import app
from chatbot.gaps import GapLog
from chatbot.rag import GAP_TEXT, RAG
from conftest import KB, BreakableEmbedder, FakeChat

KB_INTENT = json.dumps({"intent": "kb", "reply": "", "rewritten_question": "怎麼提高信用卡額度"}, ensure_ascii=False)
GREETING_INTENT = json.dumps({"intent": "greeting", "reply": "您好", "rewritten_question": "你好"}, ensure_ascii=False)
GAP_ANSWER = json.dumps({"answer": "目前查不到", "gap": True}, ensure_ascii=False)
OK_ANSWER = json.dumps({"answer": "請撥客服專線掛失【1】", "gap": False}, ensure_ascii=False)


def _rag(tmp_path: Path, chat: FakeChat) -> RAG:
    embedder = BreakableEmbedder()
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path / "vs", chat=chat)
    embedder.broken = "error"  # 退回模式，不呼叫 embedding
    return rag


def _use_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chat: FakeChat) -> Path:
    """換上假 LLM 的 RAG 與寫到 tmp_path 的缺口清單；回缺口檔路徑。"""
    monkeypatch.setattr(app, "rag", _rag(tmp_path, chat))
    path = tmp_path / "gaps" / "kb_gaps.jsonl"
    monkeypatch.setattr(app, "gap_log", GapLog(path))
    app.histories.clear()
    return path


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _post(message: str, session: str = "u1") -> dict:
    r = app.app.test_client().post("/api/chat", json={"session_id": session, "message": message, "bot": "rag_llm"})
    assert r.status_code == 200
    return r.get_json()


def test_判斷為缺口時回委婉話術且不列參考資料(tmp_path: Path) -> None:
    rag = _rag(tmp_path, FakeChat(replies=[KB_INTENT, GAP_ANSWER]))

    ans = rag.answer("怎麼提高信用額度")

    assert ans.gap is True
    assert ans.text == GAP_TEXT
    assert "(02) 2999-8888" in GAP_TEXT and "已記錄" in GAP_TEXT
    assert "回電" not in GAP_TEXT
    assert ans.sources == []


def test_判斷為缺口時缺口清單多一筆(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _use_app(monkeypatch, tmp_path, FakeChat(replies=[KB_INTENT, GAP_ANSWER]))

    body = _post("怎麼提高信用額度", session="web-1")

    assert body["answer"] == GAP_TEXT
    rows = _lines(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "web"
    assert row["user"] == "web-1"
    assert row["question"] == "怎麼提高信用額度"
    assert row["rewritten_question"] == "怎麼提高信用卡額度"
    assert row["time"][:2] == "20"  # ISO 時間


def test_正常回答時缺口清單沒有新增(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _use_app(monkeypatch, tmp_path, FakeChat(replies=[KB_INTENT, OK_ANSWER]))

    body = _post("信用卡不見了怎麼辦")

    assert "請撥客服專線掛失" in body["answer"]
    assert "【" not in body["answer"]
    assert _lines(path) == []


def test_非查詢知識庫意圖不寫缺口(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _use_app(monkeypatch, tmp_path, FakeChat(replies=[GREETING_INTENT]))

    _post("你好")

    assert _lines(path) == []


@pytest.mark.parametrize("raw", ["可以開外幣帳戶【1】", "{壞掉的 json", '{"gap": true}', '["gap"]'])
def test_第2次呼叫格式不對時當作正常回答不寫缺口(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    path = _use_app(monkeypatch, tmp_path, FakeChat(replies=[KB_INTENT, raw]))

    body = _post("我想開外幣帳戶")

    assert body["answer"]
    assert body["answer"] != GAP_TEXT
    assert _lines(path) == []


def test_LINE來的缺口記來源line與userId(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _use_app(monkeypatch, tmp_path, FakeChat(replies=[KB_INTENT, GAP_ANSWER]))
    sent: list[dict] = []
    monkeypatch.setattr(app, "line_post", lambda p, payload: sent.append(payload))
    ev = {"type": "message", "message": {"type": "text", "text": "怎麼提高信用額度"}, "source": {"userId": "U123"}, "replyToken": "r"}

    app.handle_line_event(ev)

    rows = _lines(path)
    assert [(r["source"], r["user"]) for r in rows] == [("line", "U123")]
    assert GAP_TEXT in sent[-1]["messages"][0]["text"]


def test_缺口資料夾寫不進去時仍正常回覆(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_app(monkeypatch, tmp_path, FakeChat(replies=[KB_INTENT, GAP_ANSWER]))
    blocker = tmp_path / "blocker"
    blocker.write_text("x")  # 檔案擋在資料夾的位置，mkdir 會失敗
    monkeypatch.setattr(app, "gap_log", GapLog(blocker / "kb_gaps.jsonl"))

    body = _post("怎麼提高信用額度")

    assert body["answer"] == GAP_TEXT


def test_純RAG機器人不寫缺口(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _use_app(monkeypatch, tmp_path, FakeChat())

    r = app.app.test_client().post("/api/chat", json={"message": "怎麼提高信用額度", "bot": "rag"})

    assert r.status_code == 200
    assert _lines(path) == []


def test_GapLog可指定位置並追加多筆(tmp_path: Path) -> None:
    log = GapLog(tmp_path / "eval" / "gaps.jsonl")

    log.record("web", "a", "問題一", "改寫一")
    log.record("line", "b", "問題二", "改寫二")

    rows = _lines(tmp_path / "eval" / "gaps.jsonl")
    assert [(r["source"], r["user"], r["question"]) for r in rows] == [("web", "a", "問題一"), ("line", "b", "問題二")]


def test_缺口位置由環境變數GAP_LOG_PATH設定(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAP_LOG_PATH", str(tmp_path / "x" / "g.jsonl"))

    assert GapLog.from_env().path == tmp_path / "x" / "g.jsonl"
    monkeypatch.delenv("GAP_LOG_PATH")
    assert GapLog.from_env().path == app.BASE / "gaps" / "kb_gaps.jsonl"
