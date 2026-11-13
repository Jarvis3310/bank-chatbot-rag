"""RAG＋LLM 機器人：LLM 依檢索到的段落回答、只帶最近 3 輪對話。LLM 換成假的，不呼叫 API。"""
import pytest

from conftest import FALLBACK_Q, FALLBACK_SECTION, KB, FakeChat
from chatbot import llm
from chatbot.rag import RAG, SYSTEM_PROMPT, Mode, format_for_line


def make_rag(chat: FakeChat) -> RAG:
    return RAG(KB, mode=Mode.FALLBACK, chat=chat)


def test_送給LLM的是檢索到的段落且回答不列參考資料() -> None:
    chat = FakeChat()
    rag = make_rag(chat)

    ans = rag.answer(FALLBACK_Q)

    sent = chat.calls[-1][1][-1]["content"]  # 第 2 次呼叫（第 1 次是意圖判斷）
    hits = rag.retrieve(FALLBACK_Q)
    assert 0 < len(hits) <= 4
    assert all(h.chunk.text in sent for h in hits)
    assert FALLBACK_Q in sent
    assert ans.text == "可以開外幣帳戶"  # LLM 仍輸出的【1】從文字拿掉
    assert ans.sources == []  # RAG＋LLM 機器人不列參考資料
    assert [h.chunk.section for h in ans.retrieved][0] == FALLBACK_SECTION  # 檢索結果照舊
    assert not ans.error
    assert "📄" not in format_for_line(ans)


def test_提示詞不再要求標出處() -> None:
    assert "【" not in SYSTEM_PROMPT and "句尾標註" not in SYSTEM_PROMPT


def test_聊到第4輪時只送最近3輪對話() -> None:
    chat = FakeChat()
    rag = make_rag(chat)
    history = [(f"問題{n}", f"回答{n}") for n in range(1, 5)]  # 已經聊了 4 輪

    rag.answer(FALLBACK_Q, history)

    messages = chat.calls[0][1]
    assert len(messages) == 3 * 2 + 1  # 3 輪＋這一題
    assert [m["content"] for m in messages[:-1]] == ["問題2", "回答2", "問題3", "回答3", "問題4", "回答4"]


def test_沒有OPENAI_API_KEY時回錯誤訊息而不是丟例外(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rag = RAG(KB, mode=Mode.FALLBACK)  # 沒注入假 LLM，走真正的 llm.chat

    ans = rag.answer(FALLBACK_Q)

    assert ans.error
    assert "(02) 2999-8888" in ans.text
    assert ans.sources == []


def test_LLM呼叫失敗時回錯誤訊息而不是丟例外() -> None:
    chat = FakeChat()
    chat.broken = True

    ans = make_rag(chat).answer(FALLBACK_Q)

    assert ans.error
    assert "(02) 2999-8888" in ans.text


def test_純RAG機器人不呼叫LLM() -> None:
    chat = FakeChat()

    ans = make_rag(chat).answer_rag_only(FALLBACK_Q)

    assert chat.calls == []
    assert FALLBACK_SECTION in ans.sources[0].section


def test_意圖與第2次呼叫都要求JSON輸出(monkeypatch: pytest.MonkeyPatch) -> None:
    """沒注入假 LLM 時走真正的 llm.chat；攔 requests.post，確認兩次呼叫都帶 response_format。"""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    bodies = []

    class Resp:
        status_code = 200

        def __init__(self, content: str):
            self.content = content

        def json(self) -> dict:
            return {"choices": [{"message": {"content": self.content}}]}

    replies = ['{"intent": "kb", "reply": "", "rewritten_question": "我想存日幣"}', '{"answer": "可以【1】", "gap": false}']

    def fake_post(url, headers, json, timeout):
        bodies.append(json)
        return Resp(replies[len(bodies) - 1])

    monkeypatch.setattr(llm.requests, "post", fake_post)

    ans = RAG(KB, mode=Mode.FALLBACK).answer(FALLBACK_Q)

    assert len(bodies) == 2
    assert [b.get("response_format") for b in bodies] == [{"type": "json_object"}] * 2
    assert ans.text == "可以" and not ans.error and not ans.intent_failed
