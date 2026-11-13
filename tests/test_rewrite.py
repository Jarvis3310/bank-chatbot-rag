"""RAG＋LLM 機器人：追問改寫與意圖判斷分成兩次 LLM 呼叫（#31）。LLM 換成假的，不呼叫 API。

順序：追問改寫（有對話紀錄才呼叫）→ 用改寫後的完整問題判意圖 → 檢索 → 依段落回答。
"""
import json

import pytest

from conftest import FALLBACK_Q, KB, FakeChat
from chatbot.rag import INTENT_PROMPT, REWRITE_PROMPT, SYSTEM_PROMPT, Intent, RAG, Mode

HISTORY = [("先問的第一題", "第一題的回答")]


def rewrite_json(q: str) -> str:
    return json.dumps({"rewritten_question": q}, ensure_ascii=False)


def intent_json(intent: str, reply: str = "") -> str:
    return json.dumps({"intent": intent, "reply": reply}, ensure_ascii=False)


def make_rag(chat: FakeChat) -> RAG:
    return RAG(KB, mode=Mode.FALLBACK, chat=chat)


def test_有對話紀錄時第1次呼叫只做追問改寫_帶最近3輪對話() -> None:
    chat = FakeChat(replies=[rewrite_json("完整問題"), intent_json("kb"), "答案【1】"])
    history = [(f"問題{n}", f"回答{n}") for n in range(1, 5)]

    make_rag(chat).answer("細節呢", history)

    system, messages = chat.calls[0]
    assert system == REWRITE_PROMPT
    assert [m["content"] for m in messages] == ["問題2", "回答2", "問題3", "回答3", "問題4", "回答4", "細節呢"]


def test_第2次呼叫用改寫後的完整問題判意圖_不帶對話紀錄() -> None:
    chat = FakeChat(replies=[rewrite_json("完整問題"), intent_json("kb"), "答案【1】"])

    make_rag(chat).answer("細節呢", HISTORY)

    system, messages = chat.calls[1]
    assert system == INTENT_PROMPT
    assert messages == [{"role": "user", "content": "完整問題"}]


def test_追問的完整流程是三次呼叫_改寫_意圖_回答() -> None:
    chat = FakeChat(replies=[rewrite_json("某項服務的費用是多少"), intent_json("kb"), "答案【1】"])

    ans = make_rag(chat).answer("那費用呢", HISTORY)

    assert [c[0] for c in chat.calls] == [REWRITE_PROMPT, INTENT_PROMPT, SYSTEM_PROMPT]
    assert "客戶問題：某項服務的費用是多少" in chat.calls[2][1][-1]["content"]
    assert ans.rewritten_question == "某項服務的費用是多少"
    assert ans.intent is Intent.KNOWLEDGE_BASE and not ans.intent_failed


def test_沒有對話紀錄時不呼叫改寫_直接用原問題判意圖() -> None:
    chat = FakeChat(replies=[intent_json("kb"), "答案【1】"])

    ans = make_rag(chat).answer(FALLBACK_Q)

    assert [c[0] for c in chat.calls] == [INTENT_PROMPT, SYSTEM_PROMPT]
    assert chat.calls[0][1] == [{"role": "user", "content": FALLBACK_Q}]
    assert ans.rewritten_question == FALLBACK_Q


def test_追問被判成非知識庫意圖時共2次呼叫_回覆來自意圖呼叫() -> None:
    chat = FakeChat(replies=[rewrite_json("謝謝你的協助"), intent_json("greeting", "不客氣")])

    ans = make_rag(chat).answer("謝謝", HISTORY)

    assert len(chat.calls) == 2
    assert ans.text == "不客氣" and ans.intent is Intent.GREETING and ans.retrieved == []


@pytest.mark.parametrize("bad", ["不是 JSON", rewrite_json(""), rewrite_json("   "), '{"foo": 1}', '{"rewritten_question": 5}', RuntimeError("429")])
def test_改寫失敗或空白時用原問題判意圖與檢索(bad: str | Exception) -> None:
    chat = FakeChat(replies=[bad, intent_json("kb"), "答案【1】"])

    ans = make_rag(chat).answer(FALLBACK_Q, HISTORY)

    assert chat.calls[1][1] == [{"role": "user", "content": FALLBACK_Q}]
    assert f"客戶問題：{FALLBACK_Q}" in chat.calls[2][1][-1]["content"]
    assert ans.rewritten_question == FALLBACK_Q
    assert not ans.intent_failed and not ans.error  # 改寫失敗不算意圖失敗


def test_意圖呼叫失敗時仍用改寫後的問題_當作查詢知識庫並標記() -> None:
    chat = FakeChat(replies=[rewrite_json("完整問題"), "不是 JSON", "答案【1】"])

    ans = make_rag(chat).answer("細節呢", HISTORY)

    assert ans.intent is Intent.KNOWLEDGE_BASE and ans.intent_failed
    assert ans.rewritten_question == "完整問題"
    assert "客戶問題：完整問題" in chat.calls[2][1][-1]["content"]


def test_意圖提示詞不再要求輸出改寫欄位() -> None:
    assert "rewritten_question" not in INTENT_PROMPT


def test_改寫提示詞寫明規則且不含知識庫內容() -> None:
    assert "完整問題" in REWRITE_PROMPT and "原問題" in REWRITE_PROMPT and "新話題" in REWRITE_PROMPT
    assert "rewritten_question" in REWRITE_PROMPT
    for term in ("阿兩", "1958", "創辦人", "嘉義"):
        assert term not in REWRITE_PROMPT
