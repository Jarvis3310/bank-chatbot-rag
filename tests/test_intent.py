"""RAG＋LLM 機器人：先判斷意圖，只有「查詢知識庫」才檢索。LLM 換成假的，不呼叫 API。

第 1 次 LLM 呼叫回結構化結果（intent、reply、rewritten_question）；查詢知識庫才有第 2 次（依段落回答）。
"""
import json

import pytest

from conftest import FALLBACK_Q, KB, FakeChat
from chatbot.rag import INTENT_PROMPT, Intent, RAG, Mode


def intent_json(intent: str, reply: str = "", rewritten: str = "") -> str:
    return json.dumps({"intent": intent, "reply": reply, "rewritten_question": rewritten}, ensure_ascii=False)


def make_rag(chat: FakeChat) -> RAG:
    return RAG(KB, mode=Mode.FALLBACK, chat=chat)


@pytest.mark.parametrize(
    "question, intent, reply",
    [
        ("你好", "greeting", "您好，我是兩津商業銀行的客服助理，請問想了解哪方面的服務？"),
        ("今天天氣如何", "out_of_scope", "抱歉，天氣不在我的服務範圍，銀行業務問題歡迎詢問，或撥打 (02) 2999-8888。"),
        ("我的餘額多少", "account", "我無法查詢個人帳戶資料，請撥打 24 小時客服專線 (02) 2999-8888。"),
        ("我要找真人", "human", "請撥打 24 小時客服專線 (02) 2999-8888，由專人為您服務。"),
    ],
)
def test_非知識庫的意圖由第1次呼叫直接回覆且不檢索(question: str, intent: str, reply: str) -> None:
    chat = FakeChat(replies=[intent_json(intent, reply, question)])

    ans = make_rag(chat).answer(question)

    assert len(chat.calls) == 1  # 沒有第 2 次呼叫
    assert ans.intent is Intent(intent)
    assert ans.text == reply
    assert ans.retrieved == [] and ans.sources == []
    assert not ans.error


def test_意圖提示詞寫明5類專線與服務時間() -> None:
    for name in ("查詢知識庫", "打招呼", "範圍外", "查個人帳戶", "要求真人客服"):
        assert name in INTENT_PROMPT
    assert "(02) 2999-8888" in INTENT_PROMPT
    assert "24 小時" in INTENT_PROMPT
    assert "rewritten_question" in INTENT_PROMPT  # 預留給追問改寫


def test_信用卡不見了怎麼辦判成查詢知識庫並檢索() -> None:
    q = "信用卡不見了怎麼辦"
    chat = FakeChat(replies=[intent_json("kb", "", q), "請立即掛失【1】"])
    rag = make_rag(chat)

    ans = rag.answer(q)

    assert ans.intent is Intent.KNOWLEDGE_BASE
    assert len(chat.calls) == 2
    assert ans.retrieved, "查詢知識庫要檢索"
    assert all(h.chunk.text in chat.calls[1][1][-1]["content"] for h in ans.retrieved)
    assert ans.text == "請立即掛失"


def test_怎麼提高信用額度是本行業務不是範圍外() -> None:
    q = "怎麼提高信用額度"
    chat = FakeChat(replies=[intent_json("kb", "", q), "答案【1】"])

    ans = make_rag(chat).answer(q)

    assert ans.intent is Intent.KNOWLEDGE_BASE
    assert len(chat.calls) == 2
    assert "本行業務但知識庫沒寫" in INTENT_PROMPT  # 規則寫進提示詞，LLM 才不會判成範圍外


@pytest.mark.parametrize(
    "bad",
    [
        "這不是 JSON",
        "",
        "[1, 2]",
        '{"reply": "沒有 intent"}',
        intent_json("weather"),  # 不認得的意圖
        intent_json("greeting", ""),  # 非知識庫意圖卻沒有回覆
        "```json\n" + intent_json("kb") + "\n```x",  # 程式碼框後面多了字
    ],
)
def test_回傳格式不對時當作查詢知識庫並用原問題檢索(bad: str) -> None:
    chat = FakeChat(replies=[bad, "可以開外幣帳戶【1】"])

    ans = make_rag(chat).answer(FALLBACK_Q)

    assert ans.intent is Intent.KNOWLEDGE_BASE
    assert len(chat.calls) == 2
    assert FALLBACK_Q in chat.calls[1][1][-1]["content"]
    assert ans.retrieved
    assert ans.text == "可以開外幣帳戶" and not ans.error


def test_意圖呼叫API失敗時當作查詢知識庫() -> None:
    chat = FakeChat(replies=[RuntimeError("LLM API 錯誤 429"), "可以開外幣帳戶【1】"])

    ans = make_rag(chat).answer(FALLBACK_Q)

    assert ans.intent is Intent.KNOWLEDGE_BASE
    assert ans.retrieved and ans.text == "可以開外幣帳戶" and not ans.error


def test_程式碼框包起來的JSON也能解析() -> None:
    chat = FakeChat(replies=["```json\n" + intent_json("greeting", "您好") + "\n```"])

    ans = make_rag(chat).answer("你好")

    assert ans.intent is Intent.GREETING and ans.text == "您好"


def test_意圖與改寫欄位印在服務log(capsys: pytest.CaptureFixture[str]) -> None:
    chat = FakeChat(replies=[intent_json("greeting", "您好", "你好")])

    make_rag(chat).answer("你好")

    out = capsys.readouterr().out
    assert "意圖" in out and "打招呼" in out


def test_意圖呼叫帶最近3輪對話供追問改寫使用() -> None:
    chat = FakeChat(replies=[intent_json("greeting", "您好")])
    history = [(f"問題{n}", f"回答{n}") for n in range(1, 5)]

    make_rag(chat).answer("你好", history)

    system, messages = chat.calls[0]
    assert system == INTENT_PROMPT
    assert [m["content"] for m in messages[:-1]] == ["問題2", "回答2", "問題3", "回答3", "問題4", "回答4"]
    assert "你好" in messages[-1]["content"]
