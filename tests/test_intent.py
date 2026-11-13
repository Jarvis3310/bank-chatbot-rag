"""RAG＋LLM 機器人：先判斷意圖，只有「查詢知識庫」才檢索。LLM 換成假的，不呼叫 API。

意圖呼叫回結構化結果（intent、reply）；查詢知識庫才再有一次呼叫（依段落回答）。追問改寫見 test_rewrite.py。
"""
import json

import pytest

from conftest import FALLBACK_Q, KB, FakeChat
from chatbot.rag import INTENT_PROMPT, Intent, RAG, Mode


def intent_json(intent: str, reply: str = "", rewritten: str = "") -> str:  # rewritten 已不使用，留著讓舊呼叫不用改
    return json.dumps({"intent": intent, "reply": reply}, ensure_ascii=False)


def rewrite_json(rewritten: str) -> str:
    return json.dumps({"rewritten_question": rewritten}, ensure_ascii=False)


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


# ---- #24 追問改寫：檢索用改寫後的問題 ----

def _record_queries(rag: RAG, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    queries: list[str] = []
    real = rag.retrieve
    monkeypatch.setattr(rag, "retrieve", lambda q: (queries.append(q), real(q))[1])
    return queries


def test_追問時檢索用改寫後的問題_創辦人生日(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = FakeChat(replies=[rewrite_json("創辦人阿兩的生日是哪天"), intent_json("kb"), "阿兩生於 1958 年 3 月 12 日【1】"])
    rag = make_rag(chat)
    queries = _record_queries(rag, monkeypatch)
    history = [("你們創辦人是誰", "創辦人是阿兩。")]

    ans = rag.answer("他生日是哪天", history)

    assert queries == ["創辦人阿兩的生日是哪天"]
    assert ans.rewritten_question == "創辦人阿兩的生日是哪天"
    assert "創辦人" in ans.retrieved[0].chunk.label
    assert "1958 年 3 月 12 日" in ans.text


def test_追問時檢索用改寫後的問題_補發手續費(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = FakeChat(replies=[rewrite_json("信用卡補發新卡的手續費要多少錢"), intent_json("kb"), "補發手續費 200 元【1】"])
    rag = make_rag(chat)
    queries = _record_queries(rag, monkeypatch)
    history = [("信用卡不見了", "請立即撥打掛失專線。")]

    ans = rag.answer("補發要多少錢", history)

    assert queries == ["信用卡補發新卡的手續費要多少錢"]
    assert "信用卡" in ans.retrieved[0].chunk.label
    assert "200 元" in ans.retrieved[0].chunk.text and "200 元" in ans.text


def test_新話題時改寫結果就是原問題_不被上一題帶偏(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = FakeChat(replies=[rewrite_json("房貸最多可以貸幾成"), intent_json("kb"), "最高可貸八成【1】"])
    rag = make_rag(chat)
    queries = _record_queries(rag, monkeypatch)
    history = [("信用卡不見了", "請立即撥打掛失專線。")]

    ans = rag.answer("房貸最多可以貸幾成", history)

    assert queries == ["房貸最多可以貸幾成"]  # 不含「信用卡」
    assert "房" in ans.retrieved[0].chunk.label


def test_改寫欄位空白時退回用原問題檢索(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = FakeChat(replies=[rewrite_json(""), intent_json("kb"), "可以【1】"])
    rag = make_rag(chat)
    queries = _record_queries(rag, monkeypatch)

    rag.answer(FALLBACK_Q, [("信用卡不見了", "請撥打掛失專線。")])

    assert queries == [FALLBACK_Q]


def test_不再用規則判斷追問() -> None:
    import chatbot.rag as rag_module

    assert not hasattr(rag_module, "is_followup")
    assert not hasattr(rag_module, "FOLLOWUP_START")


def test_提示詞寫明關於本行的事都算查詢知識庫_拿不準選查詢知識庫() -> None:
    kb_line = next(line for line in INTENT_PROMPT.splitlines() if "（kb）" in line)
    for word in ("創辦人", "營業時間", "據點", "介紹"):
        assert word in kb_line  # 銀行本身的事（不只產品）也歸查詢知識庫
    assert "拿不準" in INTENT_PROMPT and "查詢知識庫" in INTENT_PROMPT.split("拿不準")[1][:20]
    oos_line = next(line for line in INTENT_PROMPT.splitlines() if "（out_of_scope）" in line)
    assert "跟本行業務無關" in oos_line


def test_提示詞寫明查個人帳戶只限查詢自己的資料_操作問題算查詢知識庫() -> None:
    account_line = next(line for line in INTENT_PROMPT.splitlines() if "（account）" in line)
    assert "解鎖" in account_line and "查詢知識庫" in account_line


def test_提示詞要求範圍外回覆一定附客服專線() -> None:
    rule = next(line for line in INTENT_PROMPT.splitlines() if line.startswith("3. 範圍外"))
    assert "一定" in rule and "(02) 2999-8888" in rule


def test_LLM判成查詢知識庫與呼叫失敗退回查詢知識庫可以分辨() -> None:
    judged = make_rag(FakeChat(replies=[intent_json("kb", "", FALLBACK_Q)])).answer(FALLBACK_Q)
    api_down = make_rag(FakeChat(replies=[RuntimeError("429")])).answer(FALLBACK_Q)
    bad_format = make_rag(FakeChat(replies=["不是 JSON"])).answer(FALLBACK_Q)

    assert judged.intent is Intent.KNOWLEDGE_BASE and not judged.intent_failed
    for ans in (api_down, bad_format):
        assert ans.intent is Intent.KNOWLEDGE_BASE and ans.intent_failed


# ---- #31 追問：第 2 次呼叫的客戶問題用改寫後的問題 ----

def test_追問時第2次呼叫的客戶問題用改寫後的問題() -> None:
    chat = FakeChat(replies=[rewrite_json("某某事項的細節是什麼"), intent_json("kb"), "答案【1】"])
    history = [("先問的第一題", "第一題的回答")]

    make_rag(chat).answer("細節呢", history)

    last = chat.calls[-1][1][-1]["content"]
    assert "客戶問題：某某事項的細節是什麼" in last
    assert "細節呢" not in last  # 原問題沒了前文看不懂，不再送出


def test_改寫欄位空白時第2次呼叫的客戶問題退回原問題() -> None:
    chat = FakeChat(replies=[intent_json("kb", "", ""), "答案【1】"])

    make_rag(chat).answer(FALLBACK_Q)

    assert f"客戶問題：{FALLBACK_Q}" in chat.calls[-1][1][-1]["content"]


def test_提示詞不含知識庫段落裡的專有名詞與數字() -> None:
    import re

    from chatbot.rag import REWRITE_PROMPT, SYSTEM_PROMPT

    about = (KB / "08_about.md").read_text(encoding="utf-8")
    terms = set(re.findall(r"「([^」]+)」", about)) - {"兩津商業銀行"}  # 引號內的名稱（人名、理念）
    terms |= set(re.findall(r"\d{4}", about))  # 年份等數字
    terms |= {"阿兩", "1958", "嘉義", "茶"}
    assert "阿兩" in terms
    for prompt in (INTENT_PROMPT, SYSTEM_PROMPT, REWRITE_PROMPT):
        for term in terms:
            assert term not in prompt, f"提示詞含知識庫內容：{term}"
