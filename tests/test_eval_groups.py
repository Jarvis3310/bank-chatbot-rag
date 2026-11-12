"""eval --answer 的意圖題組、追問題組、知識庫缺口題：題組結構、計分、不污染正式缺口清單。用假 LLM、假 embedding，不呼叫 API。"""
import json
import tempfile
from pathlib import Path

import pytest

import eval as eval_module
from chatbot.gaps import DEFAULT_PATH, GapLog
from chatbot.rag import INTENT_PROMPT, RAG, SYSTEM_PROMPT, Intent
from conftest import KB, FakeEmbedder


class ScriptedChat:
    """依最後一句使用者訊息回覆的假 LLM。intents：問題→意圖代碼（沒列的當 kb）；answers：問題→第 2 次呼叫回的 JSON。"""

    def __init__(self, intents: dict[str, str] | None = None, answers: dict[str, dict] | None = None, rewrites: dict[str, str] | None = None):
        self.intents = intents or {}
        self.answers = answers or {}
        self.rewrites = rewrites or {}
        self.calls: list[tuple[str, list[dict]]] = []

    def __call__(self, system: str, messages: list[dict]) -> str:
        self.calls.append((system, messages))
        last = messages[-1]["content"]
        if system == INTENT_PROMPT:
            code = self.intents.get(last, "kb")
            reply = "" if code == "kb" else "回覆"
            return json.dumps({"intent": code, "reply": reply, "rewritten_question": self.rewrites.get(last, last)}, ensure_ascii=False)
        assert system == SYSTEM_PROMPT
        question = last.split("客戶問題：", 1)[1]
        return json.dumps(self.answers.get(question, {"answer": "好的【1】", "gap": False}), ensure_ascii=False)


def _rag(chat: ScriptedChat, tmp_path: Path) -> RAG:
    return RAG(KB, embedder=FakeEmbedder({}), vector_dir=tmp_path / "vs", chat=chat)


def test_意圖題組五類各至少兩題且含邊界題() -> None:
    by_intent: dict[Intent, list[str]] = {}
    for q, intent in eval_module.INTENT_CASES:
        by_intent.setdefault(intent, []).append(q)

    assert set(by_intent) == set(Intent)
    assert all(len(qs) >= 2 for qs in by_intent.values())
    assert ("怎麼提高信用額度", Intent.KNOWLEDGE_BASE) in eval_module.INTENT_CASES


def test_追問題組是兩輪對話且第二題有關鍵字() -> None:
    assert eval_module.FOLLOWUP_CASES
    assert ("你們創辦人是誰", "他生日是哪天", "1958") in [(c.first, c.second, c.keyword) for c in eval_module.FOLLOWUP_CASES]


def test_意圖題組全部判對時正確率滿分(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    intents = {q: i.value for q, i in eval_module.INTENT_CASES}

    passed, total = eval_module.run_intent_group(_rag(ScriptedChat(intents), tmp_path))

    assert (passed, total) == (len(eval_module.INTENT_CASES), len(eval_module.INTENT_CASES))
    assert f"【意圖題組】正確：{total}/{total}" in capsys.readouterr().out


def test_意圖判錯的題目計為答錯並列出(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    intents = {q: i.value for q, i in eval_module.INTENT_CASES}
    intents["怎麼提高信用額度"] = "out_of_scope"  # 邊界題判錯

    passed, total = eval_module.run_intent_group(_rag(ScriptedChat(intents), tmp_path))

    out = capsys.readouterr().out
    assert passed == total - 1
    assert "❌ 怎麼提高信用額度" in out
    assert f"【意圖題組】正確：{total - 1}/{total}" in out


def test_意圖題組不檢查非知識庫意圖的回覆措辭(tmp_path: Path) -> None:
    chat = ScriptedChat({q: i.value for q, i in eval_module.INTENT_CASES})  # 回覆固定是「回覆」，與規則措辭無關

    passed, total = eval_module.run_intent_group(_rag(chat, tmp_path))

    assert passed == total


def test_追問題組第二題答案含關鍵字才算對(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    case = eval_module.FOLLOWUP_CASES[0]
    answers = {
        case.first: {"answer": "創辦人是阿兩", "gap": False},
        case.second: {"answer": f"生日是 {case.keyword} 年", "gap": False},
    }
    others = {c.second: {"answer": "不知道", "gap": False} for c in eval_module.FOLLOWUP_CASES[1:]}
    chat = ScriptedChat(answers=answers | others)

    passed, total = eval_module.run_followup_group(_rag(chat, tmp_path))

    assert passed == 1 and total == len(eval_module.FOLLOWUP_CASES)
    assert f"【追問題組】正確：1/{total}" in capsys.readouterr().out
    second_call = [c for c in chat.calls if c[0] == INTENT_PROMPT][1]  # 第 2 題的意圖呼叫帶了第 1 題的對話
    assert second_call[1][0]["content"] == case.first


def test_缺口題判成缺口才寫進缺口清單且不碰正式清單(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    gap_log = GapLog(tmp_path / "eval_gaps.jsonl")
    q = eval_module.GAP_CASES[0]
    chat = ScriptedChat(answers={q: {"answer": "沒有", "gap": True}})  # 只有第 1 題判成缺口

    passed, total = eval_module.run_gap_group(_rag(chat, tmp_path), gap_log)

    rows = [json.loads(line) for line in (tmp_path / "eval_gaps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["question"] for r in rows] == [q]
    assert rows[0]["source"] == "eval"
    assert passed == 1 and total == len(eval_module.GAP_CASES)
    assert f"【知識庫缺口題】判成缺口：1/{total}" in capsys.readouterr().out


def test_eval的缺口清單在暫存資料夾不是正式清單() -> None:
    log = eval_module.eval_gap_log()

    assert log.path != DEFAULT_PATH
    assert Path(tempfile.gettempdir()) in log.path.parents


def test_加了answer才跑意圖與追問題組(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    chat = ScriptedChat({q: i.value for q, i in eval_module.INTENT_CASES})
    rag = _rag(chat, tmp_path)
    rags = {name: rag for name in eval_module.COLUMNS.values()}

    eval_module.run(rags, with_answer=True, gap_log=GapLog(tmp_path / "g.jsonl"))

    out = capsys.readouterr().out
    assert "【意圖題組】" in out and "【追問題組】" in out and "【知識庫缺口題】" in out


def test_沒加answer不跑這兩組也不呼叫LLM(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    chat = ScriptedChat()
    rag = _rag(chat, tmp_path)
    rags = {name: rag for name in eval_module.COLUMNS.values()}

    eval_module.run(rags, with_answer=False)

    out = capsys.readouterr().out
    assert "【意圖題組】" not in out and "【追問題組】" not in out
    assert chat.calls == []


def test_意圖題組含本行自身資訊的邊界題() -> None:
    cases = dict(eval_module.INTENT_CASES)
    assert cases["你們創辦人是誰"] is Intent.KNOWLEDGE_BASE
    assert cases["你們幾點開門"] is Intent.KNOWLEDGE_BASE
