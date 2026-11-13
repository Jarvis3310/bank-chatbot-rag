"""LLM 呼叫送出的參數。requests.post 換成假的，不呼叫 API。"""
import pytest

from chatbot import llm

RESPONSES = {
    "openai": {"choices": [{"message": {"content": "好的"}}]},
}


class FakeResponse:
    status_code = 200

    def __init__(self, data: dict):
        self.data = data

    def json(self) -> dict:
        return self.data


def test_回答時temperature設為0點2讓客服回答一致(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    sent = []

    def fake_post(url, headers, json, timeout):
        sent.append(json)
        return FakeResponse(RESPONSES["openai"])

    monkeypatch.setattr(llm.requests, "post", fake_post)

    assert llm.chat("系統", [{"role": "user", "content": "你好"}]) == "好的"
    assert sent[0]["temperature"] == 0.2


def test_不論LLM_PROVIDER設什麼都呼叫OpenAI的gpt_4o_mini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    sent = []

    def fake_post(url, headers, json, timeout):
        sent.append((url, headers, json))
        return FakeResponse(RESPONSES["openai"])

    monkeypatch.setattr(llm.requests, "post", fake_post)

    assert llm.chat("系統", [{"role": "user", "content": "你好"}]) == "好的"
    url, headers, body = sent[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test"
    assert body["model"] == "gpt-4o-mini"


def test_沒有OPENAI_API_KEY時丟出說明清楚的錯誤(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.chat("系統", [])
