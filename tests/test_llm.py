"""LLM 呼叫送出的參數。requests.post 換成假的，不呼叫 API。"""
import pytest

import llm

RESPONSES = {
    "anthropic": {"content": [{"type": "text", "text": "好的"}]},
    "openai": {"choices": [{"message": {"content": "好的"}}]},
}


class FakeResponse:
    status_code = 200

    def __init__(self, data: dict):
        self.data = data

    def json(self) -> dict:
        return self.data


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_回答時temperature設為0點2讓客服回答一致(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    sent = []

    def fake_post(url, headers, json, timeout):
        sent.append(json)
        return FakeResponse(RESPONSES[provider])

    monkeypatch.setattr(llm.requests, "post", fake_post)

    assert llm.chat("系統", [{"role": "user", "content": "你好"}]) == "好的"
    assert sent[0]["temperature"] == 0.2
