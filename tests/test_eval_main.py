"""eval 主程式：混合檢索在跑題途中壞掉時的結束方式。用假 embedding，不呼叫 API。"""
from pathlib import Path

import pytest

import eval as eval_module
from conftest import KB, FakeEmbedder
from rag import RAG, Mode


class WrongKeyEmbedder(FakeEmbedder):
    """建庫時正常，之後每次查詢都回 401（模擬 key 錯誤，錯誤內容跟 OpenAI 一樣是多行 JSON）。"""

    def __init__(self) -> None:
        super().__init__({})
        self.built = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.built:
            raise RuntimeError('Embedding API 錯誤 401：{\n  "error": {\n    "code": "invalid_api_key"\n  }\n}')
        return super().embed(texts)


def run_eval_with_wrong_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SystemExit:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-wrong-key-for-test")
    embedder = WrongKeyEmbedder()

    def make_rag(kb_dir, mode, strict):
        rag = RAG(KB, mode=mode, embedder=embedder, vector_dir=tmp_path, strict=strict)
        embedder.built = embedder.built or mode is Mode.HYBRID  # 建庫成功，之後查詢才開始錯
        return rag

    monkeypatch.setattr(eval_module, "RAG", make_rag)
    with pytest.raises(SystemExit) as exc:
        eval_module.main(with_answer=False)
    return exc.value


def test_key錯誤時eval只印一行錯誤訊息並以非0結束(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exit_ = run_eval_with_wrong_key(tmp_path, monkeypatch)

    assert isinstance(exit_.code, str)  # sys.exit(字串)：印到 stderr、結束代碼 1
    assert exit_.code.startswith("混合檢索用不了：")
    assert "401" in exit_.code
    assert "\n" not in exit_.code
