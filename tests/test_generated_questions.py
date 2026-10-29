"""建庫時 LLM 替每段生成口語問法，跟段落一起算向量；問法依段落快取，生成失敗時退回只用段落、冷卻後重試。"""
import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import KB, FakeClock, FakeEmbedder, FakeGenerator
from rag import RAG, load_chunks
from vector_index import VectorIndex

CHUNKS = load_chunks(KB)
FEE_RULES = {"跨行轉帳手續費": 0, "ATM 跨行": 0}
FEE_Q = "用 ATM 跨行轉帳要多少手續費"


def test_生成的問法加進embedding的文字(tmp_path: Path) -> None:
    embedder, generator = FakeEmbedder({}), FakeGenerator()
    VectorIndex(CHUNKS, embedder, tmp_path, generator=generator).close()

    first = CHUNKS[0]
    title = first.embed_text.splitlines()[0]
    assert generator.calls.count(first.embed_text) == 1
    assert f"{first.embed_text}\n{title} 問法1\n{title} 問法2\n{title} 問法3" in embedder.calls
    assert len(embedder.calls) == len(CHUNKS)


def test_段落沒變不重新生成_改了只重新生成那一段(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder({}), tmp_path, generator=FakeGenerator()).close()

    generator = FakeGenerator()
    VectorIndex(CHUNKS, FakeEmbedder({}), tmp_path, generator=generator).close()
    assert generator.calls == []

    edited = CHUNKS.copy()
    edited[3] = dataclasses.replace(edited[3], text=edited[3].text + "\n新增一行。")
    embedder, generator = FakeEmbedder({}), FakeGenerator()
    VectorIndex(edited, embedder, tmp_path, generator=generator).close()
    assert generator.calls == [edited[3].embed_text]
    assert len(embedder.calls) == 1 and embedder.calls[0].startswith(edited[3].embed_text + "\n")


def test_換生成模型時整庫重算(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder({}), tmp_path, generator=FakeGenerator("舊模型")).close()

    embedder, generator = FakeEmbedder({}), FakeGenerator("新模型")
    VectorIndex(CHUNKS, embedder, tmp_path, generator=generator).close()

    assert len(generator.calls) == len(CHUNKS) and len(embedder.calls) == len(CHUNKS)


def wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)


def test_生成失敗時只用段落算向量_仍能回答_冷卻後重試(tmp_path: Path) -> None:
    embedder, generator, clock = FakeEmbedder(FEE_RULES), FakeGenerator(), FakeClock()
    generator.broken = True
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path, clock=clock, generator=generator)
    try:
        assert sorted(embedder.calls) == sorted(c.embed_text for c in CHUNKS)  # 沒有問法
        hits = rag.retrieve(FEE_Q)
        assert hits[0].chunk.section == "跨行轉帳手續費" and hits[0].similarity is not None  # 混合檢索照常

        generator.broken = False
        generator.calls.clear()
        clock.now += 59
        rag.retrieve(FEE_Q)
        assert generator.calls == []  # 冷卻中不重試

        clock.now += 1.5
        rag.retrieve(FEE_Q)  # 觸發背景補建
        wait_for(lambda: rag.vectors.rebuilding is not None and not rag.vectors.rebuilding.is_alive())
        assert sorted(generator.calls) == sorted(c.embed_text for c in CHUNKS)
        assert sum("問法1" in t for t in embedder.calls) == len(CHUNKS)
    finally:
        rag.close()

    generator = FakeGenerator()
    VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES), tmp_path, generator=generator).close()
    assert generator.calls == []  # 補建成功後重開不再生成


def test_strict時生成失敗就報錯(tmp_path: Path) -> None:
    generator = FakeGenerator()
    generator.broken = True
    with pytest.raises(RuntimeError):
        VectorIndex(CHUNKS, FakeEmbedder({}), tmp_path, strict=True, generator=generator)
