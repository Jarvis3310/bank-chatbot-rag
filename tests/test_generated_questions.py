"""建庫時 LLM 替每段生成口語問法，跟段落一起算向量；問法存在 JSON 檔（進版控），生成失敗時退回只用段落、冷卻後重試。"""
import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import KB, FakeClock, FakeEmbedder, FakeGenerator
from rag import RAG, Chunk, load_chunks
from vector_index import VectorIndex, chunk_hash

CHUNKS = load_chunks(KB)
FEE_RULES = {"跨行轉帳手續費": 0, "ATM 跨行": 0}
FEE_Q = "用 ATM 跨行轉帳要多少手續費"


def build(tmp_path: Path, generator: FakeGenerator, chunks: list[Chunk] = CHUNKS, store: str = "vectors", **kw) -> FakeEmbedder:
    """建一次向量庫就關掉；問法檔固定是 tmp_path/questions.json。回傳這次用的假 embedding。"""
    embedder = kw.pop("embedder", FakeEmbedder({}))
    VectorIndex(chunks, embedder, tmp_path / store, generator=generator, questions_path=tmp_path / "questions.json", **kw).close()
    return embedder


def test_生成的問法加進embedding的文字(tmp_path: Path) -> None:
    generator = FakeGenerator()
    embedder = build(tmp_path, generator)

    first = CHUNKS[0]
    title = first.embed_text.splitlines()[0]
    assert generator.calls.count(first.embed_text) == 1
    assert f"{first.embed_text}\n{title} 問法1\n{title} 問法2\n{title} 問法3" in embedder.calls
    assert len(embedder.calls) == len(CHUNKS)


def test_問法存成以段落key為鍵的JSON檔_記錄hash與生成模型(tmp_path: Path) -> None:
    build(tmp_path, FakeGenerator())

    saved = json.loads((tmp_path / "questions.json").read_text(encoding="utf-8"))
    first = CHUNKS[0]
    title = first.embed_text.splitlines()[0]
    assert saved[first.key] == {"hash": chunk_hash(first), "model": "fake-gen", "questions": [f"{title} 問法{n}" for n in (1, 2, 3)]}
    assert list(saved) == sorted(c.key for c in CHUNKS)


def test_問法檔相符時刪掉向量庫重建也不呼叫LLM_向量相同(tmp_path: Path) -> None:
    first = build(tmp_path, FakeGenerator(), store="a")

    generator = FakeGenerator()
    second = build(tmp_path, generator, store="b")

    assert generator.calls == []
    assert sorted(second.calls) == sorted(first.calls)


def test_段落沒變不重新生成_改了只重新生成那一段(tmp_path: Path) -> None:
    build(tmp_path, FakeGenerator())

    generator = FakeGenerator()
    assert build(tmp_path, generator).calls == []
    assert generator.calls == []

    edited = CHUNKS.copy()
    edited[3] = dataclasses.replace(edited[3], text=edited[3].text + "\n新增一行。")
    generator = FakeGenerator()
    embedder = build(tmp_path, generator, edited)
    assert generator.calls == [edited[3].embed_text]
    assert len(embedder.calls) == 1 and embedder.calls[0].startswith(edited[3].embed_text + "\n")


def test_換生成模型時整庫重新生成(tmp_path: Path) -> None:
    build(tmp_path, FakeGenerator("舊模型"))

    generator = FakeGenerator("新模型")
    build(tmp_path, generator)

    saved = json.loads((tmp_path / "questions.json").read_text(encoding="utf-8"))
    assert len(generator.calls) == len(CHUNKS)
    assert {e["model"] for e in saved.values()} == {"新模型"}


def test_問法變了該段向量就重算(tmp_path: Path) -> None:
    build(tmp_path, FakeGenerator())
    qfile = tmp_path / "questions.json"
    saved = json.loads(qfile.read_text(encoding="utf-8"))
    saved[CHUNKS[0].key]["questions"] = ["手動改過的問法"]
    qfile.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    embedder = build(tmp_path, FakeGenerator())

    assert embedder.calls == [f"{CHUNKS[0].embed_text}\n手動改過的問法"]


def wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)


def test_生成失敗時只用段落算向量_不寫回檔案_仍能回答_冷卻後重試(tmp_path: Path) -> None:
    embedder, generator, clock = FakeEmbedder(FEE_RULES), FakeGenerator(), FakeClock()
    qfile = tmp_path / "questions.json"
    generator.broken = True
    rag = RAG(KB, embedder=embedder, vector_dir=tmp_path / "vectors", clock=clock, generator=generator, questions_path=qfile)
    try:
        assert sorted(embedder.calls) == sorted(c.embed_text for c in CHUNKS)  # 沒有問法
        assert not qfile.exists()  # 生成失敗不寫回
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
    assert build(tmp_path, generator).calls == []  # 補建成功後重開不再生成、不再重算
    assert generator.calls == []


def test_strict時生成失敗就報錯(tmp_path: Path) -> None:
    generator = FakeGenerator()
    generator.broken = True
    with pytest.raises(RuntimeError):
        build(tmp_path, generator, strict=True)


def test_知識庫載入不會把問法檔當成段落(tmp_path: Path) -> None:
    (tmp_path / "01.md").write_text("# 文件\n\n## 段落\n內容。\n", encoding="utf-8")
    (tmp_path / "questions.json").write_text('{"x": {"questions": ["## 不是段落"]}}', encoding="utf-8")

    assert [c.section for c in load_chunks(tmp_path)] == ["段落"]
