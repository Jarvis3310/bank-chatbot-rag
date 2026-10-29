"""向量索引：同步段落、查詢逾時、失敗冷卻、背景補建。用不了時 search 回 None。"""
import dataclasses
import time
from pathlib import Path

import pytest

from conftest import KB, BreakableEmbedder, BrokenAtStartup, FakeClock, FakeEmbedder, GatedAfterStartup
from rag import Chunk, load_chunks
from vector_index import VectorIndex

CHUNKS = load_chunks(KB)
FEE_RULES = {"跨行轉帳手續費": 0, "ATM 跨行": 0}
FEE_Q = "用 ATM 跨行轉帳要多少手續費"


def section_of(hits: list[tuple[int, float]], chunks: list[Chunk] = CHUNKS) -> str:
    return chunks[hits[0][0]].section


def test_搜尋回傳過門檻的段落編號與相似度(tmp_path: Path) -> None:
    index = VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES), tmp_path)

    hits = index.search(FEE_Q)
    index.close()

    assert section_of(hits) == "跨行轉帳手續費" and hits[0][1] > 0.9
    assert all(sim >= 0.4 for _, sim in hits)


def test_重開時只重算內容有變的段落(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder({}), tmp_path).close()
    edited = CHUNKS.copy()
    edited[0] = dataclasses.replace(edited[0], text=edited[0].text + "\n新增一行。")

    embedder = FakeEmbedder({})
    VectorIndex(edited, embedder, tmp_path).close()

    assert embedder.calls == [edited[0].embed_text]


def test_知識庫拿掉的段落從向量庫移除(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES), tmp_path).close()
    remaining = [c for c in CHUNKS if c.section != "跨行轉帳手續費"]

    index = VectorIndex(remaining, FakeEmbedder(FEE_RULES), tmp_path)
    hits = index.search(FEE_Q)
    index.close()

    assert all(sim < 0.9 for _, sim in hits)


def test_查詢embedding卡住時3秒內回None(tmp_path: Path) -> None:
    embedder = BreakableEmbedder()
    index = VectorIndex(CHUNKS, embedder, tmp_path)
    embedder.broken = "slow"

    t0 = time.monotonic()
    assert index.search(FEE_Q) is None
    assert time.monotonic() - t0 < 3.5


def test_查詢失敗後冷卻60秒內不再試(tmp_path: Path) -> None:
    embedder, clock = BreakableEmbedder(), FakeClock()
    index = VectorIndex(CHUNKS, embedder, tmp_path, clock=clock)
    embedder.broken = "error"
    assert index.search(FEE_Q) is None
    embedder.broken = None
    embedder.calls.clear()

    clock.now += 59
    assert index.search(FEE_Q) is None
    assert embedder.calls == []

    clock.now += 1.5
    assert index.search(FEE_Q) is not None
    assert embedder.calls == [FEE_Q]
    index.close()


def test_建庫失敗後冷卻時間到在背景補建_補建完成才搜得到(tmp_path: Path) -> None:
    embedder, clock = GatedAfterStartup(FEE_RULES), FakeClock()
    index = VectorIndex(CHUNKS, embedder, tmp_path, clock=clock)
    embedder.broken = False
    try:
        clock.now += 59
        assert index.search(FEE_Q) is None  # 向量庫全空
        assert embedder.calls == []  # 冷卻中不重試

        clock.now += 1.5
        t0 = time.monotonic()
        assert index.search(FEE_Q) is None  # 觸發背景補建，這一題不等
        assert index.search(FEE_Q) is None  # 補建中不會再開一次
        assert time.monotonic() - t0 < 1.0

        embedder.gate.set()
        deadline = time.monotonic() + 5
        while (hits := index.search(FEE_Q)) is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert section_of(hits) == "跨行轉帳手續費"
        assert sum(c.embed_text in embedder.calls for c in CHUNKS) == len(CHUNKS)  # 只補建一次
    finally:
        embedder.gate.set()
        index.close()


def test_背景補建算embedding時不佔住向量庫_查詢照常(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES), tmp_path).close()
    edited = [dataclasses.replace(c, text=c.text + "\n改版。") if c.section in ("營業時間", "客服專線") else c for c in CHUNKS]
    embedder, clock = GatedAfterStartup(FEE_RULES), FakeClock()
    index = VectorIndex(edited, embedder, tmp_path, clock=clock)  # 改過的兩段重算失敗，其他段沿用舊向量
    embedder.broken = False
    try:
        clock.now += 60
        t0 = time.monotonic()
        hits = index.search(FEE_Q)  # 觸發背景補建，補建卡在 gate
        assert time.monotonic() - t0 < 1.0
        assert section_of(hits, edited) == "跨行轉帳手續費"
    finally:
        embedder.gate.set()
        index.close()


def test_strict時失敗就丟例外(tmp_path: Path) -> None:
    with pytest.raises(ConnectionError):
        VectorIndex(CHUNKS, BrokenAtStartup({}), tmp_path / "startup", strict=True)

    embedder = BreakableEmbedder()
    index = VectorIndex(CHUNKS, embedder, tmp_path / "query", strict=True)
    embedder.broken = "error"
    with pytest.raises(RuntimeError):
        index.search(FEE_Q)
    index.close()
