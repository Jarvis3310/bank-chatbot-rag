"""向量索引：同步段落、查詢逾時、失敗冷卻、背景補建。用不了時 search 回 None。"""
import dataclasses
import threading
import time
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

import vector_index

from conftest import CHUNKS, FEE_Q, FEE_RULES, BreakableEmbedder, BrokenAtStartup, FakeClock, FakeEmbedder, GatedAfterStartup, wait_until
from rag import Chunk
from vector_index import VectorIndex


def section_of(hits: list[tuple[int, float]], chunks: list[Chunk] = CHUNKS) -> str:
    return chunks[hits[0][0]].section


def test_搜尋回傳過門檻的段落編號與相似度(tmp_path: Path) -> None:
    index = VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES), tmp_path)

    hits = index.search(FEE_Q)
    index.close()

    assert section_of(hits) == "跨行轉帳手續費" and hits[0][1] > 0.9
    assert all(sim >= vector_index.MIN_SIMILARITY for _, sim in hits)


def test_重開時只重算內容有變的段落(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder({}), tmp_path).close()
    edited = CHUNKS.copy()
    edited[0] = dataclasses.replace(edited[0], text=edited[0].text + "\n新增一行。")

    embedder = FakeEmbedder({})
    VectorIndex(edited, embedder, tmp_path).close()

    assert embedder.calls == [edited[0].embed_text]


def test_換embedding模型時整庫重算_不混用不同維度(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES, model="舊模型"), tmp_path).close()

    embedder = FakeEmbedder(FEE_RULES, model="新模型", dim=48)
    index = VectorIndex(CHUNKS, embedder, tmp_path)
    hits = index.search(FEE_Q)
    index.close()

    assert sorted(embedder.calls[:-1]) == sorted(c.embed_text for c in CHUNKS)  # 最後一筆是問題
    assert section_of(hits) == "跨行轉帳手續費" and hits[0][1] > 0.9


def test_換模型重算後再重開不再重算(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES, model="舊模型"), tmp_path).close()
    VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES, model="新模型", dim=48), tmp_path).close()

    embedder = FakeEmbedder(FEE_RULES, model="新模型", dim=48)
    index = VectorIndex(CHUNKS, embedder, tmp_path)
    hits = index.search(FEE_Q)
    index.close()

    assert embedder.calls == [FEE_Q]
    assert section_of(hits) == "跨行轉帳手續費" and hits[0][1] > 0.9


def test_舊版沒記模型的向量庫整庫重算(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder({}), tmp_path).close()
    store = QdrantClient(path=str(tmp_path))  # 模擬升級前建的向量庫：payload 沒有 model
    points, _ = store.scroll(vector_index.COLLECTION, limit=len(CHUNKS), with_payload=True)
    store.delete_payload(vector_index.COLLECTION, keys=["model"], points=[p.id for p in points])
    store.close()

    embedder = FakeEmbedder({})
    VectorIndex(CHUNKS, embedder, tmp_path).close()

    assert len(embedder.calls) == len(CHUNKS)


def test_知識庫拿掉的段落從向量庫移除(tmp_path: Path) -> None:
    VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES), tmp_path).close()
    remaining = [c for c in CHUNKS if c.section != "跨行轉帳手續費"]

    index = VectorIndex(remaining, FakeEmbedder(FEE_RULES), tmp_path)
    hits = index.search(FEE_Q)
    index.close()

    assert all(sim < 0.9 for _, sim in hits)


def test_查詢embedding卡住時2秒多就回None(tmp_path: Path) -> None:
    embedder = BreakableEmbedder()
    index = VectorIndex(CHUNKS, embedder, tmp_path)
    embedder.broken = "slow"

    t0 = time.monotonic()
    assert index.search(FEE_Q) is None
    assert time.monotonic() - t0 < 2.5


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
        assert wait_until(lambda: index.search(FEE_Q) is not None)
        assert section_of(index.search(FEE_Q)) == "跨行轉帳手續費"
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


def start_gated_rebuild(tmp_path: Path) -> tuple[VectorIndex, GatedAfterStartup, FakeClock]:
    """建庫失敗、冷卻時間到後觸發背景補建，補建卡在 gate。"""
    embedder, clock = GatedAfterStartup(FEE_RULES), FakeClock()
    index = VectorIndex(CHUNKS, embedder, tmp_path, clock=clock)
    embedder.broken = False
    clock.now += 60
    assert embedder.waiting.wait(5)  # 沒人發問，背景自己開始補建
    return index, embedder, clock


def test_建庫失敗後沒人發問_冷卻時間到背景自動補建(tmp_path: Path) -> None:
    embedder, clock = BrokenAtStartup(FEE_RULES), FakeClock()
    index = VectorIndex(CHUNKS, embedder, tmp_path, clock=clock)
    embedder.broken = False
    try:
        clock.now += 59
        time.sleep(0.3)
        assert not index.has_vectors  # 冷卻中不重試

        clock.now += 1.5
        assert wait_until(lambda: index.has_vectors)  # 全程沒有呼叫 search
        assert section_of(index.search(FEE_Q)) == "跨行轉帳手續費"
    finally:
        index.close()


def test_背景補建成功時印出已回到混合檢索_正常啟動建庫不印(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    VectorIndex(CHUNKS, FakeEmbedder(FEE_RULES), tmp_path / "ok").close()
    assert "向量庫已重建" not in capsys.readouterr().out

    embedder, clock = BrokenAtStartup(FEE_RULES), FakeClock()
    index = VectorIndex(CHUNKS, embedder, tmp_path / "retry", clock=clock)
    embedder.broken = False
    try:
        capsys.readouterr()
        clock.now += 60
        assert wait_until(lambda: index.has_vectors)
        index.rebuilding.join(5)  # print 在 has_vectors 之後，等背景重試結束再看輸出
        assert "向量庫已重建，回到混合檢索" in capsys.readouterr().out
    finally:
        index.close()


def test_背景重試又失敗時再隔60秒重試_同一時間最多一個(tmp_path: Path) -> None:
    embedder, clock = GatedAfterStartup(FEE_RULES), FakeClock()
    index = VectorIndex(CHUNKS, embedder, tmp_path, clock=clock)
    embedder.broken = False
    try:
        clock.now += 60
        assert embedder.waiting.wait(5)  # 第一次重試開始，卡在 gate
        clock.now += 600  # 重試還沒結束，再久也不會再開第二個
        time.sleep(0.3)
        assert embedder.batches == 2  # 啟動那次加上正在卡住的這次，沒有第三次
        embedder.broken = True
        embedder.gate.set()  # 第一次重試失敗（gate 之後 embedding 報錯）
        assert wait_until(lambda: index.sync_failed_at == clock.now)

        embedder.broken = False
        calls = len(embedder.calls)
        clock.now += 59
        time.sleep(0.3)
        assert len(embedder.calls) == calls  # 失敗後重新計 60 秒

        clock.now += 1.5
        assert wait_until(lambda: index.has_vectors)
    finally:
        embedder.gate.set()
        index.close()


def test_沒有失敗時close不會被背景重試卡住(tmp_path: Path) -> None:
    embedder, clock = BrokenAtStartup(FEE_RULES), FakeClock()
    index = VectorIndex(CHUNKS, embedder, tmp_path, clock=clock)  # 失敗，背景正在等冷卻

    t0 = time.monotonic()
    index.close()

    assert time.monotonic() - t0 < 1.0
    assert not index.rebuilding.is_alive()


def test_補建中close會等補建結束_不拋例外也不印錯誤_之後不再補建(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index, embedder, clock = start_gated_rebuild(tmp_path)
    capsys.readouterr()
    threading.Timer(0.2, embedder.gate.set).start()

    index.close()
    calls = len(embedder.calls)
    clock.now += 120
    assert index.search(FEE_Q) is None

    time.sleep(0.5)  # 補建若沒等完，這段時間內會寫入已關閉的向量庫而印錯誤
    assert capsys.readouterr().out == ""
    assert len(embedder.calls) == calls


def test_embedding卡住時close不無限等_之後補建也不寫入不印錯誤(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vector_index, "CLOSE_TIMEOUT", 0.2)
    index, embedder, _ = start_gated_rebuild(tmp_path)
    capsys.readouterr()

    t0 = time.monotonic()
    index.close()
    assert time.monotonic() - t0 < 1.0

    embedder.gate.set()  # 卡住的補建這時才回來
    index.rebuilding.join(5)
    assert capsys.readouterr().out == ""


class EmptyEmbedder(FakeEmbedder):
    """API 回了空的結果：沒有任何向量寫進向量庫。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []


def test_strict時向量庫全空要報錯_不默默退回(tmp_path: Path) -> None:
    index = VectorIndex(CHUNKS, EmptyEmbedder({}), tmp_path / "strict", strict=True)
    with pytest.raises(RuntimeError, match="向量庫"):
        index.search(FEE_Q)
    index.close()

    index = VectorIndex(CHUNKS, EmptyEmbedder({}), tmp_path / "normal")
    assert index.search(FEE_Q) is None  # 平常仍走退回模式
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
