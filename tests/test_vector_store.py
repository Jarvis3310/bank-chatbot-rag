"""向量庫增量更新：重啟時只重算有變的段落，重算失敗不阻擋啟動。"""
import shutil
import time
from pathlib import Path

from conftest import FEE_Q, FEE_RULES, KB, BrokenAtStartup, FakeClock, FakeEmbedder, GatedAfterStartup, wait_until
from rag import RAG


def copy_kb(tmp_path: Path) -> Path:
    """複製一份知識庫到暫存資料夾，測試才能改內容。"""
    kb = tmp_path / "kb"
    shutil.copytree(KB, kb)
    return kb


def test_知識庫不變時重啟不重算任何段落(tmp_path: Path) -> None:
    kb, store = copy_kb(tmp_path), tmp_path / "vectors"
    RAG(kb, embedder=FakeEmbedder({}), vector_dir=store).close()

    embedder = FakeEmbedder({})
    RAG(kb, embedder=embedder, vector_dir=store).close()

    assert embedder.calls == []


def edit_kb(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_改一段後重建只重算那一段(tmp_path: Path) -> None:
    kb, store = copy_kb(tmp_path), tmp_path / "vectors"
    RAG(kb, embedder=FakeEmbedder({}), vector_dir=store).close()
    edit_kb(kb / "02_transfer_fees.md", "ATM 轉帳（非約定）：每日上限 3 萬元。", "ATM 轉帳（非約定）：每日上限 5 萬元。")

    embedder = FakeEmbedder({})
    RAG(kb, embedder=embedder, vector_dir=store).close()

    assert len(embedder.calls) == 1
    assert embedder.calls[0].startswith("轉帳、提款與手續費 轉帳限額")
    assert "ATM 轉帳（非約定）：每日上限 5 萬元。" in embedder.calls[0]


def test_刪一段後重建該段檢索不到(tmp_path: Path) -> None:
    kb, store = copy_kb(tmp_path), tmp_path / "vectors"
    RAG(kb, embedder=FakeEmbedder(FEE_RULES), vector_dir=store).close()
    path = kb / "02_transfer_fees.md"
    text = path.read_text(encoding="utf-8")
    start = text.index("## 跨行轉帳手續費")
    end = text.index("\n## ", start + 1)
    path.write_text(text[:start] + text[end + 1 :], encoding="utf-8")

    rag = RAG(kb, embedder=FakeEmbedder(FEE_RULES), vector_dir=store)
    hits = rag.retrieve(FEE_Q)
    rag.close()

    assert "跨行轉帳手續費" not in [h.chunk.section for h in hits]
    # 只有被刪的那段含假 embedding 的關鍵字；它的舊向量若還在，會冒充成別段出現
    assert all(h.similarity is None or h.similarity < 0.9 for h in hits)


def test_重算失敗仍能啟動且沒重算的段落可用BM25檢索(tmp_path: Path) -> None:
    kb, store = copy_kb(tmp_path), tmp_path / "vectors"
    RAG(kb, embedder=FakeEmbedder(FEE_RULES), vector_dir=store).close()
    edit_kb(kb / "02_transfer_fees.md", "ATM 轉帳（非約定）：每日上限 3 萬元。", "ATM 轉帳（非約定）：每日上限 5 萬元。")

    embedder = BrokenAtStartup(FEE_RULES)
    rag = RAG(kb, embedder=embedder, vector_dir=store)
    embedder.broken = False
    changed = rag.retrieve("ATM 轉帳每日上限多少")
    unchanged = rag.retrieve(FEE_Q)
    rag.close()

    hit = next(h for h in changed if h.chunk.section == "轉帳限額")
    assert hit.bm25 is not None and hit.similarity is None
    # 沒變的段落沿用舊向量
    assert unchanged[0].chunk.section == "跨行轉帳手續費" and unchanged[0].similarity > 0.9


def test_第一次啟動就重算失敗時向量庫全空_改用退回模式(tmp_path: Path) -> None:
    # 「不見」要靠同義詞表才對得到「遺失」；純 BM25 的第一名會是「轉錯帳怎麼辦」
    embedder = BrokenAtStartup({})
    rag = RAG(copy_kb(tmp_path), embedder=embedder, vector_dir=tmp_path / "vectors")
    hits = rag.retrieve("信用卡不見了怎麼辦？")
    rag.close()

    assert hits[0].chunk.section == "信用卡遺失或被盜刷"
    assert all(h.similarity is None for h in hits)


def wait_for_similarity(rag: RAG, query: str) -> list:
    """背景補建完成前一直是退回模式，輪詢到出現向量相似度為止。"""
    assert wait_until(lambda: (hits := rag.retrieve(query)) and hits[0].similarity is not None), "背景補建沒有完成"
    return rag.retrieve(query)


def test_建庫失敗後冷卻時間到_背景補建期間發問立刻用退回模式_補建完成回到混合檢索(tmp_path: Path) -> None:
    embedder, clock = GatedAfterStartup(FEE_RULES), FakeClock()
    rag = RAG(copy_kb(tmp_path), embedder=embedder, vector_dir=tmp_path / "vectors", clock=clock)
    embedder.broken = False
    try:
        clock.now += 59
        rag.retrieve(FEE_Q)
        assert embedder.calls == []  # 冷卻中不重試

        clock.now += 1.5
        t0 = time.monotonic()
        hits = rag.retrieve("信用卡不見了怎麼辦？")  # 觸發背景補建，補建卡在 gate
        rag.retrieve("信用卡不見了怎麼辦？")  # 補建中再問一題，不會再開一次補建
        assert time.monotonic() - t0 < 1.0
        assert hits[0].chunk.section == "信用卡遺失或被盜刷" and hits[0].similarity is None

        embedder.gate.set()
        hits = wait_for_similarity(rag, FEE_Q)
        assert hits[0].chunk.section == "跨行轉帳手續費" and hits[0].similarity > 0.9
        chunk_texts = {c.embed_text for c in rag.chunks}
        assert sum(c in chunk_texts for c in embedder.calls) == len(rag.chunks)  # 只補建一次
    finally:
        embedder.gate.set()
        rag.close()


def test_第一次啟動就斷網_網路恢復後沒人發問也在60秒後自動回到混合檢索(tmp_path: Path) -> None:
    embedder, clock = BrokenAtStartup(FEE_RULES), FakeClock()
    rag = RAG(copy_kb(tmp_path), embedder=embedder, vector_dir=tmp_path / "vectors", clock=clock)
    try:
        assert rag.retrieve(FEE_Q)[0].similarity is None  # 斷網：退回模式

        embedder.broken = False
        clock.now += 60
        assert wait_until(lambda: rag.vectors.has_vectors)  # 這段時間不發問，等背景自己補建

        hits = rag.retrieve(FEE_Q)
        assert hits[0].chunk.section == "跨行轉帳手續費" and hits[0].similarity > 0.9
    finally:
        rag.close()
