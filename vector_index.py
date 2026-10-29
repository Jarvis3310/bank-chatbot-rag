"""向量索引：知識庫段落的 embedding 存在 Qdrant 本機模式，負責增量同步、查詢逾時與失敗冷卻。

向量檢索用不了（API 錯誤、逾時、冷卻中、向量庫全空）時 search 回 None，由 RAG 改走退回模式；
strict（eval 用）時改成丟例外。
專案慣例是不裝廠商 SDK、直接打 REST；qdrant-client 是例外，因為 Qdrant 本機模式（不架伺服器）只有它提供。
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from embedding import Embedder

if TYPE_CHECKING:
    from rag import Chunk

# 向量相似度門檻。eval（large）：範圍外最高 0.312、口語題應命中段落最低 0.414，0.32–0.41 都達標，取中間偏高
MIN_SIMILARITY = 0.37
QUERY_TIMEOUT = 3.0  # 問題的 embedding 超過幾秒就放棄
COOLDOWN = 60.0  # 失敗後幾秒內不再試
CLOSE_TIMEOUT = 5.0  # close 最多等背景補建幾秒（embedding 卡住時不無限等）
COLLECTION = "chunks"


def point_id(chunk: Chunk) -> str:
    """向量庫的 point id，由「來源檔＋章節標題」決定；章節改名就是另一段。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.key))


def chunk_hash(chunk: Chunk) -> str:
    """拿去 embedding 的整段文字的指紋，內容一改就變。"""
    return hashlib.sha256(chunk.embed_text.encode("utf-8")).hexdigest()


class Point(NamedTuple):
    """知識庫目前的一段，在向量庫裡應有的樣子。"""

    id: str  # point id
    index: int  # 段落編號
    chunk: Chunk
    hash: str


def degrade(strict: bool, message: str, e: Exception) -> None:
    """向量檢索用不了：strict（eval）時丟出例外，不默默退回 BM25；平常只印出原因。"""
    if strict:
        raise e
    print(message, repr(e))


class VectorIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        path: str | Path,
        clock: Callable[[], float] = time.monotonic,
        strict: bool = False,
    ):
        """打開向量庫並同步段落。向量庫被別的程式鎖住時丟例外；embedding 失敗不丟（strict 除外），冷卻後補建。"""
        self.chunks = chunks
        self.embedder = embedder
        self.clock = clock
        self.strict = strict
        self.store = QdrantClient(path=str(path))
        self.lock = threading.Lock()  # 背景補建和查詢不能同時操作 Qdrant 本機模式
        self.search_failed_at: float | None = None
        self.sync_failed_at: float | None = None
        self.has_vectors = False
        self.rebuilding: threading.Thread | None = None
        self.closed = False  # 關閉後不再查詢、補建，背景補建也不再寫入
        self._sync()

    def close(self) -> None:
        """釋放向量庫資料夾的鎖（Qdrant 本機模式同一時間只能有一個程式打開）。先等背景補建結束，最多 CLOSE_TIMEOUT 秒。"""
        self.closed = True
        if self.rebuilding is not None:
            self.rebuilding.join(CLOSE_TIMEOUT)
        with self.lock:
            self.store.close()

    def search(self, query: str) -> list[tuple[int, float]] | None:
        """過門檻的段落（段落編號, 相似度），相似度高的在前；向量檢索用不了時回 None。"""
        if self.closed:
            return None
        # 失敗後冷卻一段時間不再試，避免斷網時每題都多等 QUERY_TIMEOUT 秒
        if self.search_failed_at is not None and self.clock() - self.search_failed_at < COOLDOWN:
            return None
        # 同步失敗過：冷卻時間到就重試。strict 同步重試、失敗就報錯；平常在背景補建，不讓這一題等
        if self.sync_failed_at is not None and self.clock() - self.sync_failed_at >= COOLDOWN:
            if self.strict:
                self._sync()
            else:
                self._sync_in_background()
        if not self.has_vectors:
            return None
        try:
            vec = self._embed_query(query)
            with self.lock:
                points = self.store.query_points(COLLECTION, query=vec, limit=len(self.chunks)).points
        except Exception as e:  # API 錯誤、逾時
            degrade(self.strict, "向量檢索失敗，改用退回模式：", e)
            self.search_failed_at = self.clock()
            return None
        self.search_failed_at = None
        return [(p.payload["index"], p.score) for p in points if p.score >= MIN_SIMILARITY]

    def _sync(self) -> None:
        """只重算 hash 變了或新加的段落，並移除已不在知識庫裡的段落；embedding 模型換了就整庫重算。

        重算失敗不丟例外：沒變的段落沿用舊向量，沒重算到的段落從向量庫移除，冷卻 COOLDOWN 秒後由下一題重試。
        """
        current = [Point(point_id(c), i, c, chunk_hash(c)) for i, c in enumerate(self.chunks)]
        model = self.embedder.model
        stored: dict[str, str] = {}  # point id → hash
        with self.lock:
            if self.closed:
                return
            if self.store.collection_exists(COLLECTION):
                models: set[str | None] = set()
                offset = None
                while True:
                    points, offset = self.store.scroll(COLLECTION, limit=256, offset=offset, with_payload=True)
                    stored.update({str(p.id): p.payload.get("hash", "") for p in points})
                    models.update(p.payload.get("model") for p in points)
                    if offset is None:
                        break
                # 換了 embedding 模型（或舊版沒記模型）：舊向量的維度、語意都對不上，整庫重算
                if models - {model}:
                    # 先刪光 point：Windows 上本機模式刪不掉還開著的資料檔，重建同名 collection 會讀回舊 point
                    self.store.delete(COLLECTION, points_selector=list(stored))
                    self.store.delete_collection(COLLECTION)
                    stored = {}
            # 沒變的段落只更新段落編號（知識庫增刪段落後編號會位移）
            for pt in current:
                if stored.get(pt.id) == pt.hash:
                    self.store.set_payload(COLLECTION, payload={"index": pt.index}, points=[pt.id])

        current_ids = {pt.id for pt in current}
        stale = [pid for pid in stored if pid not in current_ids]
        todo = [pt for pt in current if stored.get(pt.id) != pt.hash]
        vectors: list[list[float]] = []
        if todo:  # 算 embedding 時不鎖向量庫，補建期間其他問題照常查詢
            try:
                vectors = self.embedder.embed([pt.chunk.embed_text for pt in todo])
            except Exception as e:  # noqa: BLE001 — 啟動不能因為 embedding 失敗而中斷
                if self.closed:
                    return
                degrade(self.strict, f"向量重算失敗，{len(todo)} 段先只用 BM25 檢索：", e)
                stale += [pt.id for pt in todo if pt.id in stored]
                self.sync_failed_at = self.clock()
            else:
                self.sync_failed_at = None

        with self.lock:
            if self.closed:  # 等 embedding 時服務已關閉
                return
            if vectors:
                if not self.store.collection_exists(COLLECTION):
                    self.store.create_collection(COLLECTION, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
                self.store.upsert(
                    COLLECTION,
                    points=[
                        PointStruct(id=pt.id, vector=v, payload={"index": pt.index, "key": pt.chunk.key, "hash": pt.hash, "model": model})
                        for pt, v in zip(todo, vectors)
                    ],
                )
            if stale:
                self.store.delete(COLLECTION, points_selector=stale)
            self.has_vectors = self.store.collection_exists(COLLECTION) and self.store.count(COLLECTION).count > 0

    def _sync_in_background(self) -> None:
        """在背景補建向量（同一時間最多一個），這一題不等它，照當下的向量庫回答。"""
        if self.rebuilding is not None and self.rebuilding.is_alive():
            return
        self.sync_failed_at = None  # 補建中不再觸發；失敗時 _sync 會重新記下時間

        def run() -> None:
            try:
                self._sync()
            except Exception as e:  # noqa: BLE001 — 背景補建的錯誤不能影響服務
                if not self.closed:
                    print("背景補建向量失敗：", repr(e))
                    self.sync_failed_at = self.clock()

        self.rebuilding = threading.Thread(target=run, daemon=True)
        self.rebuilding.start()

    def _embed_query(self, query: str) -> list[float]:
        """問題的 embedding，超過 QUERY_TIMEOUT 秒就放棄（卡住的請求留在背景執行緒自己結束）。"""
        result: dict = {}

        def run() -> None:
            try:
                result["vec"] = self.embedder.embed([query])[0]
            except Exception as e:
                result["error"] = e

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(QUERY_TIMEOUT)
        if worker.is_alive():
            raise TimeoutError(f"embedding 超過 {QUERY_TIMEOUT:g} 秒")
        if "error" in result:
            raise result["error"]
        return result["vec"]
