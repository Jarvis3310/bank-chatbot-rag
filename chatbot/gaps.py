"""知識庫缺口清單：RAG＋LLM 機器人讀完段落後判斷「段落沒有答案」的問題，一行一筆（JSON Lines）記在本機檔案。

客戶問題屬客戶資料，缺口資料夾不進版控、不打包進 image。寫入位置：環境變數 GAP_LOG_PATH，預設 <專案>/gaps/kb_gaps.jsonl；
eval 等其他用途直接 GapLog(自訂路徑) 即可。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "gaps" / "kb_gaps.jsonl"

SOURCE_WEB = "web"
SOURCE_LINE = "line"


class GapLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()  # LINE 每個事件一條執行緒，避免兩筆寫在同一行

    @classmethod
    def from_env(cls) -> GapLog:
        return cls(os.getenv("GAP_LOG_PATH") or DEFAULT_PATH)

    def record(self, source: str, user: str, question: str, rewritten_question: str) -> None:
        """追加一筆。寫不進去（權限、磁碟）只印 log，不影響回覆客戶。"""
        row = {
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": source,
            "user": user,
            "question": question,
            "rewritten_question": rewritten_question,
        }
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as e:
            print("知識庫缺口寫入失敗：", repr(e))
