"""退回機制的共用判斷：rag.py 與 vector_index.py 遇到向量檢索用不了時都走這裡。"""
from __future__ import annotations


def degrade(strict: bool, message: str, e: Exception) -> None:
    """向量檢索用不了：strict（eval）時丟出例外，不默默退回 BM25；平常只印出原因。"""
    if strict:
        raise e
    print(message, repr(e))
