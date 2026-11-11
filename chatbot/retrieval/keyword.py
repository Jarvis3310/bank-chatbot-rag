"""關鍵字檢索：字元 bigram 斷詞、同義詞表、BM25。退回模式只靠這裡，混合檢索也拿它跟向量檢索合併。"""
from __future__ import annotations

import math
import re
from collections import Counter

# 口語 → 知識庫用語。BM25 只比對字面，這張表補上常見說法。
SYNONYMS = {
    "不見": "遺失 掛失",
    "弄丟": "遺失 掛失",
    "掉了": "遺失 掛失",
    "被偷刷": "盜刷 冒用 掛失",
    "盜用": "盜刷 冒用",
    "匯錯": "轉錯帳 錯誤轉帳",
    "轉錯": "轉錯帳 錯誤轉帳",
    "上限": "限額",
    "額度": "限額 上限",
    "幾點": "營業時間",
    "開門": "營業時間",
    "電話": "客服專線",
    "客服": "客服專線",
    "app": "行動銀行",
    "手機銀行": "行動銀行",
    "網銀": "網路銀行",
    "換手機": "綁定 新手機",
    "鎖住": "鎖定 解鎖",
    "鎖卡": "鎖定",
    "買房": "房屋貸款",
    "房貸": "房屋貸款",
    "信貸": "信用貸款",
    "還清": "提前清償",
    "提前還": "提前清償",
    "銷戶": "結清",
    "關帳戶": "結清 銷戶",
    "換錢": "換匯 外幣",
    "美金": "美元",
    "日幣": "日圓",
    "詐騙": "防詐",
    "刷卡分期": "分期付款",
    "利息": "利率",
}

CJK = re.compile(r"[一-鿿]+")
WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def tokenize(text: str) -> list[str]:
    """中文切成單字 + 相鄰兩字（bigram），英數字保留整個詞。"""
    text = text.lower()
    tokens: list[str] = []
    for seg in CJK.findall(text):
        tokens.extend(seg)
        tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    tokens.extend(WORD.findall(text))
    return tokens


def expand_query(q: str) -> str:
    extra = [v for k, v in SYNONYMS.items() if k in q.lower()]
    return q + " " + " ".join(extra) if extra else q


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / len(docs)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out
