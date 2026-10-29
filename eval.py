"""評測：檢查「有沒有找到對的段落」，設定 LLM 後也檢查「答案有沒有關鍵數字」。

檢索分數分三欄：退回模式（BM25＋同義詞）、純 BM25（不加同義詞，對照組）、混合檢索（服務實際使用）。
範圍外題目以「檢索結果為空」判定通過。答案只用混合檢索產生。
混合檢索用不了（沒有 OPENAI_API_KEY、embedding API 錯誤）時直接報錯結束，不會默默改用退回模式。
服務（app.py）跑著時不能同時跑 eval：Qdrant 本機模式會鎖住向量庫資料夾。

用法：
  python eval.py            # 只測檢索（需要 OPENAI_API_KEY 做 embedding）
  python eval.py --answer   # 也呼叫 LLM 產生答案（需在 .env 設定 LLM）
"""
import os
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

import llm  # noqa: E402
from rag import BM25_ONLY, FALLBACK, HYBRID, RAG  # noqa: E402

# (問題, 應該找到的章節, 答案應包含的字)；章節為 None 表示超出範圍，檢索結果應為空
CASES = [
    ("信用卡不見了怎麼辦？", "信用卡遺失或被盜刷", "0800-123-456"),
    ("卡片被偷刷，我要負擔多少？", "信用卡遺失或被盜刷", "3,000"),
    ("網銀轉帳一天最多可以轉多少？", "轉帳限額", "10 萬"),
    ("轉帳轉錯人了可以拿回來嗎", "轉錯帳怎麼辦", "2999-8888"),
    ("用 ATM 跨行轉帳要多少手續費", "跨行轉帳手續費", "15"),
    ("開戶要帶什麼證件？", "臨櫃開立台幣存款帳戶需要帶什麼", "雙證件"),
    ("可以用手機線上開戶嗎", "線上開立數位帳戶", "10 分鐘"),
    ("房貸最多可以貸幾成？", "房屋貸款", "8 成"),
    ("信貸提前還款要付違約金嗎", "提前清償", "1%"),
    ("換新手機後 app 要怎麼用", "換手機後如何綁定", "OTP"),
    ("網銀密碼打錯被鎖住了", "忘記密碼或使用者代號", "5 次"),
    ("你們幾點開門？", "營業時間", "9:00"),
    ("美金線上換匯有優惠嗎", "線上換匯優惠", "3 分"),
    ("信用卡繳款截止日是哪天", "繳款方式與期限", "20 日"),
    ("可以幫我買台積電股票嗎？", None, "查不到"),
    ("明天會下雨嗎", None, "查不到"),
]


# (欄位名稱, 檢索模式)
COLUMNS = [("退回模式", FALLBACK), ("純 BM25", BM25_ONLY), ("混合檢索", HYBRID)]
ANSWER_COLUMN = "混合檢索"  # 答案只用服務實際使用的模式產生


@dataclass
class Score:
    hit1: int = 0
    hitk: int = 0
    n_in: int = 0
    out_hits: int = 0  # 範圍外題目撞到段落的題數
    n_out: int = 0


def main(with_answer: bool) -> None:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        sys.exit("混合檢索需要 OPENAI_API_KEY（只測檢索也需要），請先在 .env 設定")
    try:
        rags = {name: RAG("kb", mode=mode) for name, mode in COLUMNS}
    except RuntimeError as e:
        sys.exit(f"混合檢索用不了：{e}")
    if with_answer and not llm.enabled():
        sys.exit("請先在 .env 設定 LLM_PROVIDER 與 API key")

    scores = {name: Score() for name in rags}
    ans_ok = 0
    times = []
    for q, expect, keyword in CASES:
        lines = [q]
        for name, rag in rags.items():
            sections = [h.chunk.section for h in rag.retrieve(q)]
            s = scores[name]
            if expect:
                s.n_in += 1
                s.hit1 += bool(sections) and sections[0] == expect
                s.hitk += expect in sections
                r_ok = expect in sections
            else:
                s.n_out += 1
                s.out_hits += bool(sections)
                r_ok = not sections  # 範圍外題目：檢索結果為空才算通過
            lines.append(f"   {'✅' if r_ok else '❌'} {name}：{' / '.join(sections) or '（無）'}")

        if with_answer:
            t0 = time.time()
            ans = rags[ANSWER_COLUMN].answer(q)
            times.append(time.time() - t0)
            ok = keyword in ans.text
            ans_ok += ok
            lines.append(f"   {'✅' if ok else '❌'} 答案（應含「{keyword}」，{times[-1]:.1f}s）：{ans.text[:120]}")
        print("\n".join(lines))

    top_k = rags[ANSWER_COLUMN].top_k
    print()
    for name, s in scores.items():
        print(f"【{name}】檢索第一名正確：{s.hit1}/{s.n_in}　前 {top_k} 名有找到：{s.hitk}/{s.n_in}　範圍外撞到段落：{s.out_hits}/{s.n_out}")
    if with_answer:
        print(f"\n答案含關鍵字（{ANSWER_COLUMN}）：{ans_ok}/{len(CASES)}　平均回應：{sum(times) / len(times):.1f} 秒（{llm.provider()}）")


if __name__ == "__main__":
    main("--answer" in sys.argv)
