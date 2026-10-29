"""評測：檢查「有沒有找到對的段落」，設定 LLM 後也檢查「答案有沒有關鍵數字」。

用法：
  python eval.py            # 只測檢索，不需要 API key
  python eval.py --answer   # 也呼叫 LLM 產生答案（需在 .env 設定 LLM）
"""
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import llm  # noqa: E402
from rag import RAG  # noqa: E402

# (問題, 應該找到的章節, 答案應包含的字)；章節為 None 表示超出範圍，應該拒答
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


def main(with_answer: bool) -> None:
    rag = RAG("kb")
    if with_answer and not llm.enabled():
        sys.exit("請先在 .env 設定 LLM_PROVIDER 與 API key")

    hit1 = hitk = ans_ok = n_in = 0
    times = []
    for q, expect, keyword in CASES:
        hits = rag.retrieve(q)
        sections = [c.section for c, _ in hits]
        if expect:
            n_in += 1
            hit1 += bool(sections) and sections[0] == expect
            hitk += expect in sections
            r_ok = expect in sections
        else:
            r_ok = True  # 範圍外題目由 LLM 判斷是否拒答
        line = f"{'✅' if r_ok else '❌'} {q}\n   檢索：{' / '.join(sections) or '（無）'}"

        if with_answer:
            t0 = time.time()
            ans = rag.answer(q)
            times.append(time.time() - t0)
            ok = keyword in ans.text
            ans_ok += ok
            line += f"\n   {'✅' if ok else '❌'} 答案（應含「{keyword}」，{times[-1]:.1f}s）：{ans.text[:120]}"
        print(line)

    print(f"\n檢索第一名正確：{hit1}/{n_in}　前 {rag.top_k} 名有找到：{hitk}/{n_in}")
    if with_answer:
        print(f"答案含關鍵字：{ans_ok}/{len(CASES)}　平均回應：{sum(times) / len(times):.1f} 秒（{llm.provider()}）")


if __name__ == "__main__":
    main("--answer" in sys.argv)
