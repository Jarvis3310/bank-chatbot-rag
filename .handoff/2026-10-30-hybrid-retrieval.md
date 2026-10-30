# 交接：混合檢索收尾（2026-10-30）

混合檢索（spec #1）已經實作完成，也驗收完畢。PR #8 是 ready for review、可合併，但還沒合併。這份交接文件放在 `handoff/2026-10-30` branch，沒有放進 PR。

## 現況

- **PR**：https://github.com/Jarvis3310/bank-chatbot-rag/pull/8 ，內文有改動摘要、驗收數據與合併風險。
- **驗收結果**：#7 的留言 https://github.com/Jarvis3310/bank-chatbot-rag/issues/7#issuecomment-6519775569 。
- **跟原 spec 不同的決策**：分別記在 #1、#13、#14、#16 的留言。
- **Branch**：
  - `feat/hybrid-retrieval`：tip 235cbf8，已 push。
  - 本機的 worktree 和備份 branch 都已清掉。
- **還開著的 issue**：#1–#7、#9–#16，PR 合併後會自動關閉。

## 待辦

1. **合併 PR #8**：用「Create a merge commit」，不要 squash，要保留各 ticket 的 commit。
   - 合併後確認 #1–#7、#9–#16 都已自動關閉。
   - 確認後刪除遠端的 `feat/hybrid-retrieval`。
2. **在 demo 筆電實際斷一次網路**：之前的斷網測試都是用 Docker 模擬，還沒在筆電上真的關過 Wi-Fi。
   - 操作步驟：啟動 `python app.py` → 發問，網頁應顯示「相似度｜關鍵字」 → 關 Wi-Fi → 發問，網頁只顯示「關鍵字」，回應 < 3 秒 → 開 Wi-Fi，不必發問，約 60 秒後主控台會印「向量庫已重建，回到混合檢索」。
3. **測 LINE 端**：這次只用網頁和 eval 驗證過新的檢索，LINE webhook（`/callback`）還沒用新的檢索實際測過。
4. **更新簡報**：
   - 數據改用 #7 留言裡的表格。
   - 簡報題組照 #6 的定稿：一般、知識庫沒有、個資、別家銀行、誘導，共 14 題。
5. **在新機器上第一次啟動**：
   - 需要 `.env`，裡面要有 `OPENAI_API_KEY`。就算 `LLM_PROVIDER` 不是 openai 也要設，因為 embedding 要用。
   - 第一次啟動會建 `vector_store/`，大約 10 秒。問法直接讀 `kb/questions.json`，不會呼叫 LLM。
   - 服務跑著時不能同時跑 `eval.py`，因為 Qdrant 會鎖住資料夾。

## 已知限制（不用修，簡報被問到時可以說明）

- 驗證題組第一名 11/15。答錯的題目，正確段落都在前 4 名，LLM 回答時仍讀得到。
- 「知識庫查不到」的題目，大多仍會檢索到段落，要靠 LLM 拒答。`--answer` 和簡報題組都驗證過 LLM 會正確拒答。
- 改了 `kb/` 之後，重建時會自動補生成問法，記得把 `kb/questions.json` 一起 commit。

## 建議使用的 skill

- `dev:code-review`：合併前如果又有改動，再審一次。
- `dev:diagnosing-bugs`：LINE 端或實際斷網測試出問題時使用。
- `dev:tdd`：要修程式時使用。

## 其他

- 這次工作的暫存檔（實驗腳本、eval 輸出）放在舊機器的 scratchpad，換機器後讀不到。需要的數據都已經寫進 #7、#13、#14 的留言，驗證題目也在 #14 的內文。
