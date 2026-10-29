# 兩津商業銀行 AI 客服（RAG + LINE Bot demo）

客戶在 LINE 問問題 → 系統從知識庫找出最相關的段落 → LLM 只根據這些段落回答，並附上出處；知識庫沒有的就回答查不到，不亂編。

知識庫（`kb/`）是虛構的「兩津商業銀行」客服資料，共 7 份文件、36 個段落。

```
LINE / 網頁 ──► app.py ──► rag.py：檢索最相關的 4 段 ──► llm.py：依段落回答 ──► 回覆 + 參考資料
```

| 檔案 | 用途 |
|---|---|
| `kb/*.md` | 知識庫。每個 `## 標題` 是一段，直接改或新增 .md 就會生效（重啟後） |
| `rag.py` | 切段、中文檢索（BM25，不需 embedding）、組 prompt、解析出處 |
| `llm.py` | 可切換 LLM：Claude / OpenAI / Azure OpenAI / none |
| `app.py` | Flask 服務：LINE webhook、網頁聊天 |
| `eval.py` | 16 題評測：退回模式、純 BM25 兩欄的檢索命中，以及答案是否含關鍵數字 |
| `tests/` | 單元測試，執行 `python -m pytest` |

---

## 1. 在電腦上跑起來（Windows）

需要 Python 3.10 以上。在專案資料夾開 PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

打開 http://localhost:8000 就是網頁聊天。這時 `.env` 的 `LLM_PROVIDER=none`，系統會直接回最相關的段落，可以先確認檢索正常。

> 公司網路若 `pip install` 失敗，通常是要設定 proxy：`pip install --proxy http://公司proxy:port -r requirements.txt`。
> 程式呼叫 LLM 和 LINE 時，也會自動使用環境變數 `HTTPS_PROXY`。

## 2. 接上 LLM

編輯 `.env`，三選一：

```ini
# Claude
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Azure OpenAI（DEPLOYMENT 是 Azure 上的「部署名稱」）
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<資源名稱>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=...
```

預設模型：Claude 用 `claude-sonnet-5-5`、OpenAI 用 `gpt-4o-mini`，要換就設定 `LLM_MODEL=`。重啟 `python app.py` 後，再跑一次評測：

```powershell
python eval.py --answer
```

## 3. 建立 LINE 官方帳號（約 10 分鐘）

1. 到 [LINE Official Account Manager](https://manager.line.biz/) 用 LINE 帳號登入，建立一個官方帳號（名稱可取「兩津商業銀行客服 demo」）。
2. 進入該帳號 › 右上「設定」› **Messaging API** › 啟用。第一次會要求建立 Provider，隨便取名即可。
3. 在同一頁的 **回應設定**：
   - 「聊天」關閉
   - 「自動回應訊息」關閉
   - 「Webhook」開啟
4. 到 [LINE Developers Console](https://developers.line.biz/console/) › 選剛剛的 Provider › 點 channel：
   - **Basic settings** 分頁 › 複製 `Channel secret`
   - **Messaging API** 分頁 › 最下方 `Channel access token` › Issue › 複製
5. 填進 `.env`：
   ```ini
   LINE_CHANNEL_SECRET=...
   LINE_CHANNEL_ACCESS_TOKEN=...
   ```
6. 在 Messaging API 分頁掃 QR code，用手機把官方帳號加為好友。

## 4. 讓 LINE 連到你的電腦

LINE 需要一個公開的 HTTPS 網址。用 ngrok（或 cloudflared）把本機的 8000 port 開出去：

```powershell
# ngrok：到 ngrok.com 註冊、下載後
ngrok config add-authtoken <你的token>
ngrok http 8000

# 或 cloudflared（免註冊）
cloudflared tunnel --url http://localhost:8000
```

把它給你的 `https://xxxx.ngrok-free.app` 加上 `/callback`，貼到 LINE Developers Console › Messaging API › **Webhook URL**，按 **Verify** 出現 Success，並確認 **Use webhook** 已開啟。

> ngrok 免費版每次重開網址都會變，重開後要重新貼 Webhook URL。
> 公司網路若擋 ngrok，可以改用手機熱點，或直接用網頁聊天版 demo。

## 5. Demo 建議腳本

| 想展示的 | 問法 |
|---|---|
| 口語也聽得懂 | 「卡不見了怎麼辦」（知識庫寫的是「遺失」） |
| 精準數字 + 出處 | 「網銀轉帳一天最多轉多少？」 |
| 追問能接上 | 接著問「那手續費呢？」 |
| 不會亂答 | 「可以幫我買台積電股票嗎？」 |
| 防詐話術 | 「銀行打電話叫我給 OTP 是真的嗎」 |
| 改知識庫立即生效 | 現場改 `kb/` 某個數字 → 重啟 → 再問一次 |

網頁版每個回答都能點開「檢索到的段落」，可以讓聽眾看到 AI 拿到了哪些資料，說明它為什麼不會亂編。

輸入「重來」可清除該使用者的對話紀錄。

## 6. 這個 demo 刻意省略的（正式專案要補）

- 檢索：知識庫大了要改用向量檢索（embedding）或混合檢索
- 文件來源：PDF、Word、網頁的解析與定期更新
- 對話紀錄：目前存在記憶體，重啟就清空
- 部署、監控、權限、個資遮罩、轉真人客服
