# 兩津商業銀行 AI 客服（RAG + LINE Bot demo）

客戶在 LINE 問問題 → 系統從知識庫找出最相關的段落 → LLM 只根據這些段落回答，並附上出處；知識庫沒有的就回答查不到，不亂編。

知識庫（`kb/`）是虛構的「兩津商業銀行」客服資料，共 8 份文件、37 個段落。

```
LINE / 網頁 ──► app.py ──► rag.py：混合檢索最相關的 4 段 ──► llm.py：依段落回答 ──► 回覆 + 參考資料
```

| 檔案 | 用途 |
|---|---|
| `kb/*.md` | 知識庫。每個 `## 標題` 是一段，直接改或新增 .md 就會生效（重啟後） |
| `kb/questions.json` | 每段的生成問法（進版控），建庫時自動補生成，改知識庫後記得一起 commit；知識庫只讀 `.md`，不會把它當成段落 |
| `rag.py` | 切段、混合檢索（向量＋BM25（含同義詞表），用 RRF 合併）、組 prompt、解析出處 |
| `vector_index.py` | 向量索引：知識庫段落的向量增量同步、查詢逾時、失敗冷卻與背景補建；用不了時讓 `rag.py` 改走退回模式 |
| `embedding.py` | 把文字轉成向量：OpenAI `text-embedding-3-large`，一律用 `OPENAI_API_KEY` |
| `questions.py` | 建庫時替每段生成 3 個客戶可能的口語問法，跟段落一起算向量：OpenAI `gpt-4o-mini`，一律用 `OPENAI_API_KEY` |
| `vector_store/` | 向量庫（Qdrant 本機模式），啟動時自動建立，換 embedding 或生成問法的模型時自動整庫重算，不進版控 |
| `llm.py` | 可切換 LLM：Claude / OpenAI / Azure OpenAI / none |
| `app.py` | Flask 服務：LINE webhook、網頁聊天 |
| `eval.py` | 16 題評測：退回模式、純 BM25、混合檢索三欄的檢索命中，以及答案是否含關鍵數字；另有 6 題口語題（同義詞表沒涵蓋的說法）與 20 題驗證題組（沒拿來調門檻，只回報）只測檢索；`--answer` 時另跑 14 題簡報題組 |
| `tests/` | 單元測試（用假 embedding，不呼叫 API），共用設定在 `tests/conftest.py`；安裝與執行見下方「跑測試」 |

---

## 1. 在電腦上跑起來（Windows）

需要 Python 3.10 以上。在專案資料夾開 PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# 編輯 .env 填入 OPENAI_API_KEY（混合檢索的 embedding 需要）
python app.py
```

打開 http://localhost:8000 就是網頁聊天。這時 `.env` 的 `LLM_PROVIDER=none`，系統會直接回最相關的段落，可以先確認檢索正常。

檢索預設用混合檢索，embedding 一律呼叫 OpenAI `text-embedding-3-large`，所以就算還沒接 LLM、只測檢索，也需要在 `.env` 設定 `OPENAI_API_KEY`（跟 `LLM_PROVIDER` 選哪家無關）。只測檢索可以跑 `python eval.py`，沒有 `OPENAI_API_KEY` 時會直接報錯。

建庫時還會額外呼叫 OpenAI `gpt-4o-mini`，替每段寫 3 個客戶可能的口語問法，跟段落一起算向量，讓口語問題比較容易對到正確段落（同樣用 `OPENAI_API_KEY`，跟 `LLM_PROVIDER` 無關）。從零生成（目前 37 段）每段呼叫一次，約 10 秒、生成問法約 0.002 美元（加上 embedding 約 0.003 美元）。生成失敗（沒額度、逾時、斷線）時，該段先只用段落算向量，服務照常，冷卻 60 秒後再重試。

生成的問法存在 `kb/questions.json`（進版控），每段記錄段落指紋、生成模型和問法。建庫時段落和生成模型都相符就直接用檔案裡的問法、不呼叫 LLM，所以刪掉 `vector_store/` 重建的結果每次都一樣（LLM 每次生成的問法不完全相同，會影響檢索排名）。改了知識庫後重建（重啟服務或跑 `eval.py`），改過或新增的段落會自動補生成並寫回這個檔案，**記得把 `kb/questions.json` 跟知識庫一起 commit**。生成失敗的段落不會寫進檔案，之後重試成功才補上。

> 服務（`python app.py`）跑著時不能同時跑 `eval.py`：Qdrant 本機模式會鎖住 `vector_store/` 資料夾，請先停掉服務再跑評測。

### 跑測試

測試相依（pytest）放在 `requirements-dev.txt`，它會連同 `requirements.txt` 一起安裝。在已啟用的 `.venv` 裡：

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

測試不需要 `.env` 或 `OPENAI_API_KEY`。

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

預設模型：Claude 用 `claude-sonnet-5-5`、OpenAI 用 `gpt-4o-mini`，要換就設定 `LLM_MODEL=`。不論 LLM 選哪家，`OPENAI_API_KEY` 都要設定（混合檢索的 embedding 用）。停掉服務後跑一次評測，再重啟 `python app.py`：

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

- 文件來源：PDF、Word、網頁的解析與定期更新
- 對話紀錄：目前存在記憶體，重啟就清空
- 部署、監控、權限、個資遮罩、轉真人客服
