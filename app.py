"""服務入口（Flask）。

- POST /callback  ：LINE webhook
- GET  /          ：網頁聊天（LINE 出狀況時的備案，也方便展示檢索過程）
- POST /api/chat  ：網頁聊天用的 API；body 的 bot 指定 "rag"（純 RAG）或 "rag_llm"（預設）
- GET  /health    ：檢查設定

啟動：python app.py   （預設 http://localhost:8000）
"""
from __future__ import annotations

import atexit
import base64
import hashlib
import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_file

load_dotenv()

from chatbot import llm  # noqa: E402
from chatbot.gaps import SOURCE_LINE, SOURCE_WEB, GapLog  # noqa: E402
from chatbot.rag import HOTLINE, RAG, format_for_line  # noqa: E402

BASE = Path(__file__).parent
app = Flask(__name__)

# 服務使用的 RAG：第一次用到才建立（import 時不碰 API 和向量庫），測試可以直接換掉 rag
rag: RAG | None = None
_rag_lock = threading.Lock()


def get_rag() -> RAG:
    global rag
    with _rag_lock:  # 向量庫資料夾只能被一個 client 打開，不能同時建兩個
        if rag is None:
            rag = RAG(BASE / "kb")
        return rag


@atexit.register
def close_rag() -> None:
    """程式結束前關閉向量庫，否則 Python 結束時 Qdrant 才關，會印「Python is likely shutting down」。沒用過就不建立。"""
    with _rag_lock:
        if rag is not None:
            rag.close()

# 兩個機器人：共用同一個 RAG（同一個向量庫連線），對話紀錄各記各的
BOT_RAG = "rag"  # 純 RAG 機器人
BOT_RAG_LLM = "rag_llm"  # RAG＋LLM 機器人（預設；LINE 也接這個）
BOTS = (BOT_RAG, BOT_RAG_LLM)

# RAG＋LLM 機器人每位使用者保留最近 3 輪對話，讓追問接得上（重啟後清空）。key 是 (機器人, 使用者)
NO_LLM = "none"  # 顯示用：這個機器人／這個服務沒有在用 LLM


def llm_label() -> str:
    return "openai" if llm.enabled() else NO_LLM


histories: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=3))


# 知識庫缺口清單：位置由 GAP_LOG_PATH 設定（預設 gaps/kb_gaps.jsonl），測試可以直接換掉
gap_log = GapLog.from_env()


def ask(session: str, question: str, bot: str = BOT_RAG_LLM, source: str = SOURCE_WEB):
    t0 = time.time()
    if question.strip() in {"重來", "清除", "reset"}:  # 兩個機器人都吃：清掉 RAG＋LLM 的紀錄（純 RAG 本來就不記）
        histories[(BOT_RAG_LLM, session)].clear()
        return None, 0.0
    if bot == BOT_RAG:  # 純 RAG：不記對話、不處理追問、不用 LLM
        return get_rag().answer_rag_only(question), time.time() - t0
    key = (bot, session)
    ans = get_rag().answer(question, list(histories[key]))
    if ans.gap:  # 知識庫缺口：記下來交給後台補資料
        gap_log.record(source, session, question, ans.rewritten_question)
    if not ans.error:  # LLM 失敗的錯誤訊息不算一輪對話
        histories[key].append((question, ans.text))
    return ans, time.time() - t0


# ---------- LINE ----------
LINE_API = "https://api.line.me/v2/bot"


def line_ready() -> bool:
    return bool(os.getenv("LINE_CHANNEL_SECRET") and os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))


def verify_signature(body: bytes, signature: str) -> bool:
    secret = os.environ["LINE_CHANNEL_SECRET"].encode()
    expected = base64.b64encode(hmac.new(secret, body, hashlib.sha256).digest()).decode()
    return hmac.compare_digest(expected, signature)


def line_post(path: str, payload: dict) -> None:
    r = requests.post(
        f"{LINE_API}{path}",
        headers={"Authorization": f"Bearer {os.environ['LINE_CHANNEL_ACCESS_TOKEN']}"},
        json=payload,
        timeout=10,
    )
    if r.status_code >= 400:
        print(f"LINE API 錯誤 {path} {r.status_code}：{r.text[:200]}")


def handle_line_event(ev: dict) -> None:
    if ev.get("type") != "message" or ev["message"].get("type") != "text":
        return
    user_id = ev["source"].get("userId", "unknown")
    # 顯示「輸入中…」動畫（一對一聊天才有效）
    line_post("/chat/loading/start", {"chatId": user_id, "loadingSeconds": 20})
    try:
        ans, secs = ask(user_id, ev["message"]["text"], source=SOURCE_LINE)
        text = "已清除對話紀錄，請問有什麼可以幫您？" if ans is None else format_for_line(ans)
        print(f"[LINE] {ev['message']['text']!r} → {secs:.1f}s")
    except Exception as e:  # demo 時避免使用者完全沒收到回應
        print("回答失敗：", repr(e))
        text = f"系統忙碌中，請稍後再試，或撥打客服專線 {HOTLINE}。"
    line_post("/message/reply", {"replyToken": ev["replyToken"], "messages": [{"type": "text", "text": text[:4900]}]})


@app.post("/callback")
def callback():
    if not line_ready():
        abort(503, "尚未設定 LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN")
    body = request.get_data()
    if not verify_signature(body, request.headers.get("X-Line-Signature", "")):
        abort(400, "簽章驗證失敗，請確認 LINE_CHANNEL_SECRET")
    # 先回 200 給 LINE，再在背景產生答案（LLM 可能要好幾秒）
    for ev in json.loads(body).get("events", []):
        threading.Thread(target=handle_line_event, args=(ev,), daemon=True).start()
    return "OK"


# ---------- 網頁聊天 ----------
@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True)
    bot = data.get("bot", BOT_RAG_LLM)
    if bot not in BOTS:
        return jsonify(error=f"bot 只能是 {' 或 '.join(BOTS)}"), 400
    ans, secs = ask(data.get("session_id", "web"), data.get("message", ""), bot)
    if ans is None:
        return jsonify(answer="已清除對話紀錄。", sources=[], retrieved=[], seconds=0)
    return jsonify(
        answer=ans.text,
        sources=[c.label for c in ans.sources],
        retrieved=[h.to_dict() for h in ans.retrieved],
        seconds=round(secs, 2),
        provider="openai" if bot == BOT_RAG_LLM else NO_LLM,
    )


@app.get("/")
def index():
    return send_file(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return jsonify(chunks=len(get_rag().chunks), llm=llm_label(), line=line_ready())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"知識庫段落數：{len(get_rag().chunks)}　LLM：{llm_label()}　LINE：{'已設定' if line_ready() else '未設定'}")
    print(f"網頁聊天：http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
