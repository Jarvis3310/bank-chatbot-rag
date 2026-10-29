"""服務入口（Flask）。

- POST /callback  ：LINE webhook
- GET  /          ：網頁聊天（LINE 出狀況時的備案，也方便展示檢索過程）
- POST /api/chat  ：網頁聊天用的 API
- GET  /health    ：檢查設定

啟動：python app.py   （預設 http://localhost:8000）
"""
from __future__ import annotations

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

import llm  # noqa: E402
from rag import RAG, format_for_line  # noqa: E402

BASE = Path(__file__).parent
rag = RAG(BASE / "kb")
app = Flask(__name__)

# 每位使用者保留最近 3 輪對話，讓追問接得上（重啟後清空）
histories: dict[str, deque] = defaultdict(lambda: deque(maxlen=3))


def ask(session: str, question: str):
    t0 = time.time()
    if question.strip() in {"重來", "清除", "reset"}:
        histories[session].clear()
        return None, 0.0
    ans = rag.answer(question, list(histories[session]))
    histories[session].append((question, ans.text))
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
        ans, secs = ask(user_id, ev["message"]["text"])
        text = "已清除對話紀錄，請問有什麼可以幫您？" if ans is None else format_for_line(ans)
        print(f"[LINE] {ev['message']['text']!r} → {secs:.1f}s")
    except Exception as e:  # demo 時避免使用者完全沒收到回應
        print("回答失敗：", repr(e))
        text = "系統忙碌中，請稍後再試，或撥打客服專線 (02) 2999-8888。"
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
    ans, secs = ask(data.get("session_id", "web"), data.get("message", ""))
    if ans is None:
        return jsonify(answer="已清除對話紀錄。", sources=[], retrieved=[], seconds=0)
    return jsonify(
        answer=ans.text,
        sources=[c.label for c in ans.sources],
        retrieved=[{"label": c.label, "score": round(s, 1), "text": c.text} for c, s in ans.retrieved],
        seconds=round(secs, 2),
        provider=llm.provider(),
    )


@app.get("/")
def index():
    return send_file(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return jsonify(chunks=len(rag.chunks), llm=llm.provider(), line=line_ready())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"知識庫段落數：{len(rag.chunks)}　LLM：{llm.provider()}　LINE：{'已設定' if line_ready() else '未設定'}")
    print(f"網頁聊天：http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
