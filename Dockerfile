FROM python:3.14-slim

WORKDIR /app
# 讓 print 立刻出現在 docker logs（退回模式、恢復混合檢索的訊息都靠 print）
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY app.py eval.py ./
COPY chatbot/ chatbot/
COPY kb/ kb/
COPY static/ static/

EXPOSE 8000
# docker stop 改送 SIGINT：Python 才會跑 atexit 關閉向量庫（預設的 SIGTERM 會直接結束）
STOPSIGNAL SIGINT
CMD ["python", "app.py"]
