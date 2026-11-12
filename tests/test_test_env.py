"""測試環境本身的保護：就算專案目錄有 .env，測試也不能帶著真的 key 呼叫外部 API。"""
import os

import app  # noqa: F401 — import 時會 load_dotenv，模擬服務實際載入 .env 的情況


def test_測試時不帶真的API_key() -> None:
    for name in ("OPENAI_API_KEY", "LINE_CHANNEL_SECRET", "LINE_CHANNEL_ACCESS_TOKEN"):
        assert not os.getenv(name), f"{name} 不應出現在測試環境"
