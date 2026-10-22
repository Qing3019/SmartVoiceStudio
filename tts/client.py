"""CosyVoice3 本机 TTS 客户端：保留 JSON 协议，显示明确的服务器错误信息。"""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# 启动器根据本地端口注入环境变量；直接运行 GUI 默认 8765。
BASE_URL = os.getenv("SMARTVOICE_TTS_BASE_URL", "http://127.0.0.1:8765").rstrip("/")


def health(timeout=3):
    try:
        with urlopen(BASE_URL + "/health", timeout=timeout) as response:
            return json.load(response)
    except (OSError, URLError) as error:
        raise ConnectionError("无法连接 CosyVoice3 本机服务，请先运行 tts.local_server") from error


def synthesize(text, reference_wav, reference_text, language, dialect, output_path, mode="auto"):
    payload = {
        "text": text,
        "reference_wav": str(reference_wav),
        "reference_text": reference_text,
        "language": language,
        "dialect": dialect,
        "output_path": str(output_path),
        "mode": mode
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(BASE_URL + "/synthesize", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=1800) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            content = json.loads(error.read().decode("utf-8"))
            message = content.get("error", "服务器处理失败")
        except (ValueError, UnicodeError):
            message = error.reason
        raise RuntimeError(f"CosyVoice3 合成失败（HTTP {error.code}）：{message}") from error
    except (OSError, URLError) as error:
        raise ConnectionError("TTS 服务连接已断开，请检查 cosyvoice3 终端日志") from error
