"""SmartVoiceStudio 的 CosyVoice3 本机离线 TTS 服务（在 cosyvoice3 环境运行）。"""
import argparse
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

PREFIX = "You are a helpful assistant."
END_PROMPT = "<|endofprompt|>"
LANGUAGES = {"中文", "英语", "日语", "韩语", "德语", "西班牙语", "法语", "意大利语", "俄语"}
LANGUAGE_INSTRUCTIONS = {
    "中文": "请用普通话表达。",
    "英语": "Please speak in English.",
    "日语": "Please speak in Japanese.",
    "韩语": "Please speak in Korean.",
    "德语": "Please speak in German.",
    "西班牙语": "Please speak in Spanish.",
    "法语": "Please speak in French.",
    "意大利语": "Please speak in Italian.",
    "俄语": "Please speak in Russian."
}
DIALECT_INSTRUCTIONS = {
    "普通话": "",
    "四川话": "请用自然的四川话表达。",
    "广东话": "请用广东话表达。",
    "东北话": "请用东北话表达。",
    "上海话": "请用上海话表达。",
    "闽南话": "请用闽南话表达。",
    "山东话": "请用山东话表达。",
    "天津话": "请用天津话表达。"
}


def compose_instruction(language, dialect):
    """仅在方言模式中传给 inference_instruct2()。"""
    if language not in LANGUAGES:
        raise ValueError(f"不支持的语言：{language}")
    if language != "中文" and dialect != "普通话":
        raise ValueError("非中文合成请将方言设置为普通话")
    if dialect == "普通话":
        return PREFIX + END_PROMPT
    phrase = DIALECT_INSTRUCTIONS.get(dialect, dialect.strip())
    if not phrase or len(phrase) > 80:
        raise ValueError("方言指令长度应为1～80字符")
    return PREFIX + " " + phrase + END_PROMPT


def prepare_inference(text, reference_text, language, dialect, mode="auto"):
    """根据语言、方言及推理策略选择官方模型的三个入口。"""
    if language not in LANGUAGES:
        raise ValueError(f"不支持的语言：{language}")
    if language != "中文" and dialect != "普通话":
        raise ValueError("非中文合成不可同时选择中文方言，请在界面切换语言后重试")
    if mode not in {"auto", "zero_shot", "cross_lingual", "instruct"}:
        raise ValueError(f"无效的推理策略：{mode}")

    if mode == "auto":
        if language == "中文" and dialect == "普通话":
            mode = "zero_shot"
        elif language == "中文":
            mode = "instruct"
        else:
            mode = "cross_lingual"

    if mode == "zero_shot":
        if not reference_text:
            raise ValueError("零样本模式需要参考音频逐字文本")
        return mode, (text, PREFIX + END_PROMPT + reference_text)

    if mode == "cross_lingual":
        if language == "中文" and dialect != "普通话":
            raise ValueError("中文方言请使用「智能选择」或「指令模式」")
        return mode, (PREFIX + END_PROMPT + text,)

    if language == "中文":
        instruct = compose_instruction(language, dialect)
    else:
        instruct = PREFIX + " " + LANGUAGE_INSTRUCTIONS[language] + END_PROMPT
    return "instruct", (text, instruct)


REQUIRED_FILES = (
    "cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt",
    "campplus.onnx", "speech_tokenizer_v3.onnx"
)


def check_model_directory(repo, model_path):
    """在调用 AutoModel 前先检查源码与模型文件，避免含糊的模型识别错误。"""
    source = repo / "cosyvoice" / "cli" / "cosyvoice.py"
    if not source.is_file():
        raise FileNotFoundError(f"CosyVoice 源码不存在：{source}")

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"模型目录不存在：{model_path}\n"
            "请先下载 FunAudioLLM/Fun-CosyVoice3-0.5B-2512 模型，或者修正 --model 路径"
        )

    missing = [name for name in REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        found = sorted(str(item) for item in model_path.glob("*"))[:20]
        nearby = list(model_path.parent.glob("*/cosyvoice3.yaml"))
        info = "\n".join(found) if found else "（目录为空）"
        hints = "\n".join(map(str, nearby)) if nearby else "（未发现）"
        raise FileNotFoundError(
            f"模型目录缺少文件：{', '.join(missing)}\n"
            f"检查目录：{model_path}\n"
            f"目前目录内容：\n{info}\n"
            f"同级目录中找到的 cosyvoice3.yaml：\n{hints}\n"
            "请下载完整模型，不要把 CosyVoice 源码目录当作模型目录"
        )

    print(f"CosyVoice3 模型目录检查通过：{model_path}", flush=True)


def build_model(repo, model_path):
    """模型只在服务启动时加载一次。CosyVoice3 不支持 load_jit 构造参数。"""
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel
    return AutoModel(model_dir=str(model_path), load_trt=False, load_vllm=False, fp16=False)


def run_inference(model, text, reference_text, reference_wav, language, dialect, mode="auto"):
    mode, parameters = prepare_inference(text, reference_text, language, dialect, mode=mode)
    if mode == "instruct":
        chunks = model.inference_instruct2(*parameters, reference_wav, stream=False)
    elif mode == "cross_lingual":
        chunks = model.inference_cross_lingual(*parameters, reference_wav, stream=False)
    else:
        chunks = model.inference_zero_shot(*parameters, reference_wav, stream=False)

    outputs = []
    for part in chunks:
        data = part["tts_speech"].detach().cpu().numpy().reshape(-1)
        outputs.append(np.asarray(data, dtype=np.float32))

    if not outputs:
        raise RuntimeError("CosyVoice3 没有返回任何音频")

    result = np.concatenate(outputs)
    if not np.all(np.isfinite(result)) or len(result) == 0:
        raise ValueError("模型返回的音频数据无效")
    return result


def make_handler(model):
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, {"ok": True, "model": "CosyVoice3", "offline": True, "sample_rate": int(model.sample_rate)})
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/synthesize":
                return self.send_json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 2 <= length <= 65536:
                    raise ValueError("请求数据大小不合法")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                text = str(payload["text"]).strip()
                reference_text = str(payload.get("reference_text", "")).strip()
                reference_wav = Path(payload["reference_wav"]).expanduser().resolve()
                output_path = Path(payload["output_path"]).expanduser().resolve()
                language = str(payload.get("language", "中文"))
                dialect = str(payload.get("dialect", "普通话"))
                mode = str(payload.get("mode", "auto"))

                if not text or len(text) > 300:
                    raise ValueError("合成文本不能为空，单次不得超过300字符")
                if len(reference_text) > 500:
                    raise ValueError("参考音频逐字文本最多500字符")
                if not reference_wav.is_file() or reference_wav.suffix.lower() != ".wav":
                    raise ValueError("请提供有效的参考 WAV 文件")
                if output_path.suffix.lower() != ".wav":
                    raise ValueError("输出必须为 WAV 文件")

                # 先验证策略和文本要求，再进入单机大模型的串行推理
                selected_mode, _ = prepare_inference(text, reference_text, language, dialect, mode=mode)
                print(f"开始合成：语言={language}，方言={dialect}，模式={selected_mode}，字符数={len(text)}", flush=True)
                with inference_lock:
                    audio = run_inference(model, text, reference_text, str(reference_wav), language, dialect, mode=mode)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), audio, model.sample_rate, subtype="FLOAT")
                self.send_json(200, {"output_path": str(output_path), "sample_rate": int(model.sample_rate), "duration": len(audio) / model.sample_rate, "language": language, "mode": selected_mode})
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self.send_json(500, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 本地推理服务")
    parser.add_argument("--repo", required=True, help="官方 CosyVoice 源码目录")
    parser.add_argument("--model", required=True, help="本地 Fun-CosyVoice3 模型目录")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check-only", action="store_true", help="只检查模型路径和必要文件，不加载模型")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    model_path = Path(args.model).resolve()
    check_model_directory(repo, model_path)
    if args.check_only:
        print("仅文件检查完成，尚未加载模型。", flush=True)
        return

    print("正在加载 CosyVoice3 模型（CPU模式，可能需要等待）...", flush=True)
    model = build_model(repo, model_path)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model))
    print(f"离线 CosyVoice 服务已启动：http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
