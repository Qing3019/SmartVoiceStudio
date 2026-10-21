"""测试 SmartVoiceStudio 到本地 CosyVoice3 的完整语音合成流程。

保存到 E:\\SmartVoiceStudio\\demo\\demo_23_tts_service_test.py
运行：conda activate SmartVoiceStudio
      cd E:\\SmartVoiceStudio
      python -m demo.demo_23_tts_service_test
"""
import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import soundfile as sf

from config import RESULT_DIR
from tts.client import health, synthesize


DEFAULT_REFERENCE = Path(r"E:\CosyVoiceRuntime\CosyVoice\asset\zero_shot_prompt.wav")
DEFAULT_REFERENCE_TEXT = "希望你以后能够做的比我还好呦。"


def parse_args():
    parser = argparse.ArgumentParser(description="CosyVoice3 本机服务的真实 TTS 请求测试")
    parser.add_argument("--text", default="你好，欢迎使用智能人声音效台。")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-text", default=DEFAULT_REFERENCE_TEXT)
    parser.add_argument("--language", default="中文")
    parser.add_argument("--dialect", default="普通话")
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "tts_api_first.wav")
    return parser.parse_args()


def main():
    args = parse_args()
    reference = args.reference.expanduser().resolve()
    output = args.output.expanduser().resolve()

    print("[1/4] 检查本机服务...", flush=True)
    try:
        status = health(timeout=5)
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("无法连接本机 CosyVoice 服务，请保持 cosyvoice3 终端运行") from exc
    if status.get("ok") is not True:
        raise RuntimeError(f"服务未就绪：{status}")
    print(f"      模型：{status.get('model')}，采样率：{status.get('sample_rate')} Hz")

    print("[2/4] 检查参考音频与输出路径...", flush=True)
    if not reference.is_file():
        raise FileNotFoundError(f"参考音频不存在：{reference}")
    if not args.reference_text.strip() or not args.text.strip():
        raise ValueError("待合成文本和参考逐字文本都不能为空")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == reference:
        raise ValueError("输出文件不能覆盖参考音频")
    print(f"      参考音频：{reference}")
    print(f"      输出文件：{output}")

    print("[3/4] 发送真实推理请求（首次 CPU 推理可能需要较久）...", flush=True)
    try:
        response = synthesize(
            text=args.text,
            reference_wav=str(reference),
            reference_text=args.reference_text,
            language=args.language,
            dialect=args.dialect,
            output_path=str(output)
        )
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TTS 服务返回 HTTP {exc.code}：{body}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("TTS 请求失败：请检查服务终端中的错误和网络回环连接") from exc

    print("[4/4] 验证生成的 WAV...", flush=True)
    actual_path = Path(response["output_path"]).resolve()
    if actual_path != output:
        raise RuntimeError(f"服务返回的输出路径与预期不符：{actual_path}")
    if not output.is_file() or output.stat().st_size <= 44:
        raise RuntimeError("服务返回成功，但 WAV 文件不存在或为空")
    wav_info = sf.info(str(output))
    if wav_info.frames <= 0 or wav_info.samplerate <= 0:
        raise RuntimeError("WAV 文件没有有效的音频帧")
    print(f"      成功：{output}")
    print(f"      采样率：{wav_info.samplerate} Hz")
    print(f"      时长：{wav_info.duration:.2f} 秒")
    print("完整 TTS 请求测试通过，可以进行 GUI 联调。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n测试失败：{error}", file=sys.stderr)
        sys.exit(1)
