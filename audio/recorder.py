import sounddevice as sd
import soundfile as sf

from config import SAMPLE_RATE
from config import RECORD_SECONDS
from config import CHANNELS
from config import INPUT_DEVICE

def record_audio(
        file_path,
        seconds=RECORD_SECONDS,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        input_device=INPUT_DEVICE
):
    print("开始录音……")
    # 单声道音频采样，共有 samples 个采样点，并将录音数据存储在 audio 数组中
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype='float32',
        device=INPUT_DEVICE
    )
    
    sd.wait()

    print("录音结束。")

    # 将录音数据从numpy数组保存为 WAV 文件
    sf.write(file_path, audio, sample_rate)

    print(f"录音已保存到 {file_path}")
    return audio