import sounddevice as sd
from config import OUTPUT_DEVICE

def play_audio(audio, sample_rate, output_device=OUTPUT_DEVICE):
    print("开始播放音频……")
    sd.play(audio, samplerate=sample_rate, device=output_device)
    sd.wait()
    print("音频播放结束。")