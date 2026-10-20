import sounddevice as sd
from config import OUTPUT_DEVICE

def play_audio_async(audio, sample_rate, output_device=OUTPUT_DEVICE):
    sd.play(audio, samplerate=sample_rate, device=output_device, blocking=False)

def stop_audio():
    sd.stop()

def play_audio(audio, sample_rate, output_device=OUTPUT_DEVICE):
    print("开始播放音频……")
    play_audio_async(audio, sample_rate, output_device)
    sd.wait()
    print("音频播放结束。")