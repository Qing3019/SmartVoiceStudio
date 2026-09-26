from audio.wav_io import load_audio, save_audio
from config import RECORDING_DIR

file_path = RECORDING_DIR / "record_01.wav"

audio, sample_rate = load_audio(file_path)

print("采样率：")
print(sample_rate)
print("音频数组形状：")
print(audio.shape)
print("采样点数量：")
print(len(audio))

duration = len(audio) / sample_rate
print("音频时长（秒）：")
print(duration)

print("前10个采样点：")
print(audio[:10])