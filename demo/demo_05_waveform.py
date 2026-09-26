import numpy as np
import matplotlib.pyplot as plt

from audio.wav_io import load_audio
from config import RECORDING_DIR

file_path = RECORDING_DIR / "record_01.wav"

audio, sample_rate = load_audio(file_path)

time = np.arange(len(audio)) / sample_rate

plt.figure(figsize=(12, 4))
plt.plot(time, audio)

plt.title("Waveform of the Audio Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()
plt.tight_layout()
plt.show()