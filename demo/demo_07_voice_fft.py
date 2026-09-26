import numpy as np
import matplotlib.pyplot as plt

from audio.wav_io import load_audio
from dsp.fft import compute_fft, amplitude_to_db
from config import RECORDING_DIR

def main():
    file_path = RECORDING_DIR / "record_01.wav"
    audio, sample_rate = load_audio(file_path)
    time = np.arange(len(audio)) / sample_rate
    frequencies, amplitude = compute_fft(audio, sample_rate)

    peak_index = np.argmax(amplitude[1:]) + 1 # 不要考虑直流分量
    peak_frequency = frequencies[peak_index]
    print(f"峰值频率: {peak_frequency:.2f} Hz, 峰值幅度: {amplitude[peak_index]:.4f}")

    amplitude_db = amplitude_to_db(amplitude)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(frequencies, amplitude)
    plt.xlim(0, 4000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Voice FFT")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(time, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Voice Signal")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(frequencies, amplitude_db)
    plt.xlim(0, 4000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("Voice FFT (dB)")
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
