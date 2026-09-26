import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from audio.wav_io import load_audio
from config import RECORDING_DIR, HOP_LENGTH

from dsp.stft import compute_stft, magnitude_to_db
from dsp.fft import compute_fft, amplitude_to_db

def main():
    file_path = RECORDING_DIR / "record_01.wav"

    audio, sample_rate = load_audio(file_path)
    # 计算时间轴
    time = np.arange(len(audio)) / sample_rate

    # 计算STFT
    stft_matrix, magnitude, phase = compute_stft(audio)
    magnitude_db = magnitude_to_db(magnitude)

    # 计算FFT
    frequencies, amplitude = compute_fft(audio, sample_rate)
    amplitude_db = amplitude_to_db(amplitude)

    # 绘制图形
    plt.figure(figsize=(12, 8))


    plt.subplot(3, 1, 1)
    plt.plot(frequencies, amplitude_db)
    plt.xlim(0, 4000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("Voice FFT (dB)") 

    plt.subplot(3, 1, 2)
    plt.plot(time, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Voice Signal")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(magnitude_db, sr=sample_rate, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.ylim(0, 4000)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Speech Spectrogram")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()