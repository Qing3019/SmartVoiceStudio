import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
import librosa.display
from audio.wav_io import load_audio, save_audio

from config import RECORDING_DIR, RESULT_DIR
from config import HOP_LENGTH

from dsp.fft import compute_fft, amplitude_to_db
from dsp.stft import compute_stft, magnitude_to_db

def main():
    input_file = RECORDING_DIR / "record_01.wav"
    processed_file = RESULT_DIR / "output_denoised.wav"

    audio, sample_rate = load_audio(input_file)
    processed_audio, _ = load_audio(processed_file)

    original_time = np.linspace(0, len(audio) / sample_rate, len(audio))
    processed_time = np.linspace(0, len(processed_audio) / sample_rate, len(processed_audio))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(original_time, audio, label='原始音频', color='blue')
    plt.title('原始音频波形')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(processed_time, processed_audio, label='处理后音频', color='red')
    plt.title('处理后音频波形')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid()

    original_frequencies, original_amplitude = compute_fft(audio, sample_rate)
    processed_frequencies, processed_amplitude = compute_fft(processed_audio, sample_rate)
    original_amplitude_db = amplitude_to_db(original_amplitude)
    processed_amplitude_db = amplitude_to_db(processed_amplitude)

    plt.subplot(2, 3, 3)
    plt.plot(original_frequencies, original_amplitude_db, label='原始音频', color='blue')
    plt.title('原始音频频谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度 (dB)')
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(processed_frequencies, processed_amplitude_db, label='处理后音频', color='red')
    plt.title('处理后音频频谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度 (dB)')
    plt.legend()
    plt.grid()


    _,original_magnitude, _ = compute_stft(audio, sample_rate, hop_length=HOP_LENGTH)
    _,processed_magnitude, _ = compute_stft(processed_audio, sample_rate, hop_length=HOP_LENGTH)
    original_magnitude_db = magnitude_to_db(original_magnitude)
    processed_magnitude_db = magnitude_to_db(processed_magnitude)

    plt.subplot(2, 3, 5)
    librosa.display.specshow(original_magnitude_db, sr=sample_rate, hop_length=HOP_LENGTH, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('原始音频频谱图')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')

    plt.subplot(2, 3, 6)
    librosa.display.specshow(processed_magnitude_db, sr=sample_rate, hop_length=HOP_LENGTH, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('处理后音频频谱图')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.show()

if __name__ == "__main__":
    main()