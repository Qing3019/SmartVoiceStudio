import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体

from audio.wav_io import load_audio, save_audio
from audio.player import play_audio

from config import RECORDING_DIR, RESULT_DIR
from dsp.enhance import clip_audio, apply_gain, normalize_audio

from dsp.fft import compute_fft



def main():
    input_file = RECORDING_DIR / "record_01.wav"
    output_file = RESULT_DIR / "record_01_gain_normalize.wav"
    normalize_output_file = RESULT_DIR / "record_01_normalized.wav"

    time = np.linspace(0, 5, 44100 * 5)  # 生成时间轴

    audio, sample_rate = load_audio(input_file)
    audio_processed = apply_gain(audio, 2.0)  
    audio_processed = clip_audio(audio_processed, -1.0, 1.0)
    audio_normalized = normalize_audio(audio, peak=0.95)



    save_audio(output_file, audio_processed, sample_rate)
    print(f"已将增益和归一化后的音频保存到 {output_file}")

    frequencies_processed, amplitude_processed = compute_fft(audio_processed, sample_rate)
    frequencies_original, amplitude_original = compute_fft(audio, sample_rate)
    frequencies_normalized, amplitude_normalized = compute_fft(audio_normalized, sample_rate)

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 2, 1)
    plt.plot(frequencies_original, amplitude_original, label='原始音频', color='blue')
    plt.title('原始音频频谱')
    plt.ylim(0, 0.01)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(frequencies_processed, amplitude_processed, label='处理后音频', color='red')
    plt.title('处理后音频频谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(frequencies_normalized, amplitude_normalized, label='归一化音频', color='green')
    plt.title('归一化音频频谱')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time, audio, label='原始音频', color='blue')
    plt.title('原始音频与处理后音频波形')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(time, audio_processed, label='处理后音频', color='red')
    plt.title('处理后音频波形')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.legend()


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()