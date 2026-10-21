
import math
import numpy as np
import librosa


def get_waveform_envelope(audio, sample_rate, max_points=6000):

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0:
        return np.empty(0), np.empty(0)

    # 短音频直接显示，长音频提取峰值包络
    step = max(1, math.ceil(len(audio) / (max_points // 2)))

    if step == 1:
        times = np.arange(len(audio)) / sample_rate
        return times, audio

    indices = np.arange(0, len(audio), step)
    minimums = np.minimum.reduceat(audio, indices)
    maximums = np.maximum.reduceat(audio, indices)

    times = np.repeat(indices / sample_rate, 2)
    values = np.column_stack((minimums, maximums)).ravel()

    return times, values


def compute_spectrum(audio, sample_rate, max_frequency=8000, max_points=4000):

    audio = np.asarray(audio, dtype=np.float64)
    length = len(audio)

    if length < 2:
        raise ValueError("FFT分析至少需要两个采样点")

    # 去除直流分量，不修改输入音频
    signal = audio - np.mean(audio)
    window = np.hanning(length)

    spectrum = np.fft.rfft(signal * window)
    frequencies = np.fft.rfftfreq(length, d=1 / sample_rate)

    # 计算单边幅度谱
    denominator = max(float(np.sum(window)), 1e-12)
    amplitude = 2 * np.abs(spectrum) / denominator
    amplitude[0] /= 2

    if length % 2 == 0:
        amplitude[-1] /= 2

    spectrum_db = 20 * np.log10(np.maximum(amplitude, 1e-7))

    # 限制绘图频段
    frequency_limit = min(max_frequency, sample_rate / 2)
    count = np.searchsorted(frequencies, frequency_limit, side="right")

    frequencies = frequencies[:count]
    spectrum_db = spectrum_db[:count]

    # 对频谱进行峰值保持降采样
    if len(frequencies) > max_points:
        step = math.ceil(len(frequencies) / max_points)
        indices = np.arange(0, len(frequencies), step)

        spectrum_db = np.maximum.reduceat(spectrum_db, indices)
        frequencies = frequencies[indices]

    return frequencies, spectrum_db


def compute_spectrogram(audio, sample_rate, max_frequency=8000, max_frames=900):

    audio = np.asarray(audio, dtype=np.float32)

    n_fft = 2048
    hop_length = max(512, math.ceil(len(audio) / max_frames))

    stft_matrix = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann"
    )

    # 固定参考幅度，使不同音频可以公平比较
    window_sum = np.sum(np.hanning(n_fft + 1)[:-1]) / 2
    magnitude = np.abs(stft_matrix) / max(window_sum, 1e-12)
    spectrogram_db = 20 * np.log10(np.maximum(magnitude, 1e-6))

    frequency_limit = min(max_frequency, sample_rate / 2)
    frequency_count = min(
        stft_matrix.shape[0],
        int(frequency_limit * n_fft / sample_rate) + 1
    )

    spectrogram_db = spectrogram_db[:frequency_count, :]
    spectrogram_db = np.clip(spectrogram_db, -90, 12)

    return spectrogram_db.astype(np.float32), frequency_limit


def analyze_audio(audio, sample_rate):

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or len(audio) < 2:
        raise ValueError("音频必须是包含至少两个采样点的一维数组")

    if sample_rate <= 0 or not np.all(np.isfinite(audio)):
        raise ValueError("音频采样率或数据无效")

    waveform_time, waveform_value = get_waveform_envelope(audio, sample_rate)
    fft_frequency, fft_db = compute_spectrum(audio, sample_rate)
    spectrogram_db, spectrogram_frequency = compute_spectrogram(audio, sample_rate)

    return {
        "waveform_time": waveform_time,
        "waveform_value": waveform_value,
        "fft_frequency": fft_frequency,
        "fft_db": fft_db,
        "spectrogram_db": spectrogram_db,
        "spectrogram_frequency": spectrogram_frequency,
        "duration": len(audio) / sample_rate
    }
