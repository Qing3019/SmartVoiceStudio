import numpy as np


def compute_fft(audio, sample_rate):

    # 必须复制数据，不能在频谱分析时修改原声
    audio = np.asarray(audio, dtype=np.float32).copy()

    if audio.ndim != 1 or len(audio) == 0 or sample_rate <= 0:
        raise ValueError("FFT输入音频或采样率无效")

    if len(audio) == 1:
        return np.array([0.0]), np.array([0.0])

    audio -= np.mean(audio)
    number_of_samples = len(audio)

    # 汉宁窗抑制频谱泄漏
    window = np.hanning(number_of_samples)
    spectrum = np.fft.rfft(audio * window)
    frequencies = np.fft.rfftfreq(number_of_samples, 1 / sample_rate)
    amplitude = np.abs(spectrum) / np.sum(window)

    if number_of_samples % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2

    return frequencies, amplitude


def amplitude_to_db(amplitude, reference=None):

    amplitude = np.asarray(amplitude, dtype=np.float32)

    if amplitude.size == 0:
        return amplitude.copy()

    # reference=None保留旧Demo的峰值归一化行为
    # GUI使用reference=1.0，让原声和处理后的dB值可比较
    if reference is None:
        reference = float(np.max(amplitude))

    if reference <= 0:
        return np.full_like(amplitude, -120.0)

    return 20 * np.log10(np.maximum(amplitude / reference, 1e-10))
