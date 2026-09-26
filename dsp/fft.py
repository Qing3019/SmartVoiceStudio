import numpy as np

def compute_fft(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    audio -= np.mean(audio) # 去掉直流偏置
    number_of_samples = len(audio)

    # 加汉宁窗防止频谱泄露
    window = np.hanning(number_of_samples)
    windowed_audio = audio * window
    spectrum = np.fft.rfft(windowed_audio)
    frequencies = np.fft.rfftfreq(number_of_samples, 1 / sample_rate)
    window_gain = np.mean(window)
    amplitude = (np.abs(spectrum) / (window_gain * number_of_samples))

    if number_of_samples % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2

    return frequencies, amplitude

def amplitude_to_db(amplitude):
    amplitude = np.asarray(amplitude, dtype=np.float32)
    maximum_amplitude = np.max(amplitude)
    if maximum_amplitude == 0:
        return np.zeros_like(amplitude)
    normalized_amplitude = amplitude / maximum_amplitude
    db = 20 * np.log10(np.maximum(normalized_amplitude, 1e-10))  # 避免对数零
    return db