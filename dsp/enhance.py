import numpy as np
def clip_audio(audio, min_value=-1.0, max_value=1.0):
    audio = np.asarray(audio, dtype=np.float32)
    return np.clip(audio, min_value, max_value)

def apply_gain(audio, gain):
    audio = np.asarray(audio, dtype=np.float32)
    return audio * gain

def normalize_audio(audio, peak=0.95):
    audio = np.asarray(audio, dtype=np.float32)
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio.copy()  # 避免除以零
    normalization_factor = peak / max_val
    return audio * normalization_factor


def add_white_noise(audio, snr_db):
    audio = np.asarray(audio, dtype=np.float32)

    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio.copy()

    snr_linear = 10 ** (snr_db / 10)
    noise_power = (signal_power / snr_linear)

    noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)

    noisy_audio = (audio + noise)

    return np.asarray(noisy_audio, dtype=np.float32)