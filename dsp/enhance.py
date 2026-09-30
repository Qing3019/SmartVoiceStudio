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
    