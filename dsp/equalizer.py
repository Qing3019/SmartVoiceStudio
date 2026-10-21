
import numpy as np
from scipy.signal import sosfilt


def biquad_filter(audio, sample_rate, frequency, gain_db, filter_type, q=1.0):

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0 or abs(gain_db) < 1e-8:
        return audio.copy()

    if sample_rate <= 0:
        raise ValueError("采样率必须大于0")

    # 计算滤波器参数
    frequency = min(frequency, sample_rate * 0.45)
    A = 10 ** (gain_db / 40)
    omega = 2 * np.pi * frequency / sample_rate
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)

    if filter_type == "peaking":
        alpha = sin_w / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * cos_w
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w
        a2 = 1 - alpha / A

    elif filter_type == "low_shelf":
        beta = np.sqrt(2 * A) * sin_w

        b0 = A * ((A + 1) - (A - 1) * cos_w + beta)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w)
        b2 = A * ((A + 1) - (A - 1) * cos_w - beta)
        a0 = (A + 1) + (A - 1) * cos_w + beta
        a1 = -2 * ((A - 1) + (A + 1) * cos_w)
        a2 = (A + 1) + (A - 1) * cos_w - beta

    elif filter_type == "high_shelf":
        beta = np.sqrt(2 * A) * sin_w

        b0 = A * ((A + 1) + (A - 1) * cos_w + beta)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w)
        b2 = A * ((A + 1) + (A - 1) * cos_w - beta)
        a0 = (A + 1) - (A - 1) * cos_w + beta
        a1 = 2 * ((A - 1) - (A + 1) * cos_w)
        a2 = (A + 1) - (A - 1) * cos_w - beta

    else:
        raise ValueError("不支持的均衡器类型")

    # 转换为二阶节系数
    sos = np.array([[
        b0 / a0, b1 / a0, b2 / a0,
        1.0, a1 / a0, a2 / a0
    ]], dtype=np.float64)

    filtered = sosfilt(sos, audio.astype(np.float64))

    return filtered.astype(np.float32)


def three_band_eq(audio, sample_rate, low_db=0, mid_db=0, high_db=0):

    audio = np.asarray(audio, dtype=np.float32)

    # 低频均衡
    result = biquad_filter(
        audio, sample_rate, 200, low_db, "low_shelf"
    )

    # 中频均衡
    result = biquad_filter(
        result, sample_rate, 1500, mid_db, "peaking", q=0.8
    )

    # 高频均衡
    result = biquad_filter(
        result, sample_rate, 4500, high_db, "high_shelf"
    )

    return result
