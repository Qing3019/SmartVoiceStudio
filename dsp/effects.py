
import numpy as np
import librosa

from dsp.equalizer import three_band_eq
from dsp.filter import bandpass_filter


def change_speed(audio, speed=1.0):

    audio = np.asarray(audio, dtype=np.float32)

    if speed <= 0:
        raise ValueError("播放倍率必须大于0")

    if len(audio) == 0 or abs(speed - 1.0) < 1e-8:
        return audio.copy()

    result = librosa.effects.time_stretch(audio, rate=speed)

    return np.asarray(result, dtype=np.float32)


def change_pitch(audio, sample_rate, semitones=0):

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0 or semitones == 0:
        return audio.copy()

    result = librosa.effects.pitch_shift(
        audio,
        sr=sample_rate,
        n_steps=semitones
    )

    return np.asarray(result, dtype=np.float32)


def add_echo(audio, sample_rate, delay_ms=350, strength=0.45, repeats=3):

    audio = np.asarray(audio, dtype=np.float32)

    if not 0 <= strength < 1:
        raise ValueError("回声强度必须在0到1之间")

    if delay_ms <= 0 or repeats < 0:
        raise ValueError("回声延迟或重复次数无效")

    if len(audio) == 0 or strength == 0 or repeats == 0:
        return audio.copy()

    delay_samples = max(1, int(delay_ms * sample_rate / 1000))
    output_length = len(audio) + delay_samples * repeats
    output = np.zeros(output_length, dtype=np.float64)

    # 保留原声
    output[:len(audio)] = audio

    # 叠加逐渐衰减的回声
    for i in range(1, repeats + 1):
        start = delay_samples * i
        end = start + len(audio)
        output[start:end] += audio * (strength ** i)

    return output.astype(np.float32)


def telephone_effect(audio, sample_rate):

    nyquist = sample_rate / 2
    low_cutoff = 300
    high_cutoff = min(3400, nyquist * 0.9)

    if high_cutoff <= low_cutoff:
        raise ValueError("采样率过低，无法生成电话音")

    result = bandpass_filter(
        audio, sample_rate, low_cutoff, high_cutoff
    )

    return np.asarray(result, dtype=np.float32)


def magnetic_voice(audio, sample_rate):

    return three_band_eq(
        audio,
        sample_rate,
        low_db=3,
        mid_db=1,
        high_db=-1
    )


def baby_voice(audio, sample_rate):

    result = change_pitch(audio, sample_rate, semitones=4)

    return three_band_eq(
        result,
        sample_rate,
        low_db=-2,
        mid_db=1,
        high_db=3
    )


def deep_voice(audio, sample_rate):

    result = change_pitch(audio, sample_rate, semitones=-4)

    return three_band_eq(
        result,
        sample_rate,
        low_db=3,
        mid_db=1,
        high_db=-2
    )


def apply_preset(audio, sample_rate, preset_name):

    if preset_name == "磁性人声":
        return magnetic_voice(audio, sample_rate)

    elif preset_name == "娃娃音":
        return baby_voice(audio, sample_rate)

    elif preset_name == "大叔音":
        return deep_voice(audio, sample_rate)

    elif preset_name == "电话音":
        return telephone_effect(audio, sample_rate)

    elif preset_name == "山谷回声":
        return add_echo(audio, sample_rate)

    else:
        raise ValueError(f"未知的音效预设：{preset_name}")
