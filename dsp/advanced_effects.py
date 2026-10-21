"""高级音效：Pedalboard 内置效果 + NumPy 环形调制。"""
import numpy as np
from pedalboard import Pedalboard, Reverb, Chorus, Delay, Compressor, Gain, Bitcrush, LowpassFilter


EFFECT_SETTINGS = {
    "房间混响": ("房间大小", 0.1, 0.95, 0.05, 0.65, ""),
    "机械电音": ("调制频率", 40, 180, 5, 75, " Hz"),
    "合唱音效": ("调制速率", 0.1, 5.0, 0.1, 0.8, " Hz"),
    "广播压缩": ("压缩阈值", -45, -10, 1, -26, " dB"),
    "8-bit数字音": ("量化位数", 4, 16, 1, 8, " bit"),
    "空间复合音效": ("回声延迟", 0.1, 0.8, 0.05, 0.28, " s")
}


def process_with_board(audio, sample_rate, effects):
    board = Pedalboard(effects)
    output = board(np.ascontiguousarray(audio, dtype=np.float32), sample_rate)
    return np.asarray(output, dtype=np.float32).reshape(-1)


def ring_modulation(audio, sample_rate, frequency):
    t = np.arange(len(audio), dtype=np.float64) / sample_rate
    carrier = np.cos(2 * np.pi * frequency * t)
    output = np.tanh(2.5 * audio.astype(np.float64) * carrier) / np.tanh(2.5)
    return output.astype(np.float32)


def apply_advanced_effect(audio, sample_rate, effect_name, intensity=0.6, parameter=None):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("请输入有效的非空单声道音频")
    if sample_rate <= 0:
        raise ValueError("采样率必须大于0")
    if not 0 <= intensity <= 1:
        raise ValueError("音效强度必须在0到1之间")
    if effect_name not in EFFECT_SETTINGS:
        raise ValueError(f"未知高级音效：{effect_name}")
    if intensity == 0:
        return audio.copy()

    setting = EFFECT_SETTINGS[effect_name]
    parameter = float(setting[4] if parameter is None else parameter)
    if not setting[1] <= parameter <= setting[2]:
        raise ValueError("音效参数超出范围")

    dry = audio.copy()
    if effect_name == "房间混响":
        dry = np.pad(audio, (0, int(sample_rate * 1.5)))
        wet = process_with_board(dry, sample_rate, [Reverb(room_size=parameter, damping=0.5, wet_level=0.6, dry_level=0.0)])
    elif effect_name == "机械电音":
        wet = ring_modulation(audio, sample_rate, parameter)
    elif effect_name == "合唱音效":
        wet = process_with_board(audio, sample_rate, [Chorus(rate_hz=parameter, depth=0.5, centre_delay_ms=8.0, feedback=0.1, mix=0.8)])
    elif effect_name == "广播压缩":
        wet = process_with_board(audio, sample_rate, [Compressor(threshold_db=parameter, ratio=4.0, attack_ms=8.0, release_ms=120.0), Gain(gain_db=3.0)])
    elif effect_name == "8-bit数字音":
        wet = process_with_board(audio, sample_rate, [Bitcrush(bit_depth=parameter), LowpassFilter(cutoff_frequency_hz=min(3500.0, sample_rate * 0.4))])
    else:
        dry = np.pad(audio, (0, int(sample_rate * 2.0)))
        wet = process_with_board(dry, sample_rate, [Chorus(rate_hz=0.7, depth=0.35, mix=0.45), Delay(delay_seconds=parameter, feedback=0.35, mix=0.35), Reverb(room_size=0.7, damping=0.45, wet_level=0.35, dry_level=0.65)])

    if len(wet) != len(dry):
        raise ValueError("处理结果长度异常")
    result = (1.0 - intensity) * dry + intensity * wet
    if not np.all(np.isfinite(result)):
        raise ValueError("音效产生无效采样值")
    return result.astype(np.float32)
