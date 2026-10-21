from pathlib import Path
import numpy as np
import soundfile as sf


def load_audio(file_path, mono=True):

    audio, sample_rate = sf.read(file_path, dtype="float32")

    if mono and audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    return audio, sample_rate


def save_audio(file_path, audio, sample_rate):

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("音频为空或包含无效采样值")

    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 超过±1时保存32位浮点WAV，防止默认PCM_16静默削波
    subtype = "FLOAT" if np.max(np.abs(audio)) > 1.0 else "PCM_16"
    sf.write(output_path, audio, sample_rate, subtype=subtype)
