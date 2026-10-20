
import numpy as np
import librosa

from config import N_FFT, HOP_LENGTH, WIN_LENGTH


def spectral_subtraction(audio, sample_rate, noise_duration=0.3, alpha=1.2, beta=0.02):

    audio = np.asarray(audio, dtype=np.float32)

    # STFT变换
    stft_matrix = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # 估计噪声频谱
    noise_samples = int(noise_duration * sample_rate)
    noise_frames = max(1, int(noise_samples / HOP_LENGTH))
    noise_frames = min(noise_frames, magnitude.shape[1])

    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1)

    # 频谱减法
    estimated_clean = magnitude - alpha * noise_profile[:, None]

    # 设置频谱底，减轻音乐噪声
    spectral_floor = beta * noise_profile[:, None]
    estimated_clean = np.maximum(estimated_clean, spectral_floor)

    # 保留原始相位，重建复数频谱
    clean_stft = estimated_clean * np.exp(1j * phase)

    # ISTFT恢复时域音频
    clean_audio = librosa.istft(
        clean_stft,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        length=len(audio)
    )

    return np.asarray(clean_audio, dtype=np.float32)
