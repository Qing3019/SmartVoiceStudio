import numpy as np
import librosa

from config import N_FFT, HOP_LENGTH, WIN_LENGTH


def compute_stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):

    stft_matrix = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window="hann"
    )

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    return stft_matrix, magnitude, phase


def magnitude_to_db(magnitude, reference=None):

    # 旧Demo使用每段音频自身的最大值作为参考
    # GUI可使用固定参考值，以便直接比较不同处理结果
    if reference is None:
        return librosa.amplitude_to_db(magnitude, ref=np.max)

    return librosa.amplitude_to_db(magnitude, ref=reference, top_db=None)
