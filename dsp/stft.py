import numpy as np
import librosa

from config import N_FFT, HOP_LENGTH, WIN_LENGTH

def compute_stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')

    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    return stft_matrix, magnitude, phase
# stft_matrix: 复数矩阵，表示短时傅里叶变换的结果，包含频率点和时间帧的信息
# magnitude: 幅度谱矩阵，表示每个时间帧和频率分量的幅度
# phase: 相位谱矩阵，表示每个时间帧和频率分量的相位信息

def magnitude_to_db(magnitude):
    db = librosa.amplitude_to_db(magnitude, ref=np.max)
    return db