import numpy as np
from scipy.signal import butter, sosfiltfilt

def lowpass_filter(audio, sample_rate, cutoff_freq, order=4):
    sos = butter(order, cutoff_freq, btype='lowpass', fs=sample_rate, output='sos')

    output = sosfiltfilt(sos, audio)
    return np.asanyarray(output, dtype=np.float32)

def highpass_filter(audio, sample_rate, cutoff_freq, order=4):
    sos = butter(order, cutoff_freq, btype='highpass', fs=sample_rate, output='sos')

    output = sosfiltfilt(sos, audio)
    return np.asanyarray(output, dtype=np.float32)

def bandpass_filter(audio, sample_rate, low_cutoff_freq, high_cutoff_freq, order=4):
    sos = butter(order, [low_cutoff_freq, high_cutoff_freq], btype='bandpass', fs=sample_rate, output='sos')

    output = sosfiltfilt(sos, audio)
    return np.asanyarray(output, dtype=np.float32)