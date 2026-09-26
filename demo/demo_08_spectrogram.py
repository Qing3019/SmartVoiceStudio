import matplotlib.pyplot as plt
import librosa.display

from audio.wav_io import load_audio
from config import RECORDING_DIR, HOP_LENGTH
from dsp.stft import compute_stft, magnitude_to_db

def main():
    file_path = RECORDING_DIR / "record_01.wav"

    audio, sample_rate = load_audio(file_path)
    stft_matrix, magnitude, phase = compute_stft(audio)
    magnitude_db = magnitude_to_db(magnitude)

    print(f"STFT矩阵形状: {stft_matrix.shape}")
    print(f"幅度谱形状: {magnitude.shape}")
    print(f"相位谱形状: {phase.shape}")
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(magnitude_db, sr=sample_rate, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.ylim(0, 4000)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Speech Spectrogram")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":  
    main()
