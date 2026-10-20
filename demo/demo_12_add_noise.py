from audio.wav_io import load_audio, save_audio
from config import RECORDING_DIR, RESULT_DIR
from dsp.enhance import add_white_noise, normalize_audio

def main():
    input_file = RECORDING_DIR / "record_01.wav"
    output_file_noisy = RESULT_DIR / "output_noisy.wav"

    audio, sample_rate = load_audio(input_file)

    noisy_audio = add_white_noise(audio, snr_db=10)
    normalized_noisy_audio = normalize_audio(noisy_audio, peak=0.95)

    save_audio(output_file_noisy, normalized_noisy_audio, sample_rate)

if __name__ == "__main__":
    main()