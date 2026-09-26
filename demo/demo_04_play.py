from audio.wav_io import load_audio
from audio.player import play_audio
from config import RECORDING_DIR

file_path = RECORDING_DIR / "record_01.wav"

audio, sample_rate = load_audio(file_path)

play_audio(audio, sample_rate)