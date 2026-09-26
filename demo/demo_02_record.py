from audio.recorder import record_audio
from config import RECORDING_DIR

output_file = RECORDING_DIR / "record_01.wav"

record_audio(file_path=output_file, seconds=5, sample_rate=44100, channels=1) # 可修改，否则使用config默认值



