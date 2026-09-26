from pathlib import Path

# 音频路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECORDING_DIR = DATA_DIR / "recordings"
RESULT_DIR = BASE_DIR / "results"

# 音频参数
SAMPLE_RATE = 44100
CHANNELS = 1
RECORD_SECONDS = 5

# STFT参数
N_FFT = 2048 # 每次计算的FFT点数
HOP_LENGTH = 512 # 步长
WIN_LENGTH = 2048 # 窗口长度

# 设备参数
INPUT_DEVICE = None  
OUTPUT_DEVICE = None

RECORDING_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)