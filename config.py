
import sys
from pathlib import Path


# 判断是否在PyInstaller打包环境中运行
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent


# 数据与结果目录
DATA_DIR = BASE_DIR / "data"
RECORDING_DIR = DATA_DIR / "recordings"
RESULT_DIR = BASE_DIR / "results"


# 音频录制参数
SAMPLE_RATE = 44100
CHANNELS = 1
RECORD_SECONDS = 5

INPUT_DEVICE = None
OUTPUT_DEVICE = None


# STFT参数
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048


# GUI录音配置
MAX_RECORD_SECONDS = 60
RECORD_PREVIEW_SECONDS = 2
GUI_UPDATE_INTERVAL = 100


# 创建运行目录
RECORDING_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
