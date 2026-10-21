"""SmartVoiceStudio 路径和音频参数；兼容源码启动及 PyInstaller 打包。"""
import os
import sys
from pathlib import Path

# 打包后把音频写到用户可持久保存的位置，避免 PyInstaller 清理 dist 时误删录音
if getattr(sys, "frozen", False):
    DEFAULT_DATA_HOME = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "SmartVoiceStudio"
else:
    DEFAULT_DATA_HOME = Path(__file__).resolve().parent

BASE_DIR = Path(os.getenv("SMARTVOICE_DATA_HOME", str(DEFAULT_DATA_HOME))).expanduser().resolve()
DATA_DIR = BASE_DIR / "data"
RECORDING_DIR = DATA_DIR / "recordings"
RESULT_DIR = BASE_DIR / "results"

SAMPLE_RATE = 44100
CHANNELS = 1
RECORD_SECONDS = 5
INPUT_DEVICE = None
OUTPUT_DEVICE = None
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
MAX_RECORD_SECONDS = 60
RECORD_PREVIEW_SECONDS = 2
GUI_UPDATE_INTERVAL = 100

RECORDING_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
