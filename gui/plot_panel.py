"""异步 FFT/STFT 绘图与跨面板 LRU 缓存。"""
from collections import OrderedDict
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, QThreadPool, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from dsp.analysis import get_waveform_envelope
from gui.analysis_worker import AnalysisWorker

_ANALYSIS_CACHE = OrderedDict()
_CACHE_LIMIT = 8


def get_cached_analysis(audio, sample_rate):
    key = (id(audio), sample_rate)
    entry = _ANALYSIS_CACHE.get(key)
    if entry is None or entry[0] is not audio:
        return None
    _ANALYSIS_CACHE.move_to_end(key)
    return entry[1]


def save_cached_analysis(audio, sample_rate, result):
    key = (id(audio), sample_rate)
    _ANALYSIS_CACHE[key] = (audio, result)
    _ANALYSIS_CACHE.move_to_end(key)
    while len(_ANALYSIS_CACHE) > _CACHE_LIMIT:
        _ANALYSIS_CACHE.popitem(last=False)


class AudioAnalysisPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.request_id = 0
        self.current_audio = None
        self.current_sample_rate = None
        self.pending_jobs = {}
        self.thread_pool = QThreadPool.globalInstance()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setTitle("Waveform")
        self.waveform_plot.setLabel("bottom", "Time", units="s")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.2)
        self.waveform_curve = self.waveform_plot.plot(pen=pg.mkPen("#2563EB", width=1.2))
        self.waveform_plot.setYRange(-3.2, 3.2, padding=0)
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setTitle("FFT Spectrum")
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fft_plot.setLabel("left", "Amplitude", units="dBFS")
        self.fft_plot.showGrid(x=True, y=True, alpha=0.2)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen("#EC7B39", width=1.2))
        self.fft_plot.setYRange(-100, 15, padding=0)
        self.fft_plot.setXRange(0, 8000, padding=0)
        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.setTitle("STFT Spectrogram")
        self.spectrogram_plot.setLabel("bottom", "Time", units="s")
        self.spectrogram_plot.setLabel("left", "Frequency", units="Hz")
        self.spectrogram_image = pg.ImageItem(axisOrder="row-major")
        self.spectrogram_plot.addItem(self.spectrogram_image)
        colors = np.array([[20,22,64],[51,61,140],[39,135,170],[63,192,133],[218,213,80],[250,174,52]], dtype=np.uint8)
        self.spectrogram_image.setLookupTable(pg.ColorMap([0,.2,.4,.6,.8,1], colors).getLookupTable(nPts=256))
        self.spectrogram_image.hide()
        self.spectrogram_plot.setYRange(0, 8000, padding=0)
        self.spectrogram_plot.setXLink(self.waveform_plot)
        splitter.addWidget(self.waveform_plot)
        splitter.addWidget(self.fft_plot)
        splitter.addWidget(self.spectrogram_plot)
        splitter.setSizes([280, 260, 300])
        layout.addWidget(splitter)

    def clear_audio(self):
        self.request_id += 1
        self.current_audio = None
        self.current_sample_rate = None
        self.waveform_curve.setData([], [])
        self.fft_curve.setData([], [])
        self.spectrogram_image.hide()
        self.waveform_plot.setTitle("Waveform")
        self.fft_plot.setTitle("FFT Spectrum")
        self.spectrogram_plot.setTitle("STFT Spectrogram")

    def show_waveform(self, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0 or sample_rate <= 0:
            return
        times, values = get_waveform_envelope(audio, sample_rate, max_points=5000)
        self.waveform_curve.setData(times, values)
        self.waveform_plot.setXRange(0, max(len(audio) / sample_rate, 0.1), padding=0)

    def show_audio(self, audio, sample_rate):
        if audio is None or sample_rate is None:
            self.clear_audio()
            return
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or len(audio) < 2:
            self.clear_audio()
            return
        # 重复选择同一数组时，保留正在运行的请求，不重复入队。
        if self.current_audio is audio and self.current_sample_rate == sample_rate and self.request_id in self.pending_jobs:
            return
        self.request_id += 1
        request_id = self.request_id
        self.current_audio = audio
        self.current_sample_rate = sample_rate
        cached = get_cached_analysis(audio, sample_rate)
        if cached is not None:
            self.display_analysis(cached)
            return
        self.waveform_curve.setData([], [])
        self.fft_curve.setData([], [])
        self.spectrogram_image.hide()
        self.waveform_plot.setTitle("Waveform - Analyzing...")
        self.fft_plot.setTitle("FFT - Analyzing...")
        self.spectrogram_plot.setTitle("STFT - Analyzing...")
        worker = AnalysisWorker(audio, sample_rate, request_id)
        worker.signals.completed.connect(self.on_analysis_completed)
        worker.signals.failed.connect(self.on_analysis_failed)
        self.pending_jobs[request_id] = worker
        self.thread_pool.start(worker)

    @Slot(object, int, object, int)
    def on_analysis_completed(self, audio, sample_rate, result, request_id):
        self.pending_jobs.pop(request_id, None)
        save_cached_analysis(audio, sample_rate, result)
        if request_id != self.request_id or audio is not self.current_audio or sample_rate != self.current_sample_rate:
            return
        self.display_analysis(result)

    @Slot(str, int)
    def on_analysis_failed(self, message, request_id):
        self.pending_jobs.pop(request_id, None)
        if request_id == self.request_id:
            self.waveform_plot.setTitle("Analysis Failed")
            self.fft_plot.setTitle("Analysis Failed")
            self.spectrogram_plot.setTitle("Analysis Failed")
            print(f"音频分析失败：{message}")

    def display_analysis(self, result):
        self.waveform_curve.setData(result["waveform_time"], result["waveform_value"])
        self.fft_curve.setData(result["fft_frequency"], result["fft_db"])
        self.spectrogram_image.setImage(result["spectrogram_db"], autoLevels=False, levels=(-90, 12))
        self.spectrogram_image.setRect(QRectF(0, 0, result["duration"], result["spectrogram_frequency"]))
        self.spectrogram_image.show()
        self.waveform_plot.setXRange(0, max(result["duration"], 0.1), padding=0)
        self.fft_plot.setXRange(0, max(float(result["fft_frequency"][-1]), 1.0), padding=0)
        self.spectrogram_plot.setYRange(0, result["spectrogram_frequency"], padding=0)
        self.waveform_plot.setTitle("Waveform")
        self.fft_plot.setTitle("FFT Spectrum")
        self.spectrogram_plot.setTitle("STFT Spectrogram")
