
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import QRectF, QThreadPool, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter
from PySide6.QtCore import Qt

from dsp.analysis import get_waveform_envelope
from gui.analysis_worker import AnalysisWorker


# 所有分析面板共享缓存
_ANALYSIS_CACHE = OrderedDict()
_CACHE_LIMIT = 8


def get_cached_analysis(audio, sample_rate):

    key = (id(audio), sample_rate)
    item = _ANALYSIS_CACHE.get(key)

    if item is None:
        return None

    # 防止不同数组复用相同内存地址
    cached_audio, result = item

    if cached_audio is not audio:
        return None

    _ANALYSIS_CACHE.move_to_end(key)

    return result


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
        layout.setSpacing(5)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # 时域波形
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setTitle("Waveform")
        self.waveform_plot.setLabel("bottom", "Time", units="s")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.2)

        self.waveform_curve = self.waveform_plot.plot(
            pen=pg.mkPen("#38BDF8", width=1.3)
        )

        # 固定幅度标尺，便于比较音量增减
        self.waveform_plot.setYRange(-3.2, 3.2, padding=0)

        # FFT频谱
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setTitle("FFT Amplitude Spectrum")
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fft_plot.setLabel("left", "Amplitude", units="dBFS")
        self.fft_plot.showGrid(x=True, y=True, alpha=0.2)

        self.fft_curve = self.fft_plot.plot(
            pen=pg.mkPen("#22C55E", width=1.3)
        )

        self.fft_plot.setYRange(-100, 15, padding=0)
        self.fft_plot.setXRange(0, 8000, padding=0)

        # STFT语谱图
        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.setTitle("STFT Spectrogram")
        self.spectrogram_plot.setLabel("bottom", "Time", units="s")
        self.spectrogram_plot.setLabel("left", "Frequency", units="Hz")

        self.spectrogram_image = pg.ImageItem(axisOrder="row-major")
        self.spectrogram_plot.addItem(self.spectrogram_image)

        # 统一语谱图配色
        colors = np.array([
            [20, 22, 64],
            [51, 61, 140],
            [39, 135, 170],
            [63, 192, 133],
            [218, 213, 80],
            [250, 174, 52]
        ], dtype=np.uint8)

        color_map = pg.ColorMap(
            [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            colors
        )

        self.spectrogram_image.setLookupTable(
            color_map.getLookupTable(nPts=256)
        )

        self.spectrogram_image.hide()
        self.spectrogram_plot.setYRange(0, 8000, padding=0)

        # 同步波形与语谱图的时间坐标
        self.spectrogram_plot.setXLink(self.waveform_plot)

        splitter.addWidget(self.waveform_plot)
        splitter.addWidget(self.fft_plot)
        splitter.addWidget(self.spectrogram_plot)
        splitter.setSizes([280, 260, 300])

        layout.addWidget(splitter)


    def clear_audio(self):

        # 使之前尚未完成的后台请求失效
        self.request_id += 1
        self.current_audio = None
        self.current_sample_rate = None

        self.waveform_curve.setData([], [])
        self.fft_curve.setData([], [])
        self.spectrogram_image.hide()

        self.waveform_plot.setTitle("Waveform")
        self.fft_plot.setTitle("FFT Amplitude Spectrum")
        self.spectrogram_plot.setTitle("STFT Spectrogram")


    def show_waveform(self, audio, sample_rate):

        # 用于录音过程中的实时波形预览
        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) == 0 or sample_rate <= 0:
            return

        times, values = get_waveform_envelope(
            audio,
            sample_rate,
            max_points=5000
        )

        self.waveform_curve.setData(times, values)
        self.waveform_plot.setXRange(
            0,
            max(len(audio) / sample_rate, 0.1),
            padding=0
        )


    def show_audio(self, audio, sample_rate):

        if audio is None or sample_rate is None:
            self.clear_audio()
            return

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim != 1 or len(audio) < 2:
            self.clear_audio()
            return

        self.request_id += 1
        request_id = self.request_id

        self.current_audio = audio
        self.current_sample_rate = sample_rate

        # 如果缓存存在，立即绘图
        cached = get_cached_analysis(audio, sample_rate)

        if cached is not None:
            self.display_analysis(cached)
            return

        # 否则清空旧图，并交给后台线程计算
        self.waveform_curve.setData([], [])
        self.fft_curve.setData([], [])
        self.spectrogram_image.hide()

        self.waveform_plot.setTitle("Waveform - Analyzing...")
        self.fft_plot.setTitle("FFT - Analyzing...")
        self.spectrogram_plot.setTitle("STFT - Analyzing...")

        worker = AnalysisWorker(audio, sample_rate, request_id)

        worker.signals.completed.connect(self.on_analysis_completed)
        worker.signals.failed.connect(self.on_analysis_failed)

        # 持有工作对象，直到任务完成
        self.pending_jobs[request_id] = worker
        self.thread_pool.start(worker)


    @Slot(object, int, object, int)
    def on_analysis_completed(self, audio, sample_rate, result, request_id):

        self.pending_jobs.pop(request_id, None)

        # 保存结果，其他面板也可以直接复用
        save_cached_analysis(audio, sample_rate, result)

        # 只显示当前请求对应的结果
        if request_id != self.request_id:
            return

        if audio is not self.current_audio:
            return

        if sample_rate != self.current_sample_rate:
            return

        self.display_analysis(result)


    @Slot(str, int)
    def on_analysis_failed(self, message, request_id):

        self.pending_jobs.pop(request_id, None)

        if request_id != self.request_id:
            return

        self.waveform_plot.setTitle("Waveform - Analysis Failed")
        self.fft_plot.setTitle("FFT - Analysis Failed")
        self.spectrogram_plot.setTitle("STFT - Analysis Failed")

        print(f"音频分析失败：{message}")


    def display_analysis(self, result):

        # 显示波形
        self.waveform_curve.setData(
            result["waveform_time"],
            result["waveform_value"]
        )

        # 显示FFT频谱
        self.fft_curve.setData(
            result["fft_frequency"],
            result["fft_db"]
        )

        # 显示STFT语谱图
        self.spectrogram_image.setImage(
            result["spectrogram_db"],
            autoLevels=False,
            levels=(-90, 12)
        )

        self.spectrogram_image.setRect(
            QRectF(
                0,
                0,
                result["duration"],
                result["spectrogram_frequency"]
            )
        )

        self.spectrogram_image.show()

        # 更新坐标范围
        self.waveform_plot.setXRange(
            0,
            max(result["duration"], 0.1),
            padding=0
        )

        self.fft_plot.setXRange(
            0,
            max(result["fft_frequency"][-1], 1),
            padding=0
        )

        self.spectrogram_plot.setYRange(
            0,
            result["spectrogram_frequency"],
            padding=0
        )

        # 恢复标题
        self.waveform_plot.setTitle("Waveform")
        self.fft_plot.setTitle("FFT Amplitude Spectrum")
        self.spectrogram_plot.setTitle("STFT Spectrogram")
