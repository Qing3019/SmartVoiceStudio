
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSplitter,
    QSizePolicy
)

from dsp.fft import compute_fft, amplitude_to_db
from dsp.stft import compute_stft, magnitude_to_db


class AudioAnalysisPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()


    def init_ui(self):

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        self.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        title = QLabel("音频信号分析")
        title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #243247;
            }
        """)
        layout.addWidget(title)

        # 垂直分割布局
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        layout.addWidget(self.splitter, stretch=1)

        # 时域波形
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setBackground("w")
        self.waveform_plot.setTitle("Waveform")
        self.waveform_plot.setLabel("bottom", "Time", units="s")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.25)
        self.waveform_plot.setMinimumHeight(120)
        self.waveform_plot.setXRange(0, 5)
        self.waveform_plot.setYRange(-1, 1)

        self.waveform_curve = self.waveform_plot.plot(
            pen=pg.mkPen(color="#2563eb", width=1)
        )
        self.waveform_curve.setClipToView(True)
        self.waveform_curve.setDownsampling(auto=True, method="peak")

        # FFT频谱
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setBackground("w")
        self.fft_plot.setTitle("FFT Spectrum")
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fft_plot.setLabel("left", "Magnitude", units="dB")
        self.fft_plot.showGrid(x=True, y=True, alpha=0.25)
        self.fft_plot.setMinimumHeight(120)
        self.fft_plot.setXRange(0, 8000)
        self.fft_plot.setYRange(-100, 5)

        self.fft_curve = self.fft_plot.plot(
            pen=pg.mkPen(color="#ef6c35", width=1)
        )
        self.fft_curve.setClipToView(True)
        self.fft_curve.setDownsampling(auto=True, method="peak")

        # STFT语谱图
        self.spectrogram_plot = pg.PlotWidget()
        self.spectrogram_plot.setBackground("w")
        self.spectrogram_plot.setTitle("Spectrogram")
        self.spectrogram_plot.setLabel("bottom", "Time", units="s")
        self.spectrogram_plot.setLabel("left", "Frequency", units="Hz")
        self.spectrogram_plot.setMinimumHeight(150)

        self.spectrogram_image = pg.ImageItem(axisOrder="row-major")
        self.spectrogram_plot.addItem(self.spectrogram_image)

        color_map = pg.colormap.get("viridis", source="matplotlib")
        self.spectrogram_image.setColorMap(color_map)

        # 添加绘图区
        self.splitter.addWidget(self.waveform_plot)
        self.splitter.addWidget(self.fft_plot)
        self.splitter.addWidget(self.spectrogram_plot)
        self.splitter.setSizes([220, 220, 280])

        # 波形与语谱图时间轴联动
        self.spectrogram_plot.setXLink(self.waveform_plot)


    def show_audio(self, audio, sample_rate):

        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) == 0:
            self.clear_audio()
            return

        self.show_waveform(audio, sample_rate)
        self.show_fft(audio, sample_rate)
        self.show_spectrogram(audio, sample_rate)


    def show_waveform(self, audio, sample_rate):

        time = np.arange(len(audio)) / sample_rate
        self.waveform_curve.setData(time, audio)

        duration = len(audio) / sample_rate
        peak = float(np.max(np.abs(audio)))

        # 防止纯静音导致Y轴范围为0
        peak = max(peak, 0.05)

        self.waveform_plot.setXRange(0, duration, padding=0.02)
        self.waveform_plot.setYRange(-peak * 1.1, peak * 1.1, padding=0.02)


    def show_fft(self, audio, sample_rate):

        frequencies, amplitude = compute_fft(audio, sample_rate)
        amplitude_db = amplitude_to_db(amplitude)

        self.fft_curve.setData(frequencies, amplitude_db)

        max_frequency = min(8000, sample_rate / 2)
        self.fft_plot.setXRange(0, max_frequency, padding=0)
        self.fft_plot.setYRange(-100, 5, padding=0)


    def show_spectrogram(self, audio, sample_rate):

        stft_matrix, magnitude, phase = compute_stft(audio)
        magnitude_db = magnitude_to_db(magnitude)

        # 限制语谱图动态范围
        magnitude_db = np.maximum(magnitude_db, -80)

        self.spectrogram_image.setImage(
            magnitude_db,
            autoLevels=False,
            levels=(-80, 0)
        )

        # 将图像坐标映射到实际时间与频率
        duration = len(audio) / sample_rate
        nyquist = sample_rate / 2

        self.spectrogram_image.setRect(
            QRectF(0, 0, duration, nyquist)
        )

        max_frequency = min(8000, nyquist)
        self.spectrogram_plot.setYRange(0, max_frequency, padding=0)
        self.spectrogram_plot.setXRange(0, duration, padding=0)


    def clear_audio(self):

        self.waveform_curve.clear()
        self.fft_curve.clear()
        self.spectrogram_image.clear()
