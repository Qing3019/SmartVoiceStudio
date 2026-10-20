
from pathlib import Path
from datetime import datetime
from math import gcd
import numpy as np
import sounddevice as sd

from scipy.signal import resample_poly
from PySide6.QtCore import Qt, QTimer, QElapsedTimer, QThread, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QFrame,
    QComboBox,
    QSlider,
    QFileDialog,
    QMessageBox
)

from config import (
    RECORDING_DIR,
    RESULT_DIR,
    MAX_RECORD_SECONDS,
    RECORD_PREVIEW_SECONDS,
    GUI_UPDATE_INTERVAL
)

from audio.devices import get_audio_devices, get_input_sample_rate
from audio.recorder import StreamRecorder
from audio.wav_io import load_audio, save_audio
from audio.player import play_audio_async, stop_audio

from dsp.enhance import apply_gain, normalize_audio
from dsp.filter import lowpass_filter, highpass_filter, bandpass_filter
from dsp.denoise import spectral_subtraction

from gui.plot_panel import AudioAnalysisPanel
from gui.control_panel import DSPControlPanel
from gui.workers import DSPWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.original_audio = None
        self.processed_audio = None
        self.sample_rate = None
        self.file_path = None
        self.operations = []

        # DSP后台线程与撤销记录
        self.is_processing = False
        self.processing_thread = None
        self.processing_worker = None
        self.processing_description = ""
        self.processing_snapshot = None
        self.undo_stack = []
        self.max_undo_steps = 5

        # 录音状态
        self.recorder = StreamRecorder()
        self.record_preview = np.empty(0, dtype=np.float32)

        self.record_timer = QTimer(self)
        self.record_timer.setInterval(GUI_UPDATE_INTERVAL)
        self.record_timer.timeout.connect(self.update_recording)

        # 播放状态
        self.playback_audio = None
        self.playback_name = ""
        self.playback_duration = 0.0
        self.playback_start = 0.0
        self.is_playing = False

        self.play_elapsed = QElapsedTimer()
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(GUI_UPDATE_INTERVAL)
        self.play_timer.timeout.connect(self.update_playback)

        self.init_ui()


    def init_ui(self):

        self.setWindowTitle("Smart Voice Studio")
        self.resize(1350, 850)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # 左侧滚动控制区域
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(355)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        title = QLabel("Smart Voice Studio")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        left_layout.addWidget(title)

        subtitle = QLabel("智能人声音效处理平台")
        left_layout.addWidget(subtitle)

        # 音频设备管理
        device_title = QLabel("音频设备管理")
        device_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(device_title)

        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.refresh_button = QPushButton("刷新音频设备")

        left_layout.addWidget(QLabel("输入设备"))
        left_layout.addWidget(self.input_combo)
        left_layout.addWidget(QLabel("输出设备"))
        left_layout.addWidget(self.output_combo)
        left_layout.addWidget(self.refresh_button)

        # 麦克风录音
        record_title = QLabel("麦克风录音")
        record_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(record_title)

        self.record_button = QPushButton("开始录音")
        self.stop_record_button = QPushButton("停止录音")
        self.record_time_label = QLabel("录音时长：00:00.0")

        record_layout = QHBoxLayout()
        record_layout.addWidget(self.record_button)
        record_layout.addWidget(self.stop_record_button)

        left_layout.addLayout(record_layout)
        left_layout.addWidget(self.record_time_label)

        # 音频管理
        audio_title = QLabel("音频管理")
        audio_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(audio_title)

        self.open_button = QPushButton("导入 WAV 音频")
        self.play_original_button = QPushButton("播放原声")
        self.play_processed_button = QPushButton("播放处理后")
        self.stop_button = QPushButton("停止播放")
        self.save_button = QPushButton("保存处理结果")

        left_layout.addWidget(self.open_button)
        left_layout.addWidget(self.play_original_button)
        left_layout.addWidget(self.play_processed_button)
        left_layout.addWidget(self.stop_button)
        left_layout.addWidget(self.save_button)

        # 播放进度
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)

        self.progress_label = QLabel("00:00 / 00:00")
        left_layout.addWidget(self.progress_slider)
        left_layout.addWidget(self.progress_label)

        # 音频文件信息
        info_title = QLabel("音频文件信息")
        info_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(info_title)

        self.file_info = QLabel("尚未导入音频")
        self.file_info.setWordWrap(True)
        left_layout.addWidget(self.file_info)

        # 原声与处理后图形切换
        view_title = QLabel("可视化信号选择")
        view_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(view_title)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["原声", "处理后"])
        left_layout.addWidget(self.view_combo)

        self.process_info = QLabel("当前处理：无")
        self.process_info.setWordWrap(True)
        left_layout.addWidget(self.process_info)

        # DSP参数控制区域
        self.control_panel = DSPControlPanel()
        left_layout.addWidget(self.control_panel)
        left_layout.addStretch()

        left_scroll.setWidget(left_panel)

        # 右侧音频三联分析图
        self.analysis_panel = AudioAnalysisPanel()

        main_layout.addWidget(left_scroll)
        main_layout.addWidget(self.analysis_panel, stretch=1)

        # 音频设备与录音
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.output_combo.currentIndexChanged.connect(self.on_output_changed)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)

        # 音频播放与文件管理
        self.open_button.clicked.connect(self.open_audio_file)
        self.play_original_button.clicked.connect(self.play_original)
        self.play_processed_button.clicked.connect(self.play_processed)
        self.stop_button.clicked.connect(self.stop_playback)
        self.save_button.clicked.connect(self.save_processed_audio)
        self.progress_slider.sliderReleased.connect(self.seek_playback)
        self.view_combo.currentIndexChanged.connect(self.update_visualization)

        # DSP功能
        self.control_panel.gain_button.clicked.connect(self.apply_gain_processing)
        self.control_panel.normalize_button.clicked.connect(self.apply_normalization)
        self.control_panel.filter_button.clicked.connect(self.apply_filter_processing)
        self.control_panel.denoise_button.clicked.connect(self.apply_denoise_processing)
        self.control_panel.undo_button.clicked.connect(self.undo_processing)
        self.control_panel.reset_button.clicked.connect(self.reset_processing)

        self.set_audio_controls(False)
        self.stop_record_button.setEnabled(False)
        self.refresh_devices()

        self.statusBar().showMessage("就绪，请导入音频或开始录音")


    def set_audio_controls(self, enabled):

        self.play_original_button.setEnabled(enabled)
        self.play_processed_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.view_combo.setEnabled(enabled)
        self.control_panel.setEnabled(enabled)
        self.progress_slider.setEnabled(enabled)
        self.stop_button.setEnabled(False)

        self.control_panel.undo_button.setEnabled(
            enabled and len(self.undo_stack) > 0
        )


    def set_processing_controls(self, enabled):

        self.open_button.setEnabled(enabled)
        self.record_button.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)
        self.input_combo.setEnabled(enabled)
        self.output_combo.setEnabled(enabled)

        has_audio = self.original_audio is not None
        self.set_audio_controls(enabled and has_audio)


    # ==========================================
    # 音频设备管理
    # ==========================================

    def refresh_devices(self):

        if self.recorder.is_recording or self.is_processing:
            return

        try:
            self.stop_playback()

            input_devices, output_devices = get_audio_devices()
            input_selected = self.input_combo.currentData()
            output_selected = self.output_combo.currentData()

            self.input_combo.blockSignals(True)
            self.output_combo.blockSignals(True)

            self.input_combo.clear()
            self.output_combo.clear()

            self.input_combo.addItem("系统默认输入设备", None)
            self.output_combo.addItem("系统默认输出设备", None)

            for index, name in input_devices:
                self.input_combo.addItem(name, index)

            for index, name in output_devices:
                self.output_combo.addItem(name, index)

            input_index = self.input_combo.findData(input_selected)
            output_index = self.output_combo.findData(output_selected)

            self.input_combo.setCurrentIndex(max(input_index, 0))
            self.output_combo.setCurrentIndex(max(output_index, 0))

            self.statusBar().showMessage("音频设备列表已更新")

        except Exception as error:
            QMessageBox.critical(self, "设备查询失败", str(error))

        finally:
            self.input_combo.blockSignals(False)
            self.output_combo.blockSignals(False)


    def on_output_changed(self, index):

        if self.is_playing:
            self.stop_playback()

        self.statusBar().showMessage("输出设备已更新")


    # ==========================================
    # 麦克风录音
    # ==========================================

    def start_recording(self):

        if self.recorder.is_recording or self.is_processing:
            return

        try:
            self.stop_playback()

            input_device = self.input_combo.currentData()
            sample_rate = get_input_sample_rate(input_device)

            self.recorder.start(
                device=input_device,
                sample_rate=sample_rate
            )

            self.record_preview = np.empty(0, dtype=np.float32)

            self.set_audio_controls(False)
            self.open_button.setEnabled(False)
            self.record_button.setEnabled(False)
            self.stop_record_button.setEnabled(True)
            self.input_combo.setEnabled(False)
            self.output_combo.setEnabled(False)
            self.refresh_button.setEnabled(False)

            self.record_time_label.setText("录音时长：00:00.0")
            self.analysis_panel.clear_audio()
            self.analysis_panel.waveform_plot.setTitle("Recording - Last 2 Seconds")

            self.record_timer.start()
            self.statusBar().showMessage(f"正在录音：{sample_rate} Hz")

        except Exception as error:
            QMessageBox.critical(self, "录音启动失败", str(error))


    def update_recording(self):

        if not self.recorder.is_recording:
            return

        new_audio = self.recorder.read_available()

        if len(new_audio) > 0:
            self.record_preview = np.concatenate((self.record_preview, new_audio))

            max_samples = int(RECORD_PREVIEW_SECONDS * self.recorder.sample_rate)
            self.record_preview = self.record_preview[-max_samples:]

            self.analysis_panel.show_waveform(
                self.record_preview,
                self.recorder.sample_rate
            )

        duration = self.recorder.get_duration()
        minutes = int(duration // 60)
        seconds = duration % 60

        self.record_time_label.setText(
            f"录音时长：{minutes:02d}:{seconds:04.1f}"
        )

        if duration >= MAX_RECORD_SECONDS:
            self.stop_recording()


    def stop_recording(self):

        if not self.recorder.is_recording:
            return

        self.record_timer.stop()

        try:
            audio = self.recorder.stop()
            sample_rate = self.recorder.sample_rate

            if len(audio) < sample_rate * 0.2:
                raise ValueError("录音时间过短，请重新录制")

            filename = datetime.now().strftime("record_%Y%m%d_%H%M%S.wav")
            output_file = RECORDING_DIR / filename

            save_audio(output_file, audio, sample_rate)
            self.load_audio_data(audio, sample_rate, output_file)

            if self.recorder.overflow_count > 0 or self.recorder.dropped_blocks > 0:
                QMessageBox.warning(
                    self,
                    "录音质量提醒",
                    f"采集异常 {self.recorder.overflow_count} 次，"
                    f"丢失数据块 {self.recorder.dropped_blocks} 个"
                )

            self.statusBar().showMessage(f"录音已保存：{filename}")

        except Exception as error:
            QMessageBox.critical(self, "录音保存失败", str(error))
            self.update_visualization()

        finally:
            self.record_button.setEnabled(True)
            self.stop_record_button.setEnabled(False)
            self.open_button.setEnabled(True)
            self.input_combo.setEnabled(True)
            self.output_combo.setEnabled(True)
            self.refresh_button.setEnabled(True)

            self.set_audio_controls(self.original_audio is not None)
            self.analysis_panel.waveform_plot.setTitle("Waveform")


    # ==========================================
    # 音频文件管理
    # ==========================================

    def load_audio_data(self, audio, sample_rate, file_path):

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim != 1 or len(audio) == 0 or sample_rate <= 0:
            raise ValueError("音频数据无效")

        if not np.all(np.isfinite(audio)):
            raise ValueError("音频存在无效采样值")

        self.stop_playback()

        self.original_audio = audio.copy()
        self.processed_audio = audio.copy()
        self.sample_rate = sample_rate
        self.file_path = Path(file_path)

        # 清空上一段音频的处理历史
        self.operations.clear()
        self.undo_stack.clear()
        self.processing_snapshot = None

        duration = len(audio) / sample_rate

        self.file_info.setText(
            f"文件名：{self.file_path.name}\n"
            f"采样率：{sample_rate} Hz\n"
            f"采样点数：{len(audio)}\n"
            f"时长：{duration:.2f} s\n"
            f"处理声道：单声道"
        )

        self.process_info.setText("当前处理：无")
        self.set_audio_controls(True)

        self.playback_audio = None
        self.playback_duration = duration
        self.update_progress_display(0, duration)

        self.view_combo.setCurrentIndex(0)
        self.update_visualization()

        self.control_panel.undo_button.setEnabled(False)


    def open_audio_file(self):

        if self.is_processing or self.recorder.is_recording:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 WAV 音频文件",
            str(RECORDING_DIR),
            "WAV 音频 (*.wav)"
        )

        if not file_path:
            return

        try:
            audio, sample_rate = load_audio(file_path)
            self.load_audio_data(audio, sample_rate, file_path)
            self.statusBar().showMessage(f"已加载：{Path(file_path).name}")

        except Exception as error:
            QMessageBox.critical(self, "音频加载失败", str(error))


    def update_visualization(self, index=None):

        if self.original_audio is None or self.recorder.is_recording:
            return

        if self.view_combo.currentIndex() == 0:
            audio = self.original_audio
            name = "原声"
        else:
            audio = self.processed_audio
            name = "处理后"

        self.analysis_panel.show_audio(audio, self.sample_rate)
        self.statusBar().showMessage(f"当前显示：{name}")


    # ==========================================
    # 音频播放与进度管理
    # ==========================================

    def play_original(self):

        self.playback_audio = self.original_audio
        self.playback_name = "原声"
        self.start_playback(0)


    def play_processed(self):

        self.playback_audio = self.processed_audio
        self.playback_name = "处理后"
        self.start_playback(0)


    def start_playback(self, start_seconds=0):

        if self.playback_audio is None or self.recorder.is_recording or self.is_processing:
            return

        self.stop_playback(reset_progress=False)

        try:
            audio = self.playback_audio
            self.playback_duration = len(audio) / self.sample_rate

            if start_seconds >= self.playback_duration:
                start_seconds = 0

            start_seconds = max(0, start_seconds)
            start_sample = int(start_seconds * self.sample_rate)

            playback_audio = audio[start_sample:]
            playback_rate = self.sample_rate
            output_device = self.output_combo.currentData()

            # 如果设备不支持原采样率，则重采样播放
            try:
                sd.check_output_settings(
                    device=output_device,
                    channels=1,
                    samplerate=playback_rate,
                    dtype="float32"
                )

            except (sd.PortAudioError, ValueError):
                device_info = sd.query_devices(output_device, "output")
                target_rate = int(round(device_info["default_samplerate"]))

                if target_rate <= 0:
                    raise ValueError("输出设备采样率无效")

                sd.check_output_settings(
                    device=output_device,
                    channels=1,
                    samplerate=target_rate,
                    dtype="float32"
                )

                factor = gcd(playback_rate, target_rate)
                playback_audio = resample_poly(
                    playback_audio,
                    target_rate // factor,
                    playback_rate // factor
                ).astype(np.float32)

                playback_rate = target_rate

            play_audio_async(
                playback_audio,
                playback_rate,
                output_device=output_device
            )

            self.playback_start = start_seconds
            self.play_elapsed.start()
            self.is_playing = True
            self.play_timer.start()

            self.play_original_button.setEnabled(False)
            self.play_processed_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.update_progress_display(start_seconds, self.playback_duration)
            self.statusBar().showMessage(f"正在播放：{self.playback_name}")

        except Exception as error:
            self.stop_playback()
            QMessageBox.critical(self, "音频播放失败", str(error))


    def update_playback(self):

        if not self.is_playing:
            return

        elapsed = self.play_elapsed.elapsed() / 1000
        position = min(self.playback_start + elapsed, self.playback_duration)

        self.update_progress_display(position, self.playback_duration)

        if position >= self.playback_duration:
            self.on_playback_finished()


    def update_progress_display(self, position, duration):

        if duration <= 0:
            self.progress_label.setText("00:00 / 00:00")
            self.progress_slider.setValue(0)
            return

        progress = int(position / duration * 1000)

        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(min(1000, max(0, progress)))

        self.progress_label.setText(
            f"{self.format_time(position)} / {self.format_time(duration)}"
        )


    def format_time(self, seconds):

        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"


    def seek_playback(self):

        if self.playback_audio is None:
            return

        position = self.progress_slider.value() / 1000 * self.playback_duration

        if self.is_playing:
            self.start_playback(position)
        else:
            self.update_progress_display(position, self.playback_duration)


    def stop_playback(self, reset_progress=True):

        self.play_timer.stop()
        stop_audio()
        self.is_playing = False

        enabled = self.original_audio is not None
        enabled = enabled and not self.recorder.is_recording and not self.is_processing

        self.play_original_button.setEnabled(enabled)
        self.play_processed_button.setEnabled(enabled)
        self.stop_button.setEnabled(False)

        if reset_progress:
            self.update_progress_display(0, self.playback_duration)


    def on_playback_finished(self):

        self.play_timer.stop()
        self.is_playing = False

        enabled = not self.recorder.is_recording and not self.is_processing

        self.play_original_button.setEnabled(enabled)
        self.play_processed_button.setEnabled(enabled)
        self.stop_button.setEnabled(False)

        self.update_progress_display(self.playback_duration, self.playback_duration)
        self.statusBar().showMessage("播放已结束")


    def invalidate_playback(self):

        self.playback_audio = None
        self.playback_name = ""

        if self.original_audio is not None:
            self.playback_duration = len(self.original_audio) / self.sample_rate
        else:
            self.playback_duration = 0.0

        self.update_progress_display(0, self.playback_duration)


    # ==========================================
    # DSP后台线程管理
    # ==========================================

    def apply_processing(self, process_func, description):

        if self.processed_audio is None or self.recorder.is_recording:
            return

        if self.is_processing:
            QMessageBox.information(self, "提示", "当前音频处理任务尚未完成")
            return

        self.stop_playback()
        self.invalidate_playback()

        # 保存处理前状态
        self.processing_snapshot = (
            self.processed_audio.copy(),
            self.operations.copy()
        )
        self.processing_description = description

        # 创建后台工作线程
        self.processing_thread = QThread(self)
        self.processing_worker = DSPWorker(self.processed_audio, process_func)
        self.processing_worker.moveToThread(self.processing_thread)

        self.processing_thread.started.connect(self.processing_worker.run)
        self.processing_worker.result_ready.connect(self.on_processing_success)
        self.processing_worker.error_occurred.connect(self.on_processing_error)

        # 结束任务并释放对象
        self.processing_worker.finished.connect(self.processing_thread.quit)
        self.processing_worker.finished.connect(self.processing_worker.deleteLater)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)
        self.processing_thread.finished.connect(self.on_processing_thread_finished)

        self.is_processing = True
        self.set_processing_controls(False)

        self.statusBar().showMessage(f"正在处理：{description} ……")
        self.processing_thread.start()


    @Slot(object)
    def on_processing_success(self, result):

        try:
            result = np.asarray(result, dtype=np.float32)

            if result.ndim != 1 or len(result) != len(self.processed_audio):
                raise ValueError("处理结果的长度或维度错误")

            if not np.all(np.isfinite(result)):
                raise ValueError("处理结果包含无效采样值")

            # 峰值保护
            peak = float(np.max(np.abs(result)))

            if peak > 0.98:
                result = result * (0.98 / peak)

            # 保存撤销记录，最多保留5步
            if self.processing_snapshot is not None:
                self.undo_stack.append(self.processing_snapshot)

            if len(self.undo_stack) > self.max_undo_steps:
                self.undo_stack.pop(0)

            # 更新处理结果
            self.processed_audio = result
            self.operations.append(self.processing_description)

            self.process_info.setText(
                "当前处理：" + " → ".join(self.operations[-3:])
            )

            if self.view_combo.currentIndex() != 1:
                self.view_combo.setCurrentIndex(1)
            else:
                self.update_visualization()

            self.statusBar().showMessage(
                f"处理完成：{self.processing_description}"
            )

        except Exception as error:
            self.on_processing_error(str(error))


    @Slot(str)
    def on_processing_error(self, message):

        self.statusBar().showMessage("音频处理失败")
        QMessageBox.critical(self, "音频处理失败", message)


    @Slot()
    def on_processing_thread_finished(self):

        self.is_processing = False
        self.processing_thread = None
        self.processing_worker = None
        self.processing_snapshot = None
        self.processing_description = ""

        self.set_processing_controls(True)


    # ==========================================
    # DSP处理算法
    # ==========================================

    def apply_gain_processing(self):

        gain = self.control_panel.gain_slider.value() / 100

        self.apply_processing(
            lambda audio: apply_gain(audio, gain),
            f"增益 {gain:.2f}×"
        )


    def apply_normalization(self):

        self.apply_processing(
            lambda audio: normalize_audio(audio, peak=0.95),
            "峰值归一化"
        )


    def apply_filter_processing(self):

        if self.sample_rate is None or self.is_processing:
            return

        # 提前读取全部参数，工作线程不访问GUI
        sample_rate = self.sample_rate
        filter_type = self.control_panel.filter_combo.currentIndex()
        low_cutoff = self.control_panel.low_freq_spin.value()
        high_cutoff = self.control_panel.high_freq_spin.value()

        nyquist = sample_rate / 2

        if filter_type == 0:
            if not 0 < high_cutoff < nyquist:
                QMessageBox.warning(self, "参数错误", "低通截止频率必须小于奈奎斯特频率")
                return

            self.apply_processing(
                lambda audio: lowpass_filter(audio, sample_rate, high_cutoff),
                f"低通 {high_cutoff} Hz"
            )

        elif filter_type == 1:
            if not 0 < low_cutoff < nyquist:
                QMessageBox.warning(self, "参数错误", "高通截止频率必须小于奈奎斯特频率")
                return

            self.apply_processing(
                lambda audio: highpass_filter(audio, sample_rate, low_cutoff),
                f"高通 {low_cutoff} Hz"
            )

        else:
            if not 0 < low_cutoff < high_cutoff < nyquist:
                QMessageBox.warning(self, "参数错误", "请确保 0 < 低截止 < 高截止 < 奈奎斯特频率")
                return

            self.apply_processing(
                lambda audio: bandpass_filter(
                    audio, sample_rate, low_cutoff, high_cutoff
                ),
                f"带通 {low_cutoff}～{high_cutoff} Hz"
            )


    def apply_denoise_processing(self):

        if self.processed_audio is None or self.is_processing:
            return

        # 提前读取全部参数，避免后台线程访问主窗口
        sample_rate = self.sample_rate
        noise_duration = self.control_panel.noise_duration_spin.value()
        alpha = self.control_panel.alpha_spin.value()
        beta = self.control_panel.beta_spin.value()

        duration = len(self.processed_audio) / sample_rate

        if noise_duration >= duration:
            QMessageBox.warning(self, "参数错误", "噪声采样时长必须小于音频总时长")
            return

        self.apply_processing(
            lambda audio: spectral_subtraction(
                audio,
                sample_rate,
                noise_duration=noise_duration,
                alpha=alpha,
                beta=beta
            ),
            f"频谱去噪 α={alpha:.2f}"
        )


    # ==========================================
    # 撤销与重置
    # ==========================================

    def undo_processing(self):

        if self.is_processing or self.recorder.is_recording:
            return

        if not self.undo_stack:
            QMessageBox.information(self, "提示", "没有可以撤销的操作")
            return

        self.stop_playback()
        self.invalidate_playback()

        audio, operations = self.undo_stack.pop()

        self.processed_audio = audio
        self.operations = operations

        if self.operations:
            self.process_info.setText(
                "当前处理：" + " → ".join(self.operations[-3:])
            )
        else:
            self.process_info.setText("当前处理：无")

        if self.view_combo.currentIndex() != 1:
            self.view_combo.setCurrentIndex(1)
        else:
            self.update_visualization()

        self.control_panel.undo_button.setEnabled(len(self.undo_stack) > 0)
        self.statusBar().showMessage("已撤销上一步处理")


    def reset_processing(self):

        if self.original_audio is None or self.is_processing:
            return

        self.stop_playback()
        self.invalidate_playback()

        self.processed_audio = self.original_audio.copy()
        self.operations.clear()
        self.undo_stack.clear()

        self.process_info.setText("当前处理：无")
        self.control_panel.undo_button.setEnabled(False)

        if self.view_combo.currentIndex() != 1:
            self.view_combo.setCurrentIndex(1)
        else:
            self.update_visualization()

        self.statusBar().showMessage("处理结果已重置为原声")


    # ==========================================
    # 音频导出
    # ==========================================

    def save_processed_audio(self):

        if self.processed_audio is None or self.is_processing:
            return

        default_path = RESULT_DIR / "processed_audio.wav"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存处理后的音频",
            str(default_path),
            "WAV 音频 (*.wav)"
        )

        if not file_path:
            return

        output_path = Path(file_path)

        if output_path.suffix.lower() != ".wav":
            output_path = output_path.with_suffix(".wav")

        if self.file_path is not None and output_path.resolve() == self.file_path.resolve():
            QMessageBox.warning(self, "保存失败", "不能覆盖原始音频文件")
            return

        try:
            save_audio(output_path, self.processed_audio, self.sample_rate)
            self.statusBar().showMessage(f"已保存：{output_path.name}")

        except Exception as error:
            QMessageBox.critical(self, "保存失败", str(error))


    # ==========================================
    # 程序退出
    # ==========================================

    def closeEvent(self, event):

        if self.is_processing:
            QMessageBox.information(
                self,
                "处理尚未完成",
                "请等待当前音频处理任务结束后再关闭程序"
            )
            event.ignore()
            return

        if self.recorder.is_recording:
            self.stop_recording()

        self.record_timer.stop()
        self.play_timer.stop()
        stop_audio()

        super().closeEvent(event)
