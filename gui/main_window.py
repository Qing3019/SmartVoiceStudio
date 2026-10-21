"""Smart Voice Studio V0.9：录音、播放、DSP、多版本、可视化、高级音效。"""
from pathlib import Path
from datetime import datetime
import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame, QComboBox, QSlider, QTabWidget, QSplitter,
    QFileDialog, QMessageBox
)
from config import RECORDING_DIR, RESULT_DIR, MAX_RECORD_SECONDS, RECORD_PREVIEW_SECONDS, GUI_UPDATE_INTERVAL
from audio.devices import get_audio_devices, get_input_sample_rate
from audio.recorder import StreamRecorder
from audio.player import StreamPlayer
from audio.wav_io import load_audio, save_audio
from dsp.enhance import apply_gain, normalize_audio
from dsp.filter import lowpass_filter, highpass_filter, bandpass_filter
from dsp.denoise import spectral_subtraction
from dsp.equalizer import three_band_eq
from dsp.effects import apply_preset, add_echo, change_speed, change_pitch
from dsp.advanced_effects import apply_advanced_effect
from gui.plot_panel import AudioAnalysisPanel
from gui.control_panel import DSPControlPanel
from gui.effects_panel import EffectsPanel
from gui.advanced_effects_panel import AdvancedEffectsPanel
from gui.workers import DSPWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 音频状态
        self.original_audio = None
        self.processed_audio = None
        self.sample_rate = None
        self.file_path = None
        self.operations = []
        self.version_snapshots = {}
        self.undo_stack = []
        self.max_undo_steps = 5
        # DSP 线程状态
        self.is_processing = False
        self.processing_thread = None
        self.processing_worker = None
        self.processing_description = ""
        self.processing_stage = ""
        self.processing_snapshot = None
        # 录音状态
        self.recorder = StreamRecorder()
        self.record_preview = np.empty(0, dtype=np.float32)
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(GUI_UPDATE_INTERVAL)
        self.record_timer.timeout.connect(self.update_recording)
        # 音频播放
        self.player = StreamPlayer()
        self.playback_name = ""
        self.playback_duration = 0.0
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(GUI_UPDATE_INTERVAL)
        self.play_timer.timeout.connect(self.update_playback)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Smart Voice Studio V0.9")
        self.resize(1450, 900)
        center = QWidget()
        self.setCentralWidget(center)
        main_layout = QHBoxLayout(center)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(355)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_widget = QWidget()
        left_widget.setObjectName("leftPanel")
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(16, 16, 16, 16)
        left.setSpacing(12)

        title = QLabel("Smart Voice Studio")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        left.addWidget(title)
        left.addWidget(QLabel("智能人声音效处理平台 · V0.9"))

        # 声卡设备
        left.addWidget(self.section_title("音频设备管理"))
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.refresh_button = QPushButton("刷新音频设备")
        left.addWidget(QLabel("输入设备"))
        left.addWidget(self.input_combo)
        left.addWidget(QLabel("输出设备"))
        left.addWidget(self.output_combo)
        left.addWidget(self.refresh_button)

        # 麦克风录音
        left.addWidget(self.section_title("麦克风录音"))
        self.record_button = QPushButton("开始录音")
        self.stop_record_button = QPushButton("停止录音")
        self.record_time_label = QLabel("录音时长：00:00.0")
        record_row = QHBoxLayout()
        record_row.addWidget(self.record_button)
        record_row.addWidget(self.stop_record_button)
        left.addLayout(record_row)
        left.addWidget(self.record_time_label)

        # 文件和播放器
        left.addWidget(self.section_title("音频管理与播放"))
        self.open_button = QPushButton("导入 WAV 音频")
        self.play_original_button = QPushButton("播放原声")
        self.play_processed_button = QPushButton("播放处理后")
        self.play_selected_button = QPushButton("播放选中版本")
        self.pause_button = QPushButton("暂停播放")
        self.stop_button = QPushButton("停止播放")
        self.save_button = QPushButton("保存处理结果")
        left.addWidget(self.open_button)
        play_row = QHBoxLayout()
        play_row.addWidget(self.play_original_button)
        play_row.addWidget(self.play_processed_button)
        left.addLayout(play_row)
        left.addWidget(self.play_selected_button)
        transport_row = QHBoxLayout()
        transport_row.addWidget(self.pause_button)
        transport_row.addWidget(self.stop_button)
        left.addLayout(transport_row)
        left.addWidget(self.save_button)
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_label = QLabel("00:00 / 00:00")
        left.addWidget(self.progress_slider)
        left.addWidget(self.progress_label)

        # 版本管理
        left.addWidget(self.section_title("音频文件信息"))
        self.file_info = QLabel("尚未导入音频")
        self.file_info.setWordWrap(True)
        left.addWidget(self.file_info)
        left.addWidget(self.section_title("音频版本管理"))
        self.view_combo = QComboBox()
        self.view_combo.addItem("原声", "original")
        self.view_combo.addItem("当前处理", "processed")
        self.process_info = QLabel("当前处理：无")
        self.process_info.setWordWrap(True)
        left.addWidget(self.view_combo)
        left.addWidget(self.process_info)

        # 所有 DSP 面板
        self.control_panel = DSPControlPanel()
        self.effects_panel = EffectsPanel()
        self.advanced_effects_panel = AdvancedEffectsPanel()
        left.addWidget(self.control_panel)
        left.addWidget(self.effects_panel)
        left.addWidget(self.advanced_effects_panel)
        left.addStretch()
        left_scroll.setWidget(left_widget)

        # 右侧标签页
        self.analysis_tabs = QTabWidget()
        self.analysis_panel = AudioAnalysisPanel()
        self.analysis_tabs.addTab(self.analysis_panel, "单路信号分析")
        compare_widget = QWidget()
        compare_layout = QHBoxLayout(compare_widget)
        compare_layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        origin_column = QWidget()
        origin_layout = QVBoxLayout(origin_column)
        origin_label = QLabel("原始音频")
        origin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        origin_layout.addWidget(origin_label)
        self.compare_original_panel = AudioAnalysisPanel()
        origin_layout.addWidget(self.compare_original_panel)
        result_column = QWidget()
        result_layout = QVBoxLayout(result_column)
        self.compare_result_label = QLabel("当前处理结果")
        self.compare_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.compare_result_label)
        self.compare_result_panel = AudioAnalysisPanel()
        result_layout.addWidget(self.compare_result_panel)
        splitter.addWidget(origin_column)
        splitter.addWidget(result_column)
        splitter.setSizes([500, 500])
        compare_layout.addWidget(splitter)
        self.analysis_tabs.addTab(compare_widget, "处理前后对比")
        main_layout.addWidget(left_scroll)
        main_layout.addWidget(self.analysis_tabs, stretch=1)

        # 设备/录音/播放器信号
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.output_combo.currentIndexChanged.connect(self.on_output_changed)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.open_button.clicked.connect(self.open_audio_file)
        self.play_original_button.clicked.connect(self.play_original)
        self.play_processed_button.clicked.connect(self.play_processed)
        self.play_selected_button.clicked.connect(self.play_selected)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_playback)
        self.save_button.clicked.connect(self.save_processed_audio)
        self.progress_slider.sliderReleased.connect(self.seek_playback)
        self.view_combo.currentIndexChanged.connect(self.update_visualization)
        self.analysis_tabs.currentChanged.connect(self.update_visualization)
        # 基础DSP信号
        self.control_panel.gain_button.clicked.connect(self.apply_gain_processing)
        self.control_panel.normalize_button.clicked.connect(self.apply_normalization)
        self.control_panel.filter_button.clicked.connect(self.apply_filter_processing)
        self.control_panel.denoise_button.clicked.connect(self.apply_denoise_processing)
        self.control_panel.undo_button.clicked.connect(self.undo_processing)
        self.control_panel.reset_button.clicked.connect(self.reset_processing)
        # 音效信号
        self.effects_panel.preset_button.clicked.connect(self.apply_effect_preset)
        self.effects_panel.eq_button.clicked.connect(self.apply_eq_processing)
        self.effects_panel.eq_reset_button.clicked.connect(self.effects_panel.reset_eq_parameters)
        self.effects_panel.echo_button.clicked.connect(self.apply_echo_processing)
        self.effects_panel.speed_button.clicked.connect(self.apply_speed_processing)
        self.effects_panel.pitch_button.clicked.connect(self.apply_pitch_processing)
        # 高级音效信号
        self.advanced_effects_panel.apply_button.clicked.connect(self.apply_advanced_processing)
        self.set_audio_controls(False)
        self.stop_record_button.setEnabled(False)
        self.refresh_devices()
        self.statusBar().showMessage("就绪，请录音或导入 WAV")

    def section_title(self, name):
        label = QLabel(name)
        label.setStyleSheet("font-size: 16px; font-weight: bold;")
        return label

    def set_audio_controls(self, enabled):
        self.play_original_button.setEnabled(enabled)
        self.play_processed_button.setEnabled(enabled)
        self.play_selected_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.view_combo.setEnabled(enabled)
        self.control_panel.setEnabled(enabled)
        self.effects_panel.setEnabled(enabled)
        self.advanced_effects_panel.setEnabled(enabled)
        self.control_panel.undo_button.setEnabled(enabled and bool(self.undo_stack))
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_slider.setEnabled(False)

    def set_processing_controls(self, enabled):
        self.open_button.setEnabled(enabled)
        self.record_button.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)
        self.input_combo.setEnabled(enabled)
        self.output_combo.setEnabled(enabled)
        self.set_audio_controls(enabled and self.original_audio is not None)

    # ==========================================
    # 设备与录音
    # ==========================================
    def refresh_devices(self):
        if self.recorder.is_recording or self.is_processing:
            return
        try:
            self.stop_playback()
            inputs, outputs = get_audio_devices()
            input_selected = self.input_combo.currentData()
            output_selected = self.output_combo.currentData()
            self.input_combo.blockSignals(True)
            self.output_combo.blockSignals(True)
            self.input_combo.clear()
            self.output_combo.clear()
            self.input_combo.addItem("系统默认输入设备", None)
            self.output_combo.addItem("系统默认输出设备", None)
            for index, name in inputs:
                self.input_combo.addItem(name, index)
            for index, name in outputs:
                self.output_combo.addItem(name, index)
            self.input_combo.setCurrentIndex(max(0, self.input_combo.findData(input_selected)))
            self.output_combo.setCurrentIndex(max(0, self.output_combo.findData(output_selected)))
            self.statusBar().showMessage("音频设备列表已更新")
        except Exception as error:
            QMessageBox.critical(self, "设备查询失败", str(error))
        finally:
            self.input_combo.blockSignals(False)
            self.output_combo.blockSignals(False)

    def on_output_changed(self, index):
        if self.player.is_active:
            self.stop_playback()

    def start_recording(self):
        if self.recorder.is_recording or self.is_processing:
            return
        try:
            self.stop_playback()
            device = self.input_combo.currentData()
            sample_rate = get_input_sample_rate(device)
            self.recorder.start(device=device, sample_rate=sample_rate)
            self.record_preview = np.empty(0, dtype=np.float32)
            self.set_audio_controls(False)
            for control in (self.open_button, self.record_button, self.input_combo, self.output_combo, self.refresh_button):
                control.setEnabled(False)
            self.stop_record_button.setEnabled(True)
            self.record_time_label.setText("录音时长：00:00.0")
            self.analysis_tabs.setCurrentIndex(0)
            self.analysis_panel.clear_audio()
            self.analysis_panel.waveform_plot.setTitle("Recording - Last 2 Seconds")
            self.record_timer.start()
            self.statusBar().showMessage(f"正在录音：{sample_rate} Hz")
        except Exception as error:
            if self.recorder.is_recording:
                try:
                    self.recorder.stop()
                except Exception:
                    pass
            self.record_timer.stop()
            self.restore_recording_controls()
            self.update_visualization()
            QMessageBox.critical(self, "录音启动失败", str(error))

    def update_recording(self):
        if not self.recorder.is_recording:
            return
        new_audio = self.recorder.read_available()
        if len(new_audio):
            self.record_preview = np.concatenate((self.record_preview, new_audio))
            max_samples = int(RECORD_PREVIEW_SECONDS * self.recorder.sample_rate)
            self.record_preview = self.record_preview[-max_samples:]
            self.analysis_panel.show_waveform(self.record_preview, self.recorder.sample_rate)
        duration = self.recorder.get_duration()
        self.record_time_label.setText(f"录音时长：{int(duration // 60):02d}:{duration % 60:04.1f}")
        if duration >= MAX_RECORD_SECONDS:
            self.stop_recording()

    def restore_recording_controls(self):
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.open_button.setEnabled(True)
        self.input_combo.setEnabled(True)
        self.output_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.set_audio_controls(self.original_audio is not None)
        self.analysis_panel.waveform_plot.setTitle("Waveform")

    def stop_recording(self):
        if not self.recorder.is_recording:
            return
        self.record_timer.stop()
        try:
            audio = self.recorder.stop()
            sample_rate = self.recorder.sample_rate
            if len(audio) < sample_rate * 0.2:
                raise ValueError("录音时间过短，请重新录制")
            name = datetime.now().strftime("record_%Y%m%d_%H%M%S.wav")
            output = RECORDING_DIR / name
            save_audio(output, audio, sample_rate)
            self.load_audio_data(audio, sample_rate, output)
            if self.recorder.overflow_count or self.recorder.dropped_blocks:
                QMessageBox.warning(self, "录音质量提醒", f"采集异常 {self.recorder.overflow_count} 次，丢失数据块 {self.recorder.dropped_blocks} 个")
            self.statusBar().showMessage(f"录音已保存：{name}")
        except Exception as error:
            QMessageBox.critical(self, "录音保存失败", str(error))
        finally:
            self.restore_recording_controls()
            self.update_visualization()

    # ==========================================
    # 文件与音频版本
    # ==========================================
    def update_file_info(self):
        if self.original_audio is None:
            self.file_info.setText("尚未导入音频")
            return
        original_duration = len(self.original_audio) / self.sample_rate
        result_duration = len(self.processed_audio) / self.sample_rate
        self.file_info.setText(
            f"文件名：{self.file_path.name}\n采样率：{self.sample_rate} Hz\n"
            f"原声时长：{original_duration:.2f} s\n处理后时长：{result_duration:.2f} s\n"
            f"原声采样点：{len(self.original_audio)}\n处理采样点：{len(self.processed_audio)}"
        )

    def load_audio_data(self, audio, sample_rate, file_path):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or sample_rate <= 0 or not np.all(np.isfinite(audio)):
            raise ValueError("音频数据无效")
        self.stop_playback()
        self.original_audio = audio.copy()
        self.processed_audio = audio.copy()
        self.sample_rate = sample_rate
        self.file_path = Path(file_path)
        self.operations.clear()
        self.undo_stack.clear()
        self.version_snapshots.clear()
        self.processing_snapshot = None
        self.process_info.setText("当前处理：无")
        self.update_file_info()
        self.refresh_version_combo("original")
        self.set_audio_controls(True)
        self.playback_duration = len(audio) / sample_rate
        self.update_progress_display(0, self.playback_duration)
        self.update_visualization()

    def open_audio_file(self):
        if self.is_processing or self.recorder.is_recording:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择 WAV 音频文件", str(RECORDING_DIR), "WAV 音频 (*.wav)")
        if not path:
            return
        try:
            audio, sample_rate = load_audio(path)
            self.load_audio_data(audio, sample_rate, path)
            self.statusBar().showMessage(f"已加载：{Path(path).name}")
        except Exception as error:
            QMessageBox.critical(self, "音频加载失败", str(error))

    def refresh_version_combo(self, selected_key=None):
        if selected_key is None:
            selected_key = self.view_combo.currentData()
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItem("原声", "original")
        self.view_combo.addItem("当前处理", "processed")
        for key, name in (("denoise", "去噪结果"), ("enhance", "增强结果"), ("effect", "音效结果")):
            if key in self.version_snapshots:
                self.view_combo.addItem(name, key)
        self.view_combo.setCurrentIndex(max(0, self.view_combo.findData(selected_key)))
        self.view_combo.blockSignals(False)

    def get_audio_by_key(self, key):
        if key == "original":
            return self.original_audio, "原声"
        if key == "processed":
            return self.processed_audio, "当前处理"
        names = {"denoise": "去噪结果", "enhance": "增强结果", "effect": "音效结果"}
        return self.version_snapshots.get(key), names.get(key, "未知版本")

    def get_selected_audio(self):
        return self.get_audio_by_key(self.view_combo.currentData())

    def update_visualization(self, index=None):
        if self.original_audio is None or self.recorder.is_recording:
            return
        audio, name = self.get_selected_audio()
        if audio is None:
            return
        if self.analysis_tabs.currentIndex() == 0:
            self.analysis_panel.show_audio(audio, self.sample_rate)
        else:
            self.compare_original_panel.show_audio(self.original_audio, self.sample_rate)
            self.compare_result_panel.show_audio(audio, self.sample_rate)
            self.compare_result_label.setText(name)
        self.statusBar().showMessage(f"当前显示：{name}")

    # ==========================================
    # 播放与进度
    # ==========================================
    def play_original(self):
        if self.original_audio is not None:
            self.start_playback(self.original_audio, "原声")

    def play_processed(self):
        if self.processed_audio is not None:
            self.start_playback(self.processed_audio, "当前处理")

    def play_selected(self):
        audio, name = self.get_selected_audio()
        if audio is not None:
            self.start_playback(audio, name)

    def start_playback(self, audio, name):
        if audio is None or self.recorder.is_recording or self.is_processing:
            return
        self.stop_playback()
        try:
            self.player.start(audio, self.sample_rate, device=self.output_combo.currentData())
            self.playback_name = name
            self.playback_duration = self.player.get_duration()
            self.play_original_button.setEnabled(False)
            self.play_processed_button.setEnabled(False)
            self.play_selected_button.setEnabled(False)
            self.pause_button.setText("暂停播放")
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.progress_slider.setEnabled(True)
            self.play_timer.start()
            self.update_progress_display(0, self.playback_duration)
            self.statusBar().showMessage(f"正在播放：{name}")
        except Exception as error:
            self.stop_playback()
            QMessageBox.critical(self, "音频播放失败", str(error))

    def toggle_pause(self):
        if not self.player.is_active:
            return
        if self.player.is_paused:
            self.player.resume()
            self.pause_button.setText("暂停播放")
            self.play_timer.start()
            self.statusBar().showMessage(f"继续播放：{self.playback_name}")
        else:
            self.player.pause()
            self.pause_button.setText("继续播放")
            self.play_timer.stop()
            self.statusBar().showMessage(f"已暂停：{self.playback_name}")
        self.update_progress_display(self.player.get_position(), self.player.get_duration())

    def update_playback(self):
        if not self.player.is_active:
            return
        self.update_progress_display(self.player.get_position(), self.player.get_duration())
        if self.player.is_finished:
            self.on_playback_finished()

    def seek_playback(self):
        if not self.player.is_active:
            return
        duration = self.player.get_duration()
        position = self.progress_slider.value() / 1000 * duration
        self.player.seek(position)
        self.update_progress_display(position, duration)
        if self.player.is_finished:
            self.on_playback_finished()

    def stop_playback(self):
        self.play_timer.stop()
        self.player.stop()
        enabled = self.original_audio is not None and not self.recorder.is_recording and not self.is_processing
        self.play_original_button.setEnabled(enabled)
        self.play_processed_button.setEnabled(enabled)
        self.play_selected_button.setEnabled(enabled)
        self.pause_button.setText("暂停播放")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_slider.setEnabled(False)
        self.update_progress_display(0, self.playback_duration)

    def on_playback_finished(self):
        duration = self.playback_duration
        self.stop_playback()
        self.update_progress_display(duration, duration)
        self.statusBar().showMessage("播放结束")

    def update_progress_display(self, position, duration):
        if duration <= 0:
            self.progress_slider.setValue(0)
            self.progress_label.setText("00:00 / 00:00")
            return
        progress = int(position / duration * 1000)
        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(min(1000, max(0, progress)))
        self.progress_label.setText(f"{self.format_time(position)} / {self.format_time(duration)}")

    def format_time(self, seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def invalidate_playback(self, audio=None):
        self.stop_playback()
        if audio is None:
            audio = self.original_audio
        self.playback_duration = len(audio) / self.sample_rate if audio is not None and self.sample_rate else 0.0
        self.update_progress_display(0, self.playback_duration)

    # ==========================================
    # DSP 后台执行 + 版本管理
    # ==========================================
    def apply_processing(self, process_func, description, stage="enhance"):
        if self.processed_audio is None or self.recorder.is_recording or self.is_processing:
            return
        self.invalidate_playback(self.processed_audio)
        self.processing_snapshot = (self.processed_audio, self.operations.copy(), self.version_snapshots.copy())
        self.processing_description = description
        self.processing_stage = stage
        self.processing_thread = QThread(self)
        self.processing_worker = DSPWorker(self.processed_audio, process_func)
        self.processing_worker.moveToThread(self.processing_thread)
        self.processing_thread.started.connect(self.processing_worker.run)
        self.processing_worker.result_ready.connect(self.on_processing_success)
        self.processing_worker.error_occurred.connect(self.on_processing_error)
        self.processing_worker.finished.connect(self.processing_thread.quit)
        self.processing_worker.finished.connect(self.processing_worker.deleteLater)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)
        self.processing_thread.finished.connect(self.on_processing_thread_finished)
        self.is_processing = True
        self.set_processing_controls(False)
        self.statusBar().showMessage(f"正在处理：{description}")
        self.processing_thread.start()

    @Slot(object)
    def on_processing_success(self, result):
        try:
            result = np.asarray(result, dtype=np.float32)
            if result.ndim != 1 or len(result) == 0 or not np.all(np.isfinite(result)):
                raise ValueError("DSP处理结果无效")
            self.processed_audio = result
            self.operations.append(self.processing_description)
            self.version_snapshots[self.processing_stage] = result
            if self.processing_snapshot is not None:
                self.undo_stack.append(self.processing_snapshot)
            if len(self.undo_stack) > self.max_undo_steps:
                self.undo_stack.pop(0)
            self.process_info.setText("当前处理：" + " → ".join(self.operations[-3:]))
            self.update_file_info()
            self.invalidate_playback(self.processed_audio)
            self.refresh_version_combo("processed")
            self.update_visualization()
            peak = float(np.max(np.abs(result)))
            if peak > 1.0:
                self.statusBar().showMessage(f"处理完成：{self.processing_description}（峰值 {peak:.2f} 超出 ±1，播放会限幅）")
            else:
                self.statusBar().showMessage(f"处理完成：{self.processing_description}")
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
        self.processing_stage = ""
        self.set_processing_controls(True)

    def apply_gain_processing(self):
        if self.processed_audio is None or self.is_processing:
            return
        gain = self.control_panel.gain_slider.value() / 100
        self.apply_processing(lambda a: apply_gain(a, gain), f"增益 {gain:.2f}×", stage="enhance")

    def apply_normalization(self):
        if self.processed_audio is None or self.is_processing:
            return
        self.apply_processing(lambda a: normalize_audio(a, peak=0.95), "峰值归一化", stage="enhance")

    def apply_filter_processing(self):
        if self.sample_rate is None or self.is_processing:
            return
        sr = self.sample_rate
        mode = self.control_panel.filter_combo.currentIndex()
        low = self.control_panel.low_freq_spin.value()
        high = self.control_panel.high_freq_spin.value()
        if mode == 0:
            if not 0 < high < sr / 2:
                QMessageBox.warning(self, "参数错误", "低通截止频率必须在奈奎斯特频率以内")
                return
            self.apply_processing(lambda a: lowpass_filter(a, sr, high), f"低通 {high} Hz")
        elif mode == 1:
            if not 0 < low < sr / 2:
                QMessageBox.warning(self, "参数错误", "高通截止频率必须在奈奎斯特频率以内")
                return
            self.apply_processing(lambda a: highpass_filter(a, sr, low), f"高通 {low} Hz")
        else:
            if not 0 < low < high < sr / 2:
                QMessageBox.warning(self, "参数错误", "带通频率设置无效")
                return
            self.apply_processing(lambda a: bandpass_filter(a, sr, low, high), f"带通 {low}～{high} Hz")

    def apply_denoise_processing(self):
        if self.processed_audio is None or self.is_processing:
            return
        sr = self.sample_rate
        duration = self.control_panel.noise_duration_spin.value()
        alpha = self.control_panel.alpha_spin.value()
        beta = self.control_panel.beta_spin.value()
        if duration >= len(self.processed_audio) / sr:
            QMessageBox.warning(self, "参数错误", "噪声估计时长必须小于音频总时长")
            return
        self.apply_processing(lambda a: spectral_subtraction(a, sr, noise_duration=duration, alpha=alpha, beta=beta), f"频谱去噪 α={alpha:.2f}", stage="denoise")

    def apply_effect_preset(self):
        if self.processed_audio is None or self.is_processing:
            return
        sr = self.sample_rate
        name = self.effects_panel.preset_combo.currentText()
        self.apply_processing(lambda a: apply_preset(a, sr, name), f"预设：{name}", stage="effect")

    def apply_eq_processing(self):
        if self.processed_audio is None or self.is_processing:
            return
        sr = self.sample_rate
        low = self.effects_panel.low_slider.value()
        mid = self.effects_panel.mid_slider.value()
        high = self.effects_panel.high_slider.value()
        if low == mid == high == 0:
            QMessageBox.information(self, "提示", "请先调整EQ参数")
            return
        self.apply_processing(lambda a: three_band_eq(a, sr, low_db=low, mid_db=mid, high_db=high), f"EQ 低{low:+d} 中{mid:+d} 高{high:+d} dB", stage="enhance")

    def apply_echo_processing(self):
        if self.processed_audio is None or self.is_processing:
            return
        sr = self.sample_rate
        delay = self.effects_panel.delay_spin.value()
        strength = self.effects_panel.strength_spin.value()
        repeats = self.effects_panel.repeats_spin.value()
        self.apply_processing(lambda a: add_echo(a, sr, delay_ms=delay, strength=strength, repeats=repeats), f"回声 {delay}ms，强度{strength:.2f}", stage="effect")

    def apply_speed_processing(self):
        if self.processed_audio is None or self.is_processing:
            return
        speed = self.effects_panel.speed_spin.value()
        if abs(speed - 1.0) < 1e-8:
            QMessageBox.information(self, "提示", "倍率为1.0，不会产生变化")
            return
        self.apply_processing(lambda a: change_speed(a, speed), f"变速 {speed:.2f}×", stage="effect")

    def apply_pitch_processing(self):
        if self.processed_audio is None or self.is_processing:
            return
        sr = self.sample_rate
        steps = self.effects_panel.pitch_spin.value()
        if steps == 0:
            QMessageBox.information(self, "提示", "变调为0半音")
            return
        self.apply_processing(lambda a: change_pitch(a, sr, steps), f"变调 {steps:+d} 半音", stage="effect")

    def apply_advanced_processing(self):
        if self.processed_audio is None or self.is_processing:
            return
        sr = self.sample_rate
        name = self.advanced_effects_panel.effect_combo.currentText()
        intensity = self.advanced_effects_panel.strength_slider.value() / 100
        parameter = self.advanced_effects_panel.parameter_spin.value()
        if intensity == 0:
            QMessageBox.information(self, "提示", "音效强度为零，结果与原声一致")
            return
        self.apply_processing(lambda a: apply_advanced_effect(a, sr, name, intensity=intensity, parameter=parameter), f"高级音效：{name}，强度{intensity:.2f}", stage="effect")

    def undo_processing(self):
        if self.is_processing or self.recorder.is_recording or not self.undo_stack:
            return
        self.stop_playback()
        audio, operations, versions = self.undo_stack.pop()
        self.processed_audio = audio
        self.operations = operations
        self.version_snapshots = versions
        self.process_info.setText("当前处理：" + " → ".join(operations[-3:]) if operations else "当前处理：无")
        self.update_file_info()
        self.invalidate_playback(self.processed_audio)
        self.refresh_version_combo("processed")
        self.update_visualization()
        self.control_panel.undo_button.setEnabled(bool(self.undo_stack))
        self.statusBar().showMessage("已撤销上一步")

    def reset_processing(self):
        if self.original_audio is None or self.is_processing:
            return
        self.stop_playback()
        self.processed_audio = self.original_audio.copy()
        self.operations.clear()
        self.undo_stack.clear()
        self.version_snapshots.clear()
        self.process_info.setText("当前处理：无")
        self.control_panel.undo_button.setEnabled(False)
        self.update_file_info()
        self.invalidate_playback(self.original_audio)
        self.refresh_version_combo("processed")
        self.update_visualization()
        self.statusBar().showMessage("已恢复原声")

    def save_processed_audio(self):
        if self.processed_audio is None or self.is_processing:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存处理后的音频", str(RESULT_DIR / "processed_audio.wav"), "WAV 音频 (*.wav)")
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() != ".wav":
            output = output.with_suffix(".wav")
        if self.file_path is not None and output.resolve() == self.file_path.resolve():
            QMessageBox.warning(self, "保存失败", "不能覆盖当前原声文件")
            return
        try:
            save_audio(output, self.processed_audio, self.sample_rate)
            self.statusBar().showMessage(f"已保存：{output.name}")
        except Exception as error:
            QMessageBox.critical(self, "保存失败", str(error))

    def closeEvent(self, event):
        if self.is_processing:
            QMessageBox.information(self, "正在处理", "请等待DSP任务完成后关闭程序")
            event.ignore()
            return
        if self.recorder.is_recording:
            try:
                self.stop_recording()
            except Exception:
                pass
        self.record_timer.stop()
        self.play_timer.stop()
        self.player.stop()
        super().closeEvent(event)
