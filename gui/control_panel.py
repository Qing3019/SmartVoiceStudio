
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox
)


class DSPControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()


    def init_ui(self):

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(12)

        # 声音增强
        gain_group = QGroupBox("声音增强")
        gain_layout = QVBoxLayout(gain_group)

        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 300)
        self.gain_slider.setValue(100)

        self.gain_value_label = QLabel("1.00 ×")
        self.gain_slider.valueChanged.connect(self.update_gain_label)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("增益"))
        gain_row.addWidget(self.gain_slider)
        gain_row.addWidget(self.gain_value_label)

        self.gain_button = QPushButton("应用增益")
        self.normalize_button = QPushButton("峰值归一化")

        gain_layout.addLayout(gain_row)
        gain_layout.addWidget(self.gain_button)
        gain_layout.addWidget(self.normalize_button)
        layout.addWidget(gain_group)

        # 数字滤波
        filter_group = QGroupBox("数字滤波")
        filter_layout = QFormLayout(filter_group)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["低通滤波", "高通滤波", "带通滤波"])

        self.low_freq_spin = QSpinBox()
        self.low_freq_spin.setRange(1, 20000)
        self.low_freq_spin.setValue(300)
        self.low_freq_spin.setSuffix(" Hz")

        self.high_freq_spin = QSpinBox()
        self.high_freq_spin.setRange(1, 20000)
        self.high_freq_spin.setValue(3400)
        self.high_freq_spin.setSuffix(" Hz")

        self.filter_button = QPushButton("应用滤波")

        filter_layout.addRow("滤波类型", self.filter_combo)
        filter_layout.addRow("低截止频率", self.low_freq_spin)
        filter_layout.addRow("高截止频率", self.high_freq_spin)
        filter_layout.addRow(self.filter_button)
        layout.addWidget(filter_group)

        # 频谱减法去噪
        denoise_group = QGroupBox("语音去噪")
        denoise_layout = QFormLayout(denoise_group)

        self.noise_duration_spin = QDoubleSpinBox()
        self.noise_duration_spin.setRange(0.1, 2.0)
        self.noise_duration_spin.setSingleStep(0.1)
        self.noise_duration_spin.setValue(0.4)
        self.noise_duration_spin.setSuffix(" s")

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.5, 4.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(1.2)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 0.3)
        self.beta_spin.setSingleStep(0.01)
        self.beta_spin.setDecimals(2)
        self.beta_spin.setValue(0.02)

        self.denoise_button = QPushButton("一键频谱去噪")

        denoise_layout.addRow("噪声采样时长", self.noise_duration_spin)
        denoise_layout.addRow("过减系数 α", self.alpha_spin)
        denoise_layout.addRow("频谱底系数 β", self.beta_spin)
        denoise_layout.addRow(self.denoise_button)
        layout.addWidget(denoise_group)

        # 处理历史
        self.undo_button = QPushButton("撤销上一步")
        self.undo_button.setEnabled(False)

        self.reset_button = QPushButton("重置全部处理")

        layout.addWidget(self.undo_button)
        layout.addWidget(self.reset_button)
        layout.addStretch()


    def update_gain_label(self, value):
        self.gain_value_label.setText(f"{value / 100:.2f} ×")
