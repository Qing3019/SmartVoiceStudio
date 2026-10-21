
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


class EffectsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()


    def init_ui(self):

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(12)

        # 音效预设
        preset_group = QGroupBox("音效预设")
        preset_layout = QVBoxLayout(preset_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "磁性人声",
            "娃娃音",
            "大叔音",
            "电话音",
            "山谷回声"
        ])

        self.preset_button = QPushButton("应用音效预设")

        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.preset_button)
        layout.addWidget(preset_group)

        # 三段均衡器
        eq_group = QGroupBox("三段均衡器 EQ")
        eq_layout = QVBoxLayout(eq_group)

        self.low_slider = self.create_eq_slider(eq_layout, "低频 200 Hz")
        self.mid_slider = self.create_eq_slider(eq_layout, "中频 1500 Hz")
        self.high_slider = self.create_eq_slider(eq_layout, "高频 4500 Hz")

        self.eq_button = QPushButton("应用均衡器")
        self.eq_reset_button = QPushButton("重置EQ参数")

        eq_layout.addWidget(self.eq_button)
        eq_layout.addWidget(self.eq_reset_button)
        layout.addWidget(eq_group)

        # 回声效果
        echo_group = QGroupBox("回声效果")
        echo_layout = QFormLayout(echo_group)

        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(50, 1500)
        self.delay_spin.setSingleStep(50)
        self.delay_spin.setValue(350)
        self.delay_spin.setSuffix(" ms")

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 0.9)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setValue(0.45)

        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 6)
        self.repeats_spin.setValue(3)

        self.echo_button = QPushButton("应用回声")

        echo_layout.addRow("回声延迟", self.delay_spin)
        echo_layout.addRow("回声强度", self.strength_spin)
        echo_layout.addRow("重复次数", self.repeats_spin)
        echo_layout.addRow(self.echo_button)
        layout.addWidget(echo_group)

        # 变速和变调
        voice_group = QGroupBox("变速与变调")
        voice_layout = QFormLayout(voice_group)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix(" ×")

        self.pitch_spin = QSpinBox()
        self.pitch_spin.setRange(-8, 8)
        self.pitch_spin.setSingleStep(1)
        self.pitch_spin.setValue(0)
        self.pitch_spin.setSuffix(" 半音")

        self.speed_button = QPushButton("应用变速")
        self.pitch_button = QPushButton("应用变调")

        voice_layout.addRow("播放倍率", self.speed_spin)
        voice_layout.addRow(self.speed_button)
        voice_layout.addRow("音调变化", self.pitch_spin)
        voice_layout.addRow(self.pitch_button)
        layout.addWidget(voice_group)

        layout.addStretch()


    def create_eq_slider(self, layout, name):

        row = QHBoxLayout()

        name_label = QLabel(name)
        value_label = QLabel("0 dB")

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-12, 12)
        slider.setValue(0)

        slider.valueChanged.connect(
            lambda value: value_label.setText(f"{value:+d} dB")
        )

        row.addWidget(name_label)
        row.addWidget(slider)
        row.addWidget(value_label)

        layout.addLayout(row)

        return slider


    def reset_eq_parameters(self):

        self.low_slider.setValue(0)
        self.mid_slider.setValue(0)
        self.high_slider.setValue(0)
