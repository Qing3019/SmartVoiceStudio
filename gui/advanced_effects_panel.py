"""高级音效参数界面。"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QPushButton, QComboBox, QSlider, QDoubleSpinBox
from dsp.advanced_effects import EFFECT_SETTINGS


class AdvancedEffectsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        group = QGroupBox("发挥任务：高级音效")
        group_layout = QVBoxLayout(group)

        self.effect_combo = QComboBox()
        self.effect_combo.addItems(list(EFFECT_SETTINGS.keys()))
        group_layout.addWidget(QLabel("音效类型"))
        group_layout.addWidget(self.effect_combo)

        strength_row = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(60)
        self.strength_label = QLabel("60%")
        strength_row.addWidget(self.strength_slider)
        strength_row.addWidget(self.strength_label)
        group_layout.addWidget(QLabel("音效强度"))
        group_layout.addLayout(strength_row)

        parameter_form = QFormLayout()
        self.parameter_label = QLabel("效果参数")
        self.parameter_spin = QDoubleSpinBox()
        parameter_form.addRow(self.parameter_label, self.parameter_spin)
        group_layout.addLayout(parameter_form)

        self.apply_button = QPushButton("应用高级音效")
        self.reset_button = QPushButton("重置音效参数")
        group_layout.addWidget(self.apply_button)
        group_layout.addWidget(self.reset_button)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        group_layout.addWidget(self.description_label)
        layout.addWidget(group)

        self.effect_combo.currentTextChanged.connect(self.update_parameter)
        self.strength_slider.valueChanged.connect(self.update_strength_label)
        self.reset_button.clicked.connect(self.reset_parameters)
        self.update_parameter(self.effect_combo.currentText())

    def update_strength_label(self, value):
        self.strength_label.setText(f"{value}%")

    def update_parameter(self, effect_name):
        if effect_name not in EFFECT_SETTINGS:
            return
        name, minimum, maximum, step, default, suffix = EFFECT_SETTINGS[effect_name]
        self.parameter_label.setText(name)
        self.parameter_spin.setDecimals(0 if step >= 1 else 2)
        self.parameter_spin.setRange(minimum, maximum)
        self.parameter_spin.setSingleStep(step)
        self.parameter_spin.setSuffix(suffix)
        self.parameter_spin.setValue(default)
        descriptions = {
            "房间混响": "模拟室内反射和混响尾音",
            "机械电音": "环形调制与非线性失真",
            "合唱音效": "周期性延迟调制，增加声音层次",
            "广播压缩": "压缩响度动态范围",
            "8-bit数字音": "位深量化，模拟复古数字音质",
            "空间复合音效": "合唱、延迟、混响串联处理"
        }
        self.description_label.setText(descriptions.get(effect_name, ""))

    def reset_parameters(self):
        self.strength_slider.setValue(60)
        self.update_parameter(self.effect_combo.currentText())
