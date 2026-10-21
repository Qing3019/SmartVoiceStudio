"""可折叠面板：用于缩短 SmartVoiceStudio 左侧滚动区域。"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QToolButton, QFrame, QSizePolicy


class CollapsibleSection(QWidget):
    toggled = Signal(bool)

    def __init__(self, title, content, expanded=False, subtitle=""):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.setObjectName("collapseSection")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QToolButton()
        self.header.setObjectName("collapseHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.setMinimumHeight(51)
        self.header.setAccessibleName(title)
        layout.addWidget(self.header)

        self.body = QFrame()
        self.body.setObjectName("collapseBody")
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(12, 9, 12, 12)
        body_layout.setSpacing(0)
        body_layout.addWidget(content)
        layout.addWidget(self.body)

        self.header.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.body.setVisible(expanded)
        if self.header.isChecked() != expanded:
            self.header.setChecked(expanded)
        indicator = "▾" if expanded else "▸"
        self.header.setText(f"{indicator}   {self.title}")
        if self.subtitle:
            self.header.setToolTip(self.subtitle)
        self.toggled.emit(expanded)

    def is_expanded(self):
        return self.header.isChecked()
