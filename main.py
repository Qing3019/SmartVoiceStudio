"""SmartVoiceStudio V0.11 桌面应用入口与统一视觉主题。"""
import sys
from PySide6.QtWidgets import QApplication
import pyqtgraph as pg
from gui.main_window import MainWindow


STYLE = """
QMainWindow { background: #f3f6fc; }
QWidget { color: #26324b; font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 12px; }
QWidget#appRoot { background: #f3f6fc; }
QFrame#headerBar { background: #19243a; border-radius: 14px; }
QLabel#brandTitle { color: #ffffff; font-size: 21px; font-weight: 700; }
QLabel#brandSubtitle { color: #aebbd5; font-size: 11px; }
QLabel#versionBadge { color: #dce6ff; background: #304365; padding: 6px 10px; border-radius: 10px; font-weight: 600; }
QFrame#transportBar, QFrame#sidebarFrame, QFrame#rightFrame { background: #ffffff; border: 1px solid #e4eaf4; border-radius: 12px; }
QLabel#sidebarTitle { font-size: 15px; font-weight: 700; color: #1d2b48; }
QLabel#hintLabel, QLabel#helperText { color: #718099; font-size: 11px; }
QLabel#summaryLabel { color: #51617a; font-size: 11px; }
QLabel#toolbarLabel { color: #5e6e87; font-size: 11px; }
QPushButton { background: #ecf1fb; color: #263953; border: none; border-radius: 8px; padding: 7px 12px; font-weight: 600; min-height: 25px; }
QPushButton:hover { background: #dee7f9; }
QPushButton:pressed { background: #cfddfa; }
QPushButton:disabled { background: #eef1f6; color: #a6afc0; }
QPushButton[role="primary"] { background: #3565e8; color: white; }
QPushButton[role="primary"]:hover { background: #2855d4; }
QPushButton[role="primary"]:disabled { background: #cad4ec; color: #f4f6fd; }
QPushButton[role="danger"] { background: #fff0ee; color: #c5534f; }
QPushButton[role="danger"]:hover { background: #fee0dd; }
QPushButton[role="subtle"] { background: #f6f8fc; border: 1px solid #e7ebf3; }
QPushButton[role="text"] { background: transparent; color: #416adc; padding: 4px 6px; }
QPushButton[role="text"]:hover { background: #ecf2ff; }
QToolButton#collapseHeader { text-align: left; background: #ffffff; color: #26344f; border: none; border-bottom: 1px solid #eef1f6; border-radius: 7px; padding-left: 12px; font-size: 13px; font-weight: 600; }
QToolButton#collapseHeader:hover { background: #f7f9ff; }
QToolButton#collapseHeader:checked { color: #2c58d4; background: #f3f6ff; }
QFrame#collapseBody { background: #ffffff; border: none; border-bottom: 1px solid #ecf0f7; }
QWidget#collapseSection { background: white; border-radius: 8px; }
QFrame#referenceCard { background: #f6f8fd; border: 1px solid #ebeff6; border-radius: 8px; }
QGroupBox { font-weight: 600; border: 1px solid #e8edf7; border-radius: 9px; margin-top: 11px; padding: 12px 7px 6px 7px; background: #fff; }
QGroupBox::title { subcontrol-origin: margin; left: 11px; padding: 0 4px; color: #314466; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit { color: #223251; background: #fff; border: 1px solid #dae3f0; border-radius: 7px; padding: 5px 7px; selection-background-color: #8cacf3; }
QComboBox:hover, QLineEdit:hover, QTextEdit:hover { border-color: #a9bce5; }
QComboBox:focus, QLineEdit:focus, QTextEdit:focus { border: 1px solid #4773e8; }
QComboBox { min-height: 25px; }
QComboBox::drop-down { width: 18px; border: none; }
QSlider::groove:horizontal { background: #dbe4f4; height: 5px; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #527bea; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; border: 2px solid #4973e4; width: 13px; height: 13px; margin: -6px 0; border-radius: 7px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 7px; background: transparent; }
QScrollBar::handle:vertical { background: #cbd5e6; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QTabWidget::pane { border: 0; background: #ffffff; }
QTabBar::tab { color: #718199; background: transparent; padding: 9px 20px; font-weight: 600; }
QTabBar::tab:selected { color: #2f5fe0; border-bottom: 3px solid #416fe7; }
QStatusBar { color: #73809a; background: #f3f6fc; }
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Smart Voice Studio")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    pg.setConfigOptions(antialias=True, background="#ffffff", foreground="#52627a")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
