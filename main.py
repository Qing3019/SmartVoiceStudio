import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Smart Voice Studio")
    app.setStyle("Fusion")

    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f7fb;
        }
        QWidget#leftPanel {
            background-color: #ffffff;
            border-radius: 10px;
        }
        QLabel {
            color: #243247;
            font-size: 14px;
        }
        QPushButton {
            background-color: #2563eb;
            color: white;
            border: none;
            border-radius: 7px;
            padding: 12px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #1d4ed8;
        }
        QPushButton:disabled {
            background-color: #cbd5e1;
            color: #64748b;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()