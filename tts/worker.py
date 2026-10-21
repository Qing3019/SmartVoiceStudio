"""界面线程之外执行耗时的本机TTS调用。"""
from PySide6.QtCore import QObject, Signal, Slot
from tts.client import synthesize


class TTSWorker(QObject):
    completed = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, arguments):
        super().__init__()
        self.arguments = arguments

    @Slot()
    def run(self):
        try:
            result = synthesize(**self.arguments)
            self.completed.emit(result["output_path"])
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            self.finished.emit()
