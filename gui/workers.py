
import numpy as np

from PySide6.QtCore import QObject, Signal, Slot


class DSPWorker(QObject):

    result_ready = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, audio, process_func):
        super().__init__()

        self.audio = np.asarray(audio, dtype=np.float32).copy()
        self.process_func = process_func


    @Slot()
    def run(self):

        try:
            result = self.process_func(self.audio)
            result = np.asarray(result, dtype=np.float32)

            self.result_ready.emit(result)

        except Exception as error:
            self.error_occurred.emit(str(error))

        finally:
            self.finished.emit()
