
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from dsp.analysis import analyze_audio


class AnalysisSignals(QObject):

    completed = Signal(object, int, object, int)
    failed = Signal(str, int)


class AnalysisWorker(QRunnable):
    def __init__(self, audio, sample_rate, request_id):
        super().__init__()

        self.audio = audio
        self.sample_rate = sample_rate
        self.request_id = request_id
        self.signals = AnalysisSignals()


    @Slot()
    def run(self):

        try:
            result = analyze_audio(self.audio, self.sample_rate)

            self.signals.completed.emit(
                self.audio,
                self.sample_rate,
                result,
                self.request_id
            )

        except Exception as error:
            self.signals.failed.emit(str(error), self.request_id)
