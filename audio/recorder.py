
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import SAMPLE_RATE, CHANNELS, INPUT_DEVICE


def record_audio(file_path, seconds=5, sample_rate=SAMPLE_RATE, channels=CHANNELS, input_device=INPUT_DEVICE):

    # 固定时长录音
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=input_device
    )

    sd.wait()

    # 保存录音文件
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(file_path, audio, sample_rate)

    return audio


class StreamRecorder:
    def __init__(self):

        self.stream = None
        self.sample_rate = None
        self.audio_queue = queue.Queue(maxsize=512)
        self.audio_chunks = []
        self.total_samples = 0
        self.is_recording = False
        self.overflow_count = 0
        self.dropped_blocks = 0


    def audio_callback(self, indata, frames, time_info, status):

        # 音频回调只负责接收采样数据
        if status:
            self.overflow_count += 1

        try:
            self.audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped_blocks += 1


    def start(self, device=None, sample_rate=SAMPLE_RATE):

        if self.is_recording:
            raise RuntimeError("当前已经处于录音状态")

        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=512)
        self.audio_chunks = []
        self.total_samples = 0
        self.overflow_count = 0
        self.dropped_blocks = 0

        # 检查输入设备和采样率
        sd.check_input_settings(
            device=device,
            channels=1,
            samplerate=sample_rate,
            dtype="float32"
        )

        stream = None

        try:
            stream = sd.InputStream(
                device=device,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self.audio_callback
            )

            stream.start()
            self.stream = stream
            self.is_recording = True

        except Exception:
            if stream is not None:
                stream.close()
            raise


    def read_available(self):

        chunks = []

        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break

            self.audio_chunks.append(chunk)
            self.total_samples += len(chunk)
            chunks.append(chunk)

        if not chunks:
            return np.empty(0, dtype=np.float32)

        return np.concatenate(chunks)


    def get_duration(self):

        if self.sample_rate is None:
            return 0.0

        return self.total_samples / self.sample_rate


    def stop(self):

        if not self.is_recording:
            raise RuntimeError("当前没有正在进行的录音")

        try:
            self.stream.stop()

        finally:
            self.stream.close()
            self.stream = None
            self.is_recording = False

        # 读取剩余音频数据
        self.read_available()

        if not self.audio_chunks:
            return np.empty(0, dtype=np.float32)

        return np.concatenate(self.audio_chunks)
