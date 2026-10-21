
import threading
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from config import OUTPUT_DEVICE


class StreamPlayer:
    def __init__(self):

        self.stream = None
        self.audio = None
        self.sample_rate = None
        self.cursor = 0
        self.paused = False
        self.finished = False
        self.lock = threading.Lock()


    def audio_callback(self, outdata, frames, time_info, status):

        # 暂停或结束时输出静音
        outdata.fill(0)

        with self.lock:

            if self.audio is None or self.paused or self.finished:
                return

            remaining = len(self.audio) - self.cursor
            count = min(frames, remaining)

            if count > 0:
                outdata[:count, 0] = self.audio[self.cursor:self.cursor + count]
                self.cursor += count

            if self.cursor >= len(self.audio):
                self.finished = True


    def start(self, audio, sample_rate, device=None, start_seconds=0):

        self.stop()

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim != 1 or len(audio) == 0:
            raise ValueError("播放音频必须是非空单声道数组")

        if not np.all(np.isfinite(audio)):
            raise ValueError("播放音频包含无效采样值")

        if sample_rate <= 0:
            raise ValueError("播放采样率必须大于0")

        playback_rate = int(sample_rate)

        # 检查输出设备采样率
        try:
            sd.check_output_settings(
                device=device,
                channels=1,
                samplerate=playback_rate,
                dtype="float32"
            )

        except (sd.PortAudioError, ValueError):
            device_info = sd.query_devices(device, "output")
            target_rate = int(round(device_info["default_samplerate"]))

            if target_rate <= 0:
                raise ValueError("输出设备没有有效的默认采样率")

            sd.check_output_settings(
                device=device,
                channels=1,
                samplerate=target_rate,
                dtype="float32"
            )

            factor = gcd(playback_rate, target_rate)
            audio = resample_poly(
                audio,
                target_rate // factor,
                playback_rate // factor
            ).astype(np.float32)

            playback_rate = target_rate

        audio = np.ascontiguousarray(audio)
        start_sample = min(len(audio), max(0, int(start_seconds * playback_rate)))

        # 先建立播放状态，再启动声卡流
        with self.lock:
            self.audio = audio
            self.sample_rate = playback_rate
            self.cursor = start_sample
            self.paused = False
            self.finished = start_sample >= len(audio)

        stream = None

        try:
            stream = sd.OutputStream(
                device=device,
                samplerate=playback_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self.audio_callback
            )

            stream.start()
            self.stream = stream

        except Exception:
            if stream is not None:
                stream.close()

            with self.lock:
                self.audio = None
                self.sample_rate = None
                self.cursor = 0
                self.paused = False
                self.finished = False

            raise


    def pause(self):

        with self.lock:
            if self.stream is not None and not self.finished:
                self.paused = True


    def resume(self):

        with self.lock:
            if self.stream is not None and not self.finished:
                self.paused = False


    def seek(self, seconds):

        with self.lock:

            if self.audio is None or self.sample_rate is None:
                return

            position = int(seconds * self.sample_rate)
            self.cursor = min(len(self.audio), max(0, position))
            self.finished = self.cursor >= len(self.audio)


    def get_position(self):

        with self.lock:
            if self.sample_rate is None:
                return 0.0

            return self.cursor / self.sample_rate


    def get_duration(self):

        with self.lock:
            if self.audio is None or self.sample_rate is None:
                return 0.0

            return len(self.audio) / self.sample_rate


    @property
    def is_active(self):

        with self.lock:
            return self.stream is not None


    @property
    def is_paused(self):

        with self.lock:
            return self.paused


    @property
    def is_finished(self):

        with self.lock:
            return self.finished


    def stop(self):

        with self.lock:
            stream = self.stream
            self.stream = None
            self.audio = None
            self.sample_rate = None
            self.cursor = 0
            self.paused = False
            self.finished = False

        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

def play_audio_async(audio, sample_rate, output_device=OUTPUT_DEVICE):

    sd.play(
        np.asarray(audio, dtype=np.float32),
        samplerate=sample_rate,
        device=output_device,
        blocking=False
    )


def play_audio(audio, sample_rate, output_device=OUTPUT_DEVICE):

    play_audio_async(audio, sample_rate, output_device)
    sd.wait()


def stop_audio():
    sd.stop()
