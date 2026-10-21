"""基于 OutputStream 的播放器，支持暂停和采样点级跳转。"""
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
        outdata.fill(0)
        with self.lock:
            if self.audio is None or self.paused or self.finished:
                return
            count = min(frames, len(self.audio) - self.cursor)
            if count > 0:
                outdata[:count, 0] = np.clip(self.audio[self.cursor:self.cursor + count], -1, 1)
                self.cursor += count
            if self.cursor >= len(self.audio):
                self.finished = True

    def start(self, audio, sample_rate, device=None, start_seconds=0):
        self.stop()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or len(audio) == 0 or not np.all(np.isfinite(audio)):
            raise ValueError("必须输入有效的非空单声道音频")
        if sample_rate <= 0:
            raise ValueError("采样率必须大于0")
        rate = int(sample_rate)
        try:
            sd.check_output_settings(device=device, channels=1, samplerate=rate, dtype="float32")
        except (sd.PortAudioError, ValueError):
            device_info = sd.query_devices(device, "output")
            target_rate = int(round(device_info["default_samplerate"]))
            sd.check_output_settings(device=device, channels=1, samplerate=target_rate, dtype="float32")
            factor = gcd(rate, target_rate)
            audio = resample_poly(audio, target_rate // factor, rate // factor).astype(np.float32)
            rate = target_rate
        audio = np.ascontiguousarray(audio)
        start_sample = min(len(audio), max(0, int(start_seconds * rate)))
        with self.lock:
            self.audio = audio
            self.sample_rate = rate
            self.cursor = start_sample
            self.paused = False
            self.finished = start_sample >= len(audio)
        stream = None
        try:
            stream = sd.OutputStream(device=device, samplerate=rate, channels=1, dtype="float32", blocksize=1024, callback=self.audio_callback)
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
            if self.audio is None or not self.sample_rate:
                return
            self.cursor = min(len(self.audio), max(0, int(seconds * self.sample_rate)))
            self.finished = self.cursor >= len(self.audio)

    def get_position(self):
        with self.lock:
            return self.cursor / self.sample_rate if self.sample_rate else 0.0

    def get_duration(self):
        with self.lock:
            return len(self.audio) / self.sample_rate if self.sample_rate and self.audio is not None else 0.0

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
    sd.play(np.asarray(audio, dtype=np.float32), samplerate=sample_rate, device=output_device, blocking=False)


def play_audio(audio, sample_rate, output_device=OUTPUT_DEVICE):
    play_audio_async(audio, sample_rate, output_device)
    sd.wait()


def stop_audio():
    sd.stop()
