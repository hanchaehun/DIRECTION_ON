"""음성 유틸: RMS, 스트리밍 리샘플러(24k→16k), 앞 무음 정리, WAV 입출력."""
from __future__ import annotations

import math
import wave

import numpy as np
from scipy.signal import firwin, lfilter

RATE = 16000
BYTES_PER_MS = RATE * 2 // 1000   # 32
FRAME_BYTES = 640                 # 20ms


def as_f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32)


def rms(pcm: bytes) -> float:
    a = as_f32(pcm)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def silence(ms: float, rate: int = RATE) -> bytes:
    return b"\0\0" * int(rate * ms / 1000)


def pcm_ms(pcm: bytes, rate: int = RATE) -> float:
    return len(pcm) / 2 / rate * 1000


def chunks(pcm: bytes, ms: int, rate: int = RATE):
    step = max(2, int(rate * ms / 1000) * 2)
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]


def apply_gain(pcm: bytes, gain: float) -> bytes:
    if abs(gain - 1.0) < 1e-6:
        return pcm
    a = as_f32(pcm) * gain
    return np.clip(np.round(a), -32768, 32767).astype(np.int16).tobytes()


def read_wav(path) -> tuple[bytes, int]:
    with wave.open(str(path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: mono 16-bit PCM만 지원")
        return w.readframes(w.getnframes()), w.getframerate()


def write_wav(path, pcm: bytes, rate: int = RATE) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def frame_rms(pcm: bytes, frame_ms: int = 20, rate: int = RATE) -> np.ndarray:
    a = as_f32(pcm)
    n = int(rate * frame_ms / 1000)
    k = len(a) // n
    if k == 0:
        return np.zeros(0)
    return np.sqrt((a[: k * n].reshape(k, n) ** 2).mean(axis=1))


class StreamResampler:
    """조각 단위로 들어오는 PCM을 상태를 유지하며 변환한다 (정수비 up/down, FIR + lfilter 상태)."""

    def __init__(self, src: int, dst: int = RATE, taps: int = 127):
        self.src, self.dst = src, dst
        self.passthrough = src == dst
        g = math.gcd(src, dst)
        self.up, self.down = dst // g, src // g
        if not self.passthrough:
            fs_mid = src * self.up
            cutoff = 0.45 * min(src, dst)
            self.h = firwin(taps, cutoff, fs=fs_mid) * self.up
            self.zi = np.zeros(len(self.h) - 1)
        self.phase = 0
        self.carry = b""

    def process(self, pcm: bytes) -> bytes:
        pcm = self.carry + pcm
        cut = len(pcm) // 2 * 2
        self.carry = pcm[cut:]
        pcm = pcm[:cut]
        if self.passthrough or not pcm:
            return pcm
        x = as_f32(pcm).astype(np.float64)
        u = np.zeros(len(x) * self.up)
        u[:: self.up] = x
        y, self.zi = lfilter(self.h, 1.0, u, zi=self.zi)
        out = y[self.phase:: self.down]
        taken = len(out)
        self.phase = self.phase + taken * self.down - len(y)
        return np.clip(np.round(out), -32768, 32767).astype(np.int16).tobytes()


class LeadTrimmer:
    """앞 무음 정리: 말소리가 시작되는 지점의 pre-roll 앞부터 살리고 짧은 페이드인.
    칼같이 자르면 약한 초성(ㅅ, ㅎ, ㄱ)이 잘린다 (피드백 반영)."""

    def __init__(self, rate: int = RATE, thr: float = 300.0, preroll_ms: int = 25,
                 fade_ms: int = 5, win_ms: int = 5, max_buffer_ms: int = 3000):
        self.rate, self.thr = rate, thr
        self.win = max(1, int(rate * win_ms / 1000))
        self.preroll = int(rate * preroll_ms / 1000)
        self.fade = int(rate * fade_ms / 1000)
        self.max_samples = int(rate * max_buffer_ms / 1000)
        self.buf = b""
        self.done = False

    def process(self, pcm: bytes) -> bytes:
        if self.done:
            return pcm
        self.buf += pcm
        even = len(self.buf) // 2 * 2
        a = as_f32(self.buf[:even])
        n = len(a) // self.win
        if n:
            r = np.sqrt((a[: n * self.win].reshape(n, self.win) ** 2).mean(axis=1))
            idx = np.flatnonzero(r >= self.thr)
            if len(idx):
                start = max(0, int(idx[0]) * self.win - self.preroll)
                out = a[start:].copy()
                f = min(self.fade, len(out))
                if f:
                    out[:f] *= np.linspace(0.0, 1.0, f, endpoint=False)
                self.done = True
                tail = self.buf[even:]
                self.buf = b""
                return np.clip(np.round(out), -32768, 32767).astype(np.int16).tobytes() + tail
        if len(a) >= self.max_samples:      # 말소리가 끝내 없으면 그대로 내보낸다
            self.done = True
            out, self.buf = self.buf, b""
            return out
        return b""

    def flush(self) -> bytes:
        self.done = True
        self.buf = b""
        return b""                           # 끝까지 무음이면 버린다
