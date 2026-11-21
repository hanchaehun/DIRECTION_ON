"""TTS: Gemini 2.5 Flash TTS (비동기 스트리밍) 또는 가짜. 출력은 항상 16kHz PCM.

TTSEngine.stream(text)  디스크 캐시 → 없으면 합성. 24k→16k 변환, 앞 무음 정리, 첫 음성 시간 제한과 재시도.
TTSJob                  합성을 미리 시작해 큐에 받아 두는 작업 (추측 합성, 답 음성에 공용).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path

import httpx
import numpy as np

from .audio import RATE, LeadTrimmer, StreamResampler, apply_gain
from .config import RUNTIME, env
from .ratelimit import LIMITER

log = logging.getLogger("tts")


class TTSError(Exception):
    pass


def _audio_parts(obj):
    """응답 JSON 어디에 있든 음성 조각을 찾는다: Interactions의 {"type":"audio","data":...},
    generateContent의 inlineData/inline_data. (base64, mime, sample_rate) 를 낸다."""
    if isinstance(obj, dict):
        if obj.get("type") == "audio" and isinstance(obj.get("data"), str):
            yield obj["data"], str(obj.get("mime_type") or obj.get("mimeType") or ""), obj.get("sample_rate")
            return
        inl = obj.get("inlineData") or obj.get("inline_data")
        if isinstance(inl, dict) and isinstance(inl.get("data"), str):
            yield inl["data"], str(inl.get("mimeType") or inl.get("mime_type") or ""), None
            return
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from _audio_parts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _audio_parts(v)


def _decode(b64: str, mime: str, rate, default_rate: int) -> tuple[bytes, int]:
    raw = base64.b64decode(b64)
    m = re.search(r"rate=(\d+)", mime or "")
    sr = int(m.group(1)) if m else int(rate or default_rate)
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":         # 3.8 단건 응답은 audio/wav
        sr = int.from_bytes(raw[24:28], "little") or sr
        i = raw.find(b"data", 12)
        raw = raw[i + 8:] if i >= 0 else raw[44:]
    return raw, sr


class GeminiBackend:
    """Gemini TTS.
    interactions  POST /v1beta/interactions, stream=true (3.8 Flash/Flash-Lite TTS 문서의 방식). 말투는 speech_metadata.style
    generate      POST /v1beta/models/{m}:streamGenerateContent?alt=sse (2.5 계열 방식, 예비)"""

    def __init__(self, http: httpx.AsyncClient, s):
        self.http, self.s = http, s
        self.base = env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
        self.key = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
        self.api = s.gemini_tts_api
        self.last_error: dict | None = None

    @property
    def url(self) -> str:
        if self.api == "speech":                 # BAZE 게이트웨이 OpenAI형 /audio/speech/ (원시 PCM 24k)
            return env("TTS_SPEECH_URL", self.base.rstrip("/") + "/audio/speech/")
        if self.api == "interactions":
            return f"{self.base}/v1beta/interactions?alt=sse"
        return f"{self.base}/v1beta/models/{self.s.gemini_tts_model}:streamGenerateContent?alt=sse"

    def body(self, text: str) -> dict:
        s = self.s
        if self.api == "speech":
            b = {"model": s.gemini_tts_model, "input": text, "voice": s.gemini_voice}
            if s.tts_style and "3.8" in s.gemini_tts_model:      # instructions는 3.8 flash tts에서만 받는다 (게이트웨이 문서)
                b["instructions"] = s.tts_style
            return b
        if self.api == "interactions":
            item = {"type": "text", "text": text}
            if s.tts_style:
                item["annotations"] = [{"type": "speech_metadata", "style": s.tts_style}]
            fmt = {"type": "audio", "mime_type": "audio/l16"}
            if s.tts_sample_rate in (8000, 16000, 24000):
                fmt["sample_rate"] = s.tts_sample_rate
            return {"model": s.gemini_tts_model, "stream": True,
                    "input": [{"type": "user_input", "content": [item]}],
                    "response_format": fmt,
                    "generation_config": {"speech_config": [{"voice": s.gemini_voice}]}}
        prompt = f'{s.tts_style_prefix}:\n"{text}"' if s.tts_style_prefix else text
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["AUDIO"],
                                     "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": s.gemini_voice}}}}}

    async def stream_raw(self, text: str):
        if not self.key:
            raise TTSError("GEMINI_API_KEY 없음")
        await LIMITER.acquire()
        if self.api == "speech":
            async with self.http.stream("POST", self.url, json=self.body(text),
                                        headers={"Authorization": f"Bearer {self.key}"},
                                        timeout=httpx.Timeout(20.0, connect=4.0)) as r:
                if r.status_code != 200:
                    raw = (await r.aread()).decode("utf-8", "replace")
                    self.last_error = {"status": r.status_code, "body": raw[:1500]}
                    if r.status_code == 429:
                        LIMITER.note_429(2.0)
                    raise TTSError(f"speech {r.status_code}: {raw[:300]}")
                async for b in r.aiter_bytes():
                    if b:
                        yield b, 24000
            return
        default_rate = self.s.tts_sample_rate if self.api == "interactions" else 24000
        async with self.http.stream("POST", self.url, json=self.body(text),
                                    headers={"x-goog-api-key": self.key, "Accept": "text/event-stream"},
                                    timeout=httpx.Timeout(20.0, connect=4.0)) as r:
            if r.status_code != 200:
                raw = (await r.aread()).decode("utf-8", "replace")
                self.last_error = {"status": r.status_code, "body": raw[:1500]}
                raise TTSError(f"gemini {r.status_code}: {raw[:300]}")
            if "json" in (r.headers.get("content-type") or "") and "event-stream" not in (r.headers.get("content-type") or ""):
                ev = json.loads(await r.aread())                  # 스트리밍이 아닌 응답이 온 경우
                for b64, mime, rate in _audio_parts(ev):
                    yield _decode(b64, mime, rate, default_rate)
                return
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                ev = json.loads(data)
                if isinstance(ev, dict) and ev.get("error"):
                    self.last_error = {"status": 200, "body": data[:1500]}
                    raise TTSError(f"gemini stream error: {data[:300]}")
                for b64, mime, rate in _audio_parts(ev):
                    yield _decode(b64, mime, rate, default_rate)


class FakeBackend:
    """음절마다 모음 비슷한 소리를 낸다. 24kHz, 앞 무음 120ms (리샘플러와 무음 정리를 시험)."""

    def __init__(self, s):
        self.s = s

    @staticmethod
    def synth(text: str, rate: int = 24000) -> bytes:
        out = [np.zeros(int(rate * 0.12))]
        f0 = 170.0
        for i, ch in enumerate(text):
            if ch.isspace():
                out.append(np.zeros(int(rate * 0.06)))
                continue
            if ch in ".,?!":
                out.append(np.zeros(int(rate * 0.18)))
                continue
            n = int(rate * 0.13)
            t = np.arange(n) / rate
            f = f0 * (1 + 0.08 * math.sin(i))
            w = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6))
            env_ = np.minimum(1, np.minimum(t / 0.015, (t[-1] - t) / 0.03))
            out.append(w * env_ * 3500)
        a = np.concatenate(out)
        return np.clip(a, -32768, 32767).astype(np.int16).tobytes()

    async def stream_raw(self, text: str):
        await asyncio.sleep(self.s.fake_tts_ttfb_ms / 1000)
        pcm = self.synth(text)
        step = int(24000 * 0.1) * 2
        for i in range(0, len(pcm), step):
            yield pcm[i:i + step], 24000
            await asyncio.sleep(0.03)            # 실시간의 약 3배 속도


class TTSEngine:
    def __init__(self, s, backend, name: str):
        self.s, self.backend = s, backend
        self.sem = asyncio.Semaphore(max(1, s.tts_concurrency))
        voice = s.gemini_voice if name == "gemini" else "fake"
        self.cache_dir = RUNTIME / "tts_cache" / f"{name}_{s.gemini_tts_model}_{voice}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests = 0
        self.cache_hits = 0
        self.errors = 0

    def _key(self, text: str) -> Path:
        h = hashlib.sha1(f"{self.s.gemini_tts_api}|{self.s.tts_style}|{self.s.tts_style_prefix}|{text}".encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{h}.pcm"

    def cached(self, text: str) -> bytes | None:
        p = self._key(text)
        return p.read_bytes() if p.exists() else None

    async def stream(self, text: str, stats: dict | None = None, persist: bool | None = None):
        """16kHz PCM 조각을 낸다. stats에 req/first/cache 시각을 기록한다.
        persist: 디스크 캐시에 저장할지 (None이면 설정 persist_dynamic_tts)."""
        persist = self.s.persist_dynamic_tts if persist is None else persist
        stats = stats if stats is not None else {}
        hit = self.cached(text)
        if hit is not None:
            self.cache_hits += 1
            stats["cache"] = True
            stats["first"] = time.perf_counter()
            for i in range(0, len(hit), 3200):
                yield hit[i:i + 3200]
            return
        last_err = None
        for attempt in range(1 + self.s.tts_retries):
            collected: list[bytes] = []
            got_audio = False
            try:
                async with self.sem:
                    self.requests += 1
                    stats.setdefault("req", time.perf_counter())
                    agen = self.backend.stream_raw(text)
                    rs, trim = None, LeadTrimmer(thr=self.s.silence_rms)
                    while True:
                        timeout = self.s.tts_first_audio_timeout_s if not got_audio else 15.0
                        try:
                            raw, rate = await asyncio.wait_for(agen.__anext__(), timeout)
                        except StopAsyncIteration:
                            break
                        rs = rs or StreamResampler(rate, RATE)
                        out = trim.process(rs.process(raw))
                        if out:
                            out = apply_gain(out, self.s.tts_gain)
                            if not got_audio:
                                stats["first"] = time.perf_counter()
                                got_audio = True
                            collected.append(out)
                            yield out
                    await agen.aclose()
                if collected and persist:
                    try:
                        self._key(text).write_bytes(b"".join(collected))
                    except OSError as e:
                        log.warning("tts 캐시 저장 실패: %r", e)
                stats["chunks"] = len(collected)
                return
            except (TTSError, asyncio.TimeoutError, httpx.HTTPError, json.JSONDecodeError) as e:
                self.errors += 1
                last_err = e
                log.warning("tts 실패 (%d): %r", attempt + 1, e)
                if got_audio:                      # 이미 소리를 냈으면 재시도하지 않는다 (중복 발화 방지)
                    raise TTSError(f"stream broken: {e!r}")
        raise TTSError(f"tts failed: {last_err!r}")

    async def synth_full(self, text: str, persist: bool = True) -> bytes:
        return b"".join([c async for c in self.stream(text, persist=persist)])


class TTSJob:
    """합성을 미리 시작해 조각을 큐에 쌓는다. 소비자는 iterate()로 받는다."""

    def __init__(self, engine: TTSEngine, text: str):
        self.text = text
        self.stats: dict = {}
        self.q: asyncio.Queue = asyncio.Queue()
        self.first = asyncio.Event()
        self.error: Exception | None = None
        self.task = asyncio.create_task(self._run(engine))

    async def _run(self, engine: TTSEngine):
        try:
            async for c in engine.stream(self.text, self.stats):
                self.q.put_nowait(c)
                self.first.set()
        except Exception as e:
            self.error = e
        finally:
            self.first.set()
            self.q.put_nowait(None)

    async def iterate(self):
        while True:
            c = await self.q.get()
            if c is None:
                if self.error:
                    raise self.error
                return
            yield c

    def cancel(self):
        if not self.task.done():
            self.task.cancel()
