"""STT 어댑터.

SonioxSTT    실제 Soniox 실시간 스트림 (stt-rt-v6)
FakeSTT      측정 모드 시험용. 사례 음성과 정답 문장으로 Soniox처럼 토큰을 흘린다.
             fake_stt_adversarial=True면 문장 중간 쉼에서도 종료 신호를 내는 '공격적인' STT를 흉내 낸다.
FakeLiveSTT  시연 모드 시험용. 대본(fake_script)을 말소리 길이에 맞춰 흘린다.

모든 구현이 같은 상태를 노출한다: final_text(), text(), end_seen, final_audio_ms,
last_token_wall, fin_event, recording. 새 메시지가 오면 on_update()를 부른다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from .audio import BYTES_PER_MS, frame_rms, read_wav, rms
from .config import PROMPTS, env

log = logging.getLogger("stt")
MARKERS = ("<end>", "<fin>")


class STTBase:
    def __init__(self, s, on_update):
        self.s = s
        self.on_update = on_update
        self.final_tokens: list[dict] = []
        self.nonfinal: list[dict] = []
        self.turn_start = 0
        self.end_seen = False
        self.fin_event = asyncio.Event()
        self.final_audio_ms = 0.0
        self.last_token_wall: float | None = None
        self.audio_ms = 0.0                 # 지금까지 넣은 음성 길이 (스트림 기준)
        self.recording: list[dict] = []
        self.error: str | None = None
        self.t0 = time.perf_counter()
        self.closed = False

    # ── 상태 ──
    def final_text(self) -> str:
        return "".join(t["text"] for t in self.final_tokens[self.turn_start:]).strip()

    def text(self) -> str:
        return ("".join(t["text"] for t in self.final_tokens[self.turn_start:])
                + "".join(t["text"] for t in self.nonfinal)).strip()

    def has_tokens(self) -> bool:
        return bool(self.text())

    def new_turn(self) -> None:
        self.turn_start = len(self.final_tokens)
        self.nonfinal = []
        self.end_seen = False
        self.fin_event.clear()

    # ── 응답 처리 (Soniox 응답 형식) ──
    def _apply(self, res: dict) -> None:
        now = time.perf_counter()
        if self.s.record_stt:
            self.recording.append({"t": round(now - self.t0, 4), "audio_ms": round(self.audio_ms, 1), "res": res})
        if res.get("error_code"):
            self.error = f"{res.get('error_code')} {res.get('error_type')} {res.get('error_message')}"
            log.warning("soniox error: %s", self.error)
        before = self.text()
        nonfinal, got_end = [], False
        for t in res.get("tokens", []):
            txt = t.get("text", "")
            if txt == "<end>":
                got_end = True
                continue
            if txt == "<fin>":
                self.fin_event.set()
                continue
            if t.get("is_final"):
                self.final_tokens.append(t)
            else:
                nonfinal.append(t)
        self.nonfinal = nonfinal
        if self.text() != before:            # 같은 비최종 토큰의 재전송은 새 말이 아니다
            self.last_token_wall = now
            if not got_end:
                self.end_seen = False        # <end> 뒤에 새 말이 나왔다
        if got_end:
            self.end_seen = True
        if res.get("final_audio_proc_ms") is not None:
            self.final_audio_ms = float(res["final_audio_proc_ms"])
        try:
            self.on_update()
        except Exception:   # 판단 쪽 오류가 수신 루프를 죽이지 않게
            log.exception("on_update failed")

    # ── 인터페이스 ──
    async def start(self) -> None: ...
    def push_audio(self, pcm: bytes) -> None: ...
    async def finalize(self, timeout: float) -> bool: return False
    async def close(self) -> None: self.closed = True
    async def send_agent_text(self, text: str, end_of_segment: bool = False) -> None: ...


# ───────────────────────── Soniox ─────────────────────────

def _read_prompt(name: str) -> str:
    p = PROMPTS / name
    if not p.exists():
        return ""
    t = p.read_text(encoding="utf-8").strip()
    return "" if t.startswith("TODO:") else t


def soniox_config(s, caller: dict | None) -> dict:
    cfg = {
        "model": s.soniox_model,
        "audio_format": "pcm_s16le", "sample_rate": 16000, "num_channels": 1,
        "language_hints": ["ko"],
        "enable_endpoint_detection": True,
        "endpoint_latency_adjustment_level": s.endpoint_latency_adjustment_level,
        "endpoint_sensitivity": s.endpoint_sensitivity,
        "max_endpoint_delay_ms": s.max_endpoint_delay_ms,
    }
    if s.live_agent_text:
        cfg["enable_agent_text_input"] = True
    if s.stt_use_context:
        general = [{"key": "domain", "value": "건강기능식품 쇼핑몰 고객센터 전화 상담"},
                   {"key": "business name", "value": s.brand_name}]
        if caller and caller.get("name"):
            general.append({"key": "caller", "value": str(caller["name"])})
        ctx: dict = {"general": general}
        terms = [ln.strip() for ln in _read_prompt("stt_terms.txt").splitlines() if ln.strip()]
        from . import knowledge
        terms += [n for n in knowledge.product_names() if not re.search(r"\d{3,}", n)]   # 숫자·코드는 넣지 않는다 (문서 권고)
        terms = list(dict.fromkeys(terms))[:120]
        if terms:
            ctx["terms"] = terms
        text = _read_prompt("stt_context.md")
        if text:
            ctx["text"] = text[:9000]
        cfg["context"] = ctx
    return cfg


class SonioxSTT(STTBase):
    def __init__(self, s, on_update, caller: dict | None = None):
        super().__init__(s, on_update)
        self.url = env("SONIOX_STT_URL", s.soniox_url)
        self.key = env("SONIOX_API_KEY")
        self.config = soniox_config(s, caller)
        self.q: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.finished = asyncio.Event()
        self.connected_at: float | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    def push_audio(self, pcm: bytes) -> None:
        self.audio_ms += len(pcm) / BYTES_PER_MS
        if not self.closed:
            self.q.put_nowait(pcm)        # 연결 전 음성도 큐에 쌓였다가 설정 메시지 뒤에 나간다

    async def _run(self) -> None:
        from websockets.asyncio.client import connect
        if not self.key:
            self.error = "SONIOX_API_KEY 없음"
            log.error(self.error)
            return
        sender = None
        try:
            async with connect(self.url, additional_headers={"Authorization": f"Bearer {self.key}"},
                               max_size=None, open_timeout=5, ping_interval=10) as ws:
                await ws.send(json.dumps(self.config, ensure_ascii=False))
                self.connected_at = time.perf_counter() - self.t0
                sender = asyncio.create_task(self._sender(ws))
                async for msg in ws:
                    if isinstance(msg, str):
                        res = json.loads(msg)
                        self._apply(res)
                        if res.get("finished"):
                            self.finished.set()
                            break
        except Exception as e:  # 연결 실패도 세션은 계속 (받아쓰기 없이 대체 응답)
            self.error = self.error or repr(e)[:300]
            log.warning("soniox stream ended: %s", self.error)
        finally:
            self.finished.set()
            self.fin_event.set()
            if sender:
                sender.cancel()

    async def _sender(self, ws) -> None:
        while True:
            item = await self.q.get()
            if isinstance(item, bytes):
                await ws.send(item)
            else:                          # 제어 메시지 (str)
                await ws.send(item)
                if item == "":
                    return

    async def finalize(self, timeout: float) -> bool:
        if self.closed or self.finished.is_set():
            return False
        self.fin_event.clear()
        self.q.put_nowait(json.dumps({"type": "finalize"}))
        try:
            await asyncio.wait_for(self.fin_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def send_agent_text(self, text: str, end_of_segment: bool = False) -> None:
        if self.closed or not self.s.live_agent_text:
            return
        msg = {"type": "agent_text", "text": text}
        if end_of_segment:
            msg["end_of_segment"] = True
        self.q.put_nowait(json.dumps(msg, ensure_ascii=False))

    async def close(self) -> None:
        """측정 모드: 커밋 직후 스트림을 닫는다 (동시 스트림 한도 10개 보호)."""
        if self.closed:
            return
        self.closed = True
        self.q.put_nowait("")             # 빈 텍스트 프레임 = 스트림 종료
        try:
            await asyncio.wait_for(self.finished.wait(), 1.5)
        except asyncio.TimeoutError:
            pass
        if self.task and not self.task.done():
            self.task.cancel()


# ───────────────────────── 가짜 STT (측정 모드) ─────────────────────────

def _load_case_text(path: str, case_id: str) -> str | None:
    if not path or not case_id or not Path(path).exists():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            if c.get("case_id") == case_id:
                return c.get("customer_utterance")
    return None


def build_schedule(text: str, pcm: bytes, s) -> tuple[list, list]:
    """정답 문장을 음성의 말소리 구간에 맞춰 배치한 사건 목록. (사건 시각은 음성 스트림 ms)"""
    r = frame_rms(pcm)
    voiced = r >= s.silence_rms
    segs: list[list[float]] = []
    i, n = 0, len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            segs.append([i * 20.0, j * 20.0])
            i = j
        else:
            i += 1
    merged: list[list[float]] = []
    for a, b in segs:
        if merged and a - merged[-1][1] < 400:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if not merged:
        return [], []
    phrases = [p.strip() for p in re.findall(r"[^,]+,?", text) if p.strip()]
    if len(phrases) != len(merged):
        phrases, merged = [text.strip()], [[merged[0][0], merged[-1][1]]]
    words: list[dict] = []
    events: list[tuple[float, tuple]] = []
    for idx, ((a, b), ph) in enumerate(zip(merged, phrases)):
        ws = ph.split()
        for k, w in enumerate(ws):
            t = a + (k + 1) / len(ws) * (b - a)
            words.append({"text": w, "t": t})
            events.append((t, ("reveal", len(words) - 1)))
        last = len(words) - 1
        is_last = idx == len(merged) - 1
        gap = float("inf") if is_last else merged[idx + 1][0] - b
        ends_final = bool(re.search(r"[.?!]$", ph))
        will_end = gap >= s.fake_endpoint_ms and (is_last or ends_final or s.fake_stt_adversarial)
        fix = will_end and s.fake_stt_adversarial and ph.endswith(",")
        if gap >= 250:
            events.append((b + 250, ("finalize", last, b, fix)))
        if will_end:
            events.append((b + s.fake_endpoint_ms, ("end",)))
    events.sort(key=lambda e: e[0])
    return words, events


class FakeSTT(STTBase):
    def __init__(self, s, on_update, case_id: str | None):
        super().__init__(s, on_update)
        self.words: list[dict] = []
        self.events: list = []
        self.state: list[str] = []       # hidden / nonfinal / final
        text = _load_case_text(s.fake_cases, case_id or "")
        wav = Path(s.fake_audio_dir) / f"{case_id}.wav" if s.fake_audio_dir and case_id else None
        if text and wav and wav.exists():
            pcm, _ = read_wav(wav)
            self.words, self.events = build_schedule(text, pcm, s)
            self.state = ["hidden"] * len(self.words)
        elif case_id:
            log.warning("fake stt: %s 정답 문장 또는 음성 없음", case_id)
        self.idx = 0

    def push_audio(self, pcm: bytes) -> None:
        self.audio_ms += len(pcm) / BYTES_PER_MS
        loop = asyncio.get_running_loop()
        while self.idx < len(self.events) and self.events[self.idx][0] <= self.audio_ms:
            ev = self.events[self.idx][1]
            self.idx += 1
            loop.call_later(self.s.fake_stt_lag_ms / 1000, self._fire, ev)

    def _tokens(self, newly_final: list[int], end: bool = False, fin: bool = False) -> dict:
        toks = [{"text": self._tok_text(i), "is_final": True} for i in newly_final]
        toks += [{"text": self._tok_text(i), "is_final": False}
                 for i, st in enumerate(self.state) if st == "nonfinal"]
        if end:
            toks.append({"text": "<end>", "is_final": True})
        if fin:
            toks.append({"text": "<fin>", "is_final": True})
        return {"tokens": toks, "final_audio_proc_ms": self.final_audio_ms}

    def _tok_text(self, i: int) -> str:
        return (" " if i else "") + self.words[i]["text"]

    def _fire(self, ev: tuple) -> None:
        if self.closed:
            return
        kind = ev[0]
        if kind == "reveal":
            if self.state[ev[1]] == "hidden":
                self.state[ev[1]] = "nonfinal"
            self._apply(self._tokens([]))
        elif kind == "finalize":
            _, last, b, fix = ev
            if fix and self.words[last]["text"].endswith(","):
                self.words[last]["text"] = self.words[last]["text"][:-1] + "."
            newly = [i for i in range(last + 1) if self.state[i] != "final"]
            for i in newly:
                self.state[i] = "final"
            self.final_audio_ms = max(self.final_audio_ms, b)
            self._apply(self._tokens(newly))
        elif kind == "end":
            newly = [i for i, st in enumerate(self.state) if st == "nonfinal"]
            for i in newly:
                self.state[i] = "final"
            self._apply(self._tokens(newly, end=True))

    async def finalize(self, timeout: float) -> bool:
        await asyncio.sleep(0.06)
        newly = [i for i, w in enumerate(self.words) if self.state[i] != "final" and w["t"] <= self.audio_ms]
        for i in newly:
            self.state[i] = "final"
        if newly:
            self.final_audio_ms = max(self.final_audio_ms, max(self.words[i]["t"] for i in newly))
        self._apply(self._tokens(newly, fin=True))
        return True


# ───────────────────────── 가짜 STT (시연 모드) ─────────────────────────

class FakeLiveSTT(STTBase):
    WORD_MS = 300

    def __init__(self, s, on_update, script: list[str] | None):
        super().__init__(s, on_update)
        self.script = list(script or [])
        self.cur: list[str] | None = None
        self.revealed = 0
        self.finalized = 0
        self.voiced_ms = 0.0
        self.gap_ms = 0.0
        self.offset = 0               # 현재 발화 첫 단어의 전체 토큰 번호

    def push_audio(self, pcm: bytes) -> None:
        dt = len(pcm) / BYTES_PER_MS
        self.audio_ms += dt
        loop = asyncio.get_running_loop()
        lag = self.s.fake_stt_lag_ms / 1000
        if rms(pcm) >= self.s.silence_rms:
            if self.cur is None:
                line = self.script.pop(0) if self.script else "(대본 없음)"
                self.cur, self.revealed, self.finalized, self.voiced_ms = line.split(), 0, 0, 0.0
            self.voiced_ms += dt
            self.gap_ms = 0.0
            want = min(len(self.cur), int(self.voiced_ms // self.WORD_MS) + 1)
            if want > self.revealed:
                self.revealed = want
                loop.call_later(lag, self._apply, self._res())
        elif self.cur is not None:
            self.gap_ms += dt
            if self.gap_ms >= 250 and self.finalized < self.revealed:
                loop.call_later(lag, self._apply, self._res(final_upto=self.revealed))
                self.finalized = self.revealed
            if self.gap_ms >= self.s.fake_endpoint_ms:
                loop.call_later(lag, self._apply, self._res(final_upto=len(self.cur), end=True))
                self.cur = None

    def _res(self, final_upto: int | None = None, end: bool = False, fin: bool = False) -> dict:
        cur = self.cur or []
        toks = []
        if final_upto is not None:
            toks += [{"text": " " + w, "is_final": True} for w in cur[self.finalized_sent():final_upto]]
            self._sent_final = final_upto
        toks += [{"text": " " + w, "is_final": False} for w in cur[self.finalized_sent():self.revealed]]
        if end:
            toks.append({"text": "<end>", "is_final": True})
            self._sent_final = 0
        if fin:
            toks.append({"text": "<fin>", "is_final": True})
        return {"tokens": toks, "final_audio_proc_ms": self.audio_ms}

    def finalized_sent(self) -> int:
        return getattr(self, "_sent_final", 0)

    async def finalize(self, timeout: float) -> bool:
        await asyncio.sleep(0.05)
        if self.cur is not None:
            self._apply(self._res(final_upto=self.revealed, fin=True))
            self.finalized = self.revealed
        else:
            self._apply({"tokens": [{"text": "<fin>", "is_final": True}], "final_audio_proc_ms": self.audio_ms})
        return True
