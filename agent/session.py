"""세션: 측정기 또는 시연 클라이언트의 연결 하나.

측정 모드  한 턴: 듣기 → 말 끝 확정(커밋) → 첫 소리 → 답 → turn.end. 그 뒤로는 측정기가 닫을 때까지 무음을 읽어 버린다.
시연 모드  session.start에 "mode": "live". 여러 턴, 끼어들기, 중단 지점 이력 정리, 화면용 이벤트.

한 턴의 흐름 (게이트웨이 실측에 맞춘 설계: LLM 약 1.7초, TTS는 스트리밍 없이 약 3초, 판정 모델 Jev 약 0.5초)
1. 듣기        음성을 Soniox에 넣고, 무음 0.45초에 Jev로 '말이 끝났는가·의도·불만'을 판정한다.
2. 커밋        무음이 기준을 넘으면 말 끝으로 확정한다. Jev가 덜 끝난 말로 보면 더 기다린다.
3. 첫 소리     미리 합성한 음성만 1.5초 안에 낼 수 있다.
               · 의도가 분명하고 답이 정해져 있으면 은행 문장으로 바로 답한다 (LLM·TTS 없음).
               · 아니면 맥락에 맞는 첫마디(공감/확인/안내)를 내고 LLM을 부른다.
4. 판단        GPT-5.4 mini가 답하거나 함수를 고른다. 번호는 맥락 번호에 맞추고, 접수 함수는 번호가 온전할 때만 실행한다.
5. 답          문장 단위로 은행 문장은 그대로, 나머지는 TTS를 동시에 걸어 순서대로 내보낸다. 실패한 문장은 건너뛴다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass

from websockets.exceptions import ConnectionClosed

from . import aftertool, directlib, facts, knowledge, pace, toolflow
from .audio import BYTES_PER_MS, chunks, rms, silence, time_stretch
from .brain import Decision, parse_now
from .clips import tool_kind
from .metrics import Timeline
from .router import choose_bank, choose_opener
from .turn import classify
from .textnorm import for_tts, norm_compare, trailing_digits
from .toolflow import GUARD_NOTE, claims_lookup_without_evidence, evidence_text, execute_calls, reask_for
from .tts import TTSError, TTSJob

log = logging.getLogger("session")
perf = time.perf_counter


def normalize_history(h) -> list[dict]:
    """공식 형식(SPEC.md): {"role": customer|agent, "text"} / {"role": tool_call, "name", "arguments"} /
    {"role": tool_result, "name", "result"}. 그대로 두고, 다른 모양은 customer/agent 글로 바꾼다."""
    out: list[dict] = []
    for item in h or []:
        if isinstance(item, str):
            m = re.match(r"^\s*(고객|손님|customer|user|상담원|상담사|agent|assistant)\s*[:：]\s*(.*)$", item, re.I)
            if m:
                role = "customer" if m.group(1).lower() in ("고객", "손님", "customer", "user") else "agent"
                out.append({"role": role, "text": m.group(2)})
            else:
                out.append({"role": "customer", "text": item})
        elif isinstance(item, dict):
            raw = str(item.get("role") or item.get("speaker") or item.get("from") or "").lower()
            if raw == "tool_call":
                out.append({"role": "tool_call", "name": item.get("name"), "arguments": item.get("arguments") or {}})
                continue
            if raw == "tool_result":
                out.append({"role": "tool_result", "name": item.get("name"), "result": item.get("result")})
                continue
            text = item.get("text") or item.get("content") or item.get("utterance") or item.get("message") or ""
            role = "agent" if raw in ("agent", "assistant", "상담원", "상담사", "bot", "ai", "counselor", "staff") else "customer"
            out.append({"role": role, "text": str(text)})
    return out


_PLEASANTRY = [(re.compile(r"^\s*(네|예)\s*,?\s*고객님\s*[,.!]?\s*"), "네, "),
               (re.compile(r"^\s*고객님\s*[,.!]\s*"), "")]
_YES = re.compile(r"^\s*(네|예)\s*[,.!]\s*")
_NO = re.compile(r"^\s*(아니요|아뇨|아니오)\s*[,.!]\s*")
_HANDOFF_SAID = re.compile(r"상담원.{0,6}연결")
_SENT_END = re.compile(r"[.?!。？！]\s*$")
_VERB_END = re.compile(r"(요|다|까|죠|니다|세요|어|아|해|네|군|지|래|나|가|고|서|데|라|자|게|걸|냐|니)$")


_CONNECTIVE_END = re.compile(r"(는데|은데|인데|던데|한데|지만|니까|으면|하면|해서|아서|어서|가지고|는지|거든|"
                             r"(하|했|되|됐|있|없|왔|갔|샀|받았)고)$")
_TAIL_DIGITS = re.compile(r"(\d[\d\s\-]*)$")


def _unfinished_tail(text: str) -> str:
    """마침표가 찍혔어도 이어 말할 가능성이 큰 끝. ''이면 해당 없음.
    - 'partial_number': 번호를 읽다 멈춤 ('010 9901.', '20261003-00.'). 다 읽은 번호는 해당 없음
    - 'connective': '요' 없이 연결어미로 끝남 ('어제 주문했는데.'). '~는데요.'는 요청을 끝맺는 말로도 흔해 제외"""
    body = re.sub(r"[.?!。？！\s]+$", "", text.strip())
    m = _TAIL_DIGITS.search(body)
    d = re.sub(r"\D", "", m.group(1)) if m else trailing_digits(body)    # '공일공 일이삼사'처럼 한글로 읽은 번호도
    if d:
        if (d.startswith("01") and len(d) < 10) or (re.match(r"0[2-6]", d) and len(d) < 9) \
                or (d.startswith("20") and 8 <= len(d) < 15):
            return "partial_number"
    if _CONNECTIVE_END.search(body):
        return "connective"
    return ""


def _noun_only(text: str) -> bool:
    """물음표 없이 서술어 없는 명사로 끝난 짧은 말인가 ('대장영양제.', '배송 문의.'). 숫자를 불러 준 말은 제외한다."""
    t = text.strip()
    if t.endswith(("?", "？")):
        return False
    body = re.sub(r"[.?!。？！\s]+$", "", t)
    if len(re.sub(r"\D", "", body)) >= 7:
        return False                       # 번호를 불러 준 말은 끝난 것으로 본다
    return len(re.sub(r"\s", "", body)) <= 14 and not _VERB_END.search(body)


@dataclass
class Spec:
    text: str
    norm: str
    task: asyncio.Task
    msgs: list
    started: float
    early: "EarlyTTS | None" = None


async def _empty():
    """빈 비동기 반복자."""
    return
    yield


class EarlyTTS:
    """LLM 답이 스트리밍되는 동안 첫 합성 단위의 음성을 미리 만든다 (첫마디 뒤 공백 단축).

    안전 장치: 최종 답을 평소처럼 다듬어(서두 정리·길이 제한·날짜 검증) 나눈 첫 단위의 글이 미리 만든 글과
    한 글자도 다르지 않을 때만 쓴다. 다르거나, 함수 호출로 바뀌었거나, 합성이 실패했으면 버리고 평소대로 합성한다.
    어떤 오류도 답을 막지 않는다 (콜백 예외는 이 객체를 끄기만 한다)."""

    def __init__(self, sess: "Session"):
        self.sess = sess
        self.text = ""
        self.job: TTSJob | None = None
        self.off = False

    def feed(self, raw: str) -> None:
        if self.off or self.job is not None:
            return
        try:
            t = self.sess._early_unit(raw)
            if t:
                self.text = t
                self.job = TTSJob(self.sess.ctx.tts, for_tts(t, self.sess.s, self.sess.now))
                self.sess.tl.count("early_tts")
        except Exception:
            log.exception("미리 합성 시작 실패 (이번 답은 평소대로)")
            self.off = True
            self.cancel()

    def take(self, text: str) -> TTSJob | None:
        """최종 첫 단위 글과 같으면 미리 만든 작업을 넘긴다 (한 번만). 아니면 버린다."""
        job, self.job = self.job, None
        if job is None:
            return None
        failed = job.task.done() and job.error is not None
        if text != self.text or failed:
            job.cancel()
            self.sess.tl.count("early_tts_miss")
            return None
        self.sess.tl.count("early_tts_hit")
        return job

    def cancel(self) -> None:
        self.off = True                     # 취소 뒤 스트림이 이어져도 새 합성을 시작하지 않는다
        job, self.job = self.job, None
        if job is not None:
            job.cancel()


class Session:
    def __init__(self, ws, ctx):
        self.ws, self.ctx, self.s = ws, ctx, ctx.settings
        self.send_lock = asyncio.Lock()
        self.closed = asyncio.Event()
        self.t_start = perf()
        self.audio_ms = 0.0
        self.first_audio_wall: float | None = None
        self.pace = pace.Profile()          # 말하는 사람의 속도·쉼 버릇 (통화 동안 유지)
        self.turn_voiced_ms = 0.0           # 이번 턴에 실제 소리가 난 시간
        self.turn_pauses: list[float] = []  # 이번 턴의 말 중간 쉼 (ms)
        self.carry_voiced_ms = 0.0
        self.play_speed = 1.0               # 상담원 음성 재생 속도 (느린 사람에게 느리게)
        self.paced_cache: dict = {}
        self.thr = self.s.silence_rms
        self.noise: list[float] = []
        self.history: list[dict] = []
        self.response_task: asyncio.Task | None = None
        self.facts: dict = {}
        self.actions: list = []
        self.tl: Timeline | None = None
        self.ctx_ids: dict = {}
        self.evid = ""                      # 이번 통화의 조회 결과 (답 검증의 근거)

    # ───────────────────────── 진입 ─────────────────────────
    async def run(self) -> None:
        try:
            start = json.loads(await asyncio.wait_for(self.ws.recv(), 30))
            if start.get("type") != "session.start":
                raise ValueError("첫 메시지가 session.start가 아님")
        except Exception as e:
            log.warning("세션 시작 실패: %r", e)
            return
        self.start = start
        self.sid = str(start.get("session_id") or f"S-{uuid.uuid4().hex[:8]}")
        self.case_id = start.get("case_id")
        self.mode = "live" if start.get("mode") == "live" else "measure"
        self.tools_url = start.get("tools_url")
        self.caller = start.get("caller") if isinstance(start.get("caller"), dict) else None
        self.now = parse_now(start.get("now"))
        self.history = normalize_history(start.get("history"))
        self._refresh_context()
        for h in self.history:
            if h.get("role") == "tool_result":
                self.evid += facts.render_tool_result(str(h.get("name")), h.get("result"), self.now) + "\n"
        self.tools_task = asyncio.create_task(self.ctx.tools.tools(self.tools_url)) if self.tools_url else None
        self.stt = self.ctx.make_stt(self, start)
        self._reset_turn()
        await self.stt.start()
        recv = asyncio.create_task(self._recv_loop())
        try:
            if self.mode == "live":
                await self._run_live()
            else:
                await self._run_measure()
            await self.closed.wait()
        except asyncio.CancelledError:
            pass
        finally:
            recv.cancel()
            for t in (self.response_task, getattr(self, "audio_tail", None)):
                if t and not t.done():
                    t.cancel()
            self._cancel_spec()
            try:
                await self.stt.close()
            except Exception:
                pass
            if self.mode == "measure" and self.tl:
                self._dump()

    def _refresh_context(self) -> None:
        self.ctx_ids = facts.context_ids(self.history, self.caller)

    def _reset_turn(self, keep_speech: bool = False) -> None:
        self.tl = Timeline(self.sid, self.case_id, self.mode)
        self.tl.info["history_len"] = len(self.history)
        self.phase = "listening"
        if not keep_speech:
            self.speech_started = False
            self.voiced_run_ms = 0.0
            self.last_voiced_audio_ms: float | None = None
            self.last_voiced_wall: float | None = None
        self.voiced_now = False
        # 이번 턴의 소리 시간·쉼 기록을 새로 시작한다. 끊긴 말을 이어 붙이는 턴이면 앞말의 소리 시간을 더한다
        self.turn_voiced_ms = (self.voiced_run_ms if keep_speech else 0.0) + self.carry_voiced_ms
        self.carry_voiced_ms = 0.0
        self.turn_pauses = []
        # 지난 턴에 쓰이지 않은 미리 합성은 버린다
        for old in (getattr(self, "early", None), getattr(getattr(self, "spec", None), "early", None)):
            if old is not None:
                old.cancel()
        self.early: EarlyTTS | None = None
        self.spec: Spec | None = None
        self.routes: dict = {}              # 받아쓰기(정규화) → Jev 판정
        self.route_pending: dict[str, float] = {}   # 받아쓰기 → 판정 요청 시각
        self.commit_evt = asyncio.Event()
        self.first_sound_sent = False
        self.filler_sent = False
        self.opener_text = ""
        self.turn_customer = ""
        self.customer_pushed = False
        self.final_task: asyncio.Task | None = None
        self.turn_end_sent = False
        self.barge = False
        self.stt_turn_reset = False
        self.audio_tail: asyncio.Task | None = None
        self.spoken: list[dict] = []        # 이번 턴에 보낸 말 (text, ms)
        self.play_end = 0.0                 # 시연 모드: 클라이언트 재생이 끝날 예상 시각
        self.speech_end_wall: float | None = None
        self.speech_end_audio_ms = 0.0
        if keep_speech:
            self.tl.mark("speech_start")

    # ───────────────────────── 수신 ─────────────────────────
    async def _recv_loop(self) -> None:
        try:
            async for m in self.ws:
                if isinstance(m, (bytes, bytearray)):
                    self._on_audio(bytes(m))
                else:
                    self._on_control(m)
        except ConnectionClosed:
            pass
        finally:
            self.closed.set()

    def _on_control(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if msg.get("type") == "debug.history":
            self._live_event("debug.history", history=self.history)

    def _on_audio(self, pcm: bytes) -> None:
        now = perf()
        dt = len(pcm) / BYTES_PER_MS
        if self.first_audio_wall is None:
            self.first_audio_wall = now
        self.audio_ms += dt
        r = rms(pcm)
        if self.s.adaptive_noise and not self.speech_started and len(self.noise) < 15 and self.audio_ms <= 400:
            self.noise.append(r)
            if len(self.noise) == 15:
                floor = sorted(self.noise)[13]
                if floor * self.s.noise_factor > self.thr:
                    self.thr = floor * self.s.noise_factor
                    log.info("%s 소음 바닥 %.0f → 임계값 %.0f", self.sid, floor, self.thr)
        voiced = r >= self.thr
        if voiced:
            if self.speech_started and self.voiced_run_ms == 0.0 and self.last_voiced_audio_ms is not None:
                gap = self.audio_ms - dt - self.last_voiced_audio_ms      # 쉬었다가 다시 말하기 시작함
                if gap >= self.s.pace_min_pause_ms and self.phase == "listening":
                    self.turn_pauses.append(gap)
            self.voiced_run_ms += dt
            if self.phase == "listening":            # 상담원이 말하는 동안의 소리(에코·잡음)는 세지 않는다
                self.turn_voiced_ms += dt
            self.last_voiced_audio_ms, self.last_voiced_wall = self.audio_ms, now
            if not self.speech_started and self.voiced_run_ms >= self.s.speech_start_ms:
                self.speech_started = True
                self.tl.mark("speech_start")
        else:
            self.voiced_run_ms = 0.0
        self.voiced_now = voiced
        self.stt.push_audio(pcm)
        if self.phase == "listening":
            self._evaluate(now)
        elif self.mode == "live":
            self._check_barge(now)

    def on_stt_update(self) -> None:
        if self.stt.end_seen:
            self.tl.mark("end_token")
        if self.phase == "listening":
            self._live_event("stt.partial", text=self._turn_text())
            self._evaluate(perf())
        elif self.mode == "live":
            self._check_barge(perf())

    # ───────────────────────── 말 끝 판단 ─────────────────────────
    def _silence_ms(self) -> float:
        if self.speech_started and self.last_voiced_audio_ms is not None:
            return self.audio_ms - self.last_voiced_audio_ms
        return 0.0

    def _similar_pending(self, norm: str) -> bool:
        """비슷한 받아쓰기로 이미 판정을 요청해 두었으면 다시 부르지 않는다 (실측: 문항당 1~5회 → 호출 수·분당 한도)."""
        if not self.route_pending:
            return False
        import difflib
        return any(difflib.SequenceMatcher(None, norm, k).ratio() >= self.s.route_reuse_ratio for k in self.route_pending)

    def _route_for(self, text: str):
        """이 받아쓰기에 대한 Jev 판정. 판정을 요청한 뒤 받아쓰기가 조금 바뀌어도(문장부호, 끝 글자) 거의 같으면 쓴다.
        (실측: 정확히 같은 글만 인정했더니 6문항 중 4문항에서 판정을 쓰지 못했다)"""
        norm = norm_compare(text)
        r = self.routes.get(norm)
        if r is None and norm and self.routes:
            import difflib
            best, score = None, 0.0
            for k, v in self.routes.items():
                sc = difflib.SequenceMatcher(None, norm, k).ratio()
                if sc > score:
                    best, score = v, sc
            if score >= self.s.route_reuse_ratio:
                r = best
        return r if (r is not None and r.ok) else None

    def _evaluate(self, now: float) -> None:
        if self.phase != "listening":
            return
        s = self.s
        text = self._turn_text()
        norm = norm_compare(text)
        silence_ms = self._silence_ms()
        quiet = self.speech_started and not self.voiced_now
        # 판정 모델: 조용해지면 한 번 부른다 (받아쓰기가 바뀌면 다시)
        pause = s.live_router_pause_ms if self.mode == "live" else s.router_pause_ms
        if (s.use_router and norm and quiet and silence_ms >= pause
                and norm not in self.routes and not self._similar_pending(norm) and self._route_for(text) is None):
            self.route_pending[norm] = perf()
            asyncio.create_task(self._run_router(text, norm))
        # 추측 실행 (시연 모드): 쉼에서 LLM을 미리 돌려 답까지의 시간을 줄인다. 점수에는 영향이 없어 측정 모드에서는 끈다
        if self._spec_enabled():
            if self.voiced_now and self.spec and not self.spec.task.done():
                self._cancel_spec()
            elif norm and quiet and silence_ms >= s.spec_pause_ms and (self.spec is None or self.spec.norm != norm):
                self._start_spec(text)
        rule = self._commit_rule(text, norm, silence_ms, now)
        if rule:
            self._commit(rule, text, now)

    def _commit_rule(self, text: str, norm: str, silence_ms: float, now: float) -> str | None:
        """커밋할 규칙 이름 또는 None. 기준은 무음 길이이고, Jev가 '덜 끝난 말'로 보면 더 기다린다.
        연습 음성 80개 실측: 말 중간 쉼이 1.0초를 넘는 것은 1개. Soniox <end>는 6개에서 말 도중에 나와 쓰지 않는다."""
        s = self.s
        if not self.speech_started and not norm:
            return None
        if self.voiced_now:
            # 소음 경로: 무음이 잡히지 않지만 STT는 끝났다고 보고 새 토큰도 없다
            # 진짜 소음이면 소리가 몇 초씩 끊김 없이 이어진다. 쉬었다가 다시 말을 시작한 순간에는 쓰지 않는다
            # (실측 P0130-01: 1.8초 쉼 뒤 다시 말하기 시작하자 이 규칙으로 커밋해 끼어듦)
            if (self.stt.end_seen and norm and self.stt.last_token_wall
                    and self.voiced_run_ms >= s.noise_commit_min_voiced_ms
                    and (now - self.stt.last_token_wall) * 1000 >= s.noise_commit_token_idle_slow_ms):
                return "noise"
            return None
        if not norm:
            return "no_text" if silence_ms >= s.no_text_commit_ms else None
        r = self._route_for(text)
        live = self.mode == "live"
        t = text.strip()
        if t.endswith((",", "，", "…", "-")):
            # 쉼표·말줄임으로 끝남: 이어서 말하는 중이다 (실측 P0130-01: '대장 영양제,' 뒤 1.8초 쉬고 이어 말함)
            thr, name = s.commit_comma_ms, "comma"
        elif _unfinished_tail(t):
            # 번호를 읽다 멈췄거나 연결어미로 끝남: Soniox가 마침표를 찍어도 이어 말하는 경우가 많다 (블라인드 감사)
            # 번호를 읽다 멈춘 경우는 실제로 번호가 깨져 끝난 말(연습 P0095-06, P0193-02)도 있어 조금만 더 기다린다
            # 시연 모드도 빨리 받아치지 않는다: 번호를 세 자리·네 자리씩 끊어 읽으며 1초 가까이 쉬는 사람이 많다
            name = _unfinished_tail(t)
            thr = s.commit_incomplete_ms if name == "partial_number" else s.commit_noun_ms
        elif classify(t, "", s) == "preamble":
            # 인사·용건 예고뿐: 뒤에 본론이 이어지는 경우가 많다
            thr, name = s.commit_preamble_ms, "preamble"
        elif _SENT_END.search(t) and _noun_only(t):
            # 서술어 없이 명사로만 끝난 말: 이어서 말할 수 있다 (실측 P0130-01: '대장영양제.' 뒤 1.8초 쉬고 이어 말함 → 끼어듦)
            thr, name = (s.live_commit_mid_ms if live else s.commit_noun_ms), "noun"
        elif _SENT_END.search(t):
            # Soniox는 말이 끝나면 문장부호를 붙인다. 실측: Jev는 '…내야 하나요?' 같은 끝난 질문도 0.2로 판정해
            # 28문항 중 4문항이 1.5초를 넘겼고, Jev 덕분에 끼어들기를 막은 사례는 없었다 → 문장부호가 있으면 끝난 말로 본다
            thr, name = (s.live_commit_fast_ms if live else s.commit_complete_ms), "punct"
        elif r is None:
            thr, name = s.commit_silence_ms, "silence"
        elif r.complete >= s.complete_min_prob:
            if live:
                thr = s.live_commit_fast_ms if r.complete >= 0.6 else s.live_commit_mid_ms
            else:
                thr = s.commit_complete_ms
            name = "complete"
        elif r.complete >= 0.25:
            thr, name = s.commit_incomplete_ms, "incomplete"
        else:
            thr, name = s.commit_very_incomplete_ms, "very_incomplete"
        base_thr = thr
        pace_info: dict = {}
        if pace.enabled(s, self.mode) and name in pace.ADJUSTABLE and self.pace.ready():
            # 느리게 말하는 사람·말 중간에 길게 쉬는 사람은 더 기다린다 (1.0~1.2배, 줄이지는 않는다)
            rate = pace.turn_rate(t, self.turn_voiced_ms, s) or self.pace.rate
            own = max(max(self.turn_pauses, default=0.0), self.pace.pause_hint())
            thr, pace_info = pace.adjust(thr, name, rate=rate, own_pause_ms=own, s=s, mode=self.mode)
        if silence_ms < thr:
            return None
        # 확정 시점에 판정이 오는 중이면 조금 더 기다린다: 판정이 있어야 고정 답 바로 응답·첫마디 선택을 할 수 있다
        # (게이트웨이 경유 판정은 p50 0.46초, 무음 0.45초에 부르면 확정 1.0초 무렵에 도착한다)
        if (r is None and s.use_router and base_thr <= s.jev_commit_wait_max_thr_ms and self._similar_pending(norm)
                and silence_ms < thr + s.jev_commit_wait_ms):
            return None
        if pace_info:
            self.tl.info["pace"] = pace_info            # 실제로 확정할 때만 남긴다
        return name

    async def _run_router(self, text: str, norm: str) -> None:
        try:
            r = await self.ctx.router.classify(self.history, text, self.caller)
        except Exception as e:      # 판정 실패는 무음 규칙으로 대체된다
            log.warning("router failed: %r", e)
            return
        finally:
            self.route_pending.pop(norm, None)
        self.routes[norm] = r
        self.tl.count("jev")
        self.tl.event("jev", ok=r.ok, ms=r.ms, complete=round(r.complete, 2), upset=round(r.upset, 2),
                      intent=r.intent, conf=round(r.conf, 2), error=r.error[:80])
        self._live_event("turn.judge", complete=round(r.complete, 2), intent=r.intent, conf=round(r.conf, 2), ms=r.ms)
        if self.phase == "listening":
            self._evaluate(perf())

    def _commit(self, rule: str, text: str, now: float) -> None:
        self.phase = "committing"
        self.speech_end_audio_ms = self.last_voiced_audio_ms or self.audio_ms
        self.speech_end_wall = self.last_voiced_wall or now
        if self.mode == "measure" and self.first_audio_wall is not None and self.last_voiced_wall is not None:
            # 측정기는 음성을 실시간으로 보내고 말 끝을 음성 시계로 잰다. 수신이 밀렸으면 음성 시계 쪽이 더 이르다.
            anchored = self.first_audio_wall + self.speech_end_audio_ms / 1000
            self.tl.info["recv_lag_ms"] = round((self.last_voiced_wall - anchored) * 1000)
            self.speech_end_wall = min(self.speech_end_wall, anchored)
        self.tl.mark("speech_end", self.speech_end_wall, overwrite=True)
        self.tl.mark("commit", now)
        self.tl.info.update(rule=rule, text_at_commit=text)
        self._live_event("turn.commit", rule=rule, text=text, pace=self.tl.info.get("pace"))
        self.commit_evt.set()

    # ───────────────────────── 추측 실행 (시연 모드) ─────────────────────────
    def _spec_enabled(self) -> bool:
        return (self.mode == "live" and self.s.spec_live) or (self.mode == "measure" and self.s.spec_measure)

    async def _tools(self):
        if self.tools_task:
            try:
                tools = await asyncio.wait_for(asyncio.shield(self.tools_task), 3.0)
                if tools:
                    self.ctx.last_tools = tools
                    return tools
            except Exception as e:
                log.warning("%s 함수 정의를 못 받음, 로컬 정의 사용: %r", self.sid, e)
        return self.ctx.last_tools or knowledge.local_tools() or None

    async def _decide(self, msgs: list, effort: str, on_text=None) -> Decision:
        try:
            return await self.ctx.brain.decide(msgs, await self._tools(), effort, on_text=on_text)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("decide failed")
            return Decision("error", error=repr(e)[:200])

    def _start_spec(self, text: str) -> None:
        self._cancel_spec()
        msgs = self.ctx.brain.build_messages(self.history, text, self.caller, self.now, ctx=self.ctx_ids)
        early = EarlyTTS(self) if self.s.early_tts else None
        task = asyncio.create_task(self._decide(msgs, self.s.reasoning_effort, early.feed if early else None))
        self.spec = Spec(text, norm_compare(text), task, msgs, perf(), early)
        self.tl.mark("spec_first")
        self.tl.count("spec")

    def _cancel_spec(self) -> None:
        sp = getattr(self, "spec", None)
        if sp and not sp.task.done():
            sp.task.cancel()
        if sp and sp.early:
            sp.early.cancel()
        self.spec = None

    # ───────────────────────── 측정 모드 ─────────────────────────
    async def _run_measure(self) -> None:
        self.emergency: str | None = None
        self.response_task = asyncio.create_task(self._measure_turn())
        dog = asyncio.create_task(self._watchdog())
        try:
            await self.response_task
        except asyncio.CancelledError:
            if self.emergency is None:           # 세션 자체가 닫히는 경우
                raise
        except Exception:
            log.exception("%s 응답 중 오류", self.sid)
            self.emergency = "error"
        finally:
            dog.cancel()
        if self.emergency and not self.turn_end_sent:
            self.tl.event("emergency", kind=self.emergency)
            self.tl.info["emergency"] = self.emergency
            self.phase = "responding"
            if not (self.emergency == "error" and len(self.spoken) > (1 if self.filler_sent else 0)):
                await self._say_clip("no_speech" if self.emergency == "no_speech" else "fallback")
            await self._send_turn_end()

    async def _measure_turn(self) -> None:
        await self.commit_evt.wait()
        await self._respond()
        await self._send_turn_end()

    async def _watchdog(self) -> None:
        """어떤 경우에도 25초 제한 전에 turn.end를 보낸다. 발동하면 응답 작업을 취소하고 _run_measure가 마무리한다."""
        while not self.turn_end_sent:
            await asyncio.sleep(0.1)
            now = perf()
            kind = None
            if not self.speech_started and not self.stt.has_tokens() and now - self.t_start > self.s.no_speech_timeout_s:
                kind = "no_speech"
            else:
                last = max([t for t in (self.last_voiced_wall, self.stt.last_token_wall) if t] or [0])
                if last and now - last > self.s.response_watchdog_s:
                    kind = "error"
            if kind and not self.turn_end_sent:
                self.emergency = kind
                if self.response_task and not self.response_task.done():
                    self.response_task.cancel()
                return

    # ───────────────────────── 시연 모드 ─────────────────────────
    async def _run_live(self) -> None:
        first, keep = True, False
        while not self.closed.is_set():
            if not first:
                self._reset_turn(keep_speech=keep)
            first, keep = False, False
            waiter = asyncio.create_task(self.commit_evt.wait())
            closer = asyncio.create_task(self.closed.wait())
            await asyncio.wait({waiter, closer}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            closer.cancel()
            if self.closed.is_set():
                break
            self.response_task = asyncio.create_task(self._respond())
            try:
                await self.response_task
            except asyncio.CancelledError:
                cur = asyncio.current_task()
                if not self.barge or (cur and cur.cancelling()):
                    raise
            except Exception:
                log.exception("%s 응답 중 오류", self.sid)
            if self.barge:
                await self._finish_interrupted()
                keep = True
            else:
                self._remember_agent_turn(interrupted=False)
                await self._send_turn_end()
            rec = self.tl.dump({"transcript": self.tl.info.get("transcript")}, redact=not self.s.log_text)
            self._live_event("turn.metrics", **rec["ms_after_speech_end"], rule=self.tl.info.get("rule"),
                             play_speed=self.tl.info.get("play_speed", 1.0), pace=self.tl.info.get("pace"),
                             counts=rec["counts"], marks=rec["marks"])
            self._live_event("debug.history", history=self.history)

    def _check_barge(self, now: float) -> None:
        if self.phase not in ("committing", "responding", "speaking") or self.barge:
            return
        by_voice = self.voiced_run_ms >= self.s.barge_in_ms
        # 텍스트 기준은 받아쓰기를 새 턴으로 넘긴 뒤에만 (그 전에는 방금 커밋한 문장이 남아 있다)
        by_text = (self.stt_turn_reset and len(norm_compare(self.stt.text())) >= self.s.barge_in_chars
                   and self.last_voiced_wall is not None and now - self.last_voiced_wall < 0.5)
        if by_voice or by_text:
            self.barge = True
            self.tl.event("barge_in", by="voice" if by_voice else "text")
            for t in (self.audio_tail, self.response_task):
                if t and not t.done():
                    t.cancel()

    def _turn_text(self) -> str:
        """이번 턴의 고객 말. 확정 직후 이어 말해 합친 경우 앞부분을 붙인다."""
        carry = getattr(self, "carry_text", "")
        return (carry + " " + self.stt.text()).strip() if carry else self.stt.text()

    def _mergeable(self) -> bool:
        """끼어든 말이 '말을 이어 간 것'인가: 상담원이 첫마디(채움말)만 했고 실제 답·함수 실행은 아직이면 그렇다.
        이때는 앞말과 이어 말을 한 문장으로 합쳐 다시 판단한다 (예: '010 1234' 확정 → 첫마디 → '5678')."""
        if not (self.turn_customer or self.tl.info.get("transcript")):
            return False
        if self.tl.counts.get("tool_calls", 0) or self.tl.counts.get("bank_direct", 0):
            return False
        return all(seg.get("text") == self.opener_text for seg in self.spoken)

    async def _finish_interrupted(self) -> None:
        await self._send_json({"type": "agent.cancel"})
        ft = getattr(self, "final_task", None)
        if ft is not None and not ft.done():
            try:
                await asyncio.wait_for(asyncio.shield(ft), 1.0)   # 끊긴 턴의 손님 말을 이력에 남기기 위해
            except Exception:
                pass
        if self.mode == "live" and self._mergeable():
            # 이력에 남기지 않고 다음 턴 앞에 붙인다. 이미 이력에 넣었으면 되돌린다
            said = self.turn_customer
            last = self.history[-1] if self.history else {}
            if self.customer_pushed and last.get("role") == "customer" and last.get("text") == said:
                self.history.pop()
            self.customer_pushed = True                    # 턴 끝에서 다시 넣지 않게
            self.carry_text = said
            self.carry_voiced_ms = max(0.0, self.turn_voiced_ms - self.voiced_run_ms)
            # 끝난 줄 알고 받아쳤는데 말을 이어 갔다: 이 사람은 그만큼 쉬는 사람이다 → 다음부터 더 기다린다
            gap = (self.audio_ms - self.voiced_run_ms) - self.speech_end_audio_ms
            self.pace.note_pause(gap)
            self.tl.event("carry", chars=len(said), gap_ms=round(gap))
            self.tl.info["interrupted_after"] = "(이어 말함: 합침)"
            return
        played = self._remember_agent_turn(interrupted=True)
        if self.s.live_agent_text:
            await self.stt.send_agent_text("", end_of_segment=True)   # 들려준 데까지만, 세그먼트 닫기
        self.tl.info["interrupted_after"] = played

    def _remember_agent_turn(self, interrupted: bool) -> str:
        """실제로 재생된 부분까지만 이력에 남긴다. 재생 위치는 보낸 길이와 클라이언트 버퍼로 추정."""
        self._push_customer()
        unplayed = max(0.0, self.play_end - perf()) * 1000 if interrupted else 0.0
        parts = []
        for seg in reversed(self.spoken):
            if unplayed <= 0:
                parts.append(seg["text"])
                continue
            if unplayed >= seg["ms"]:
                unplayed -= seg["ms"]
                continue
            frac = 1 - unplayed / seg["ms"] if seg["ms"] else 1
            parts.append(seg["text"][: max(0, int(len(seg["text"]) * frac))])
            unplayed = 0
        text = " ".join(reversed([p for p in parts if p])).strip()
        if interrupted and text:
            text = text + " [중단됨]"
        if text:
            self.history.append({"role": "agent", "text": text})
        return text

    # ───────────────────────── 응답 ─────────────────────────
    async def _finalize_transcript(self, t_commit: float) -> str:
        """확정 받아쓰기를 얻는다 (커밋 뒤 최대 finalize_wait_ms). 측정 모드는 스트림을 닫는다."""
        budget = t_commit + self.s.finalize_wait_ms / 1000
        if not self.stt.end_seen:
            await self.stt.finalize(max(0.0, budget - perf()))
        while (self.stt.final_audio_ms < self.speech_end_audio_ms - 1 and perf() < budget
               and not self.stt.error and not self.stt.closed):
            await asyncio.sleep(0.02)
        self.tl.info["final_lag_ms"] = round(self.speech_end_audio_ms - self.stt.final_audio_ms)
        final = self.stt.final_text() or self.stt.text()
        carry = getattr(self, "carry_text", "")
        if carry:                                          # 끊겼다가 이어 말한 앞부분을 붙인다 (시연 모드)
            final = (carry + " " + final).strip()
            self.carry_text = ""
            self.tl.event("merged", chars=len(carry))
        self.tl.info["transcript"] = final
        self.pace.note_rate(pace.turn_rate(final, self.turn_voiced_ms, self.s))   # 확정 받아쓰기로 잰다 (늦게 따라온 끝 글자 포함)
        self.tl.mark("final_text")
        self.turn_customer = final
        if self.mode == "measure":
            asyncio.create_task(self.stt.close())          # 동시 스트림 한도 보호
        else:
            self.stt.new_turn()
            self.stt_turn_reset = True
        return final

    def _customer_texts(self) -> list[str]:
        """이번 통화에서 고객이 한 말 (이번 턴 포함)."""
        out = [str(h.get("text") or "") for h in self.history if h.get("role") == "customer"]
        if self.turn_customer and not self.customer_pushed:
            out.append(self.turn_customer)
        return out

    def _push_customer(self) -> None:
        """이번 턴의 손님 말을 이력에 한 번만 남긴다 (끼어들기로 응답이 취소돼도 남도록 턴 끝에서도 부른다)."""
        if self.turn_customer and not self.customer_pushed:
            self.history.append({"role": "customer", "text": self.turn_customer})
            self.customer_pushed = True

    async def _respond(self) -> None:
        s = self.s
        self.phase = "responding"
        t_commit = perf()
        text0 = self._turn_text()
        route = self._route_for(text0)
        # 확정 받아쓰기는 별도 작업으로 바로 시작한다. 첫 소리를 기다리지 않게 하고, 시연 모드에서 말하는 도중
        # 끼어들어 응답이 취소돼도 받아쓰기는 다음 턴으로 넘어가게 한다 (실측: 이전 턴 글이 다음 턴에 섞였다)
        self.final_task = asyncio.create_task(self._finalize_transcript(t_commit))
        self.play_speed = pace.play_speed(self.pace, s, self.mode)    # 이전 턴까지 잰 말 속도로 이번 답의 재생 속도를 정한다
        if self.play_speed < 1.0:
            self.tl.info["play_speed"] = self.play_speed
        # 1. 첫 소리: 은행 직행 또는 첫마디. 확정 받아쓰기를 기다리지 않는다 (1.5초 안에 나가야 한다)
        bank_clip, opener_kind = None, ""
        if text0.strip():
            bank_key, why = choose_bank(text0, self.history, self.caller, route, s, mode=self.mode)
            bank_clip = self.ctx.clips.bank(bank_key) if bank_key else None
            if bank_clip is not None:
                self.tl.info.update(decision="bank", bank=bank_key, bank_why=why)
                self.tl.count("bank_direct")
                self.tl.mark("decision")
                await self._speak(bank_clip[0], bank_clip[1])
            else:
                opener_kind = choose_opener(text0, route, s)
                await self._opener(opener_kind)
        # 2. 확정 받아쓰기
        final = await asyncio.shield(self.final_task)
        if not final.strip():
            self.tl.info["decision"] = "empty"
            if bank_clip is None:
                await self._say_clip("fallback")
            return
        if bank_clip is not None:
            self._push_customer()
            await self._drain()
            return
        # 3. 판단: 추측 결과가 확정 받아쓰기와 같으면 그대로 쓴다 (시연 모드)
        sp = self.spec
        if sp and sp.norm == norm_compare(final) and not sp.task.cancelled():
            task, msgs = sp.task, sp.msgs
            early, sp.early = sp.early, None               # 추측 실행 중 미리 만든 첫 문장 음성도 넘겨받는다
            self.tl.count("spec_hit")
            self.tl.info["spec"] = "hit"
        else:
            self.tl.info["spec"] = "miss" if sp else "none"
            self._cancel_spec()
            msgs = self.ctx.brain.build_messages(self.history, final, self.caller, self.now,
                                                 ctx=self.ctx_ids, spoken=self.opener_text)
            early = EarlyTTS(self) if s.early_tts else None
            task = asyncio.create_task(self._decide(msgs, s.reasoning_effort, early.feed if early else None))
        self.early = early                                 # 턴이 바뀔 때 _reset_turn이 버릴 수 있게 남긴다
        self._push_customer()
        try:
            decision = await task
            self.tl.mark("decision")
            self.tl.info["decision"] = decision.kind
            if decision.usage:
                self.tl.info["usage"] = decision.usage
            await self._decision_loop(decision, msgs, early)
            await self._drain()
        finally:
            if early is not None:                          # 쓰이지 않은 미리 합성은 버린다 (취소·오류 포함)
                early.cancel()

    async def _call_tool(self, name: str, args: dict) -> dict:
        r = await self.ctx.tools.call(self.tools_url, self.sid, name, args, self.s.tool_timeout_s)
        ms = round((r["t1"] - r["t0"]) * 1000)
        self.tl.event("tool", name=name, args=args, status=r["status_code"], ms=ms)
        self.tl.count("tool_calls")
        self.tl.mark("tool_resp", overwrite=True)
        self.facts.update({k: v for k, v in args.items() if isinstance(v, (str, int))})
        self.actions.append({"name": name, "args": args, "status": r["status_code"],
                             "result": r["body"] if isinstance(r["body"], dict) else str(r["body"])})
        self._live_event("tool.call", name=name, args=args)
        self._live_event("tool.result", name=name, status=r["status_code"], body=r["body"], ms=ms)
        return {"status_code": r["status_code"], "body": r["body"], "ms": ms}

    async def _decision_loop(self, decision: Decision, msgs: list, early: EarlyTTS | None = None) -> None:
        """함수 호출이 끝날 때까지 돈 뒤 답을 말한다. 조회 없이 '확인해 보니'라고 답하려 하면 한 번 되돌린다."""
        s = self.s
        msgs = list(msgs)
        rounds, guard_used, after_tool = 0, False, False
        all_runs: list = []
        prior_write = toolflow.history_write_succeeded(self.history)      # 이번 턴 전에 이미 성공한 접수
        failed_before = toolflow.lookup_failed_before(self.history)
        while True:
            if decision.kind == "answer" and decision.text:
                # 직접 답의 첫 문장 번호([D번호]) → 미리 합성한 문장. 함수 결과 뒤 답에서는 번호 표시만 지운다
                decision.text = directlib.expand(decision.text, allow=(rounds == 0))
            if (decision.kind == "answer" and not guard_used
                    and claims_lookup_without_evidence(decision.text, self.evid)):
                guard_used = True
                self.tl.event("guard", why="조회 없이 확인했다고 답하려 함")
                msgs += [decision.assistant_msg, {"role": "system", "content": GUARD_NOTE}]
                decision = await self._decide(msgs, s.reasoning_effort)
                continue
            if decision.kind == "answer" and not guard_used:
                # 조건이 붙은 첫 문장 후보를 반대 조건에서 쓰려 함 (예: 쿠팡 구매 건에 자사몰 반품 기간) → 한 번 되돌린다
                bad = directlib.violations(facts.split_sentences(decision.text), self._customer_texts())
                if bad:
                    guard_used = True
                    self.tl.event("guard", why="첫 문장 후보 조건 위반")
                    self.tl.count("direct_guard")
                    note = "(시스템: " + " ".join(o.note for o in bad) + " 그 문장을 쓰지 말고 다시 답한다.)"
                    msgs += [decision.assistant_msg, {"role": "system", "content": note}]
                    decision = await self._decide(msgs, s.reasoning_effort)
                    continue
            if decision.kind == "answer" and not guard_used:
                # 성공한 접수가 없는데 '접수해 드렸습니다'라고 하려 함 → 한 번 되돌린다 (이미 접수된 건이면 문장만 바꾼다)
                fixed, issue = toolflow.write_claim_check(decision.text, all_runs, prior_write)
                if issue == "already":
                    self.tl.event("guard", why="이미 접수된 건을 접수했다고 함 → 문장 교체")
                    decision.text = fixed
                elif issue == "false_claim":
                    guard_used = True
                    self.tl.event("guard", why="성공한 접수 없이 접수했다고 함")
                    msgs += [decision.assistant_msg, {"role": "system", "content": toolflow.WRITE_NOTE}]
                    decision = await self._decide(msgs, s.reasoning_effort)
                    continue
            if (decision.kind == "answer" and not guard_used and rounds == 0 and not failed_before
                    and toolflow.needs_return_lookup(self.turn_customer, self.history, bool(self.ctx_ids.get("phones")))):
                # 반품·환불 진행 문의인데 조회 없이 답하려 함 → 반품 조회부터 하게 한 번 되돌린다 (실측 P0031-02)
                guard_used = True
                self.tl.event("guard", why="반품 진행 문의에 조회 없이 답함")
                msgs += [decision.assistant_msg, {"role": "system", "content": toolflow.RETURN_NOTE}]
                decision = await self._decide(msgs, s.reasoning_effort)
                continue
            if (decision.kind == "answer" and not guard_used and rounds == 0 and not failed_before
                    and (self.ctx_ids.get("caller_phone") or self.ctx_ids.get("orders"))
                    and toolflow.asks_for_number(decision.text)):
                # 발신자 번호·이전 대화 주문번호가 있는데 번호를 다시 물으려 함 → 조회부터 하게 한 번 되돌린다 (실측 P0010-11)
                guard_used = True
                self.tl.event("guard", why="번호가 있는데 다시 물음")
                msgs += [decision.assistant_msg, {"role": "system", "content": toolflow.NUMBER_NOTE}]
                decision = await self._decide(msgs, s.reasoning_effort)
                continue
            if not (decision.kind == "tool" and rounds < s.max_tool_rounds):
                break
            if not self.tools_url:
                self.tl.event("tool_blocked", reason="tools_url 없음")
                await self._say_clip("fallback")
                return
            if self.mode == "live" and rounds == 0:
                # 시연 모드: 사람 상담원처럼 "주문 내역 확인해 드리겠습니다"를 이어 말한다 (조회 동안 4초 넘는 침묵 방지)
                kind = tool_kind(decision.tool_calls[0].name, self.ctx.phrases)
                clip = self.ctx.clips.pick(kind + "_cont" if self.filler_sent else kind)
                if clip:
                    self._enqueue(clip[0], clip[1] + silence(self.s.filler_tail_silence_ms))
            if self.mode == "measure" and self.filler_sent and rounds == 0:
                # 첫마디가 측정기에 다 도착한 뒤에 호출한다 (꼬리가 '조회 뒤 답'으로 잡히지 않게)
                gap = s.tool_gap_after_filler_ms / 1000 - (perf() - (self.tl.t0 + self.tl.marks.get("opener_sent", 0)))
                if gap > 0:
                    await asyncio.sleep(gap)
            self.tl.mark("tool_call")
            if toolflow.add_return_lookup(decision, self.turn_customer, self.ctx_ids.get("caller_phone", "")):
                self.tl.event("co_lookup", name="get_return_info")      # 반품·환불 진행 문의: 반품 조회도 함께
            runs = await execute_calls(decision.tool_calls, self.ctx_ids, self._call_tool, s, self.now)
            all_runs += runs
            msgs.append(decision.assistant_msg)
            for run in runs:
                msgs.append({"role": "tool", "tool_call_id": run.tc.id, "content": run.content})
                self.history.append({"role": "tool_call", "name": run.tc.name, "arguments": run.args})
                self.history.append({"role": "tool_result", "name": run.tc.name, "result": run.body})
                if run.blocked:
                    self.tl.event("tool_blocked", name=run.tc.name, reason=run.blocked[:60])
                if run.notes:
                    self.tl.event("id_resolved", name=run.tc.name, notes=run.notes)
                if run.calls:
                    after_tool = True
            self.evid += evidence_text(runs) + "\n"
            self._refresh_context()
            rk = reask_for(runs)
            if rk:
                # 번호가 깨져 들려 모든 호출이 막혔다: LLM을 거치지 않고 구체적으로 다시 묻는다
                clip = self.ctx.clips.bank(rk)
                self.tl.event("reask", kind=rk)
                self.tl.count("reask")
                if clip:
                    await self._drain()
                    await self._speak(clip[0], clip[1])
                    return
            rounds += 1
            msgs.append({"role": "system", "content": aftertool.note()})
            decision = await self._decide(msgs, s.reasoning_effort_round2)
            self.tl.mark("decision2", overwrite=True)
            if decision.kind == "answer":
                decision.text = aftertool.expand(decision.text)
        if decision.kind == "answer" and decision.text:
            # 미리 합성은 첫 판단이 그대로 답이 된 경우만 (함수 호출 뒤 답·되돌린 답은 글이 달라 쓰지 않는다)
            await self._answer(decision.text, after_tool=after_tool,
                               early=early if (rounds == 0 and not guard_used) else None)
        else:
            self.tl.info["error"] = decision.error or "tool rounds exceeded"
            log.warning("%s 결정 실패: %s", self.sid, (decision.error or "")[:200])
            await self._say_clip("fallback")

    # ───────────────────────── 말하기 ─────────────────────────
    def _prepare_answer(self, text: str) -> tuple[str | None, str]:
        """(앞에 붙일 짧은 음성 종류, 본문). 상투적인 서두를 걷어 내고, '네,'·'아니요,'는 캐시 음성으로 낸다."""
        t = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        for pat, rep in _PLEASANTRY:
            t = pat.sub(rep, t, count=1)
        cust = next((h["text"] for h in reversed(self.history) if h.get("role") == "customer"), "")
        if not re.search(r"안녕|여보세요", cust):
            t = re.sub(r"^\s*안녕하세요\s*[,.!]?\s*(?=\S)", "", t)
        m = _YES.match(t)
        if m:
            return "yes", t[m.end():].strip()
        m = _NO.match(t)
        if m:
            return "no", t[m.end():].strip()
        return None, t

    def _unit_plan(self, body: str) -> list[tuple]:
        """답을 말하는 단위로 나눈다 (합성은 하지 않는다). [("clip", 글, 음성) | ("tts", 글, None)].
        첫 합성 단위는 첫 문장 하나로 따로 두고, 길면 쉼표에서 앞부분만 떼어 낸다 (TTS 시간은 글자 수에 비례해
        첫 단위가 짧을수록 답이 빨리 들린다). 첫 단위는 첫 합성 문장에만 달려 있어 뒷부분이 바뀌어도 같다."""
        s = self.s
        units: list[tuple] = []
        cur = ""
        first_done = False

        def flush():
            nonlocal cur
            if cur.strip():
                units.append(("tts", cur.strip(), None))
            cur = ""

        for sent in facts.split_sentences(body):
            clip = self.ctx.clips.match(sent)
            if clip:
                flush()
                units.append(("clip", clip[0], clip[1]))
            elif not first_done:
                first_done = True
                flush()
                head, rest = facts.short_head(sent, s.first_unit_chars, s.first_unit_min_chars)
                units.append(("tts", head, None))
                cur = rest
            elif cur and len(cur) + len(sent) > s.tts_unit_chars:
                flush()
                cur = sent
            else:
                cur = (cur + " " + sent).strip()
        flush()
        return units

    def _early_unit(self, raw: str) -> str | None:
        """스트리밍 중인 답 앞부분에서 첫 합성 단위 글을 정할 수 있으면 그 글. 최종 답에서 같은 글이 나와야만 쓰인다.
        첫 문장이 확실히 끝났을 때(뒤에 다음 문장이 시작됨)만 정한다. 자리표시·번호 선택 표시가 섞이면 하지 않는다."""
        if raw and directlib.has_mark(raw):
            raw = directlib.expand(raw)          # 첫 문장은 미리 합성한 문장, 둘째 문장부터 미리 합성
        if not raw or "{" in raw or "[" in raw:
            return None
        sents = facts.split_sentences(raw)
        if len(sents) < 2:
            return None
        done = " ".join(sents[:-1])
        if claims_lookup_without_evidence(done, self.evid):
            return None                                   # 되돌려질 답이다
        _, body = self._prepare_answer(done)
        body, _ = facts.verify_answer(body, self.evid, self.now)
        for kind, utext, _ in self._unit_plan(body):
            if kind == "tts":
                return utext
        return None

    def _plan_units(self, body: str, early: EarlyTTS | None = None) -> list[tuple]:
        """말하는 단위마다 TTS를 동시에 시작한다. 첫 합성 단위는 미리 만든 음성이 있고 글이 같으면 그것을 쓴다."""
        units: list[tuple] = []
        first_tts = True
        for kind, utext, payload in self._unit_plan(body):
            if kind == "clip":
                units.append((kind, utext, payload))
                self.tl.count("bank_sentence")
                continue
            job = early.take(utext) if (first_tts and early is not None) else None
            first_tts = False
            units.append(("tts", utext, job or TTSJob(self.ctx.tts, for_tts(utext, self.s, self.now))))
        if early is not None:
            early.cancel()                                # 첫 단위가 합성 단위가 아니었던 경우 등
        return units

    async def _answer(self, text: str, after_tool: bool = False, early: EarlyTTS | None = None) -> None:
        s = self.s
        prefix, body = self._prepare_answer(text)
        body = facts.clamp_answer(body, s.answer_max_sentences, s.answer_max_chars, is_bank=self.ctx.clips.match_text)
        body, fixes = facts.verify_answer(body, self.evid, self.now)
        if fixes:
            self.tl.event("answer_fixed", fixes=fixes)
        units = self._plan_units(body, early) if body else []
        if not body and early is not None:
            early.cancel()
        self.tl.mark("tts_req")
        if prefix and not (prefix == "yes" and self.filler_sent):     # 이미 "네"를 했으면 다시 하지 않는다
            clip = self.ctx.clips.pick(prefix)
            if clip:
                await self._drain()
                await self._speak(clip[0], clip[1], mark_answer=after_tool)
        if self.mode == "live" and _HANDOFF_SAID.search(body):
            self._send_handoff("상담원 연결 안내")
        spoke = 0
        slow = getattr(self, "play_speed", 1.0) < 1.0
        stretch = 1.0 / getattr(self, "play_speed", 1.0)              # 느리게 들려줄 때는 문장 사이 쉼도 그만큼 늘린다
        sent_gap, comma_gap = silence(s.sentence_gap_ms * stretch), silence(s.comma_gap_ms * stretch)
        try:
            for i, (kind, utext, payload) in enumerate(units):
                gap = comma_gap if utext.endswith(",") else sent_gap     # 쉼표에서 나눈 조각 뒤는 짧게 쉰다
                await self._drain()
                if kind == "clip":
                    # _speak이 음성과 뒤의 쉼을 함께 느리게 하므로 여기서는 원래 길이의 쉼을 붙인다 (이중으로 늘지 않게)
                    raw_gap = silence(s.comma_gap_ms if utext.endswith(",") else s.sentence_gap_ms)
                    await self._speak(utext, payload + (raw_gap if i < len(units) - 1 else b""), mark_answer=after_tool)
                    spoke += 1
                    continue
                job: TTSJob = payload
                seg = {"text": utext, "ms": 0.0}
                got = False
                try:
                    if slow:
                        # 느리게 들려주려면 문장 음성이 다 있어야 한다 (게이트웨이 TTS는 어차피 한 번에 온다 → 첫 소리 손해 없음)
                        whole = b"".join([c async for c in job.iterate()])
                        if whole:
                            got = True
                            await self._send_text(utext)
                            self.spoken.append(seg)
                            self.tl.mark("tts_first", job.stats.get("first"))
                            if job.stats.get("cache"):
                                self.tl.count("tts_cache")
                            await self._send_pcm(self._paced(whole), mark_answer=after_tool, seg=seg)
                    async for c in (_empty() if slow else job.iterate()):
                        if not got:
                            got = True
                            await self._send_text(utext)
                            self.spoken.append(seg)
                            self.tl.mark("tts_first", job.stats.get("first"))
                            if job.stats.get("cache"):
                                self.tl.count("tts_cache")
                        await self._send_pcm(c, mark_answer=after_tool, seg=seg)
                    if got:
                        spoke += 1
                        if i < len(units) - 1:
                            await self._send_pcm(gap)
                except TTSError as e:      # 실패한 문장은 건너뛴다. 대체 음성으로 뜻을 바꾸지 않는다
                    self.tl.event("tts_failed", text=utext[:40], error=repr(e)[:120])
                    self.tl.count("tts_failed")
                    log.warning("%s TTS 실패(건너뜀): %r", self.sid, e)
        finally:
            for kind, _, payload in units:
                if kind == "tts":
                    payload.cancel()
        if not spoke and units:
            await self._say_clip("fallback")

    async def _opener(self, kind: str) -> None:
        clip = self.ctx.clips.pick(kind) or self.ctx.clips.pick("neutral")
        if not clip:
            return
        self.filler_sent = True
        self.opener_text = clip[0]
        self.tl.event("opener", kind=kind, text=clip[0])
        self.tl.count("filler")
        self.tl.info.setdefault("filler", kind)
        pcm = clip[1] + silence(self.s.filler_tail_silence_ms)
        if self.mode == "live":
            self._enqueue(clip[0], pcm)       # 재생하는 동안 판단·조회를 계속한다
        else:
            await self._speak(clip[0], pcm)
        self.tl.mark("opener_sent")

    async def _say_clip(self, kind: str) -> None:
        clip = self.ctx.clips.pick(kind)
        if clip:
            await self._drain()
            await self._speak(clip[0], clip[1])

    def _enqueue(self, text: str, pcm: bytes, mark_answer: bool = False) -> None:
        """시연 모드: 앞선 음성이 끝난 뒤 이어서 재생되도록 줄 세운다 (기다리지 않고 돌아온다)."""
        prev = self.audio_tail

        async def run():
            if prev:
                try:
                    await prev
                except BaseException:
                    return
            await self._speak(text, pcm, mark_answer)

        self.audio_tail = asyncio.create_task(run())

    async def _drain(self) -> None:
        if self.audio_tail and not self.audio_tail.done():
            await self.audio_tail

    async def _speak(self, text: str, pcm: bytes, mark_answer: bool = False) -> None:
        await self._send_text(text)
        seg = {"text": text, "ms": 0.0}
        self.spoken.append(seg)
        await self._send_pcm(self._paced(pcm), mark_answer=mark_answer, seg=seg)

    async def _send_text(self, text: str) -> None:
        await self._send_json({"type": "agent.transcript", "text": text})

    def _paced(self, pcm: bytes) -> bytes:
        """재생 속도를 적용한 음성. 같은 녹음 문장은 한 번만 바꿔 둔다. 실패하면 원래 음성."""
        sp = getattr(self, "play_speed", 1.0)
        if sp >= 1.0 or not pcm:
            return pcm
        key = (len(pcm), hash(pcm), sp)
        cache = getattr(self, "paced_cache", None)
        if cache is None:
            cache = self.paced_cache = {}
        out = cache.get(key)
        if out is None:
            try:
                out = time_stretch(pcm, sp)
            except Exception:
                log.exception("재생 속도 변환 실패 (원래 속도로)")
                out = pcm
            if len(cache) > 200:
                cache.clear()
            cache[key] = out
        return out

    async def _send_pcm(self, pcm: bytes, mark_answer: bool = False, seg: dict | None = None) -> float:
        """보낸 길이(ms)를 돌려준다. 측정 모드는 곧바로, 시연 모드는 재생 속도에 맞춰 조금씩.
        seg가 있으면 조각마다 길이를 누적한다 (중간에 끊겨도 어디까지 보냈는지 남는다)."""
        sent = 0.0
        step = self.s.out_chunk_ms if self.mode == "measure" else 20
        for c in chunks(pcm, step):
            if self.mode == "live":
                while self.play_end - perf() > self.s.live_lead_ms / 1000:
                    await asyncio.sleep(0.01)
            self.phase = "speaking" if self.mode == "live" else self.phase
            try:
                async with self.send_lock:
                    await self.ws.send(c)
            except ConnectionClosed:
                self.closed.set()
                return sent
            dur = len(c) / BYTES_PER_MS
            now = perf()
            self.play_end = max(self.play_end, now) + dur / 1000
            sent += dur
            if seg is not None:
                seg["ms"] += dur
            if rms(c) >= self.s.silence_rms:
                if not self.first_sound_sent:
                    self.first_sound_sent = True
                    self.tl.mark("first_sound", now)
                if mark_answer:
                    self.tl.mark("answer_sound", now)
        return sent

    async def _send_turn_end(self) -> None:
        if self.turn_end_sent:
            return
        self.turn_end_sent = True
        self.tl.mark("turn_end")
        await self._send_json({"type": "turn.end"})

    async def _send_json(self, obj: dict) -> None:
        try:
            async with self.send_lock:
                await self.ws.send(json.dumps(obj, ensure_ascii=False))
        except ConnectionClosed:
            self.closed.set()

    def _live_event(self, typ: str, **kv) -> None:
        """시연 화면용 이벤트. 측정 모드에서는 보내지 않는다 (측정기가 모르는 메시지를 줄인다)."""
        if getattr(self, "mode", "measure") == "live":
            asyncio.create_task(self._send_json({"type": typ, **kv}))

    def _send_handoff(self, reason: str) -> None:
        cust = [h["text"] for h in self.history if h.get("role") == "customer"]
        self._live_event("handoff", reason=reason, caller=self.caller, facts=self.facts,
                         actions=self.actions[-5:], summary=" / ".join(cust[-3:]))

    def _dump(self) -> None:
        extra = {"stt_error": self.stt.error, "threshold": self.thr,
                 "tts": {"requests": self.ctx.tts.requests, "cache_hits": self.ctx.tts.cache_hits}}
        if self.s.record_stt and self.stt.recording:
            from .config import RUNTIME
            p = RUNTIME / "stt_rec" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', self.sid)}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"session_id": self.sid, "case_id": self.case_id,
                                     "speech_end_audio_ms": self.speech_end_audio_ms,
                                     "recording": self.stt.recording}, ensure_ascii=False), encoding="utf-8")
        rec = self.tl.dump(extra, redact=not self.s.log_text)
        d = rec["ms_after_speech_end"]
        log.info("%s %s rule=%s decision=%s first=%sms commit=%sms tool→answer=%sms opener=%s jev=%s",
                 self.sid, self.case_id, rec.get("rule"), rec.get("decision"), d.get("first_sound"),
                 d.get("commit"), d.get("after_tool"), rec.get("filler"), rec.get("counts", {}).get("jev", 0))
