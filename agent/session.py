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

from . import aftertool, facts, knowledge
from .audio import BYTES_PER_MS, chunks, rms, silence
from .brain import Decision, parse_now
from .metrics import Timeline
from .router import choose_bank, choose_opener
from .textnorm import for_tts, norm_compare
from .toolflow import GUARD_NOTE, claims_lookup_without_evidence, evidence_text, execute_calls
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


@dataclass
class Spec:
    text: str
    norm: str
    task: asyncio.Task
    msgs: list
    started: float


class Session:
    def __init__(self, ws, ctx):
        self.ws, self.ctx, self.s = ws, ctx, ctx.settings
        self.send_lock = asyncio.Lock()
        self.closed = asyncio.Event()
        self.t_start = perf()
        self.audio_ms = 0.0
        self.first_audio_wall: float | None = None
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
        self.spec: Spec | None = None
        self.routes: dict = {}              # 받아쓰기(정규화) → Jev 판정
        self.route_pending: set[str] = set()
        self.commit_evt = asyncio.Event()
        self.first_sound_sent = False
        self.filler_sent = False
        self.opener_text = ""
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
            self.voiced_run_ms += dt
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
            self._live_event("stt.partial", text=self.stt.text())
            self._evaluate(perf())
        elif self.mode == "live":
            self._check_barge(perf())

    # ───────────────────────── 말 끝 판단 ─────────────────────────
    def _silence_ms(self) -> float:
        if self.speech_started and self.last_voiced_audio_ms is not None:
            return self.audio_ms - self.last_voiced_audio_ms
        return 0.0

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
        text = self.stt.text()
        norm = norm_compare(text)
        silence_ms = self._silence_ms()
        quiet = self.speech_started and not self.voiced_now
        # 판정 모델: 조용해지면 한 번 부른다 (받아쓰기가 바뀌면 다시)
        pause = s.live_router_pause_ms if self.mode == "live" else s.router_pause_ms
        if (s.use_router and norm and quiet and silence_ms >= pause
                and norm not in self.routes and norm not in self.route_pending):
            self.route_pending.add(norm)
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
            if (self.stt.end_seen and norm and self.stt.last_token_wall
                    and (now - self.stt.last_token_wall) * 1000 >= s.noise_commit_token_idle_slow_ms):
                return "noise"
            return None
        if not norm:
            return "no_text" if silence_ms >= s.no_text_commit_ms else None
        r = self._route_for(text)
        live = self.mode == "live"
        if r is None:
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
        return name if silence_ms >= thr else None

    async def _run_router(self, text: str, norm: str) -> None:
        try:
            r = await self.ctx.router.classify(self.history, text, self.caller)
        except Exception as e:      # 판정 실패는 무음 규칙으로 대체된다
            log.warning("router failed: %r", e)
            return
        finally:
            self.route_pending.discard(norm)
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
        self._live_event("turn.commit", rule=rule, text=text)
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

    async def _decide(self, msgs: list, effort: str) -> Decision:
        try:
            return await self.ctx.brain.decide(msgs, await self._tools(), effort)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("decide failed")
            return Decision("error", error=repr(e)[:200])

    def _start_spec(self, text: str) -> None:
        self._cancel_spec()
        msgs = self.ctx.brain.build_messages(self.history, text, self.caller, self.now, ctx=self.ctx_ids)
        task = asyncio.create_task(self._decide(msgs, self.s.reasoning_effort))
        self.spec = Spec(text, norm_compare(text), task, msgs, perf())
        self.tl.mark("spec_first")
        self.tl.count("spec")

    def _cancel_spec(self) -> None:
        sp = getattr(self, "spec", None)
        if sp and not sp.task.done():
            sp.task.cancel()
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

    async def _finish_interrupted(self) -> None:
        await self._send_json({"type": "agent.cancel"})
        played = self._remember_agent_turn(interrupted=True)
        if self.s.live_agent_text:
            await self.stt.send_agent_text("", end_of_segment=True)   # 들려준 데까지만, 세그먼트 닫기
        self.tl.info["interrupted_after"] = played

    def _remember_agent_turn(self, interrupted: bool) -> str:
        """실제로 재생된 부분까지만 이력에 남긴다. 재생 위치는 보낸 길이와 클라이언트 버퍼로 추정."""
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
        self.tl.info["transcript"] = final
        self.tl.mark("final_text")
        if self.mode == "measure":
            asyncio.create_task(self.stt.close())          # 동시 스트림 한도 보호
        else:
            self.stt.new_turn()
            self.stt_turn_reset = True
        return final

    async def _respond(self) -> None:
        s = self.s
        self.phase = "responding"
        t_commit = perf()
        text0 = self.stt.text()
        route = self._route_for(text0)
        # 1. 첫 소리: 은행 직행 또는 첫마디. 확정 받아쓰기를 기다리지 않는다 (1.5초 안에 나가야 한다)
        bank_clip, opener_kind = None, ""
        if text0.strip():
            bank_key, why = choose_bank(text0, self.history, self.caller, route, s)
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
        final = await self._finalize_transcript(t_commit)
        if not final.strip():
            self.tl.info["decision"] = "empty"
            if bank_clip is None:
                await self._say_clip("fallback")
            return
        if bank_clip is not None:
            self.history.append({"role": "customer", "text": final})
            await self._drain()
            return
        # 3. 판단: 추측 결과가 확정 받아쓰기와 같으면 그대로 쓴다 (시연 모드)
        sp = self.spec
        if sp and sp.norm == norm_compare(final) and not sp.task.cancelled():
            task, msgs = sp.task, sp.msgs
            self.tl.count("spec_hit")
            self.tl.info["spec"] = "hit"
        else:
            self.tl.info["spec"] = "miss" if sp else "none"
            self._cancel_spec()
            msgs = self.ctx.brain.build_messages(self.history, final, self.caller, self.now,
                                                 ctx=self.ctx_ids, spoken=self.opener_text)
            task = asyncio.create_task(self._decide(msgs, s.reasoning_effort))
        self.history.append({"role": "customer", "text": final})
        decision = await task
        self.tl.mark("decision")
        self.tl.info["decision"] = decision.kind
        if decision.usage:
            self.tl.info["usage"] = decision.usage
        await self._decision_loop(decision, msgs)
        await self._drain()

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

    async def _decision_loop(self, decision: Decision, msgs: list) -> None:
        """함수 호출이 끝날 때까지 돈 뒤 답을 말한다. 조회 없이 '확인해 보니'라고 답하려 하면 한 번 되돌린다."""
        s = self.s
        msgs = list(msgs)
        rounds, guard_used, after_tool = 0, False, False
        while True:
            if (decision.kind == "answer" and not guard_used
                    and claims_lookup_without_evidence(decision.text, self.evid)):
                guard_used = True
                self.tl.event("guard", why="조회 없이 확인했다고 답하려 함")
                msgs += [decision.assistant_msg, {"role": "system", "content": GUARD_NOTE}]
                decision = await self._decide(msgs, s.reasoning_effort)
                continue
            if not (decision.kind == "tool" and rounds < s.max_tool_rounds):
                break
            if not self.tools_url:
                self.tl.event("tool_blocked", reason="tools_url 없음")
                await self._say_clip("fallback")
                return
            if self.mode == "measure" and self.filler_sent and rounds == 0:
                # 첫마디가 측정기에 다 도착한 뒤에 호출한다 (꼬리가 '조회 뒤 답'으로 잡히지 않게)
                gap = s.tool_gap_after_filler_ms / 1000 - (perf() - (self.tl.t0 + self.tl.marks.get("opener_sent", 0)))
                if gap > 0:
                    await asyncio.sleep(gap)
            self.tl.mark("tool_call")
            runs = await execute_calls(decision.tool_calls, self.ctx_ids, self._call_tool, s, self.now)
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
            rounds += 1
            msgs.append({"role": "system", "content": aftertool.note()})
            decision = await self._decide(msgs, s.reasoning_effort_round2)
            self.tl.mark("decision2", overwrite=True)
            if decision.kind == "answer":
                decision.text = aftertool.expand(decision.text)
        if decision.kind == "answer" and decision.text:
            await self._answer(decision.text, after_tool=after_tool)
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

    def _plan_units(self, body: str) -> list[tuple]:
        """답을 문장으로 나눠 은행 문장은 음성 그대로, 나머지는 TTS 단위로 묶는다. TTS는 여기서 모두 동시에 시작한다."""
        units: list[tuple] = []
        cur = ""

        def flush():
            nonlocal cur
            if cur.strip():
                units.append(("tts", cur.strip(), TTSJob(self.ctx.tts, for_tts(cur.strip(), self.s, self.now))))
            cur = ""

        for sent in facts.split_sentences(body):
            clip = self.ctx.clips.match(sent)
            if clip:
                flush()
                units.append(("clip", clip[0], clip[1]))
                self.tl.count("bank_sentence")
            elif cur and len(cur) + len(sent) > self.s.tts_unit_chars:
                flush()
                cur = sent
            else:
                cur = (cur + " " + sent).strip()
        flush()
        return units

    async def _answer(self, text: str, after_tool: bool = False) -> None:
        s = self.s
        prefix, body = self._prepare_answer(text)
        body = facts.clamp_answer(body, s.answer_max_sentences, s.answer_max_chars, is_bank=self.ctx.clips.match_text)
        body, fixes = facts.verify_answer(body, self.evid, self.now)
        if fixes:
            self.tl.event("answer_fixed", fixes=fixes)
        units = self._plan_units(body) if body else []
        self.tl.mark("tts_req")
        if prefix and not (prefix == "yes" and self.filler_sent):     # 이미 "네"를 했으면 다시 하지 않는다
            clip = self.ctx.clips.pick(prefix)
            if clip:
                await self._drain()
                await self._speak(clip[0], clip[1], mark_answer=after_tool)
        if self.mode == "live" and _HANDOFF_SAID.search(body):
            self._send_handoff("상담원 연결 안내")
        spoke = 0
        gap = silence(s.sentence_gap_ms)
        try:
            for i, (kind, utext, payload) in enumerate(units):
                await self._drain()
                if kind == "clip":
                    await self._speak(utext, payload + (gap if i < len(units) - 1 else b""), mark_answer=after_tool)
                    spoke += 1
                    continue
                job: TTSJob = payload
                seg = {"text": utext, "ms": 0.0}
                got = False
                try:
                    async for c in job.iterate():
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
        await self._send_pcm(pcm, mark_answer=mark_answer, seg=seg)

    async def _send_text(self, text: str) -> None:
        await self._send_json({"type": "agent.transcript", "text": text})

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
