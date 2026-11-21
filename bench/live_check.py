"""시연 모드 점검: 브라우저 대신 연습 음성을 이어 붙여 여러 턴 통화를 하고, 지정한 턴에서 상담원 말 도중에 끼어든다.

    python bench/live_check.py [--server ws://127.0.0.1:9000] [--tools http://127.0.0.1:8100]
                               [--turns P0026-01,P0019-02,P0137-01] [--barge 3] [--barge-after 1.2]

턴마다: 손님 말 끝 → 상담원 첫 소리(ms), 상담원이 한 말, 함수 호출, 끼어들기 처리(agent.cancel)를 출력한다.
마이크처럼 20ms 프레임을 실시간으로 계속 보낸다 (말 사이에는 무음).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT.parent / "directon-2026-voice-cs" / "data" / "audio" / "practice"
FRAME = 640


SLOW = 1.0     # --slow: 고객 음성을 이 배속으로 바꿔 보낸다 (느리게 말하는 사람 재현, 음높이 유지)


def load(cid: str) -> tuple[bytes, float]:
    meta = json.loads((AUDIO / "meta.json").read_text(encoding="utf-8"))
    with wave.open(str(AUDIO / f"{cid}.wav")) as w:
        pcm, end = w.readframes(w.getnframes()), meta[cid]["speech_end"]
    if abs(SLOW - 1.0) > 1e-3:
        import sys
        sys.path.insert(0, str(ROOT))
        from agent.audio import time_stretch
        pcm, end = time_stretch(pcm, SLOW), end / SLOW
    return pcm, end


def rms(b: bytes) -> float:
    a = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    return float(np.sqrt((a * a).mean())) if len(a) else 0.0


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://127.0.0.1:9000")
    ap.add_argument("--tools", default="http://127.0.0.1:8100")
    ap.add_argument("--turns", default="P0026-01,P0019-02,P0137-01")
    ap.add_argument("--barge", type=int, default=3, help="이 턴은 앞 턴의 답 도중에 시작한다 (0=없음)")
    ap.add_argument("--barge-after", type=float, default=1.2, help="앞 턴 답의 첫 소리 뒤 몇 초에 끼어들지")
    ap.add_argument("--caller-name", default="")
    ap.add_argument("--caller-phone", default="")
    ap.add_argument("--slow", type=float, default=1.0, help="고객 음성 배속 (0.75 = 느리게 말하는 사람)")
    a = ap.parse_args()
    global SLOW
    SLOW = a.slow
    turns = [load(c) for c in a.turns.split(",")]
    caller = {"name": a.caller_name or None, "phone": a.caller_phone or None, "email": None} if (a.caller_name or a.caller_phone) else None
    log: list[str] = []
    state = {"turn": 0, "speech_end": None, "first": {}, "texts": {}, "tools": {}, "cancel": [], "answer_start": {}}
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0

    async with websockets.connect(a.server, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.start", "session_id": f"LIVECHK-{int(time.time())}", "mode": "live",
                                  "now": time.strftime("%Y-%m-%d (%a) %H:%M"), "history": [], "caller": caller,
                                  "tools_url": a.tools, "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}},
                                 ensure_ascii=False))
        queue: list[tuple[int, bytes, float]] = []
        go = asyncio.Event()       # 이번 턴 상담원 첫 소리
        ended = asyncio.Event()    # 이번 턴 turn.end
        done = asyncio.Event()

        async def sender():
            k = 0
            silence = b"\0" * FRAME
            cur, pos, idx = None, 0, -1
            start = time.perf_counter()
            while not done.is_set():
                if cur is None and queue:
                    idx, cur, se = queue.pop(0)
                    pos = 0
                    state["turn"] = idx
                    state["speech_end"] = now() + se          # 이 턴의 말 끝 시각 (예정)
                    log.append(f"[{now():6.2f}] ▶ 손님 {idx}턴 말 시작")
                if cur is not None:
                    chunk = cur[pos:pos + FRAME]
                    pos += FRAME
                    if len(chunk) < FRAME:
                        chunk += b"\0" * (FRAME - len(chunk))
                        cur = None
                else:
                    chunk = silence
                await ws.send(chunk)
                k += 1
                wait = start + k * 0.02 - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)

        async def receiver():
            async for m in ws:
                t = now()
                if isinstance(m, bytes):
                    i = state["turn"]
                    if rms(m) >= 300 and i not in state["first"] and state["speech_end"] is not None and t >= state["speech_end"] - 0.05:
                        state["first"][i] = t - state["speech_end"]
                        log.append(f"[{t:6.2f}]   상담원 첫 소리 (말 끝 뒤 {state['first'][i]*1000:.0f}ms)")
                        go.set()
                    continue
                e = json.loads(m)
                typ = e.get("type")
                if typ == "agent.transcript":
                    state["texts"].setdefault(state["turn"], []).append(e.get("text", ""))
                    log.append(f"[{t:6.2f}]   상담원: {e.get('text')}")
                elif typ == "tool.call":
                    state["tools"].setdefault(state["turn"], []).append((e.get("name"), e.get("args")))
                    log.append(f"[{t:6.2f}]   함수 {e.get('name')} {json.dumps(e.get('args'), ensure_ascii=False)}")
                elif typ == "tool.result":
                    log.append(f"[{t:6.2f}]   결과 {str(e.get('body'))[:90]}")
                elif typ == "turn.commit":
                    log.append(f"[{t:6.2f}]   말 끝 확정 ({e.get('rule')}) '{e.get('text')}'")
                elif typ == "turn.judge":
                    log.append(f"[{t:6.2f}]   Jev 완결 {e.get('complete')} 의도 {e.get('intent')}({e.get('conf')}) {e.get('ms')}ms")
                elif typ == "agent.cancel":
                    state["cancel"].append(t)
                    log.append(f"[{t:6.2f}]   ■ 끼어들기 감지 → 상담원 말 중단")
                elif typ == "turn.end":
                    log.append(f"[{t:6.2f}]   turn.end")
                    ended.set()
                elif typ == "handoff":
                    log.append(f"[{t:6.2f}]   인계 카드: {e.get('reason')}")

        st = asyncio.create_task(sender())
        rt = asyncio.create_task(receiver())
        try:
            for i, (pcm, se) in enumerate(turns, 1):
                if i > 1:
                    if a.barge == i:
                        # 앞 턴 답의 첫 소리를 기다렸다가 몇 초 뒤 끼어든다
                        t_wait = time.perf_counter()
                        while (i - 1) not in state["first"] and time.perf_counter() - t_wait < 15:
                            await asyncio.sleep(0.05)
                        await asyncio.sleep(a.barge_after)
                    else:
                        await asyncio.wait_for(ended.wait(), 30)     # 상담원이 말을 다 보낸 뒤 (재생은 약 0.3초 더 남는다)
                        await asyncio.sleep(0.8)
                ended.clear()
                queue.append((i, pcm, se))
                await asyncio.sleep(se + 0.2)
            t_wait = time.perf_counter()
            while len(state["first"]) < len(turns) and time.perf_counter() - t_wait < 20:
                await asyncio.sleep(0.1)
            await asyncio.sleep(6)
        finally:
            done.set()
            st.cancel()
            rt.cancel()
    print("\n".join(log))
    print("\n요약")
    for i in range(1, len(turns) + 1):
        f = state["first"].get(i)
        print(f"  {i}턴: 첫 소리 {'-' if f is None else f'{f*1000:.0f}ms'} | 함수 {state['tools'].get(i, [])} | 말 {' '.join(state['texts'].get(i, []))[:120]}")
    print(f"  끼어들기 중단 이벤트: {len(state['cancel'])}회")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
