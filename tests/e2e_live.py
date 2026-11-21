"""시연 모드 종단 시험: 여러 턴, 끼어들기, 중단 지점 이력.

    python tests/e2e_live.py
시나리오: 1턴 주문 조회 → 상담원이 말하는 도중 고객이 끼어듦 → 2턴 취소 문의.
확인하는 것: agent.cancel이 고객 말소리 시작 후 0.6초 안에 오는가, 이력에 [중단됨]이 남는가,
           2턴이 번호를 다시 묻지 않고 답하는가, 함수 호출에 X-Session-Id가 실리는가.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.harness import KIT, start_servers  # noqa: E402
from agent.audio import read_wav  # noqa: E402

AGENT, MOCK = 19001, 18101


async def client() -> dict:
    from websockets.asyncio.client import connect
    demo = KIT / "data" / "audio" / "demo"
    turn1 = read_wav(demo / "DEMO3.wav")[0]                             # 중간 쉼이 없는 한 덩어리 발화
    barge = read_wav(demo / "DEMO3.wav")[0][int(16000 * 0.9) * 2:]     # 앞 무음 0.9초 제거
    ev: list[tuple[float, dict]] = []
    audio_in: list[tuple[float, int]] = []
    state = {"barge_voice_at": None, "agent_audio_at": None}
    t0 = time.perf_counter()
    async with connect(f"ws://127.0.0.1:{AGENT}", max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "session.start", "session_id": "LIVE-TEST", "mode": "live", "now": "2026-11-21 11:00",
            "history": [], "caller": {"name": "데모 고객", "phone": "01000001234"},
            "tools_url": f"http://127.0.0.1:{MOCK}",
            "fake_script": ["주문번호 데모 일 공 공 일 택배 어디쯤 왔어요?", "잠깐만요 그거 취소할 수 있어요?"]},
            ensure_ascii=False))

        async def recv():
            async for m in ws:
                now = time.perf_counter() - t0
                if isinstance(m, bytes):
                    audio_in.append((now, len(m)))
                    if state["agent_audio_at"] is None:
                        state["agent_audio_at"] = now
                else:
                    ev.append((now, json.loads(m)))

        async def send():
            frame = 640
            queue = [turn1[i:i + frame] for i in range(0, len(turn1), frame)]
            barge_frames = [barge[i:i + frame] for i in range(0, len(barge), frame)]
            barged = False
            k = 0
            while time.perf_counter() - t0 < 25:
                if not queue and not barged and state["agent_audio_at"] is not None \
                        and time.perf_counter() - t0 > state["agent_audio_at"] + 0.5:
                    queue, barged = list(barge_frames), True
                    state["barge_voice_at"] = time.perf_counter() - t0 + 0.08   # 잘라 낸 뒤 말소리까지 약 80ms
                chunk = queue.pop(0) if queue else b"\0" * frame
                if len(chunk) < frame:
                    chunk += b"\0" * (frame - len(chunk))
                await ws.send(chunk)
                k += 1
                ends = [e for _, e in ev if e.get("type") == "turn.end"]
                if barged and len(ends) >= 1 and any(e.get("type") == "agent.cancel" for _, e in ev):
                    if len(ends) >= 1 and time.perf_counter() - t0 > (state["barge_voice_at"] or 0) + 6:
                        break
                await asyncio.sleep(max(0, t0 + k * 0.02 - time.perf_counter()))
            await ws.send(json.dumps({"type": "debug.history"}))
            await asyncio.sleep(0.5)

        r = asyncio.create_task(recv())
        await send()
        r.cancel()
    return {"events": ev, "audio": audio_in, **state}


def main() -> int:
    td, mock, agent, log_dir = start_servers(AGENT, MOCK, tag=f"live_{time.strftime('%H%M%S')}")
    try:
        res = asyncio.run(client())
    finally:
        agent.stop()
        mock.stop()
    ev = res["events"]
    for t, e in ev:
        if e["type"] in ("turn.commit", "tool.call", "tool.result", "agent.transcript", "agent.cancel", "turn.end", "handoff"):
            brief = {k: v for k, v in e.items() if k in ("text", "rule", "name", "args", "status")}
            print(f"{t:6.2f}s {e['type']:17} {json.dumps(brief, ensure_ascii=False)[:110]}")
    fails = []
    cancel = next((t for t, e in ev if e["type"] == "agent.cancel"), None)
    if cancel is None:
        fails.append("agent.cancel 없음")
    elif res["barge_voice_at"] is not None:
        lag = cancel - res["barge_voice_at"]
        print(f"\n끼어들기 반응: 고객 말소리 시작 → agent.cancel {lag * 1000:.0f}ms")
        if lag > 0.6:
            fails.append(f"끼어들기 반응이 느림 {lag:.2f}s")
    hist = [e for _, e in ev if e["type"] == "debug.history"]
    last_hist = hist[-1]["history"] if hist else []
    print("\n이력:")
    for h in last_hist:
        print(f"  {h['role']:8} {h['text'][:90]}")
    if not any(h["role"] == "agent" and "[중단됨]" in h["text"] for h in last_hist):
        fails.append("이력에 [중단됨] 없음")
    commits = [e for _, e in ev if e["type"] == "turn.commit"]
    if len(commits) < 2:
        fails.append(f"두 번째 턴이 커밋되지 않음 ({len(commits)})")
    calls = [e for _, e in ev if e["type"] == "tool.call"]
    if not calls or calls[0]["args"] != {"order_id": "DEMO1001"}:
        fails.append(f"1턴 함수 호출이 다름: {calls[:1]}")
    mock_calls = [l for l in mock.lines if l.startswith("[call]")]
    if not any("sid=LIVE-TEST" in l for l in mock_calls):
        fails.append("모의 서버에 X-Session-Id가 기록되지 않음")
    if fails:
        print("\n실패:\n- " + "\n- ".join(fails))
        print("\n".join(agent.lines[-20:]))
        return 1
    print("\n통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
