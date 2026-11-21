"""측정 모드 종단 시험: 가짜 STT·LLM·TTS로 상담원 서버를 띄우고 스타터킷 측정기를 그대로 돌린다.

    python tests/e2e_measure.py            # 기본: 공격적인 가짜 STT (쉼마다 종료 신호)
확인하는 것: 끼어들기 0건, 25초 안에 turn.end, 함수 호출(이름·인자·X-Session-Id), 첫 소리 시간.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.harness import KIT, ROOT, start_servers  # noqa: E402

TABLE = [(1.5, 100), (2.0, 90), (2.5, 80), (3.0, 70), (3.5, 60), (4.0, 50), (5.0, 30), (6.0, 10)]
EXPECT_CALLS = {
    "DEMO1": [], "ADV1": [],
    "DEMO2": [("get_order_info", {"order_id": "DEMO1001"})],
    "ADV2": [("get_order_info", {"order_id": "DEMO1001"})],
    "ADV3": [("get_order_info", {"order_id": "DEMO1001"})],
    "DEMO3": [("request_callback", {"phone_number": "01000001234"})],
}


def pts(t):
    if t is None or t < 0:
        return 0
    return next((v for b, v in TABLE if t <= b), 0)


def main() -> int:
    if "--keep-cache" not in sys.argv:          # 매번 같은 조건: 가짜 TTS 캐시를 비워 첫마디 경로까지 시험
        import shutil
        for d in (ROOT / "runtime" / "tts_cache").glob("fake_*"):
            shutil.rmtree(d, ignore_errors=True)
    tag = f"measure_{time.strftime('%H%M%S')}"
    td, mock, agent, log_dir = start_servers(19000, 18100, tag=tag)
    out = log_dir / "eval"
    try:
        cmd = [sys.executable, str(KIT / "eval" / "run_eval.py"), "--server", "ws://127.0.0.1:19000",
               "--cases", str(td / "cases.jsonl"), "--audio", str(td / "audio"),
               "--mock", "http://127.0.0.1:18100", "--out", str(out), "--concurrency", "4"]
        r = subprocess.run(cmd, cwd=KIT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}, timeout=180)
        print(r.stdout.strip())
        if r.returncode:
            print(r.stderr[-2000:])
            return 1
    finally:
        time.sleep(0.5)
        agent.stop()
        mock.stop()
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    fails = []
    print(f"\n{'case':6} {'첫소리':>7} {'조회뒤':>7} {'속도':>4} {'끼어듦':>5} {'turn.end':>8}  함수 호출 / 상담원 말")
    for row in sorted(rows, key=lambda x: x["case_id"]):
        cid = row["case_id"]
        fs, at = row.get("first_sound"), row.get("after_tool_sound")
        sp = pts(fs) if at is None else (pts(fs) + pts(at)) / 2
        if row.get("spoke_before_end") or "error" in row or row.get("turn_end") is None:
            sp = 0
        calls = [(c["name"], c["arguments"]) for c in row.get("tool_calls", [])]
        f2 = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"{cid:6} {f2(fs):>7} {f2(at):>7} {sp:>4} "
              f"{str(bool(row.get('spoke_before_end'))):>5} {str(round(row['turn_end'], 2)) if row.get('turn_end') else '-':>8}  "
              f"{calls} / {row.get('agent_text', '')[:60]}")
        if "error" in row:
            fails.append(f"{cid}: 오류 {row['error']}")
        if row.get("spoke_before_end"):
            fails.append(f"{cid}: 말 도중에 소리를 냈다")
        if row.get("turn_end") is None or row["turn_end"] > 25:
            fails.append(f"{cid}: turn.end 없음 또는 25초 초과")
        if cid in EXPECT_CALLS and calls != EXPECT_CALLS[cid]:
            fails.append(f"{cid}: 함수 호출 {calls} != 기대 {EXPECT_CALLS[cid]}")
        if any(not c.get("matched") for c in row.get("tool_calls", [])):
            fails.append(f"{cid}: 모의 서버에서 일치하는 기록 없음")
        if fs is None or fs > 3.0:
            fails.append(f"{cid}: 첫 소리 {fs}")
    print(f"\n측정기 결과: {out}\n서버 로그: {log_dir}")
    agent_log = [l for l in agent.lines if " session: " in l]
    print("\n".join(agent_log[-12:]))
    if fails:
        print("\n실패:\n- " + "\n- ".join(fails))
        return 1
    print(f"\n통과: {len(rows)}문항")
    return 0


if __name__ == "__main__":
    sys.exit(main())
