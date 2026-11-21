"""STT만 평가: 연습 음성을 실시간 속도로 Soniox에 흘려 넣고 받아쓰기·숫자 정확도·<end> 시점을 잰다.

    python bench/stt_eval.py [--limit 80] [--concurrency 6] [--only P0019-02,...]

서버와 같은 설정(agent.stt.soniox_config)을 쓴다. 결과: runtime/stt_eval/<시각>/results.jsonl, 화면 요약.
연습용이다 (받아쓰기를 디스크에 남긴다). 시험 문항에는 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import RUNTIME, Settings, env  # noqa: E402
from agent.stt import soniox_config  # noqa: E402

OFFICIAL = ROOT.parent / "directon-2026-voice-cs"


def digit_runs(text: str, min_len: int = 4) -> list[str]:
    """글 속의 숫자 덩어리(구분자 무시)를 이어 붙여 4자리 이상만. '010 1234 5678' → '01012345678'."""
    t = re.sub(r"(?<=\d)[\s.\-,]+(?=\d)", "", text)
    return [m for m in re.findall(r"\d+", t) if len(m) >= min_len]


async def run_one(case: dict, audio_dir: Path, meta: dict, s: Settings, tail_s: float) -> dict:
    from websockets.asyncio.client import connect
    cid = case["case_id"]
    with wave.open(str(audio_dir / f"{cid}.wav")) as w:
        pcm = w.readframes(w.getnframes())
    speech_end = meta[cid]["speech_end"]
    pcm += b"\0" * int(16000 * 2 * tail_s)
    cfg = soniox_config(s, case.get("customer_profile"))
    out = {"case_id": cid, "ref": case["customer_utterance"], "final": "", "end_ms_after_speech": None, "err": None,
           "nonfinal_at_end": None}
    t0 = time.perf_counter()
    try:
        async with connect(env("SONIOX_STT_URL", s.soniox_url), additional_headers={"Authorization": f"Bearer {env('SONIOX_API_KEY')}"},
                           open_timeout=8, max_size=None) as ws:
            await ws.send(json.dumps(cfg, ensure_ascii=False))
            t_audio0 = time.perf_counter()

            async def send():
                k = 0
                for i in range(0, len(pcm), 640):
                    await ws.send(pcm[i:i + 640])
                    k += 1
                    wait = t_audio0 + k * 0.02 - time.perf_counter()
                    if wait > 0:
                        await asyncio.sleep(wait)
                await ws.send("")

            st = asyncio.create_task(send())
            async for m in ws:
                r = json.loads(m)
                now = time.perf_counter() - t_audio0
                if r.get("error_code"):
                    out["err"] = f"{r.get('error_code')} {r.get('error_message')}"
                    break
                for t in r.get("tokens", []):
                    if t.get("text") == "<end>":
                        if out["end_ms_after_speech"] is None and now >= speech_end - 0.05:
                            out["end_ms_after_speech"] = round((now - speech_end) * 1000)
                        elif now < speech_end - 0.05:
                            out.setdefault("premature_end_s", []).append(round(now, 2))
                    elif t.get("is_final") and t.get("text") != "<fin>":
                        out["final"] += t["text"]
                        if out.get("first_final_after_end_ms") is None and now >= speech_end:
                            pass
                if r.get("finished"):
                    break
            st.cancel()
    except Exception as e:
        out["err"] = repr(e)[:200]
    out["final"] = out["final"].strip()
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    ref_d, got_d = digit_runs(out["ref"]), digit_runs(out["final"])
    out["ref_digits"], out["got_digits"] = ref_d, got_d
    out["digits_ok"] = all(d in got_d or any(d in g for g in got_d) for d in ref_d) if ref_d else None
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OFFICIAL / "data" / "practice_cases.jsonl"))
    ap.add_argument("--audio", default=str(OFFICIAL / "data" / "audio" / "practice"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--tail", type=float, default=3.5, help="말 끝 뒤에 붙이는 무음(초)")
    a = ap.parse_args()
    s = Settings.load()
    audio_dir = Path(a.audio)
    meta = json.loads((audio_dir / "meta.json").read_text(encoding="utf-8"))
    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8") if l.strip()]
    cases = [c for c in cases if c["case_id"] in meta]
    if a.only:
        want = set(a.only.split(","))
        cases = [c for c in cases if c["case_id"] in want]
    if a.limit:
        cases = cases[: a.limit]
    sem = asyncio.Semaphore(a.concurrency)

    async def one(c):
        async with sem:
            r = await run_one(c, audio_dir, meta, s, a.tail)
            tag = "ERR" if r["err"] else ("숫자X" if r["digits_ok"] is False else "ok")
            print(f"{r['case_id']:9} {tag:4} end+{r['end_ms_after_speech']}ms | {r['final'][:70]}", flush=True)
            return r

    rows = await asyncio.gather(*(one(c) for c in cases))
    out_dir = RUNTIME / "stt_eval" / time.strftime("%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with_d = [r for r in rows if r["digits_ok"] is not None]
    ends = [r["end_ms_after_speech"] for r in rows if r["end_ms_after_speech"] is not None]
    summ = {"cases": len(rows), "errors": sum(1 for r in rows if r["err"]),
            "digit_cases": len(with_d), "digit_ok": sum(1 for r in with_d if r["digits_ok"]),
            "end_seen": len(ends), "end_ms_p50": statistics.median(ends) if ends else None,
            "end_ms_p90": sorted(ends)[int(len(ends) * 0.9)] if ends else None,
            "premature_end_cases": sum(1 for r in rows if r.get("premature_end_s"))}
    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(summ, ensure_ascii=False))
    for r in with_d:
        if not r["digits_ok"]:
            print(f"[숫자X] {r['case_id']}\n   원문 {r['ref'][:90]!r}\n   받아쓰기 {r['final'][:90]!r}\n   {r['ref_digits']} vs {r['got_digits']}")
    for r in rows:
        if r["err"]:
            print(f"[ERR] {r['case_id']} {r['err']}")
    print(f"\n결과: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
