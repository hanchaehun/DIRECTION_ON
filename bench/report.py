"""세션 기록 요약과 겹침 막대 도표 (설계 8.1절, 발표 6번 슬라이드).

    python bench/report.py --last 20
    python bench/report.py --eval runtime/replay/<시각>/eval/results.jsonl --html out.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = [(1.5, 100), (2.0, 90), (2.5, 80), (3.0, 70), (3.5, 60), (4.0, 50), (5.0, 30), (6.0, 10)]


def pts(t):
    if t is None or t < 0:
        return 0
    return next((v for b, v in TABLE if t <= b), 0)


def pct(vals, p):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo))


def load_sessions(paths: list[Path], last: int | None, ids: set | None) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if ids is not None:
        rows = [r for r in rows if r.get("session_id") in ids]
    return rows[-last:] if last else rows


def speed_of(e: dict) -> float:
    if e.get("spoke_before_end") or "error" in e or e.get("turn_end") is None:
        return 0
    fs, at = e.get("first_sound"), e.get("after_tool_sound")
    return pts(fs) if at is None else (pts(fs) + pts(at)) / 2


def summarize(rows: list[dict], evals: dict) -> str:
    out = []
    hdr = f"{'case':10} {'rule':5} {'추측':5} {'결정':6} {'첫마디':13} {'커밋':>5} {'결정':>5} {'첫소리':>6} {'TTS첫':>6} {'조회뒤':>6}"
    if evals:
        hdr += f" | {'측정첫':>6} {'측정조회':>7} {'끼어듦':>5} {'속도':>5}"
    out.append(hdr)
    for r in rows:
        d = r.get("ms_after_speech_end", {})
        line = (f"{str(r.get('case_id') or r.get('session_id'))[:10]:10} {str(r.get('rule')):5} {str(r.get('spec')):5} "
                f"{str(r.get('decision')):6} {str(r.get('filler') or '-'):13} {str(d.get('commit')):>5} "
                f"{str(d.get('decision')):>5} {str(d.get('first_sound')):>6} {str(d.get('tts_first')):>6} {str(d.get('after_tool')):>6}")
        e = evals.get(r.get("session_id"))
        if evals:
            if e:
                f2 = lambda v: "-" if v is None else f"{v:.2f}"
                line += (f" | {f2(e.get('first_sound')):>6} {f2(e.get('after_tool_sound')):>7} "
                         f"{str(bool(e.get('spoke_before_end'))):>5} {speed_of(e):>5}")
            else:
                line += " | (측정기 결과 없음)"
        out.append(line)
    out.append("")
    for key in ("commit", "decision", "first_sound", "tts_first", "after_tool"):
        vals = [r.get("ms_after_speech_end", {}).get(key) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            out.append(f"{key:12} n={len(vals):3}  p50={pct(vals, 50)}ms  p90={pct(vals, 90)}ms  p95={pct(vals, 95)}ms")
    n = len(rows) or 1
    spec_hit = sum(1 for r in rows if r.get("spec") == "hit")
    fill = sum(1 for r in rows if r.get("filler") in ("neutral", "neutral_info"))
    out.append(f"추측 적중 {spec_hit}/{len(rows)}  중립 첫마디 {fill}/{len(rows)} ({fill * 100 // n}%)  "
               f"규칙 {dict(sorted(_count(r.get('rule') for r in rows).items()))}")
    if evals:
        ev = [evals[r["session_id"]] for r in rows if r.get("session_id") in evals]
        if ev:
            sp = [speed_of(e) for e in ev]
            out.append(f"속도 점수 평균 {sum(sp) / len(sp):.1f}  끼어듦 {sum(1 for e in ev if e.get('spoke_before_end'))}건  "
                       f"오류 {sum(1 for e in ev if 'error' in e)}건")
    return "\n".join(out)


def _count(it):
    c = {}
    for x in it:
        c[str(x)] = c.get(str(x), 0) + 1
    return c


def svg(rows: list[dict]) -> str:
    """말 끝을 0으로 둔 겹침 막대. 추측 실행이 무음 대기와 겹치는 모습이 보인다."""
    lo, hi = -1.5, 4.0
    for r in rows:
        d = r.get("ms_after_speech_end", {})
        hi = max(hi, (d.get("turn_end") or 0) / 1000 + 0.3)
    hi = min(hi, 12.0)
    W, left, rowh = 980, 120, 26
    H = 50 + rowh * len(rows) + 40
    x = lambda s: left + (s - lo) / (hi - lo) * (W - left - 20)
    colors = {"말": "#9aa4b1", "추측 LLM": "#3b82f6", "LLM": "#60a5fa", "함수": "#f59e0b", "TTS 대기": "#a855f7", "음성": "#10b981"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for s in range(int(lo), int(hi) + 1):
        parts.append(f'<line x1="{x(s):.1f}" y1="30" x2="{x(s):.1f}" y2="{H - 30}" stroke="#e5e7eb"/>'
                     f'<text x="{x(s):.1f}" y="{H - 14}" text-anchor="middle" fill="#6b7280">{s}s</text>')
    parts.append(f'<line x1="{x(0):.1f}" y1="26" x2="{x(0):.1f}" y2="{H - 30}" stroke="#111827" stroke-dasharray="4 3"/>'
                 f'<text x="{x(0):.1f}" y="20" text-anchor="middle" fill="#111827">말 끝</text>'
                 f'<line x1="{x(1.5):.1f}" y1="26" x2="{x(1.5):.1f}" y2="{H - 30}" stroke="#ef4444" stroke-dasharray="2 3"/>'
                 f'<text x="{x(1.5):.1f}" y="20" text-anchor="middle" fill="#ef4444">1.5s</text>')
    for i, r in enumerate(rows):
        m, y = r.get("marks", {}), 40 + i * rowh
        se = m.get("speech_end")
        if se is None:
            continue
        rel = lambda k: None if m.get(k) is None else m[k] - se
        bars = [("말", rel("speech_start"), 0.0),
                ("추측 LLM" if r.get("spec") == "hit" else "LLM",
                 rel("spec_first") if r.get("spec") == "hit" else rel("commit"), rel("decision")),
                ("함수", rel("tool_call"), rel("tool_resp")),
                ("TTS 대기", rel("tts_req"), rel("tts_first")),
                ("음성", rel("first_sound"), rel("turn_end"))]
        label = html.escape(str(r.get("case_id") or r.get("session_id"))[:14])
        parts.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" fill="#111827">{label}</text>')
        for name, a, b in bars:
            if a is None or b is None or b <= a:
                continue
            a, b = max(a, lo), min(b, hi)
            parts.append(f'<rect x="{x(a):.1f}" y="{y + 3}" width="{max(1.5, x(b) - x(a)):.1f}" height="{rowh - 8}" '
                         f'fill="{colors[name]}" opacity="0.85"><title>{name} {a:.2f}~{b:.2f}s</title></rect>')
        c = rel("commit")
        if c is not None:
            parts.append(f'<line x1="{x(c):.1f}" y1="{y}" x2="{x(c):.1f}" y2="{y + rowh - 2}" stroke="#111827" stroke-width="2"/>')
    lx = left
    for name, col in colors.items():
        parts.append(f'<rect x="{lx}" y="{H - 30 + 18}" width="12" height="10" fill="{col}"/>'
                     f'<text x="{lx + 16}" y="{H - 30 + 27}" fill="#374151">{name}</text>')
        lx += 100
    parts.append(f'<line x1="{lx}" y1="{H - 12}" x2="{lx}" y2="{H - 2}" stroke="#111827" stroke-width="2"/>'
                 f'<text x="{lx + 6}" y="{H - 3}" fill="#374151">커밋</text></svg>')
    return "".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--eval", default=None, help="측정기 results.jsonl (있으면 측정기 수치와 나란히)")
    ap.add_argument("--html", default=None)
    a = ap.parse_args(argv)
    paths = [Path(p) for p in a.logs] if a.logs else sorted((ROOT / "runtime" / "logs").glob("sessions-*.jsonl"))
    evals, ids = {}, None
    if a.eval:
        for l in Path(a.eval).read_text(encoding="utf-8").splitlines():
            if l.strip():
                e = json.loads(l)
                evals[e["session_id"]] = e
        ids = set(evals)
    rows = [r for r in load_sessions(paths, a.last if not a.eval else None, ids) if r.get("mode") == "measure"]
    if not rows:
        print("기록 없음")
        return 1
    text = summarize(rows, evals)
    print(text)
    if a.html:
        Path(a.html).write_text(
            "<!doctype html><meta charset='utf-8'><title>측정 보고</title>"
            "<body style='font-family:sans-serif;background:#fff;color:#111;margin:24px'>"
            f"<h2>단계별 시간 (말 끝 = 0)</h2>{svg(rows)}<pre>{html.escape(text)}</pre></body>", encoding="utf-8")
        print(f"\nHTML: {a.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
