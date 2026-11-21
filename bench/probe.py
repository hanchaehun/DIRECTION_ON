"""09:00 실측 스크립트 (설계 7.4절 분기를 09:40까지 정하기 위한 숫자). 기본 3~5분, 상한 10분.

    python bench/probe.py --audio-dir <연습 음성 폴더> [--meta <meta.json>] [--cases <연습 문항 jsonl>]
    python bench/probe.py --fake                 # 키 없이 배관만 시험

항목
  env     키 유무, 패키지, 주소
  net     각 API까지 TCP+TLS 연결 시간, 기본 경로의 내 IP
  audio   연습 음성: 앞뒤 무음, 문장 중간 쉼 분포, 소음 바닥, 공식 말 끝과 RMS 300 기준의 차이
  soniox  리전 판별, 연습 음성 실시간 스트리밍: <end> 지연, 말 도중 <end> 수, 받아쓰기 (토큰 기록 저장)
  llm     GPT-5.4 mini 첫 토큰·완료 시간 (none/low, 캐시 전후), 함수 호출 결정 시간, 판정 호출, 동시 4건
  tts     Gemini TTS 글 길이별 첫 음성·조각 수, 동시 4건, 429 내용, 말투 지시문을 읽어 버리는지
  read    TTS가 번호·날짜·금액을 제대로 읽는지 (TTS → Soniox 되받아쓰기)
결과: runtime/probe/<시각>/report.md, probe.json, soniox_rec/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import socket
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.audio import RATE, StreamResampler, frame_rms, pcm_ms, read_wav, silence  # noqa: E402
from agent.brain import Brain, FakeBackend as FakeLLM, OpenAIBackend  # noqa: E402
from agent.clips import load_phrases  # noqa: E402
from agent.config import Settings, env  # noqa: E402
from agent.stt import FakeSTT, SonioxSTT  # noqa: E402
from agent.textnorm import for_tts  # noqa: E402
from agent.tts import FakeBackend as FakeTTS, GeminiBackend, TTSError  # noqa: E402

perf = time.perf_counter
REGIONS = {"us": "wss://stt-rt.soniox.com/transcribe-websocket", "eu": "wss://stt-rt.eu.soniox.com/transcribe-websocket",
           "jp": "wss://stt-rt.jp.soniox.com/transcribe-websocket", "in": "wss://stt-rt.in.soniox.com/transcribe-websocket"}
TTS_TEXTS = {
    "25자": "주문하신 상품은 현재 배송 중입니다.",
    "60자": "주문하신 상품은 오늘 오후에 출고되어 내일 도착할 예정입니다. 다른 문의 사항이 있으실까요?",
    "120자": ("말씀하신 주문은 이미 출고가 되어 지금은 취소가 어렵습니다. 상품을 받으신 뒤 칠 일 안에 반품을 신청하시면 "
             "환불해 드릴 수 있습니다. 반품 신청을 지금 도와드릴까요? 필요하시면 반품 절차를 다시 안내해 드리겠습니다."),
}
READ_TESTS = [  # (이름, 문장, 받아쓰기에 있어야 할 숫자열, 있으면 안 되는 말)
    ("전화(하이픈)", "연락처는 010-1234-5678로 확인됩니다.", "01012345678", ["하이픈", "다시", "빼기"]),
    ("전화(붙임)", "연락처는 01012345678로 확인됩니다.", "01012345678", []),
    ("주문번호(하루온)", "주문번호 20261102-0004321 건 확인했습니다.", "202611020004321", ["하이픈", "빼기"]),
    ("대표번호", "ARS 전화 주문은 1600-0000번입니다.", "16000000", ["하이픈", "빼기"]),
    ("고객센터", "고객센터는 02-000-0000번입니다.", "020000000", ["하이픈", "빼기"]),
    ("송장번호", "송장번호는 994268494700입니다.", "994268494700", []),
    ("날짜(ISO)", "2026-11-25에 도착할 예정입니다.", "1125", ["하이픈", "빼기"]),
    ("날짜(월일)", "6월 10일에 출고됩니다.", "610", ["육월", "십월"]),
    ("금액", "판매가는 19,800원입니다.", "19800", ["쉼표", "콤마"]),
    ("물결", "영업일 기준 3~4일 안에 회수 기사님이 방문합니다.", "34", ["물결", "틸드"]),
    ("제품명", "간 가드 맥스2와 조인플렉스 관절&연골 가드 5days를 안내해 드릴게요.", "25", ["앤드"]),
    ("퍼센트", "지금 10% 할인 중입니다.", "10", []),
    ("시각", "오후 2시 30분까지 주문하시면 당일 출고됩니다.", "230", []),
]
STYLE_TEST = "차분하고 친절한 상담원 말투로 읽어 주세요"


def _jd(o):
    """numpy 값도 JSON으로."""
    return o.item() if hasattr(o, "item") else str(o)


def pctl(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 1)


def digits(text: str) -> str:
    """받아쓰기에서 숫자열을 뽑는다. 한글 숫자 표기('삼만 구천팔백')도 값으로 바꾼다."""
    from agent.textnorm import parse_sino
    out = []
    for tok in re.findall(r"[0-9][0-9,\-]*|[영공일이삼사오육칠팔구십백천만]+", text):
        if tok[0].isdigit():
            out.append(re.sub(r"\D", "", tok))
        else:
            v = parse_sino(tok)
            if v is not None and len(tok) >= 2:
                out.append(str(v))
    return "".join(out)


class Probe:
    def __init__(self, a, s: Settings):
        self.a, self.s = a, s
        self.res: dict = {"started": datetime.now().isoformat(timespec="seconds"), "fake": a.fake}
        self.out = ROOT / "runtime" / "probe" / datetime.now().strftime("%m%d_%H%M%S")
        (self.out / "soniox_rec").mkdir(parents=True, exist_ok=True)
        self.tts_used = 0
        self.http: httpx.AsyncClient | None = None
        self.soniox_url = env("SONIOX_STT_URL", s.soniox_url)

    def log(self, *x):
        print(*x, flush=True)

    # ───────────── env / net ─────────────
    async def env(self):
        import importlib.util as u
        self.res["env"] = {
            "python": platform.python_version(), "os": platform.platform(),
            "keys": {k: bool(env(k)) for k in ("SONIOX_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")},
            "urls": {k: env(k) for k in ("OPENAI_BASE_URL", "GEMINI_BASE_URL", "SONIOX_STT_URL") if env(k)},
            "packages": {m: bool(u.find_spec(m)) for m in ("websockets", "httpx", "numpy", "scipy")},
            "models": {"stt": self.s.soniox_model, "llm": self.s.openai_model, "tts": self.s.gemini_tts_model},
        }
        self.log("[env]", self.res["env"]["keys"])

    async def net(self):
        hosts = {"openai": urlparse(env("OPENAI_BASE_URL", "https://api.openai.com/v1")).hostname,
                 "gemini": urlparse(env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")).hostname}
        hosts.update({f"soniox_{k}": urlparse(v).hostname for k, v in REGIONS.items()})

        async def tls(host):
            ts = []
            for _ in range(3):
                t0 = perf()
                try:
                    r, w = await asyncio.wait_for(asyncio.open_connection(host, 443, ssl=ssl.create_default_context(),
                                                                          server_hostname=host), 6)
                    ts.append(round((perf() - t0) * 1000))
                    w.close()
                except Exception as e:
                    return {"error": repr(e)[:120]}
            return {"ms": ts, "p50": pctl(ts, 50)}

        out = dict(zip(hosts, await asyncio.gather(*(tls(h) for h in hosts.values()))))
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.connect(("8.8.8.8", 80))
            out["default_route_ip"] = sk.getsockname()[0]
            sk.close()
        except OSError:
            out["default_route_ip"] = None
        out["host_ips"] = sorted(set(socket.gethostbyname_ex(socket.gethostname())[2]))
        self.res["net"] = out
        self.log("[net]", {k: (v.get("p50") if isinstance(v, dict) else v) for k, v in out.items()})

    # ───────────── audio ─────────────
    def wavs(self) -> list[Path]:
        d = Path(self.a.audio_dir) if self.a.audio_dir else None
        if not d or not d.exists():
            return []
        return sorted(d.glob("*.wav"))[: self.a.max_files if self.a.max_files else None]

    def meta(self) -> dict:
        p = Path(self.a.meta) if self.a.meta else (Path(self.a.audio_dir) / "meta.json" if self.a.audio_dir else None)
        if p and p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    async def audio(self):
        meta, rows = self.meta(), []
        for w in sorted(Path(self.a.audio_dir).glob("*.wav")) if self.a.audio_dir else []:
            pcm, rate = read_wav(w)
            r = frame_rms(pcm)
            v = np.flatnonzero(r >= 300)
            if not len(v):
                continue
            start, end = int(v[0]) * 20, (int(v[-1]) + 1) * 20
            gaps, i = [], v[0]
            vv = r >= 300
            while i < v[-1]:
                if not vv[i]:
                    j = i
                    while j < v[-1] and not vv[j]:
                        j += 1
                    if (j - i) * 20 >= 200:
                        gaps.append(int(j - i) * 20)
                    i = j
                else:
                    i += 1
            pre = r[: v[0]]
            row = {"file": w.name, "rate": rate, "dur_ms": round(pcm_ms(pcm, rate)), "speech_start": start, "speech_end": end,
                   "tail_ms": round(pcm_ms(pcm, rate)) - end, "max_gap": max(gaps, default=0), "gaps": gaps,
                   "noise_p90": round(float(np.percentile(pre, 90)), 1) if len(pre) else None}
            m = meta.get(w.stem)
            if isinstance(m, dict) and "speech_end" in m:
                row["meta_end_diff_ms"] = round(end - m["speech_end"] * 1000)
            rows.append(row)
        allgaps = [g for r in rows for g in r["gaps"]]
        summ = {"files": len(rows), "gaps_ge": {f"{t}ms": sum(1 for g in allgaps if g >= t) for t in (400, 600, 800, 1000, 1500, 2000)},
                "max_gap": max(allgaps, default=0), "files_with_gap_ge_1000": sum(1 for r in rows if r["max_gap"] >= 1000),
                "noise_p90_max": max((r["noise_p90"] or 0 for r in rows), default=0),
                "meta_end_diff_ms": [r.get("meta_end_diff_ms") for r in rows if "meta_end_diff_ms" in r][:40]}
        summ["suggest_silence_rms"] = max(300, round(summ["noise_p90_max"] * 2.5))
        self.res["audio"] = {"summary": summ, "files": rows}
        self.log("[audio]", summ)

    # ───────────── soniox ─────────────
    async def soniox_region(self):
        if self.a.fake or not env("SONIOX_API_KEY"):
            self.res["soniox_region"] = {"skipped": "키 없음" if not self.a.fake else "fake"}
            return
        from websockets.asyncio.client import connect

        async def try_one(name, url):
            t0 = perf()
            try:
                async with connect(url, additional_headers={"Authorization": f"Bearer {env('SONIOX_API_KEY')}"},
                                   open_timeout=6) as ws:
                    t_open = perf() - t0
                    await ws.send(json.dumps({"model": self.s.soniox_model, "audio_format": "pcm_s16le",
                                              "sample_rate": 16000, "num_channels": 1}))
                    await ws.send(silence(200))
                    await ws.send("")
                    async for m in ws:
                        d = json.loads(m)
                        if d.get("error_code"):
                            return {"ok": False, "error": f"{d.get('error_code')} {d.get('error_type')}"}
                        if d.get("finished"):
                            return {"ok": True, "open_ms": round(t_open * 1000), "total_ms": round((perf() - t0) * 1000)}
            except Exception as e:
                return {"ok": False, "error": repr(e)[:150]}
            return {"ok": False, "error": "no finished"}

        out = dict(zip(REGIONS, await asyncio.gather(*(try_one(k, v) for k, v in REGIONS.items()))))
        ok = [k for k, v in out.items() if v.get("ok")]
        best = min(ok, key=lambda k: out[k]["open_ms"]) if ok else None
        if best and not env("SONIOX_STT_URL"):
            self.soniox_url = REGIONS[best]
        self.res["soniox_region"] = {"results": out, "use": best, "url": self.soniox_url}
        self.log("[soniox region]", {k: v.get("ok") for k, v in out.items()}, "→", best)

    async def _stream(self, pcm: bytes, case_id: str | None = None, tail_ms: int = 3000) -> dict:
        """실시간 속도로 흘려 보내고 토큰 기록을 돌려준다."""
        if self.a.fake:
            stt = FakeSTT(self.s, lambda: None, case_id)
        else:
            stt = SonioxSTT(self.s, lambda: None, None)
            stt.url = self.soniox_url
        await stt.start()
        frames = [pcm[i:i + 640] for i in range(0, len(pcm), 640)] + [silence(20)] * (tail_ms // 20)
        t0 = perf()
        for k, fr in enumerate(frames):
            stt.push_audio(fr)
            d = t0 + (k + 1) * 0.02 - perf()
            if d > 0:
                await asyncio.sleep(d)
        await asyncio.sleep(0.3)
        await stt.close()
        return {"text": stt.final_text() or stt.text(), "recording": stt.recording, "error": stt.error}

    async def soniox(self):
        if not self.a.fake and not env("SONIOX_API_KEY"):
            self.res["soniox"] = {"skipped": "키 없음"}
            return
        meta, files = self.meta(), self.wavs()
        if self.a.fake:
            files = [f for f in files if f.stem in self._case_ids()]
        sem = asyncio.Semaphore(4)

        async def one(w: Path):
            async with sem:
                pcm, rate = read_wav(w)
                if rate != RATE:
                    pcm = StreamResampler(rate, RATE).process(pcm)
                r = frame_rms(pcm)
                v = np.flatnonzero(r >= 300)
                end = (meta.get(w.stem, {}).get("speech_end", 0) * 1000) or ((v[-1] + 1) * 20 if len(v) else 0)
                st = await self._stream(pcm, w.stem)
                ends = [x["audio_ms"] for x in st["recording"] if any(t.get("text") == "<end>" for t in x["res"].get("tokens", []))]
                premature = [round(e) for e in ends if e < end - 100]
                after = [e - end for e in ends if e >= end - 100]
                (self.out / "soniox_rec" / f"{w.stem}.json").write_text(json.dumps(
                    {"case_id": w.stem, "speech_end_ms": end, "recording": st["recording"]}, ensure_ascii=False), encoding="utf-8")
                return {"file": w.name, "text": st["text"], "end_delay_ms": round(after[0]) if after else None,
                        "premature_end_at_ms": premature, "error": st["error"]}

        rows = await asyncio.gather(*(one(w) for w in files))
        delays = [r["end_delay_ms"] for r in rows if r["end_delay_ms"] is not None]
        summ = {"files": len(rows), "end_delay_p50": pctl(delays, 50), "end_delay_p90": pctl(delays, 90),
                "no_end": sum(1 for r in rows if r["end_delay_ms"] is None),
                "files_with_premature_end": sum(1 for r in rows if r["premature_end_at_ms"]),
                "errors": [r["error"] for r in rows if r["error"]][:3],
                "config": {k: getattr(self.s, k) for k in ("endpoint_latency_adjustment_level", "endpoint_sensitivity", "max_endpoint_delay_ms")}}
        self.res["soniox"] = {"summary": summ, "files": rows}
        self.log("[soniox]", summ)

    def _case_ids(self) -> set:
        p = Path(self.s.fake_cases) if self.s.fake_cases else None
        if not p or not p.exists():
            return set()
        return {json.loads(l)["case_id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

    # ───────────── llm ─────────────
    async def llm(self):
        if not self.a.fake and not env("OPENAI_API_KEY"):
            self.res["llm"] = {"skipped": "키 없음"}
            return
        backend = FakeLLM(self.s) if self.a.fake else OpenAIBackend(self.http, self.s)
        brain = Brain(self.s, backend, load_phrases())
        tools_raw = json.loads(Path(self.a.tools).read_text(encoding="utf-8"))
        tools = [t if "function" in t else {"type": "function", "function": t} for t in tools_raw]
        now = datetime.now()

        async def call(text, effort, use_tools=True):
            msgs = brain.build_messages([], text, {"name": "테스트", "phone": "01000001234"}, now)
            t0 = perf()
            d = await brain.decide(msgs, tools if use_tools else None, effort)
            u = d.usage or {}
            cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
            return {"kind": d.kind, "ttft_ms": round((d.t_first - t0) * 1000) if d.t_first else None,
                    "total_ms": round(((d.t_done or perf()) - t0) * 1000), "prompt_tokens": u.get("prompt_tokens"),
                    "cached": cached, "out_tokens": u.get("completion_tokens"),
                    "reasoning": ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")),
                    "error": d.error[:160] if d.error else None,
                    "tools": [(t.name, t.args) for t in d.tool_calls], "text": d.text[:60]}

        out = {"answer_none": [], "answer_low": [], "tool_none": [], "judge_ms": [], "burst4": []}
        q_answer, q_tool = "배송비는 얼마예요?", "주문번호 20261103-0005678 택배 어디쯤 왔는지 알려 주세요."
        for _ in range(self.a.llm_reps):
            out["answer_none"].append(await call(q_answer, "none"))
        for _ in range(max(2, self.a.llm_reps // 2)):
            out["answer_low"].append(await call(q_answer, "low"))
        for _ in range(self.a.llm_reps):
            out["tool_none"].append(await call(q_tool, "none"))
        for _ in range(3):
            t0 = perf()
            j = await brain.judge("주문번호를 말씀해 주시겠어요?", "주문번호가 데모 일 공 공 일인데요.")
            out["judge_ms"].append({"ms": round((perf() - t0) * 1000), "res": j})
        out["burst4"] = await asyncio.gather(*(call(q_tool, "none") for _ in range(4)))
        summ = {}
        for k in ("answer_none", "answer_low", "tool_none", "burst4"):
            ok = [r for r in out[k] if not r["error"]]
            summ[k] = {"ttft_p50": pctl([r["ttft_ms"] for r in ok[1:] or ok], 50),
                       "total_p50": pctl([r["total_ms"] for r in ok[1:] or ok], 50),
                       "cold_total": out[k][0]["total_ms"] if out[k] else None,
                       "cached_tokens": [r["cached"] for r in out[k]], "errors": [r["error"] for r in out[k] if r["error"]][:2]}
        summ["tool_args"] = out["tool_none"][0]["tools"] if out["tool_none"] else None
        summ["judge_ms_p50"] = pctl([j["ms"] for j in out["judge_ms"]], 50)
        summ["judge_res"] = [j["res"] for j in out["judge_ms"]]
        summ["dropped_params"] = sorted(getattr(backend, "drop", []))
        self.res["llm"] = {"summary": summ, "raw": out}
        self.log("[llm]", {k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ('ttft_p50', 'total_p50')}) for k, v in summ.items()})

    # ───────────── tts ─────────────
    def _tts_backend(self):
        return FakeTTS(self.s) if self.a.fake else GeminiBackend(self.http, self.s)

    async def _synth(self, backend, text) -> dict:
        if self.tts_used >= self.a.tts_budget:
            return {"skipped": "예산 초과"}
        self.tts_used += 1
        t0 = perf()
        first, chunks, pcm, rate = None, 0, b"", 24000
        try:
            async for data, rate in backend.stream_raw(text):
                first = first or perf()
                chunks += 1
                pcm += data
        except (TTSError, httpx.HTTPError) as e:
            return {"error": repr(e)[:300], "detail": getattr(backend, "last_error", None)}
        audio16 = StreamResampler(rate, RATE).process(pcm)
        return {"ttfb_ms": round((first - t0) * 1000) if first else None, "total_ms": round((perf() - t0) * 1000),
                "chunks": chunks, "audio_ms": round(pcm_ms(pcm, rate)), "rate": rate, "pcm16": audio16}

    async def tts(self):
        if not self.a.fake and not env("GEMINI_API_KEY") and not env("GOOGLE_API_KEY"):
            self.res["tts"] = {"skipped": "키 없음"}
            return
        b = self._tts_backend()
        out = {"by_length": {}, "burst4": [], "style": None}
        for name, text in TTS_TEXTS.items():
            rs = [await self._synth(b, text) for _ in range(self.a.tts_reps)]
            out["by_length"][name] = [{k: v for k, v in r.items() if k != "pcm16"} for r in rs]
        burst = await asyncio.gather(*(self._synth(b, TTS_TEXTS["25자"]) for _ in range(4)))
        out["burst4"] = [{k: v for k, v in r.items() if k != "pcm16"} for r in burst]
        # 말투 지시문을 소리 내어 읽는가: 같은 문장을 지시문 있이/없이 합성해 길이와 받아쓰기를 비교
        # 3.8 interactions: 말투는 speech_metadata.style (tts_style). generate 경로: 본문 앞 지시문 (tts_style_prefix)
        old_style, old_prefix = self.s.tts_style, self.s.tts_style_prefix
        self.s.tts_style, self.s.tts_style_prefix = "", ""
        plain = await self._synth(b, TTS_TEXTS["25자"])
        if self.s.gemini_tts_api == "interactions":
            self.s.tts_style = old_style or "차분하고 친절한 상담원 말투"
        else:
            self.s.tts_style_prefix = old_prefix or STYLE_TEST
        styled = await self._synth(b, TTS_TEXTS["25자"])
        self.s.tts_style, self.s.tts_style_prefix = old_style, old_prefix
        style = {"plain_ms": plain.get("audio_ms"), "styled_ms": styled.get("audio_ms")}
        if styled.get("pcm16") and (self.a.fake or env("SONIOX_API_KEY")):
            st = await self._stream(styled["pcm16"], tail_ms=1500) if not self.a.fake else {"text": TTS_TEXTS["25자"]}
            style["styled_text"] = st["text"]
            style["reads_instruction"] = bool(re.search(r"말투|읽어|상담원", st["text"]))
        out["style"] = style
        ttfb = [r.get("ttfb_ms") for v in out["by_length"].values() for r in v]
        summ = {"ttfb_by_length_p50": {k: pctl([r.get("ttfb_ms") for r in v], 50) for k, v in out["by_length"].items()},
                "chunks_by_length": {k: [r.get("chunks") for r in v] for k, v in out["by_length"].items()},
                "ttfb_p50": pctl(ttfb, 50), "ttfb_p90": pctl(ttfb, 90),
                "burst4_errors": [r.get("error") for r in out["burst4"] if r.get("error")],
                "burst4_ttfb": [r.get("ttfb_ms") for r in out["burst4"]],
                "errors_detail": [r.get("detail") for r in out["burst4"] if r.get("detail")][:1],
                "style": style, "requests_used": self.tts_used}
        self.res["tts"] = {"summary": summ, "raw": out}
        self.log("[tts]", {k: summ[k] for k in ("ttfb_by_length_p50", "chunks_by_length", "burst4_errors", "style")})

    async def read(self):
        need_stt = not self.a.fake and not env("SONIOX_API_KEY")
        if need_stt or (not self.a.fake and not (env("GEMINI_API_KEY") or env("GOOGLE_API_KEY"))):
            self.res["read"] = {"skipped": "TTS 또는 Soniox 키 없음"}
            return
        b = self._tts_backend()
        sem = asyncio.Semaphore(4)

        async def check(name, text, want, bad, normalized=False):
            src = for_tts(text, self.s, datetime.now()) if normalized else text
            r = await self._synth(b, src)
            if "pcm16" not in r:
                return {"name": name, "src": src, "error": r.get("error") or r.get("skipped")}
            async with sem:
                st = await self._stream(r["pcm16"], tail_ms=1500) if not self.a.fake else {"text": text}
            heard = st["text"]
            ok = want in digits(heard) and not any(w in heard for w in bad)
            return {"name": name, "src": src, "heard": heard, "ok": ok, "digits": digits(heard)}

        raw = await asyncio.gather(*(check(*t) for t in READ_TESTS))
        fixes = []
        for t, r in zip(READ_TESTS, raw):
            if r.get("ok") is False and for_tts(t[1], self.s, datetime.now()) != t[1]:
                fixes.append(await check(*t, normalized=True))
        self.res["read"] = {"raw": raw, "normalized_retry": fixes}
        self.log("[read]", [(r["name"], r.get("ok")) for r in raw], "재시도:", [(r["name"], r.get("ok")) for r in fixes])

    # ───────────── 결정 ─────────────
    def decide(self) -> dict:
        d: dict = {}
        tts = self.res.get("tts", {}).get("summary") or {}
        ttfb = tts.get("ttfb_p50")
        burst_err = tts.get("burst4_errors") or []
        if ttfb is None:
            d["plan"] = "판단 불가 (TTS 측정 없음)"
        elif burst_err:
            d["plan"] = "C (동시 4건에서 오류) — 한도 확인 필요, 캐시 비율 70% 이상, 운영진에 한도 문의"
        elif ttfb < 1200:
            d["plan"] = "A 후보 (첫 음성 {:.0f}ms). 분당 한도가 60회 이상인지 운영진·콘솔로 확인 후 spec_tts=true".format(ttfb)
        elif ttfb < 1800:
            d["plan"] = "B (첫 음성 {:.0f}ms) — 추측 합성 끔, 정형 문장·대표 답변 캐시 확대".format(ttfb)
        else:
            d["plan"] = "C (첫 음성 {:.0f}ms) — 캐시 비율 70% 이상".format(ttfb)
        cfg = {}
        a = self.res.get("audio", {}).get("summary") or {}
        if a.get("suggest_silence_rms", 300) > 300:
            cfg["silence_rms"] = a["suggest_silence_rms"]
        if a.get("files_with_gap_ge_1000"):
            d["pauses"] = f"1초 넘는 문장 중간 쉼이 있는 파일 {a['files_with_gap_ge_1000']}개 — 종료 정책 재생 실험 필요"
        reg = self.res.get("soniox_region", {})
        if reg.get("use") and reg.get("use") != "us":
            d["soniox_url"] = reg.get("url")
        so = self.res.get("soniox", {}).get("summary") or {}
        if so.get("files_with_premature_end"):
            d["soniox_premature"] = f"말 도중 <end> {so['files_with_premature_end']}개 파일 — 완결성 규칙이 막아야 함"
        llm = self.res.get("llm", {}).get("summary") or {}
        if llm.get("answer_none", {}).get("total_p50") and llm.get("answer_low", {}).get("total_p50"):
            d["llm"] = (f"none {llm['answer_none']['total_p50']}ms / low {llm['answer_low']['total_p50']}ms "
                        f"(무음 대기 0.3~0.6초에 가려지는 쪽을 고른다)")
        if llm.get("dropped_params"):
            d["llm_dropped_params"] = llm["dropped_params"]
        if ttfb is not None:
            # 커밋 p50(약 0.6초) + TTS 첫 음성이 1.35초 안이면 중립 첫마디가 거의 안 나온다
            d["answer_filler_deadline_s"] = "1.35 유지" if ttfb + 600 <= 1350 else \
                f"1.35 유지 시 자유 답변 대부분에 중립 첫마디. 1.9로 올리면 속도 90점·첫마디 감소 (예상 도착 {(ttfb + 600) / 1000:.2f}초)"
        rd = self.res.get("read", {})
        if rd.get("raw"):
            bad = [r["name"] for r in rd["raw"] if r.get("ok") is False]
            fixed = [r["name"] for r in rd.get("normalized_retry", []) if r.get("ok")]
            d["tts_reading"] = {"틀림": bad, "변환하면 맞음": fixed}
        d["config_patch"] = cfg
        self.res["decision"] = d
        return d

    def report(self) -> Path:
        r = self.res
        lines = [f"# 09:00 실측 보고 ({r['started']}{' · fake' if r['fake'] else ''})", ""]
        lines += ["## 결정", "```json", json.dumps(r.get("decision", {}), ensure_ascii=False, indent=1, default=_jd), "```", ""]
        for k in ("env", "net", "soniox_region"):
            if k in r:
                lines += [f"## {k}", "```json", json.dumps(r[k], ensure_ascii=False, indent=1, default=_jd)[:3000], "```"]
        for k in ("audio", "soniox", "llm", "tts"):
            if k in r:
                lines += [f"## {k}", "```json", json.dumps(r[k].get("summary", r[k]), ensure_ascii=False, indent=1, default=_jd)[:4000], "```"]
        if "soniox" in r and r["soniox"].get("files"):
            lines += ["", "| 파일 | <end> 지연 | 말 도중 <end> | 받아쓰기 |", "|---|---|---|---|"]
            lines += [f"| {f['file']} | {f['end_delay_ms']} | {f['premature_end_at_ms']} | {f['text'][:60]} |" for f in r["soniox"]["files"]]
        if r.get("read", {}).get("raw"):
            lines += ["", "## read (TTS 읽기)", "| 항목 | 원문 | 되받아쓰기 | 통과 |", "|---|---|---|---|"]
            lines += [f"| {x['name']} | {x['src']} | {x.get('heard', x.get('error'))} | {x.get('ok')} |" for x in r["read"]["raw"]]
            lines += [f"| {x['name']} (변환) | {x['src']} | {x.get('heard', x.get('error'))} | {x.get('ok')} |"
                      for x in r["read"].get("normalized_retry", [])]
        p = self.out / "report.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        (self.out / "probe.json").write_text(json.dumps(self.res, ensure_ascii=False, indent=1, default=_jd), encoding="utf-8")
        return p

    async def run(self):
        only = set(self.a.only.split(",")) if self.a.only else None
        async with httpx.AsyncClient(timeout=20.0, limits=httpx.Limits(max_connections=20)) as http:
            self.http = http
            steps = [("env", self.env), ("net", self.net), ("audio", self.audio), ("soniox", self.soniox_region),
                     ("soniox", self.soniox), ("llm", self.llm), ("tts", self.tts), ("read", self.read)]
            # 서로 독립인 것은 함께 돌린다: (env, net, audio) → (soniox, llm, tts) → read
            groups = [[s for s in steps[:3]], [steps[3]], [steps[4], steps[5], steps[6]], [steps[7]]]
            for g in groups:
                tasks = [self._guard(name, fn) for name, fn in g if not only or name in only]
                await asyncio.gather(*tasks)
        self.decide()
        return self.report()

    async def _guard(self, name, fn):
        t0 = perf()
        try:
            await asyncio.wait_for(fn(), self.a.step_timeout)
        except Exception as e:
            self.res.setdefault("errors", {})[f"{name}.{fn.__name__}"] = repr(e)[:300]
            self.log(f"[{name}] 실패: {e!r}"[:300])
        self.res.setdefault("timing_s", {})[fn.__name__] = round(perf() - t0, 1)


def main():
    kit = ROOT.parent / "directon-2026-voice-cs"          # 공식 자료 (연습 음성 80개)
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default=str(kit / "data" / "audio" / "practice"))
    ap.add_argument("--meta", default=None)
    ap.add_argument("--tools", default=str(ROOT / "data" / "brand" / "tools.json"))
    ap.add_argument("--max-files", type=int, default=8)
    ap.add_argument("--llm-reps", type=int, default=4)
    ap.add_argument("--tts-reps", type=int, default=2)
    ap.add_argument("--tts-budget", type=int, default=24, help="TTS 요청 상한 (하루 한도가 낮을 수 있다)")
    ap.add_argument("--step-timeout", type=float, default=240)
    ap.add_argument("--only", default=None, help="예: net,audio,soniox")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    s = Settings.load(a.config)
    if a.fake and not s.fake_cases:
        s.fake_cases = str(kit / "data" / "practice_cases.jsonl")
        s.fake_audio_dir = a.audio_dir
    t0 = perf()
    p = asyncio.run(Probe(a, s).run())
    print(f"\n완료 {perf() - t0:.0f}초 → {p}")


if __name__ == "__main__":
    main()
