"""종단 시험 공용: 시험 데이터 만들기, 서버 띄우기."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "DIRECT_ON 2026 Starter Kit_V2" / "DIRECT_ON 2026 Starter Kit"
TESTDATA = ROOT / "runtime" / "testdata"
sys.path.insert(0, str(ROOT))
from agent.audio import frame_rms, read_wav, write_wav  # noqa: E402


def speech_end_s(pcm: bytes, thr: float = 300.0) -> float:
    r = frame_rms(pcm)
    idx = np.flatnonzero(r >= thr)
    return round((idx[-1] + 1) * 0.02, 3) if len(idx) else 0.0


def gaps(pcm: bytes, min_ms: int = 400, thr: float = 300.0) -> list[tuple[int, int]]:
    """말소리 사이의 무음 구간 (샘플 위치)."""
    v = frame_rms(pcm) >= thr
    out, i, n = [], 0, len(v)
    first = int(np.argmax(v)) if v.any() else n
    last = n - int(np.argmax(v[::-1])) if v.any() else 0
    i = first
    while i < last:
        if not v[i]:
            j = i
            while j < last and not v[j]:
                j += 1
            if (j - i) * 20 >= min_ms:
                out.append((i * 320, j * 320))
            i = j
        else:
            i += 1
    return out


def stretch_gap(pcm: bytes, gap_index: int, total_ms: int) -> bytes:
    a = np.frombuffer(pcm, np.int16)
    s, e = gaps(pcm)[gap_index]
    extra = int(16000 * total_ms / 1000) - (e - s)
    mid = (s + e) // 2
    return np.concatenate([a[:mid], np.zeros(max(0, extra), np.int16), a[mid:]]).tobytes()


def make_testdata() -> Path:
    audio = TESTDATA / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    kit_cases = {json.loads(l)["case_id"]: json.loads(l)
                 for l in (KIT / "data" / "demo_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    pcm = {k: read_wav(KIT / "data" / "audio" / "demo" / f"{k}.wav")[0] for k in kit_cases}
    cases, meta = [], {}

    def add(cid, base, wav, **over):
        c = dict(kit_cases[base], case_id=cid, room_id=cid, **over)
        write_wav(audio / f"{cid}.wav", wav)
        cases.append(c)
        meta[cid] = {"speech_end": speech_end_s(wav)}

    for k in kit_cases:
        add(k, k, pcm[k])
    # 인사 뒤 1.4초 쉼 (인사만 듣고 끼어들면 안 된다)
    add("ADV1", "DEMO1", stretch_gap(pcm["DEMO1"], 0, 1400))
    # 문장 중간 1.3초 쉼 ("…일인데요," 뒤)
    add("ADV2", "DEMO2", stretch_gap(pcm["DEMO2"], 0, 1300))
    # 번호를 물은 뒤의 답 (history 사용)
    s, e = gaps(pcm["DEMO2"])[0]
    part = np.frombuffer(pcm["DEMO2"], np.int16)[: s + 1600]
    wav3 = np.concatenate([part, np.zeros(16000, np.int16)]).tobytes()
    add("ADV3", "DEMO2", wav3, customer_utterance="주문번호가 데모 일 공 공 일인데요.",
        history=[{"role": "agent", "text": "주문번호를 말씀해 주시겠어요?"}])
    (TESTDATA / "cases.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8")
    (audio / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return TESTDATA


class Proc:
    """서버 하나를 띄우고 READY 줄을 기다린다. 출력은 로그 파일로."""

    def __init__(self, name: str, args: list[str], env: dict | None = None, log_dir: Path | None = None):
        self.name = name
        log_dir = log_dir or (ROOT / "runtime" / "e2e")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{name}.log"
        e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        e.update(env or {})
        self.p = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=e, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        self.ready = threading.Event()
        self.lines: list[str] = []
        self.t = threading.Thread(target=self._pump, daemon=True)
        self.t.start()

    def _pump(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            for line in self.p.stdout:
                self.lines.append(line.rstrip())
                f.write(line)
                f.flush()
                if line.startswith("READY"):
                    self.ready.set()

    def wait_ready(self, timeout: float = 60) -> None:
        t0 = time.time()
        while not self.ready.wait(0.2):
            if self.p.poll() is not None or time.time() - t0 > timeout:
                raise RuntimeError(f"{self.name} 시작 실패:\n" + "\n".join(self.lines[-30:]))

    def stop(self) -> None:
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()


def start_servers(agent_port: int, mock_port: int, extra_env: dict | None = None, tag: str = "e2e"):
    td = make_testdata()
    log_dir = ROOT / "runtime" / "e2e" / tag
    mock = Proc(f"mock_{mock_port}", ["mock_tools/server.py", "--port", str(mock_port), "--host", "127.0.0.1"], log_dir=log_dir)
    env = {"AGENT_FAKE_CASES": str(td / "cases.jsonl"), "AGENT_FAKE_AUDIO_DIR": str(td / "audio")}
    env.update(extra_env or {})
    agent = Proc(f"agent_{agent_port}", ["-m", "agent.server", "--port", str(agent_port), "--host", "127.0.0.1"],
                 env=env, log_dir=log_dir)
    mock.wait_ready()
    agent.wait_ready()
    return td, mock, agent, log_dir
