"""세션별 단계 시각 기록 (설계 8.1절). 한 줄에 한 세션(또는 시연 모드의 한 턴)씩 JSONL."""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from datetime import datetime

from .config import RUNTIME

_LOCK = threading.Lock()
_TEXT_KEYS = {"transcript", "text_at_commit", "text", "spec", "final", "args", "said", "body", "summary", "facts",
              "actions", "caller", "interrupted_after"}


def _redact(obj):
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k in _TEXT_KEYS and v not in (None, "") else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


class Timeline:
    def __init__(self, session_id: str, case_id: str | None, mode: str):
        self.t0 = time.perf_counter()
        self.info: dict = {"session_id": session_id, "case_id": case_id, "mode": mode,
                           "started": datetime.now().isoformat(timespec="seconds")}
        self.marks: dict[str, float] = {}
        self.events: list[dict] = []
        self.counts: Counter = Counter()

    def rel(self, t: float | None) -> float | None:
        return None if t is None else round(t - self.t0, 4)

    def mark(self, name: str, t: float | None = None, overwrite: bool = False) -> None:
        if overwrite or name not in self.marks:
            self.marks[name] = round((t if t is not None else time.perf_counter()) - self.t0, 4)

    def event(self, name: str, /, **kv) -> None:
        self.events.append({"t": round(time.perf_counter() - self.t0, 4), "e": name, **kv})

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] += n

    def derived(self) -> dict:
        m = self.marks
        se = m.get("speech_end")

        def after(k):
            return None if se is None or m.get(k) is None else round((m[k] - se) * 1000)

        d = {k: after(k) for k in ("commit", "decision", "first_sound", "tts_first", "turn_end", "end_token", "spec_first")}
        if m.get("tool_resp") is not None and m.get("answer_sound") is not None:
            d["after_tool"] = round((m["answer_sound"] - m["tool_resp"]) * 1000)
        return d

    def record(self) -> dict:
        return {**self.info, "marks": self.marks, "ms_after_speech_end": self.derived(),
                "counts": dict(self.counts), "events": self.events}

    def dump(self, extra: dict | None = None, redact: bool = False) -> dict:
        """redact=True: 고객이 한 말·번호·답 문장을 빼고 시각과 수치만 남긴다 (최종 측정용)."""
        rec = self.record()
        if extra:
            rec.update(extra)
        if redact:
            rec = _redact(rec)
        path = RUNTIME / "logs" / f"sessions-{datetime.now():%Y%m%d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
