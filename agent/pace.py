"""말하는 사람에 맞춰 '말 끝' 대기 시간을 늘린다 (시연 모드).

신호 두 가지
1. 발음 속도 = 받아쓴 글자 수 ÷ 실제 소리가 난 시간(쉼 제외). 느리게 말하는 사람은 쉼도 길다.
2. 이 사람이 보인 쉼: 이번 발화 중간의 쉼, 그리고 서버가 너무 일찍 받아쳐서 고객이 말을 이어 간 경우의 쉼.

원칙: 늘리기만 한다 (줄이면 끼어든다). 근거가 부족하면 손대지 않는다. 이미 길게 기다리는 규칙은 건드리지 않는다.
합성 음성은 속도와 쉼이 따로 놀아(연습 78개: 느린 1/3 쉼 418ms, 빠른 1/3 419ms) 측정 모드에서는 기본으로 끈다.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

ADJUSTABLE = ("punct", "silence", "complete")     # 빨리 확정하는 규칙만 늘린다
_LETTERS = re.compile(r"[가-힣A-Za-z]")
_DIGITS = re.compile(r"\d")


def text_stats(text: str) -> tuple[int, int]:
    """(글자 수, 숫자 수). 글자는 한글 음절과 영문자만 센다. 숫자는 읽는 길이가 제각각이라 속도 계산에서 뺀다."""
    t = text or ""
    return len(_LETTERS.findall(t)), len(_DIGITS.findall(t))


def turn_rate(text: str, voiced_ms: float, s) -> float | None:
    """이번 발화의 발음 속도(글자/초). 근거가 부족하면 None: 너무 짧거나, 번호 위주의 말."""
    chars, digits = text_stats(text)
    if chars < s.pace_min_chars or voiced_ms < s.pace_min_voiced_ms:
        return None
    if digits > chars * 0.5:
        return None
    return chars / (voiced_ms / 1000.0)


@dataclass
class Profile:
    """통화 한 건 동안 쌓는 말하는 사람의 버릇 (턴이 바뀌어도 유지)."""
    rate: float | None = None                                     # 지난 턴들의 발음 속도 (지수 평균)
    pauses: deque = field(default_factory=lambda: deque(maxlen=5))  # 끝난 줄 알았는데 이어 말한 쉼 (ms)

    def note_rate(self, r: float | None) -> None:
        if r is None or not (1.0 <= r <= 12.0):
            return
        self.rate = r if self.rate is None else 0.6 * self.rate + 0.4 * r

    def note_pause(self, ms: float) -> None:
        if 200 <= ms <= 4000:
            self.pauses.append(float(ms))

    def pause_hint(self) -> float:
        return max(self.pauses) if self.pauses else 0.0


def enabled(s, mode: str) -> bool:
    return s.pace_adapt == "all" or (s.pace_adapt == "live" and mode == "live")


def adjust(thr: float, name: str, *, rate: float | None, own_pause_ms: float, s, mode: str) -> tuple[float, dict]:
    """(조절한 대기 시간 ms, 기록용 숫자). 늘리기만 하고, 모드별 상한을 넘지 않는다."""
    if not enabled(s, mode) or name not in ADJUSTABLE:
        return thr, {}
    cap = thr + (s.pace_max_extra_live_ms if mode == "live" else s.pace_max_extra_measure_ms)
    out = float(thr)
    info: dict = {}
    if rate is not None and rate > 0:
        f = min(max(s.pace_ref_cps / rate, 1.0), s.pace_max_factor)
        out = max(out, thr * f)
        info.update(cps=round(rate, 1), factor=round(f, 2))
    if own_pause_ms > 0:
        out = max(out, own_pause_ms * s.pace_pause_margin)
        info["own_pause_ms"] = round(own_pause_ms)
    out = min(out, cap)
    if out <= thr:
        return thr, {}
    info.update(base_ms=round(thr), thr_ms=round(out))
    return out, info
