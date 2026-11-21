"""말하는 사람에 맞춰 '말 끝' 대기 시간을 조절한다 (시연 모드).

신호 두 가지
1. 발음 속도 = 받아쓴 글자 수 ÷ 실제 소리가 난 시간(쉼 제외). 느리게 말하는 사람은 쉼도 길다.
2. 이 사람이 보인 쉼: 이번 발화 중간의 쉼, 그리고 서버가 너무 일찍 받아쳐서 고객이 말을 이어 간 경우의 쉼.

원칙: 배율은 기준 속도 ÷ 이 사람 속도, 1.0~1.2배 (느린 사람만 길게, 줄이지는 않는다: 줄이면 끼어든다).
같은 통화의 이전 턴에서 잰 기록이 있을 때만 적용한다. 첫 턴은 기록만 한다.
→ 채점은 질문 하나뿐이라(이전 대화는 글로만 온다) 모드 설정과 상관없이 채점에서는 구조적으로 발동하지 않는다. 근거가 부족하면 손대지 않는다. 이미 길게 기다리는 규칙은 건드리지 않는다.
합성 음성은 속도와 쉼이 따로 놀아(연습 78개: 느린 1/3 쉼 418ms, 빠른 1/3 419ms) 측정 모드에서는 기본으로 끈다.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

ADJUSTABLE = ("punct", "silence", "complete")     # 빨리 확정하는 규칙만 조절한다
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

    def ready(self) -> bool:
        """이전 턴에서 잰 기록이 있는가 (이번 턴 값만으로는 적용하지 않는다)."""
        return self.rate is not None or bool(self.pauses)


def enabled(s, mode: str) -> bool:
    return s.pace_adapt == "all" or (s.pace_adapt == "live" and mode == "live")


def adjust(thr: float, name: str, *, rate: float | None, own_pause_ms: float, s, mode: str) -> tuple[float, dict]:
    """(조절한 대기 시간 ms, 기록용 숫자).
    - 속도 배율: 1.0~1.2배 (pace_min_factor~pace_max_factor)
    - 본인 쉼: 실제로 본 쉼 × 1.25까지, 늘어나는 폭은 모드별 상한(pace_max_extra_*)까지.
      속도 배율 상한(1.2배)에 묶으면 0.75초 쉬다 끼어들린 사람이 다음 턴에도 0.72초에 잘린다 → 따로 둔다."""
    if not enabled(s, mode) or name not in ADJUSTABLE:
        return thr, {}
    extra = s.pace_max_extra_live_ms if mode == "live" else s.pace_max_extra_measure_ms
    out = float(thr)
    info: dict = {}
    if rate is not None and rate > 0:
        f = min(max(s.pace_ref_cps / rate, s.pace_min_factor), s.pace_max_factor)
        out = thr * f
        info.update(cps=round(rate, 1), factor=round(f, 2))
    if own_pause_ms > 0:
        out = max(out, min(own_pause_ms * s.pace_pause_margin, thr + extra))   # 이 사람이 쉬던 만큼은 기다린다
        info["own_pause_ms"] = round(own_pause_ms)
    out = max(out, thr * s.pace_min_factor)
    if abs(out - thr) < 1:
        return thr, {}
    info.update(base_ms=round(thr), thr_ms=round(out))
    return out, info


def play_speed(profile: "Profile", s, mode: str) -> float:
    """상담원 음성 재생 속도 (1.0 = 원래 속도). 느리게 말하는 사람에게만 느리게, 시연 모드에서 이전 턴 기록이 있을 때만.
    0.05 단위로 맞춰 같은 녹음 문장의 변환 결과를 다시 쓴다."""
    if mode != "live" or not s.pace_play or profile.rate is None or profile.rate <= 0:
        return 1.0
    sp = min(1.0, max(s.pace_play_min_speed, profile.rate / s.pace_ref_cps))
    return round(round(sp / 0.05) * 0.05, 2)
