"""게이트웨이 호출 속도 조절. BAZE는 키(회원)당 60초에 120건을 넘으면 429를 낸다 (실측: 평가 두 개를 동시에 돌리자 발생).

상담 서버의 모든 게이트웨이 HTTP 호출(LLM, Jev, TTS)이 이 한도를 함께 쓴다. 최근 60초의 호출 시각을 기억해
한도에 닿으면 자리가 날 때까지 기다린다. 덜 중요한 호출(Jev)은 기다리지 않고 건너뛴다.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque


class RateLimiter:
    def __init__(self, per_minute: int = 100, window_s: float = 60.0):
        self.limit, self.window = per_minute, window_s
        self.calls: deque[float] = deque()
        self.lock = asyncio.Lock()
        self.waited = 0          # 기다린 횟수
        self.skipped = 0         # 건너뛴 횟수
        self.blocked_until = 0.0  # 429를 받았을 때 서버가 알려 준 재개 시각

    def _trim(self, now: float) -> None:
        while self.calls and now - self.calls[0] >= self.window:
            self.calls.popleft()

    def used(self) -> int:
        self._trim(time.monotonic())
        return len(self.calls)

    def note_429(self, retry_after_s: float = 2.0) -> None:
        self.blocked_until = max(self.blocked_until, time.monotonic() + max(0.5, min(retry_after_s, 20.0)))

    async def acquire(self, optional: bool = False, max_wait_s: float = 8.0) -> bool:
        """자리를 얻으면 True. optional=True면 기다리지 않고 False를 돌려준다."""
        deadline = time.monotonic() + max_wait_s
        while True:
            async with self.lock:
                now = time.monotonic()
                self._trim(now)
                if now >= self.blocked_until and len(self.calls) < self.limit:
                    self.calls.append(now)
                    return True
                wait = max(self.blocked_until - now, (self.calls[0] + self.window - now) if len(self.calls) >= self.limit else 0.0)
            if optional:
                self.skipped += 1
                return False
            if time.monotonic() + wait > deadline:
                async with self.lock:
                    self.calls.append(time.monotonic())      # 더는 못 기다린다. 보내 보고 429면 재시도 로직에 맡긴다
                return True
            self.waited += 1
            await asyncio.sleep(min(max(wait, 0.05), 1.0))


LIMITER = RateLimiter()


def configure(per_minute: int) -> None:
    LIMITER.limit = max(10, int(per_minute))
