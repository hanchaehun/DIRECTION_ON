"""상담원 서버 진입점.

    python -m agent.server --port 9000              # 설정은 config.json, 키는 .env
    AGENT_STT_BACKEND=soniox AGENT_LLM_BACKEND=openai AGENT_TTS_BACKEND=gemini python -m agent.server
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time

import httpx
from websockets.asyncio.server import serve

from . import knowledge, ratelimit
from .router import Router
from .brain import Brain
from .brain import FakeBackend as FakeLLM
from .brain import OpenAIBackend
from .clips import ClipBank, load_phrases
from .config import Settings, env
from .session import Session
from .stt import FakeLiveSTT, FakeSTT, SonioxSTT
from .tools_client import ToolsClient
from .tts import FakeBackend as FakeTTS
from .tts import GeminiBackend, TTSEngine

log = logging.getLogger("server")


class ServerContext:
    def __init__(self, s: Settings, http: httpx.AsyncClient):
        self.settings, self.http = s, http
        self.phrases = load_phrases()
        llm = OpenAIBackend(http, s) if s.llm_backend == "openai" else FakeLLM(s)
        self.brain = Brain(s, llm, self.phrases)
        tb = GeminiBackend(http, s) if s.tts_backend == "gemini" else FakeTTS(s)
        self.tts = TTSEngine(s, tb, "gemini" if s.tts_backend == "gemini" else "fake")
        self.clips = ClipBank(self.tts, self.phrases, s)
        self.tools = ToolsClient(http)
        self.router = Router(http, s)
        ratelimit.configure(s.gateway_rpm, s.gateway_jev_reserve)
        self.last_tools: list | None = knowledge.local_tools() or None   # 첫 세션 전 예열에도 같은 함수 정의를 쓴다

    def make_stt(self, session, start: dict):
        s = self.settings
        if s.stt_backend == "soniox":
            return SonioxSTT(s, session.on_stt_update, session.caller)
        if session.mode == "live":
            return FakeLiveSTT(s, session.on_stt_update, start.get("fake_script"))
        return FakeSTT(s, session.on_stt_update, session.case_id)


async def keepalive_loop(ctx: ServerContext) -> None:
    """프롬프트 캐시 유지. 최근에 실제 요청이 있었으면 건너뛴다 (호출마다 과금·분당 한도)."""
    while True:
        await asyncio.sleep(30)
        if time.perf_counter() - ctx.brain.last_call >= ctx.settings.cache_keepalive_s:
            await ctx.brain.keepalive(ctx.last_tools)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    s = Settings.load(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "websockets.server"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    keys = {k: bool(env(k)) for k in ("SONIOX_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")}
    log.info("backends stt=%s llm=%s tts=%s | keys %s", s.stt_backend, s.llm_backend, s.tts_backend, keys)
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=300)
    async with httpx.AsyncClient(limits=limits, timeout=10.0) as http:
        ctx = ServerContext(s, http)

        async def warm() -> None:
            """포트를 먼저 연 뒤 뒤에서 예열한다. 시작 직후 측정이 들어와도 연결은 받는다 (첫마디가 없으면 바로 답한다)."""
            t0 = time.perf_counter()
            stats = await ctx.clips.warm()
            log.info("첫마디·정형 문장 %s (%.1fs, TTS 요청 %d)", stats, time.perf_counter() - t0, ctx.tts.requests)
            if s.llm_backend == "openai":
                await ctx.brain.keepalive(ctx.last_tools)   # 연결과 프롬프트 캐시 예열
                await keepalive_loop(ctx)

        async def handler(ws):
            await Session(ws, ctx).run()

        async with serve(handler, a.host, a.port, max_size=None) as server:
            print(f"READY ws://{a.host}:{a.port}", flush=True)
            bg = asyncio.create_task(warm())
            try:
                await server.serve_forever()
            finally:
                bg.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
