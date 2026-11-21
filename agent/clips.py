"""첫마디·정형 문장 음성 (설계 5.7절). 서버 시작 시 미리 합성하고 디스크에 저장한다."""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import re

from .audio import silence
from .config import PROMPTS
from .textnorm import norm_compare

log = logging.getLogger("clips")
_LOOKUP = re.compile(r"(get|check|lookup|find|search|status|track|inquir|list|조회)", re.I)
_ACCEPT = re.compile(r"(create|request|cancel|register|submit|apply|update|change|refund|return|exchange|book|접수|취소)", re.I)
_HANDOFF = re.compile(r"(transfer|handoff|hand_off|connect|human|agent|escalat|상담원)", re.I)


def load_phrases() -> dict:
    return json.loads((PROMPTS / "phrases.json").read_text(encoding="utf-8"))


def tool_kind(name: str, phrases: dict) -> str:
    kinds = phrases.get("tool_kinds", {})
    if name in kinds:
        return kinds[name]
    if _HANDOFF.search(name):
        return "handoff"
    if _ACCEPT.search(name):
        return "accept"
    return "lookup"


def is_handoff_tool(name: str) -> bool:
    return bool(_HANDOFF.search(name))


_NEG = re.compile(r"(안|못|않|없|불가|어렵|아니|금지|말아)")


def same_facts(a: str, b: str) -> bool:
    """거의 같은 두 문장이 사실까지 같은가: 숫자와 부정 표현이 모두 같아야 한다.
    (예: '5만 원 이상'과 '4만 원 이상', '드셔도 됩니다'와 '드시면 안 됩니다'는 글자는 비슷해도 다른 말이다)"""
    if re.sub(r"\D", "", a) != re.sub(r"\D", "", b):
        return False
    return sorted(_NEG.findall(a)) == sorted(_NEG.findall(b))


def all_fixed_sentences(phrases: dict, with_products: bool = True, with_direct: bool = True) -> list[str]:
    """합성 없이 나가는 문장들: fixed 목록 + 은행 답을 문장으로 나눈 것 + (선택) 제품별 첫 문장."""
    from .facts import split_sentences
    out = list(phrases.get("fixed", []))
    for k, v in (phrases.get("bank") or {}).items():
        if not k.startswith("_"):
            out += split_sentences(v)
    if with_direct:
        from .directlib import sentences as direct_sentences
        out += direct_sentences()         # 직접 답의 첫 문장 후보 (미리 합성. LLM이 글자 그대로 쓰면 바로 나간다)
    if with_products:
        from .aftertool import product_openers
        out += product_openers()
    return list(dict.fromkeys(s for s in out if s))


class ClipBank:
    def __init__(self, engine, phrases: dict, s):
        from .facts import split_sentences
        self.engine, self.phrases, self.s = engine, phrases, s
        self.pcm: dict[str, bytes] = {}
        # phrases.json의 문구 목록 중 fixed(정형 문장)를 뺀 모든 종류
        self.KINDS = tuple(k for k, v in phrases.items() if isinstance(v, list) and k != "fixed" and not k.startswith("_"))
        self.cycles = {k: itertools.cycle(phrases.get(k) or [""]) for k in self.KINDS}
        self.sentences = all_fixed_sentences(phrases)
        self.fixed = {norm_compare(t): t for t in self.sentences}
        self.bank_sents = {k: split_sentences(v) for k, v in (phrases.get("bank") or {}).items() if not k.startswith("_")}
        self.now = None

    def all_texts(self) -> list[str]:
        out = []
        for k in self.KINDS:
            out += self.phrases.get(k, [])
        out += self.sentences
        return list(dict.fromkeys(t for t in out if t))

    async def warm(self, concurrency: int = 3) -> dict:
        from .textnorm import for_tts
        sem = asyncio.Semaphore(concurrency)
        failed = []

        async def one(t):
            async with sem:
                try:
                    self.pcm[t] = await self.engine.synth_full(for_tts(t, self.s))   # 읽는 형태로 바꿔 합성 (3~4일, 1600-0000)
                except Exception as e:
                    failed.append((t, repr(e)[:120]))

        await asyncio.gather(*(one(t) for t in self.all_texts()))
        for t, e in failed:
            log.warning("clip 합성 실패: %s (%s)", t, e)
        return {"ok": len(self.pcm), "failed": len(failed)}

    def pick(self, kind: str) -> tuple[str, bytes] | None:
        """같은 종류 안에서 돌려 쓴다 (같은 채움말 반복 감점 방지)."""
        variants = self.phrases.get(kind) or []
        if kind not in self.cycles:
            return None
        for _ in range(len(variants)):
            t = next(self.cycles[kind])
            if t in self.pcm:
                return t, self.pcm[t]
        return None

    def filler(self, kind: str) -> tuple[str, bytes] | None:
        c = self.pick(kind)
        if c is None:
            return None
        return c[0], c[1] + silence(self.s.filler_tail_silence_ms)

    def match_text(self, text: str) -> str | None:
        """LLM이 쓴 문장과 같거나 거의 같은 은행 문장 (없으면 None)."""
        n = norm_compare(text)
        t = self.fixed.get(n)
        if t is None and len(n) >= 8:
            import difflib
            best, score = None, 0.0
            for k, v in self.fixed.items():
                if abs(len(k) - len(n)) > 6 or not same_facts(n, k):
                    continue
                r = difflib.SequenceMatcher(None, n, k).ratio()
                if r > score:
                    best, score = v, r
            if score >= self.s.bank_fuzzy_ratio:
                t = best
        return t

    def match(self, text: str) -> tuple[str, bytes] | None:
        """은행 문장이면 합성 없이 바로 쓴다 (은행 쪽 문구로 말한다)."""
        t = self.match_text(text)
        if t and t in self.pcm:
            return t, self.pcm[t]
        return None

    def bank(self, key: str) -> tuple[str, bytes] | None:
        """은행 답 전체 (문장 음성을 이어 붙인다). 하나라도 없으면 None."""
        sents = self.bank_sents.get(key)
        if not sents or any(x not in self.pcm for x in sents):
            return None
        gap = silence(180)
        return " ".join(sents), gap.join(self.pcm[x] for x in sents)

    def bank_text(self, key: str) -> str:
        return " ".join(self.bank_sents.get(key) or [])
