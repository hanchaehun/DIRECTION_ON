"""종료 판단 (설계 5.3절). 순수 함수로 두어 단위 시험과 재생 실험에서 그대로 쓴다.

classify(): 받아쓰기가 '지금 답할 내용이 있는가'
    complete   답해도 됨
    incomplete 이어서 말할 가능성이 큼 (쉼표·연결 어미·자릿수 미달·머뭇거림)
    preamble   인사·부르는 말·용건 예고뿐
    undecided  양쪽 다 가능 (LLM 판정 요청)
    empty      받아쓰기 없음
decide_commit(): 커밋 규칙 r1~r5 + 소음 경로.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .textnorm import trailing_digits

_PREAMBLE = re.compile(
    r"(?<!\S)(안녕하세요|안녕하십니까|여보세요|수고하십니다|수고\s*많으십니다|수고\s*많으세요|수고하세요|"
    r"저기요|상담원님|상담사님|고객센터(?:죠|지요|맞죠|맞나요|인가요)?|"
    r"(?:제가\s*)?(?:문의|질문)\s*(?:좀|하나)?\s*(?:드리려고요|드릴게\s*있(?:어서요|는데요)|드릴려고요|하려고요|드려요|있어서요|있는데요)|"
    r"(?:좀\s*)?(?:여쭤|물어)\s*(?:볼게|보려고|볼\s*게)\s*(?:있(?:어서요|는데요)|요)?|"
    r"궁금한\s*(?:게|것이|점이)\s*있(?:어서요|는데요)|"
    r"네|예|넵)(?!\S)")
_FILLER = re.compile(r"(어+|음+|으+음*|아+|그+|저+|에+|흠+|그러니까|그게|뭐지|뭐더라|잠깐만요|잠시만요|잠깐만|잠깐)")
_SHORT_ANSWER = re.compile(
    r"(네|예|넵|응|아니요|아뇨|아니오|아니에요|맞아요|맞습니다|맞아|네맞아요|네맞습니다|괜찮아요|됐어요|"
    r"네부탁드려요|부탁드려요|부탁드립니다|네부탁드립니다|그렇게해주세요|네그렇게해주세요|좋아요|네좋아요|"
    r"네괜찮아요|네알겠습니다|알겠습니다|네감사합니다|감사합니다|없어요|없습니다|아니요없어요|네없어요)")
_QUESTION_END = re.compile(r"(\?|까요|나요|세요|실까요|시겠어요|시겠습니까|습니까|을까요|ㄹ까요|가요|는지요|죠)[.?!]?\s*$")
_NUMBER_REQ = re.compile(r"(번호|연락처|휴대폰|핸드폰|전화)")
_PHONE_REQ = re.compile(r"(전화|연락처|휴대폰|핸드폰)")
_INCOMPLETE_END = re.compile(
    r"(는데|은데|인데|던데|해서|라서|이라서|니까|으니까|하면|되면|이면|으면|려면|다면|지만|는지|거나|하고|이고|"
    r"이랑|하구|그리고|그래서|근데|그런데|그러면|그럼|그니까|왜냐면|혹시|만약)$")
_UNDECIDED_END = re.compile(r"(는데요|은데요|인데요|던데요|거든요|고요|구요|해서요|라서요|니까요|어서요|아서요|든요)$")
_COMPLETE_END = re.compile(r"(요|다|죠|까|니다|세요|네|군|나|지|래|어|아|해)$")


def _compact(t: str) -> str:
    return re.sub(r"[\s.,!?~…]+", "", t)


def is_question(agent_text: str) -> bool:
    return bool(agent_text) and bool(_QUESTION_END.search(agent_text.strip()))


def expected_digits(last_agent: str, s=None) -> list[int] | None:
    if not last_agent or not _NUMBER_REQ.search(last_agent):
        return None
    if _PHONE_REQ.search(last_agent):
        return list(getattr(s, "phone_digit_lengths", [10, 11]) or []) or None
    if "주문" in last_agent:
        return list(getattr(s, "order_id_digit_lengths", []) or []) or None
    return None


def classify(text: str, last_agent: str = "", s=None) -> str:
    t = (text or "").strip()
    if not t:
        return "empty"
    # 직전 상담원 질문에 대한 단답
    if last_agent and is_question(last_agent) and _SHORT_ANSWER.fullmatch(_compact(t)):
        return "complete"
    # 번호를 물었고 숫자로 답하는 중: 자릿수로 판정
    need = expected_digits(last_agent, s)
    digits = trailing_digits(t)
    if need and digits:
        return "complete" if len(digits) >= min(need) else "incomplete"
    # 인사·용건 예고·머뭇거림뿐인가
    core = re.sub(r"[.,!?~…]+", " ", t)
    had_preamble = bool(_PREAMBLE.search(core))
    rest = _PREAMBLE.sub(" ", core)
    words = [w for w in rest.split() if not _FILLER.fullmatch(w)]
    if not words:
        return "preamble" if had_preamble else "incomplete"
    # 끝 모양
    last = t.rstrip()
    if last.endswith((",", "…", "-", "·")):
        return "incomplete"
    q = last.endswith("?")
    body = re.sub(r"[.?!~]+$", "", last).rstrip()
    w = body.split()[-1] if body.split() else ""
    if _FILLER.fullmatch(w):
        return "incomplete"
    if _INCOMPLETE_END.search(body):
        return "incomplete"
    if _UNDECIDED_END.search(body):
        return "complete" if q else "undecided"
    if q:
        return "complete"
    if digits and not last.endswith((".", "!")):
        return "incomplete"            # 번호를 말하다 멈춘 것으로 본다
    if _COMPLETE_END.search(body):
        return "complete"
    return "undecided"


@dataclass
class TurnView:
    """커밋 판단에 필요한 순간 상태."""
    speech_started: bool
    has_text: bool
    silence_ms: float          # 마지막 말소리 이후 무음 (음성 스트림 기준)
    voiced_now: bool
    end_seen: bool             # Soniox <end> 이후 새 말이 없음
    verdict: str               # classify 결과
    judge: str | None          # undecided일 때 LLM 판정 (complete/incomplete)
    token_idle_ms: float       # 마지막 새 토큰 이후 경과 (벽시계)


def effective_verdict(v: TurnView, s) -> str:
    if v.verdict != "undecided":
        return v.verdict
    if v.judge in ("complete", "incomplete"):
        return v.judge
    return "incomplete" if v.silence_ms >= s.undecided_wait_ms else "undecided"


def decide_commit(v: TurnView, s) -> str | None:
    """커밋할 규칙 이름 또는 None."""
    if not v.speech_started and not v.has_text:
        return None
    ev = effective_verdict(v, s)
    if v.voiced_now:
        # 소음 경로: 무음이 잡히지 않지만 STT는 끝났다고 본다
        if v.end_seen and v.has_text:
            idle = s.noise_commit_token_idle_ms if ev == "complete" else s.noise_commit_token_idle_slow_ms
            if v.token_idle_ms >= idle:
                return "noise"
        return None
    if ev == "preamble":
        return "r4" if v.silence_ms >= min(s.r4_preamble_silence_ms, s.r5_max_silence_ms) else None
    if v.silence_ms >= s.r5_max_silence_ms:
        return "r5"
    if ev == "complete":
        if v.end_seen and v.silence_ms >= s.r1_end_silence_ms:
            return "r1"
        if v.silence_ms >= s.r2_complete_silence_ms:
            return "r2"
        return None
    if ev in ("incomplete", "empty") and v.end_seen and v.silence_ms >= s.r3_incomplete_end_silence_ms:
        return "r3"
    return None


def spec_threshold_ms(verdict: str, s) -> int:
    return s.spec_pause_slow_ms if verdict in ("incomplete", "preamble", "empty") else s.spec_pause_ms
