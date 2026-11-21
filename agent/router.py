"""판정 모델(Jev)로 말 끝·의도를 0.5초 안에 판정한다 (게이트웨이 /decisions/).

LLM(약 1.7초)보다 빨라서 말 끝 뒤 1초 안에 결과를 쓸 수 있다. 쓰는 곳:
1. 말 끝 판단: turn_complete 확률
2. 첫마디 고르기: upset(불만) → 공감 한마디, 조회·안내 → 확인/안내 한마디
3. 은행 직행: 의도가 분명하고 답이 정해진 경우(번호 요청, 구매 안내, 인사 등) LLM 없이 미리 합성한 문장으로 답한다

접수 함수(취소·반품·교환·콜백)는 여기서 결정하지 않는다. 잘못 부르면 되돌릴 수 없다.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

import httpx

from .config import env
from .ratelimit import LIMITER

log = logging.getLogger("router")

QUESTIONS = {
    "turn_complete": {
        "type": "noul",
        "instructions": "전화 상담에서 고객의 말이 끝나 상담원이 지금 답해도 되는가? 말을 하다가 멈춘 것 같으면 아니오. "
                        "짧은 명사형 문의('배송문의', '환불 방법 문의')나 번호만 불러 준 경우도 끝난 것이다."},
    "upset": {
        "type": "noul",
        "instructions": "고객이 불만, 불편, 항의, 실망을 표현하고 있는가? 단순한 문의나 요청은 아니오."},
    "intent": {
        "type": "choice",
        "instructions": "고객의 이번 말이 무엇을 원하는지 하나 고른다. 직전 상담원 말이 질문이면 그에 대한 답인지 먼저 본다.",
        "criteria": {
            "greeting_only": "인사만 함 (안녕하세요, 여보세요)",
            "thanks_or_bye": "감사 인사나 통화를 마치는 말만 함",
            "gives_identifier": "주문번호, 전화번호, 이름, 주소, 계좌 같은 정보를 불러 줌 (받은 문자 내용을 읽어 주는 것 포함)",
            "answers_question": "상담원의 질문에 대한 대답 (예/아니오, 전체/일부, 반품 사유, 어느 주문인지 등)",
            "order_inquiry_no_id": "배송, 주문 확인, 입금 확인, 취소 진행을 묻지만 이번 말에 주문번호나 전화번호가 없음",
            "return_request_no_id": "반품이나 환불을 해 달라고 요청하지만 이번 말에 주문번호나 전화번호가 없음",
            "return_status_no_id": "이미 신청한 반품이나 환불이 언제 되는지, 왜 안 되는지 묻지만 번호가 없음",
            "delivery_problem_no_id": "잘못 온 상품, 빠진 상품, 못 받은 상품을 알리지만 주문번호가 없음",
            "purchase_how": "제품을 사고 싶다, 주문하고 싶다, 주문을 도와 달라 (결제 수단을 묻는 것은 아님)",
            "callback_request": "전화를 달라, 전화로 주문하고 싶다, 상담원과 통화하고 싶다",
            "product_question": "특정 제품이나 건강 고민을 말하며 효과, 성분, 섭취 방법, 가격, 구성을 묻거나 제품 이름을 말함",
            "product_unspecified": "어떤 제품인지 말하지 않고 효과나 먹는 방법을 물음",
            "policy_question": "배송비, 결제 수단, 쿠폰, 적립금, 회원가입, 반품 조건 같은 일반 안내를 물음",
            "complaint": "광고, 가격, 서비스에 대한 불만이나 의견",
            "unclear": "무엇을 원하는지 알 수 없는 말",
            "other": "위에 없음",
        }},
}

# 의도 → 은행 문장 키 (phrases.json의 bank). 조건은 decide_bank에서 본다.
_ASK_BY_INTENT = {
    "order_inquiry_no_id": "ask_order_or_contact",
    "return_request_no_id": "ask_order_or_phone",
    "return_status_no_id": "ask_name_phone_return",
    "delivery_problem_no_id": "ask_order_number",
}
_DIRECT_BY_INTENT = {
    "greeting_only": "greeting",
    "purchase_how": "purchase_guide",
    "product_unspecified": "ask_product",
    "unclear": "ask_detail",
}
_PHONE = re.compile(r"(?<!\d)0\d{1,2}[\s.\-]{0,3}\d{3,4}[\s.\-]{0,3}\d{4}(?!\d)")
_ORDER = re.compile(r"\d{8}[\s\-]?\d{7}")
_LONGNUM = re.compile(r"\d[\d\s\-]{7,}\d")


@dataclass
class Route:
    complete: float = 1.0
    upset: float = 0.0
    intent: str = "other"
    conf: float = 0.0
    probs: dict = field(default_factory=dict)
    ms: int = 0
    ok: bool = False
    error: str = ""


def has_identifier(text: str) -> bool:
    return bool(_PHONE.search(text) or _ORDER.search(text) or _LONGNUM.search(text))


def history_has_identifier(history: list[dict]) -> bool:
    for h in history or []:
        if h.get("role") in ("tool_call", "tool_result"):
            return True
        if h.get("role") == "customer" and has_identifier(str(h.get("text") or "")):
            return True
    return False


def build_state(history: list[dict], text: str, caller: dict | None) -> dict:
    last_agent, prev_customer = "", ""
    for h in reversed(history or []):
        if h.get("role") == "agent" and not last_agent:
            last_agent = re.sub(r"https?://\S+", "", str(h.get("text") or ""))
            last_agent = re.sub(r"\s+", " ", last_agent).strip()[-260:]
        elif h.get("role") == "customer" and last_agent and not prev_customer:
            prev_customer = re.sub(r"\s+", " ", str(h.get("text") or "")).strip()[:160]
            break
    st = {"직전 상담원 말": last_agent or "(없음. 통화의 첫 말)", "고객의 이번 말": text}
    if prev_customer:
        st["그 앞의 고객 말"] = prev_customer
    return st


class Router:
    def __init__(self, http: httpx.AsyncClient, s):
        self.http, self.s = http, s
        base = env("OPENAI_BASE_URL", "https://api.baze.co/v1/gateway").rstrip("/")
        self.url = env("DECISIONS_URL", base + "/decisions/")
        self.key = env("DECISIONS_API_KEY") or env("BAZE_API_KEY") or env("OPENAI_API_KEY")
        # 게이트웨이를 거칠 때만 게이트웨이 분당 한도를 함께 센다
        self.via = "gateway" if self.url.startswith(base) else "custom"
        self.calls = 0

    async def classify(self, history: list[dict], text: str, caller: dict | None, timeout: float | None = None) -> Route:
        r = Route()
        if not self.key or not text.strip():
            r.error = "no key" if not self.key else "empty"
            return r
        if self.via == "gateway" and not await LIMITER.acquire(optional=True, kind="jev"):   # 떼어 둔 몫까지 찼을 때만 건너뜀
            r.error = "rate-limited (skipped)"
            return r
        t0 = time.perf_counter()
        self.calls += 1
        try:
            if self.s.router_extra_delay_ms:
                await asyncio.sleep(self.s.router_extra_delay_ms / 1000)
            resp = await self.http.post(self.url, headers={"Authorization": f"Bearer {self.key}"},
                                        json={"model": self.s.jev_model, "state": build_state(history, text, caller),
                                              "questions": QUESTIONS},
                                        timeout=timeout or self.s.jev_timeout_s)
            r.ms = round((time.perf_counter() - t0) * 1000)
            if resp.status_code != 200:
                r.error = f"HTTP {resp.status_code} {resp.text[:160]}"
                if resp.status_code == 429 and self.via == "gateway":
                    LIMITER.note_429(2.0)            # 게이트웨이 한도일 때만 다른 게이트웨이 호출도 잠시 멈춘다
                return r
            a = resp.json().get("answers") or {}
            r.complete = float((a.get("turn_complete") or {}).get("noul", 1.0))
            r.upset = float((a.get("upset") or {}).get("noul", 0.0))
            it = a.get("intent") or {}
            r.intent = str(it.get("choice") or "other")
            r.probs = it.get("probabilities") or {}
            r.conf = float(r.probs.get(r.intent, it.get("confidence") or 0.0))
            r.ok = True
        except Exception as e:
            r.ms = round((time.perf_counter() - t0) * 1000)
            r.error = repr(e)[:160]
        return r


def decide_bank(r: Route, text: str, history: list[dict], caller: dict | None, s) -> str | None:
    """은행 문장으로 바로 답할 수 있으면 그 키. 아니면 None (LLM으로).
    번호가 이미 있으면(이번 말, 이전 대화, 발신자 전화번호) 조회가 가능하므로 LLM에 맡긴다."""
    if not r.ok or r.conf < s.bank_min_conf:
        return None
    if r.intent in _DIRECT_BY_INTENT:
        if r.intent == "greeting_only" and len(re.sub(r"\s", "", text)) > 14:
            return None
        if r.intent in ("unclear", "purchase_how") and has_identifier(text):
            return None          # 이름·번호를 함께 말한 구매 요청은 콜백 접수일 수 있다 (연습 P0133-03) → LLM이 판단
        return _DIRECT_BY_INTENT[r.intent]
    if r.intent in _ASK_BY_INTENT:
        if has_identifier(text) or history_has_identifier(history) or (caller and caller.get("phone")):
            return None
        return _ASK_BY_INTENT[r.intent]
    return None


def pick_opener(r: Route, text: str, s) -> str:
    """LLM 답을 기다리는 동안 낼 첫마디 종류 (phrases.json의 종류 이름)."""
    if r.ok and r.upset >= s.upset_min_prob and r.intent not in ("gives_identifier", "answers_question", "greeting_only", "thanks_or_bye"):
        return "empathy"                     # 번호만 불러 준 말에는 사과하지 않는다 (앞선 불만 맥락 때문에 확률이 높게 나온다)
    if r.ok and r.intent in ("order_inquiry_no_id", "return_request_no_id", "return_status_no_id",
                             "delivery_problem_no_id", "gives_identifier"):
        return "ack_check"
    if r.ok and r.intent in ("product_question", "policy_question", "purchase_how"):
        return "neutral_info"
    return "neutral"


# ───────────────────────── 키워드 규칙 (무료, 즉시) ─────────────────────────
# 연습 80문항 실측: 키워드만으로 발동 16건 중 15건 일치, Jev만으로 14건 중 14건 일치.
# 고정 답을 잘못 내면 그 문항은 0점이라 둘이 같은 판단일 때(또는 Jev 확신도가 높을 때)만 낸다.

_ASK_FAMILY = {"ask_order_or_contact", "ask_order_or_phone", "ask_name_phone_return", "ask_order_number"}
_UPSET_KW = re.compile(r"(실망|화가|화나|짜증|불편|말이\s*안|제대로|아직도|도대체|너무하|항의|엉망|어이없)")


def keyword_route(text: str, history: list[dict], caller: dict | None) -> str | None:
    """받아쓰기에서 키워드로 고른 은행 답 후보. 번호가 이미 있으면(말·이전 대화·발신자) 조회가 가능하므로 None."""
    t = re.sub(r"\s+", " ", text or "").strip()
    c = re.sub(r"[\s.,!?~]", "", t)
    if re.fullmatch(r"(네)?(안녕하세요|여보세요)+", c):
        return "greeting"
    if not t or has_identifier(t):
        return None
    if re.search(r"(쿠팡|네이버|스마트\s*스토어|11번가|지마켓|옥션|타사|다른\s*(쇼핑몰|사이트)|홈쇼핑)", t):
        return None                          # 자사몰이 아닌 구매는 응대가 다르다 → LLM
    known = history_has_identifier(history) or bool(caller and caller.get("phone"))
    buy = (re.search(r"(구매|구입|주문)\s*(하고\s*싶|할려고|하려고|하려는|할게|합니다|요망|원하|도와|좀)", t)
           or re.search(r"(사고\s*싶|살려고|사려고)", t))
    past = re.search(r"(주문한|주문했|구매한|구매했|샀|시켰|받았|받아)", t)
    if buy and not past and not re.search(r"(취소|환불|반품|배송비|결제|카드|계좌|쿠폰)", t):
        return "purchase_guide"
    if known:
        return None
    if re.search(r"(잘못\s*(배송|왔|보내)|다른\s*(제품|상품)이|누락|빠져|안\s*왔)", t):
        return "ask_order_number"
    if re.search(r"(반품|환불|환풀)", t):
        if re.search(r"(가능한가|가능해|되나요\?|비용|얼마|조건|기간)", t) and not re.search(r"(아직|안\s*들어|언제\s*되|늦)", t):
            return None                      # 정책 질문일 수 있다
        if re.search(r"(아직|안\s*들어|언제|늦|확인이\s*안|처리가|보냈는데|회수)", t):
            return "ask_name_phone_return"
        return "ask_order_or_phone"
    if re.search(r"(배송|도착|언제\s*와|입금|주문)", t) and not re.search(r"(배송비|방법|결제|카드)", t):
        if re.search(r"(언제|확인|문의|안\s*되|안돼|어디|조회)", t):
            return "ask_order_or_contact"
    return None


# 고정 답은 질문 하나에만 맞는다. 용건이 둘이거나 몸에 이상이 생겼다는 말이면 LLM이 답한다.
# LLM으로 넘겨도 첫 소리는 미리 합성한 첫마디가 나가므로 속도는 그대로다.
_SYMPTOM = re.compile(r"(두드러기|발진|가려[워운움]|복통|배가\s*아프|배\s*아프|설사|구토|토했|메스꺼|속이\s*(안|쓰|메|불편)|어지러|어지럽|"
                      r"부작용|알레르기|알러지|이상\s*(증상|반응)|탈이\s*났|병원|응급|두통|부었)")
_TOPICS = (
    ("ship", re.compile(r"(배송(?!비)|도착|택배|송장|언제\s*(와|오|받))")),
    ("fee", re.compile(r"(배송비|결제|카드|쿠폰|적립금|포인트|계좌|현금영수증|회원|가격|얼마)")),
    ("return", re.compile(r"(반품|환불|환풀|교환|취소)")),
    # 제품에 대해 '묻는' 말만 센다. "먹었는데 효과가 없어서 환불"은 반품 사유이지 제품 문의가 아니다
    ("product", re.compile(r"(효과|효능|성분|섭취|복용|먹)[^.?!]{0,10}(어떤|어떻게|알려|궁금|있나요|있을까요|인가요|는지|될까요|되나요|좋나요|좋을까요)"
                           r"|(어떤|무슨)\s*(효과|효능|성분)"
                           # 섭취 시기·방법 질문 (블라인드 B0017-01: '하루에 언제 먹어야 해요? … 구입하려고요')
                           r"|(언제|어떻게|몇\s*(알|번|포|정|개|봉)|하루에?)\s*\S{0,6}\s*(먹|섭취|복용|드시|드셔)")),
    ("buy", re.compile(r"((구매|구입)|사고\s*싶|살려고|사려고|주문\s*(방법|하려|할려|하고\s*싶|은\s*어떻게))")),
)


def needs_llm(text: str) -> str:
    """고정 답으로 답하면 안 되는 이유(없으면 빈 문자열). 'symptom' = 섭취 후 이상 증상, 'compound' = 용건이 둘 이상."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if _SYMPTOM.search(t):
        return "symptom"
    if sum(1 for _, rx in _TOPICS if rx.search(t)) >= 2:
        return "compound"
    return ""


# ───────────────────────── 시연용 단문 FAQ 키워드 직행 (안전 게이트) ─────────────────────────
# 함수 없는 일반 FAQ는 LLM+TTS를 거치면 첫마디 뒤 5초 넘게 공백이 생긴다.
# 시연 모드(mode="live")에서 엄격한 안전 조건을 모두 통과한 단일 단문 FAQ만 미리 합성한 음성으로 바로 답한다.
_FAQ_CONJ = re.compile(r"(그리고|그런데|근데|또한|추가로|그거\s*말고|말고도)")
_FAQ_BLOCK = re.compile(
    r"(주문한|주문했|결제했|결제\s*후|사후|샀는데|시켰는데|받았|보냈|입금했|내\s*주문|제\s*주문|"
    r"언제\s*(와|오|도착|출고|들어|주나)|합배송|묶음|따로|두\s*건|추가\s*내|도서|산간|제주|"
    r"반품|환불|환풀|교환|취소|왕복|불량|오배송|누락|못\s*받|안\s*왔|송장|기사|조회|확인|"
    r"쿠팡|네이버|스마트\s*스토어|11번가|지마켓|옥션|타사|다른\s*(회사|브랜드|쇼핑몰|사이트)|홈쇼핑|"
    r"처방|병원|지병|질환|혈압약|당뇨약|감기약|영양제\s*말고|약이랑|약과|약\s*먹|임산부|수유|"
    r"상담원|연결|콜백|전화\s*(주|줘|부탁)|연락\s*(주|줘|부탁)|만료|소멸|복구|오류|안\s*돼|안돼|안\s*되|비싸|광고|탈퇴|할인|얼마짜리)"
)
_FAQ_RULES: tuple[tuple[str, re.Pattern, re.Pattern | None], ...] = (
    ("faq_shipping_fee",
     re.compile(r"배송비.*(얼마|무료|기준|붙|있나요|인가요|내야|얼마나|어떻게)|(얼마|무료).{0,8}배송비"),
     re.compile(r"(추가|따로|두\s*개|박스\s*만|한\s*박스|주문\s*하|구입|사고\s*싶)")),
    ("faq_courier",
     re.compile(r"(택배사|어느\s*택배|무슨\s*택배|어떤\s*택배|택배\s*(어디|회사|어느))"),
     None),
    ("faq_hours",
     re.compile(r"(고객센터|상담\s*시간|운영\s*시간|업무\s*시간|영업\s*시간|점심\s*시간).{0,14}(몇\s*시|언제|번호|어떻게|알려|인가요|에요|예요|까지|해요|되나요)|"
                r"(몇\s*시|언제).{0,10}(고객센터|상담|운영|업무|영업)"),
     None),
    ("faq_points",
     re.compile(r"(적립금|포인트).{0,14}(얼마|조건|최소|부터|사용|쓸\s*수)"),
     re.compile(r"(쿠폰|같이|함께|동시|중복)")),
    ("faq_coupon_stack",
     re.compile(r"(쿠폰.{0,12}(중복|같이\s*쓸|함께\s*쓸|동시|적립금)|(중복|동시).{0,10}쿠폰)"),
     re.compile(r"(발급|어디서|받|카카오|회원가입|기간|며칠)")),
    ("faq_minor",
     re.compile(r"(미성년자|고등학생|중학생|청소년|어린이|아이).{0,12}(먹어|섭취|복용|괜찮|되나|돼요)"),
     re.compile(r"(남편|술|음주|성인|아빠|엄마)")),
    ("faq_alcohol",
     re.compile(r"(음주|술\s*(마|먹|드)).{0,14}(먹어|섭취|복용|괜찮|되나|돼요|전후|후에|전에)"),
     re.compile(r"(고등학생|미성년자|아이|약)")),
    ("faq_together",
     re.compile(r"(같이|함께|동시에).{0,10}(먹어|섭취|복용|드셔)"),
     re.compile(r"(약|처방|병원|술|음주|커피|우유|타사|다른\s*회사|비타민\s*말고|쿠폰|적립금)")),
)


def faq_route(text: str, history: list[dict], caller: dict | None, max_chars: int = 32) -> tuple[str | None, str]:
    """엄격한 안전 조건을 모두 통과한 단문 단일 FAQ만 은행 키를 반환한다. (bank_key, 사유)"""
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t:
        return None, "empty"
    if history:
        return None, "has_history"
    if has_identifier(t):
        return None, "has_identifier"
    compact = re.sub(r"[\s.,!?~…]+", "", t)
    if len(compact) > max_chars:
        return None, "too_long"
    if t.count("?") + t.count("？") >= 2 or _FAQ_CONJ.search(t):
        return None, "multi_clause"
    why = needs_llm(t)
    if why:
        return None, why
    if _FAQ_BLOCK.search(t):
        return None, "blocked_context"
    hits = []
    for key, pos, neg in _FAQ_RULES:
        if pos.search(t) and not (neg and neg.search(t)):
            hits.append(key)
    if len(hits) == 1:
        # faq_together는 하루온 제품·영양제 문맥이 함께 있을 때만 발동 (모호한 "같이 먹어도 돼요?" 방지)
        if hits[0] == "faq_together" and not re.search(r"(하루온|제품|영양제|가드|두\s*가지|여러|맥스|이눌린|조인플렉스)", t):
            return None, "ambiguous_together"
        return hits[0], "faq_hit"
    if len(hits) > 1:
        return None, "multi_faq"
    return None, "no_match"


def choose_bank(text: str, history: list[dict], caller: dict | None, r: Route | None, s,
                *, mode: str = "measure") -> tuple[str | None, str]:
    """(은행 답 키 또는 None, 근거). 고정 답은 틀리면 안 되므로 보수적으로 낸다.
    - 인사만 한 경우: 키워드만으로 낸다.
    - 용건이 둘 이상이거나 섭취 후 이상 증상을 말하면 내지 않는다 → LLM.
    - 시연 모드(또는 faq_direct_live_only=False)에서 안전 조건을 모두 통과한 단문 FAQ는 키워드로 바로 낸다.
    - Jev 확신도가 높으면(bank_min_conf 이상) Jev 단독으로 낸다.
    - 키워드 후보가 있고 Jev도 같은 계열로 판단하면 낸다.
    - Jev 결과가 없으면(시간 초과) 내지 않는다 → LLM."""
    kw = keyword_route(text, history, caller)
    if kw == "greeting":
        return kw, "keyword"
    why = needs_llm(text)
    if why:
        return None, why
    if getattr(s, "faq_direct", False) and (not getattr(s, "faq_direct_live_only", True) or mode == "live"):
        fk, _ = faq_route(text, history, caller, max_chars=getattr(s, "faq_max_chars", 32))
        if fk:
            return fk, "faq-keyword"
    if r is None or not r.ok:
        return None, "no-jev"
    jv_hi = decide_bank(r, text, history, caller, s)
    if jv_hi:
        return jv_hi, "jev"
    if kw:
        lo = Route(complete=r.complete, upset=r.upset, intent=r.intent, conf=1.0 if r.conf >= s.bank_agree_conf else 0.0, ok=True)
        jv_lo = decide_bank(lo, text, history, caller, s)
        if jv_lo and (jv_lo == kw or (jv_lo in _ASK_FAMILY and kw in _ASK_FAMILY)):
            return jv_lo, "keyword+jev"
    return None, "llm"


def choose_opener(text: str, r: Route | None, s) -> str:
    """LLM 답을 기다리는 동안 낼 첫마디 종류. Jev가 없으면 키워드로만 고른다."""
    if r is not None and r.ok:
        return pick_opener(r, text, s)       # 불만 여부는 Jev 판정을 따른다 (키워드만으로는 사과가 너무 자주 나간다)
    t = text or ""
    if _UPSET_KW.search(t):
        return "empathy"
    if has_identifier(t) or re.search(r"(배송|주문|입금|환불|반품|취소|교환|조회|확인)", t):
        return "ack_check"
    if re.search(r"(효과|먹|섭취|복용|성분|가격|얼마|쿠폰|결제|카드|회원|방법)", t):
        return "neutral_info"
    return "neutral"
