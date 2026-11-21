"""직접 답(함수 없는 답)의 첫 문장 후보. 미리 합성해 두어 LLM 판단이 끝나는 즉시 답이 시작된다 (TTS 약 3초 생략).

오류를 만들지 않는 장치
1. 문장은 모두 [브랜드 정보]·[업무별 정책 표] 원문의 사실을 그대로 옮긴 것이다 (직접 대조: tests/test_units.py DirectLibTest).
2. LLM은 자기 답의 첫 문장이 이 문장과 '같은 뜻'일 때만 글자 그대로 쓴다. 미리 합성한 음성은 LLM이 쓴 글과
   글자가 같을 때(또는 숫자·부정어까지 같은 거의 같은 글일 때)만 쓰인다 → 음성이 LLM 답과 다른 말을 하는 일은 없다.
3. 조건이 붙은 문장(자사몰 구매, 하루온 제품끼리 등)은 고객 말에 반대 조건이 있으면 답을 한 번 되돌려 다시 판단한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Opener:
    text: str
    when: str = ""                       # 프롬프트에 함께 보여 줄 사용 조건
    forbid: re.Pattern | None = None     # 고객 말(이번 턴·이전 대화)에 이것이 있으면 이 문장을 쓰면 안 된다
    note: str = ""                       # 되돌릴 때 LLM에 줄 설명


_OTHER_MALL = re.compile(r"(쿠팡|네이버|스마트\s*스토어|11번가|지마켓|옥션|타사|다른\s*(쇼핑몰|사이트|곳)|홈쇼핑|약국|매장|오프라인)")
_MEDS = re.compile(r"(약을|약이랑|약과|약\s*먹|약하고|복용\s*중인\s*약|처방|지병|질환|혈압약|당뇨약|병원|타사|다른\s*(회사|브랜드))")
_RETURN_CTX = re.compile(r"(반품|교환|환불|왕복|회수)")

OPENERS: tuple[Opener, ...] = (
    # 배송
    Opener("배송비는 5만 원 이상 구매하시면 무료예요.", "주문 배송비 문의 (반품·교환 배송비가 아닐 때)", _RETURN_CTX,
           "반품·교환 배송비 문의에 주문 배송비 문장을 쓰려 했다."),
    Opener("5만 원 미만이면 기본 배송비 3,000원이 붙어요.", "주문 배송비 문의 (반품·교환 배송비가 아닐 때)", _RETURN_CTX,
           "반품·교환 배송비 문의에 주문 배송비 문장을 쓰려 했다."),
    Opener("배송은 CJ대한통운 택배로 진행됩니다.", "택배사 문의"),
    Opener("평일 오후 3시 이전 주문은 당일 출고돼요.", "출고 시점 일반 문의"),
    Opener("금요일 오후 3시 이후 주문은 다음 주 월요일에 출고돼요.", "주말 앞 출고 문의"),
    Opener("받으시기까지는 영업일 기준 2~3일 정도 걸려요.", "출고 뒤 도착까지 걸리는 기간 일반 문의"),
    Opener("제주와 도서·산간 지역은 배송비가 추가될 수 있어요.", "제주·도서산간 배송 문의"),
    # 반품·환불·교환
    Opener("자사몰에서 구매하신 제품은 받으신 날부터 10일 안에 반품하실 수 있어요.", "자사몰 구매 건의 반품 가능 기간 문의",
           _OTHER_MALL, "다른 쇼핑몰 구매 건에 자사몰 반품 정책 문장을 쓰려 했다. 다른 쇼핑몰 구매 건은 구매처 정책을 따른다."),
    Opener("환불보장제에 따라 반품 비용 없이 진행됩니다.", "자사몰 구매 건, 받은 지 10일 이내 반품 비용 문의",
           _OTHER_MALL, "다른 쇼핑몰 구매 건에 환불보장제 문장을 쓰려 했다."),
    Opener("환불은 회수한 상품이 물류센터에 입고된 뒤 진행돼요.", "반품 뒤 환불 시점 문의"),
    Opener("다른 제품으로 교환은 어렵고, 반품 후 다시 구매하셔야 해요.", "다른 제품·옵션으로 교환 문의"),
    # 결제·쿠폰·적립금
    Opener("쿠폰은 중복으로 사용하실 수 없어요.", "쿠폰 중복 사용 문의"),
    Opener("쿠폰과 적립금은 함께 사용하실 수 없어요.", "쿠폰·적립금 동시 사용 문의"),
    Opener("적립금은 4만 원 이상 구매하실 때 사용하실 수 있어요.", "적립금 사용 조건 문의"),
    # 섭취
    Opener("하루온 제품은 모두 함께 드셔도 됩니다.", "하루온 제품끼리 함께 먹어도 되는지 (약·타사 제품이 아닐 때)",
           _MEDS, "약·지병·다른 회사 제품과 함께 먹는 문의에 '하루온 제품끼리' 문장을 쓰려 했다. 지병·복용약 공통 안내를 따른다."),
    Opener("하루온 제품은 성인 기준이라 미성년자 섭취는 권장드리지 않아요.", "미성년자·청소년 섭취 문의"),
    Opener("음주 직전이나 직후에는 섭취를 권장드리지 않아요.", "음주 전후 섭취 문의"),
    Opener("지병이 있거나 드시는 약이 있으면 전문의와 먼저 상담해 보시길 권해 드려요.", "지병·복용약이 있을 때 섭취 문의"),
    Opener("효과는 개인 체질에 따라 다를 수 있어요.", "효과가 없다는 말 (제품을 특정한 효능 문의가 아닐 때)"),
    Opener("섭취를 멈추시고, 증상이 계속되면 진료를 받아 보시길 권해 드려요.", "섭취 후 이상 증상"),
    # 기타
    Opener("고객센터 운영 시간은 평일 오전 10시부터 오후 5시까지예요.", "고객센터 운영 시간 문의"),
)

_BY_TEXT = {o.text: o for o in OPENERS}


def sentences() -> list[str]:
    return [o.text for o in OPENERS]


def prompt_section() -> str:
    rows = "\n".join(f"[D{i}] {o.text}  ({o.when})" for i, o in enumerate(OPENERS, 1))
    return ("# 답의 첫 문장 후보 (미리 녹음되어 바로 나간다)\n"
            "함수를 부르지 않고 답할 때, 답의 첫 문장으로 아래 문장 중 하나가 뜻과 괄호 안 조건에 정확히 맞으면 답의 맨 앞에 그 번호를 "
            "[D번호]로 쓰고, 이어서 나머지 설명만 쓴다 (고른 문장을 다시 쓰지 않는다). 맞는 것이 없으면 번호 없이 자유롭게 쓴다. "
            "뜻이 조금이라도 다르거나 조건이 맞지 않으면 고르지 않는다. 고객이 묻지 않은 내용을 이 목록 때문에 덧붙이지 않는다. "
            "함수 결과를 받은 뒤의 답에는 이 번호를 쓰지 않는다.\n"
            "예: 고객 \"홈페이지에서 샀는데 반품 언제까지 돼요?\" → [D8] 드셔 보신 뒤에도 효과가 없으면 반품하실 수 있어요.\n" + rows)


_MARK = re.compile(r"^\s*\[\s*D\s*(\d{1,2})\s*\]\s*(.*)$", re.S | re.I)
_CMP = re.compile(r"[\s.,!?~…·'\"\-]+")


def has_mark(text: str) -> bool:
    return bool(_MARK.match(text or ""))


def expand(text: str, allow: bool = True) -> str:
    """'[D8] 이어지는 말' → '고른 문장 이어지는 말'. 번호가 범위 밖이면 번호만 뗀다 (나머지는 LLM이 쓴 그대로).
    LLM이 고른 문장을 다시 써 놓았으면 한 번만 남긴다. 번호 표시가 답 중간에 남지 않게 모두 지운다."""
    m = _MARK.match(text or "")
    if not m or not allow:
        return re.sub(r"\[\s*D\s*\d{1,2}\s*\]\s*", "", text or "", flags=re.I) if text else text
    n, rest = int(m.group(1)), re.sub(r"\[\s*D\s*\d{1,2}\s*\]\s*", "", m.group(2), flags=re.I).strip()
    if not (1 <= n <= len(OPENERS)):
        return rest
    op = OPENERS[n - 1].text
    parts = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", rest)
    if parts and _CMP.sub("", parts[0]) == _CMP.sub("", op):
        rest = "".join(parts[1:]).strip()
    return (op + " " + rest).strip()


def violations(answer_sents: list[str], customer_texts: list[str]) -> list[Opener]:
    """답에 쓴 후보 문장 중 고객 말의 조건과 어긋나는 것 (되돌려 다시 판단할 근거)."""
    ctx = " ".join(t for t in customer_texts if t)
    out = []
    for s in answer_sents:
        o = _BY_TEXT.get(s)
        if o and o.forbid is not None and o.forbid.search(ctx):
            out.append(o)
    return out
