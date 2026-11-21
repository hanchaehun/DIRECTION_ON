"""함수 결과 뒤의 답은 첫 문장을 번호로 고르게 한다.

그 문장들은 미리 합성돼 있어서 TTS(약 3초)를 기다리지 않고 바로 나간다. '조회 뒤 답' 만점 기준은 3.5초다.
실측: 자유롭게 쓰게 하면 LLM이 은행 문장과 다른 문장으로 합쳐 써서 조회 뒤 답이 4.96초(70점 구간)였고,
번호로 고르게 하자 1.25~1.64초가 됐다. 번호는 LLM이 조회 결과와 문의 맥락을 보고 고르므로
첫 문장은 채움말이 아니라 실제 답의 시작이다.
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache

from . import knowledge

STATUS_OPENERS = [
    "확인해 보니 현재 배송 중이에요.",
    "확인해 보니 아직 입금이 확인되지 않아 출고 전이에요.",
    "확인해 보니 결제는 정상적으로 완료됐어요.",
    "확인해 보니 출고 준비 중이에요.",
    "확인해 보니 배송이 완료된 것으로 나옵니다.",
    "확인해 보니 주문이 여러 건 확인됩니다.",
    "확인해 보니 이미 취소된 주문이에요.",
    "말씀하신 번호로는 주문이 확인되지 않습니다.",
    "말씀하신 번호로는 반품 신청 내역이 확인되지 않습니다.",
    "반품 접수해 드렸습니다.",
    "말씀해 주신 성함과 연락처로 상담원 전화 요청을 접수했습니다.",
    "반품하시려는 사유를 말씀해 주시겠어요?",
    "전체 반품하실지, 일부만 반품하실지 말씀해 주시겠어요?",
    "번호를 잘못 들었을 수 있어서요, 한 자리씩 천천히 다시 말씀해 주시겠어요?",
]

_NUM = re.compile(r"\s*\[(\d{1,2})\]\s*(.*)$", re.S)
_CMP = re.compile(r"[\s.,!?~…·'\"\-]+")


def product_openers() -> list[str]:
    """상품 검색 답의 첫 문장: '문의하신 제품은 ○○이에요.' (제품별로 미리 합성)."""
    out = []
    for n in knowledge.product_names():
        name = re.sub(r"\s+", " ", n).strip()
        if name:
            out.append(f"문의하신 제품은 {name}입니다.")
    return list(dict.fromkeys(out))


@lru_cache(maxsize=1)
def openers() -> tuple[str, ...]:
    return tuple(STATUS_OPENERS + product_openers())


def note() -> str:
    ops = openers()
    rows = "\n".join(f"[{i}] {x}" for i, x in enumerate(ops, 1))
    return ("(시스템: 답 형식) 답의 맨 앞에, 아래 목록에서 지금 상황에 맞는 첫 문장의 번호를 대괄호로 쓴다. 맞는 것이 없으면 [0]을 쓴다. "
            "번호 뒤에는 이어서 말할 내용만 쓴다. 고른 문장을 다시 쓰지 않는다. 덧붙일 말이 없으면 번호만 쓴다.\n"
            "예: [1] 9월 28일에 주문하신 맑은 오메가3이고, 정확한 위치는 CJ대한통운에서 송장번호로 조회하실 수 있어요.\n"
            "예: [10] 영업일 기준 3~4일 안에 회수 기사님이 방문하니 제품을 다시 포장해서 전달해 주세요. "
            "환불은 회수한 상품이 물류센터에 들어온 뒤 진행되고, 입고까지 최대 2주 정도 걸릴 수 있습니다.\n"
            "상품 검색 결과로 답할 때는 그 제품의 '문의하신 제품은 …입니다.' 번호를 고르고, 뒤에 고객이 물은 내용을 쓴다.\n" + rows)


def _same(a: str, b: str) -> bool:
    x, y = _CMP.sub("", a), _CMP.sub("", b)
    return bool(x) and (x == y or difflib.SequenceMatcher(None, x, y).ratio() >= 0.85)


def expand(text: str) -> str:
    """'[3] 이어지는 말' → '고른 문장 이어지는 말'. 번호가 없으면 그대로, [0]이면 번호만 뗀다.
    LLM이 고른 문장을 다시 써 놓았으면 한 번만 남긴다 (실측: '확인해 보니 현재 배송 중이에요.'가 두 번 나갔다)."""
    m = _NUM.match(text or "")
    if not m:
        return text
    ops = openers()
    n, rest = int(m.group(1)), m.group(2).strip()
    if not (1 <= n <= len(ops)):
        return rest or text
    op = ops[n - 1]
    parts = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", rest)
    if parts and _same(parts[0].strip(), op):
        rest = "".join(parts[1:]).strip()
    return (op + " " + rest).strip()
