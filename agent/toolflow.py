"""함수 실행 흐름 (서버와 텍스트 평가가 함께 쓴다).

LLM이 정한 호출 → 번호를 맥락에 맞춤 → 접수 함수는 번호가 온전할 때만 실행 → 조회가 비면 맥락의 다른 번호로 재조회
→ 결과를 사실 목록으로 요약. 함수 채점 규칙: 조회는 더 불러도 감점이 없고, 접수는 잘못 부르면 0점이다.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from . import facts, knowledge
from .textnorm import normalize_tool_args

LOOKUPS = {"get_order_info", "get_return_info"}
WRITES = {"cancel_order", "return_order", "return_order_item", "exchange_order", "request_callback"}


@dataclass
class ToolRun:
    tc: object                 # brain.ToolCall
    args: dict
    body: dict | None = None
    content: str = ""          # LLM에 넘길 요약
    notes: list = field(default_factory=list)
    blocked: str = ""          # 실행하지 않은 이유
    reask: str = ""            # 번호가 깨져 들려 막혔을 때 다시 물을 은행 문장 키 (reask_phone / reask_order / reask_any)
    calls: list = field(default_factory=list)   # 실제로 나간 호출 [(name, args, status_code, ms)]


_HEALTH = ("간", "눈", "관절", "연골", "기관지", "호흡", "목", "혈관", "혈당", "콜레스테롤", "배변", "변비", "장", "대장", "위",
           "소화", "전립선", "갱년기", "수면", "숙면", "멜라토닌", "프로폴리스", "면역", "비타민", "오메가", "이눌린", "피로", "숙취",
           "세트", "영양제", "가드", "맥스", "88")


# 검색어에서 빼고 보는 말: 어느 제품에나 붙는 일반어, 수량, 묻는 말투. 이것을 빼고 남는 말이 없으면 무엇을 찾는지 알 수 없다.
_GENERIC = re.compile(r"(하루온|\d+\s*(박스|개월|개|통|병|알|정|포|일|주|달|번|회)?|박스|효과|효능|섭취|복용|방법|가격|얼마|구성|성분|"
                      r"제품|상품|추천|문의|정보|관련|대해|기간|동안|"
                      r"먹(으면|어도|는데|는|고|어)?|드시(면|는)?|드세요|있(나요|어요|을까요|는지|습니까)|되(나요|는지)|"
                      r"좋(은|아요|을까요|나요)|어떤|어떻게|언제|하루에|하루|몇|이거|그거|저거|이것|그것|알려|주세요|궁금)")


def product_mentioned(query: str) -> bool:
    """검색어에 무엇을 찾는지 알 수 있는 말이 있는가. 없으면 모의 서버가 아무 제품이나 '가장 가까운 기록'으로 돌려준다
    (실측: 받아쓰기 '선박스 효과'로 검색해 기관지 제품으로 답함).
    제품명 조각·건강 고민이 있으면 통과. 그 밖에는 일반어를 뺀 뒤 두 글자 이상 남으면 통과시킨다 — 목록에 없는 성분명·제품 별칭
    (밀크씨슬, 혈관캐어)을 막지 않기 위해서다. 검색은 더 불러도 감점이 없고, 막으면 필요한 호출을 놓친다."""
    raw = str(query or "")
    q = re.sub(r"\s+", "", raw).replace("하루온", "")
    if not q:
        return False
    tokens = set()
    for n in knowledge.product_names():
        for w in re.split(r"[\s&]+", n.replace("하루온", "")):
            if len(w) >= 2:
                tokens.add(w)
    if any(t in q for t in tokens) or any(h in q for h in _HEALTH):
        return True
    for w in _GENERIC.sub(" ", raw).split():
        w = re.sub(r"(이요|요|은|는|이|가|을|를|도|에|의)$", "", re.sub(r"[^가-힣A-Za-z]", "", w))
        if len(w) >= 2:
            return True
    return False


_EXAMPLE_PHONES = facts.EXAMPLE_PHONES


def reask_kind(name: str, args: dict, blocked: str) -> str:
    """번호가 깨져 들려 막힌 경우 어떤 번호를 다시 물을지. 손님 음성의 숫자가 사람이 들어도 이상한 경우가 있어
    (연습 P0052-02) 무엇을 어떻게 말해 달라는지 구체적으로 묻는다."""
    if not blocked:
        return ""
    if "번호가 없다" in blocked:
        # 번호를 말한 적이 없는데 조회하려 했다 → '잘 안 들렸다'가 아니라 평범하게 번호를 여쭌다 (실측 P0026·P0072)
        return {"get_order_info": "ask_order_or_contact", "get_return_info": "ask_name_phone_return"}.get(name, "")
    if "정확" not in blocked:
        return ""
    if name in ("request_callback", "get_return_info"):
        return "reask_phone"
    if name in ("cancel_order", "return_order", "exchange_order"):
        return "reask_order"
    if name == "get_order_info":
        ph, oid = facts.digits(args.get("buyer_cellphone")), facts.digits(args.get("order_id"))
        guess = ph or oid
        if guess.startswith("20") and len(guess) >= 12:
            return "reask_order"                 # 날짜처럼 시작하는 긴 번호는 주문번호를 말하려던 것
        if ph or (guess and len(guess) <= 12):
            return "reask_phone"
        return "reask_any"
    return ""


def reask_for(runs: list) -> str:
    """모든 호출이 '번호가 깨져서' 막혔으면 다시 물을 은행 문장 키. LLM을 거치지 않고 바로 묻는다 (문구 고정, 약 1.7초 단축)."""
    if not runs or not all(r.blocked and r.reask for r in runs):
        return ""
    kinds = {r.reask for r in runs}
    return kinds.pop() if len(kinds) == 1 else "reask_any"


def block_reason(name: str, args: dict, ctx: dict) -> str:
    """함수를 실행하면 안 되는 경우. 번호가 깨져 들린 채로 접수·조회하면 틀린 호출이 된다."""
    # 형식 예시 번호를 그대로 넣은 호출 (실측 P0193-02: 번호가 덜 들리자 '010-1234-5678'로 조회함)
    ctx_phones = {facts.digits(p) for p in ctx.get("phones", [])}
    for k in ("buyer_cellphone", "customer_phone_number", "phone_number"):
        d = facts.digits(args.get(k))
        if d in _EXAMPLE_PHONES and d not in ctx_phones:
            return "전화번호가 정확히 들리지 않았다. 조회하지 않았다." if name in LOOKUPS else "전화번호가 정확하지 않다. 접수하지 않았다."
    if name == "search_products" and not product_mentioned(args.get("query")):
        return "검색어에 제품명이나 건강 고민이 없다. 검색하지 않았다. 고객에게 어떤 제품을 문의하시는지 여쭌다."
    if name == "request_callback":
        if not facts.valid_phone(args.get("phone_number")):
            return "전화번호가 정확하지 않다. 접수하지 않았다."
        if not str(args.get("customer_name") or "").strip():
            return "성함이 없다. 접수하지 않았다. 고객에게 성함을 여쭌다."
    if name in ("cancel_order", "return_order", "exchange_order"):
        key = "order_id" if name == "cancel_order" else "order_number"
        v = args.get(key)
        known = {facts.digits(o) for o in ctx.get("orders", [])}
        if not (facts.valid_order_id(v) or facts.digits(v) in known):
            return "주문번호가 정확하지 않다. 접수하지 않았다. 주문 조회로 주문을 먼저 확인하거나 주문번호를 다시 여쭌다."
    if name == "get_order_info" and not (facts.digits(args.get("buyer_cellphone")) or facts.digits(args.get("order_id"))):
        return "주문번호나 전화번호가 없다. 번호가 없다. 조회하지 않았다."
    if name == "get_return_info" and not facts.digits(args.get("customer_phone_number")):
        return "전화번호가 없다. 번호가 없다. 조회하지 않았다."
    if name == "get_order_info":
        ph, oid = args.get("buyer_cellphone"), args.get("order_id")
        if not ((ph and facts.valid_phone(ph)) or (oid and facts.valid_order_id(oid))):
            return "번호가 정확히 들리지 않았다 (전화번호도 주문번호도 형식이 맞지 않음). 조회하지 않았다."
    if name == "get_return_info":
        if not facts.valid_phone(args.get("customer_phone_number")):
            return "전화번호가 정확히 들리지 않았다. 조회하지 않았다."
    if name == "return_order_item":
        items = args.get("return_items")
        if not isinstance(items, list) or not items or not all(isinstance(x, dict) and x.get("order_item_code") for x in items):
            return "반품할 상품의 상세주문번호가 없다. 접수하지 않았다. 주문 조회 결과의 상세주문번호를 쓴다."
    return ""


async def execute_calls(calls: list, ctx: dict, call_fn, s, now: datetime) -> list[ToolRun]:
    """call_fn(name, args) → {"status_code": int, "body": dict, "ms": int} (비동기)."""
    runs: list[ToolRun] = []
    for tc in calls:
        args = normalize_tool_args(tc.args if isinstance(tc.args, dict) else {}, s)
        args, notes = facts.resolve_ids(tc.name, args, ctx)
        run = ToolRun(tc, args, notes=notes)
        if tc.name in WRITES or tc.name in LOOKUPS or tc.name == "search_products":
            run.blocked = block_reason(tc.name, args, ctx)
            run.reask = reask_kind(tc.name, args, run.blocked)
        runs.append(run)

    async def one(run: ToolRun) -> None:
        if run.blocked:
            run.body = {"status": "blocked", "message": run.blocked}
            return
        r = await call_fn(run.tc.name, run.args)
        run.calls.append((run.tc.name, run.args, r.get("status_code"), r.get("ms")))
        run.body = r.get("body") if isinstance(r.get("body"), dict) else {"status": "error", "message": str(r.get("body"))[:200]}
        if run.tc.name in LOOKUPS and facts.lookup_empty(run.tc.name, run.body):
            for alt in facts.retry_candidates(run.tc.name, run.args, ctx):
                r2 = await call_fn(run.tc.name, alt)
                run.calls.append((run.tc.name, alt, r2.get("status_code"), r2.get("ms")))
                b2 = r2.get("body")
                if isinstance(b2, dict) and not facts.lookup_empty(run.tc.name, b2):
                    run.notes.append(f"처음 번호로는 조회되지 않아 {json.dumps(alt, ensure_ascii=False)} 로 다시 조회함")
                    run.args, run.body = alt, b2
                    break

    await asyncio.gather(*(one(r) for r in runs))
    for run in runs:
        run.content = facts.render_tool_result(run.tc.name, run.body, now)
        if run.notes:
            run.content += "\n(시스템: " + "; ".join(run.notes) + ")"
    return runs


def evidence_text(runs: list[ToolRun]) -> str:
    return "\n".join(r.content for r in runs)


# 조회·접수 결과를 말하는 표현. '접수되어 있어요'(이전 통화의 접수를 안내)는 넣지 않는다.
# 실측: 쿠팡 반품 문의(P0043)에 조회 없이 '말씀하신 번호로는 반품 신청 내역이 확인되지 않습니다'라고 답했다.
_CLAIM = __import__("re").compile(r"(확인해\s*보니|조회해\s*보니|확인\s*결과|조회\s*결과|번호로는.{0,14}(확인|조회)되지|"
                                  r"접수해\s*드렸|접수했습니다|조회되는\s*(주문|내역)이\s*없)")
GUARD_NOTE = ("(시스템: 이번 통화에는 조회 결과가 없는데 확인했다고 말하려 했다. 그렇게 말할 수 없다. "
              "주문번호나 전화번호가 있으면 조회 함수를 부르고, 없으면 번호를 여쭌다.)")


def claims_lookup_without_evidence(answer: str, evidence: str) -> bool:
    """조회 결과가 없는데 '확인해 보니…'라고 답하려는 경우 (연습 P0175-02: 고객이 읽어 준 문자만 보고 배송 중이라고 답함)."""
    return bool(_CLAIM.search(answer or "")) and not (evidence or "").strip()
