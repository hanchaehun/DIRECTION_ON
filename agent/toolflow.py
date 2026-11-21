"""함수 실행 흐름 (서버와 텍스트 평가가 함께 쓴다).

LLM이 정한 호출 → 번호를 맥락에 맞춤 → 접수 함수는 번호가 온전할 때만 실행 → 조회가 비면 맥락의 다른 번호로 재조회
→ 결과를 사실 목록으로 요약. 함수 채점 규칙: 조회는 더 불러도 감점이 없고, 접수는 잘못 부르면 0점이다.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime

from . import facts
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
    calls: list = field(default_factory=list)   # 실제로 나간 호출 [(name, args, status_code, ms)]


def block_reason(name: str, args: dict, ctx: dict) -> str:
    """접수 함수를 실행하면 안 되는 경우. 번호가 깨져 들린 채로 접수하면 틀린 호출이 된다."""
    if name == "request_callback":
        if not facts.valid_phone(args.get("phone_number")):
            return "전화번호가 정확하지 않다. 접수하지 않았다. 고객에게 전화번호를 다시 한 번 천천히 말씀해 달라고 한다."
        if not str(args.get("customer_name") or "").strip():
            return "성함이 없다. 접수하지 않았다. 고객에게 성함을 여쭌다."
    if name in ("cancel_order", "return_order", "exchange_order"):
        key = "order_id" if name == "cancel_order" else "order_number"
        v = args.get(key)
        known = {facts.digits(o) for o in ctx.get("orders", [])}
        if not (facts.valid_order_id(v) or facts.digits(v) in known):
            return "주문번호가 정확하지 않다. 접수하지 않았다. 주문 조회로 주문을 먼저 확인하거나 주문번호를 다시 여쭌다."
    if name == "get_order_info":
        ph, oid = args.get("buyer_cellphone"), args.get("order_id")
        if not ((ph and facts.valid_phone(ph)) or (oid and facts.valid_order_id(oid))):
            return ("번호가 정확히 들리지 않았다 (전화번호도 주문번호도 형식이 맞지 않음). 조회하지 않았다. "
                    "고객에게 주문번호나 전화번호를 다시 한 번 천천히 말씀해 달라고 한다.")
    if name == "get_return_info":
        if not facts.valid_phone(args.get("customer_phone_number")):
            return "전화번호가 정확히 들리지 않았다. 조회하지 않았다. 고객에게 전화번호를 다시 한 번 천천히 말씀해 달라고 한다."
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
        if tc.name in WRITES or tc.name in LOOKUPS:
            run.blocked = block_reason(tc.name, args, ctx)
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


_CLAIM = __import__("re").compile(r"(확인해\s*보니|조회해\s*보니|확인\s*결과|조회\s*결과)")
GUARD_NOTE = ("(시스템: 이번 통화에는 조회 결과가 없는데 확인했다고 말하려 했다. 그렇게 말할 수 없다. "
              "주문번호나 전화번호가 있으면 조회 함수를 부르고, 없으면 번호를 여쭌다.)")


def claims_lookup_without_evidence(answer: str, evidence: str) -> bool:
    """조회 결과가 없는데 '확인해 보니…'라고 답하려는 경우 (연습 P0175-02: 고객이 읽어 준 문자만 보고 배송 중이라고 답함)."""
    return bool(_CLAIM.search(answer or "")) and not (evidence or "").strip()
