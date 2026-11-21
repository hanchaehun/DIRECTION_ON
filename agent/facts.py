"""정확도 장치 (지정 모델을 추가로 쓰지 않는 순수 코드).

1. 번호 맥락 우선 (resolve_ids)
   손님 음성의 숫자는 틀리게 들릴 수 있다 (연습 음성 실측: 숫자 27문항 중 5개만 원문과 일치, Soniox 확신도 0.9 이상).
   받아쓴 번호가 발신자 번호·이전 대화의 번호와 비슷하면 그 번호로 바꾼다.
2. 조회 결과 요약 (render_tool_result)
   원본 JSON의 '2026-09-28 08:25:08'을 모델이 통화 날짜와 섞어 틀리게 말했다 (실측).
   날짜·상태·경과일을 코드로 계산해 한국어 사실 목록으로 넘긴다.
3. 답 검증 (verify_answer)
   답에 나온 날짜가 근거(조회 결과, 날짜 표)에 없으면 그 문장을 뺀다.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

WEEK = "월화수목금토일"
PHONE_KEYS = ("buyer_cellphone", "customer_phone_number", "phone_number")
ORDER_KEYS = ("order_id", "order_number")
# 프롬프트·예시에 등장하는 가짜 예시 번호 (LLM이 번호를 못 들었을 때 그대로 넣어 조회하는 결함 차단)
EXAMPLE_PHONES = frozenset({
    "01012345678", "01000000000", "01012341234", "01000001234",
    "01011112222", "01022223333", "01033334444", "01044445555", "01055556666",
})


def digits(v) -> str:
    return re.sub(r"\D", "", str(v)) if v not in (None, "") else ""


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        q = [i]
        for j, cb in enumerate(b, 1):
            q.append(min(p[j] + 1, q[j - 1] + 1, p[j - 1] + (ca != cb)))
        p = q
    return p[-1]


# ───────────────────────── 1. 번호 맥락 우선 ─────────────────────────

_PHONE_RE = re.compile(r"(?<!\d)0\d{1,2}[\s.\-]{0,3}\d{3,4}[\s.\-]{0,3}\d{4}(?!\d)")
_ORDER_RE = re.compile(r"(?<!\d)(20\d{6})[\s\-]?(\d{7})(?:-(\d{2}))?(?!\d)")


def context_ids(history: list[dict], caller: dict | None) -> dict:
    """이전 대화와 발신자 정보에서 믿을 수 있는 번호를 모은다. (글로 남은 번호라 음성 오류가 없다)"""
    phones: list[str] = []
    orders: list[str] = []          # 'YYYYMMDD-NNNNNNN'
    items: list[str] = []           # 상세주문번호
    cp = digits((caller or {}).get("phone"))
    if valid_phone(cp):
        phones.append(cp)
    for h in history or []:
        role = h.get("role")
        blob = ""
        if role in ("customer", "agent"):
            blob = str(h.get("text") or "")
        elif role == "tool_call":
            blob = json.dumps(h.get("arguments") or {}, ensure_ascii=False)
        elif role == "tool_result":
            blob = json.dumps(h.get("result"), ensure_ascii=False)
        for m in _PHONE_RE.finditer(blob):
            phones.append(digits(m.group(0)))
        for m in _ORDER_RE.finditer(blob):
            orders.append(f"{m.group(1)}-{m.group(2)}")
            if m.group(3):
                items.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    uniq = lambda xs: list(dict.fromkeys(xs))
    return {"phones": uniq(phones), "orders": uniq(orders), "items": uniq(items), "caller_phone": cp if valid_phone(cp) else ""}


def _snap(value: str, candidates: list[str], max_dist: int) -> str | None:
    d = digits(value)
    if not d or not candidates:
        return None
    best, best_n = None, 99
    for c in candidates:
        cd = digits(c)
        if cd == d:
            return c
        if abs(len(cd) - len(d)) > max_dist:
            continue
        n = lev(d, cd)
        if n < best_n:
            best, best_n = c, n
    return best if best is not None and best_n <= max_dist else None


def resolve_ids(name: str, args: dict, ctx: dict, max_phone_dist: int = 3, max_order_dist: int = 4) -> tuple[dict, list[str]]:
    """함수 인자의 번호를 맥락의 번호에 맞춘다. (바뀐 인자, 바꾼 내용 설명) 을 돌려준다.
    - 전화번호: 맥락 번호와 3자 이내로 다르면 맥락 번호로.
    - 주문번호: 맥락 주문번호와 4자 이내로 다르면 맥락 번호로.
    - 조회 함수인데 번호가 없거나 형식이 깨졌고 발신자 번호가 있으면 발신자 번호를 넣는다."""
    out, notes = dict(args), []
    for k in PHONE_KEYS:
        v = out.get(k)
        if isinstance(v, str) and v.strip():
            s = _snap(v, ctx.get("phones", []), max_phone_dist)
            if s and digits(s) != digits(v):
                notes.append(f"{k}: {v} → {s} (맥락 번호)")
                out[k] = s
    for k in ORDER_KEYS:
        v = out.get(k)
        if isinstance(v, str) and v.strip():
            s = _snap(v, ctx.get("orders", []), max_order_dist)
            if s and digits(s) != digits(v):
                notes.append(f"{k}: {v} → {s} (맥락 번호)")
                out[k] = s
            elif not s:
                m = _ORDER_RE.fullmatch(v.strip())
                if m:
                    out[k] = f"{m.group(1)}-{m.group(2)}"           # 형식만 정리
    if name == "return_order_item":
        fixed = []
        for it in out.get("return_items") or []:
            if isinstance(it, dict) and it.get("order_item_code"):
                s = _snap(str(it["order_item_code"]), ctx.get("items", []), max_order_dist)
                if s and s != it["order_item_code"]:
                    notes.append(f"order_item_code: {it['order_item_code']} → {s}")
                    it = {**it, "order_item_code": s}
            fixed.append(it)
        if fixed:
            out["return_items"] = fixed
    cp = ctx.get("caller_phone")
    ctx_phones = {digits(p) for p in ctx.get("phones", [])}
    if cp and name == "get_order_info":
        ph, oid = digits(out.get("buyer_cellphone")), digits(out.get("order_id"))
        bad_phone = bool(ph) and (not valid_phone(ph) or (ph in EXAMPLE_PHONES and ph not in ctx_phones))
        bad_order = bool(oid) and not valid_order_id(out.get("order_id"))
        # 주문번호가 깨져 들렸으면(형식이 안 맞음) 발신자 번호로도 찾을 수 있게 한다. 멀쩡한 주문번호에는 덧붙이지 않는다
        # (다른 번호로 주문했을 수 있다).
        if bad_phone or (not ph and (not oid or bad_order)):
            notes.append(f"buyer_cellphone: {out.get('buyer_cellphone')} → {cp} (발신자 번호)")
            out["buyer_cellphone"] = cp
    if cp and name == "get_return_info":
        ph = digits(out.get("customer_phone_number"))
        if not valid_phone(ph) or (ph in EXAMPLE_PHONES and ph not in ctx_phones):
            notes.append(f"customer_phone_number: {out.get('customer_phone_number')} → {cp} (발신자 번호)")
            out["customer_phone_number"] = cp
    return out, notes


def valid_phone(v) -> bool:
    d = digits(v)
    if d.startswith("010"):
        return len(d) == 11              # 휴대폰은 11자리. 덜 들린 번호(실측 P0193-02 '010-3710-35')로 조회하지 않는다
    return 9 <= len(d) <= 11 and d.startswith("0")


def valid_order_id(v) -> bool:
    """하루온 주문번호: YYYYMMDD + 7자리 (예: 20261101-0001234)."""
    d = digits(v)
    if len(d) != 15 or not d.startswith("20"):
        return False
    return 1 <= int(d[4:6]) <= 12 and 1 <= int(d[6:8]) <= 31


def retry_candidates(name: str, args: dict, ctx: dict) -> list[dict]:
    """조회가 not_found/빈 결과일 때 다시 시도할 인자들 (조회 함수는 더 불러도 감점이 없다)."""
    tried = {digits(args.get(k)) for k in PHONE_KEYS + ORDER_KEYS if digits(args.get(k))}
    out = []
    if name == "get_order_info":
        for o in ctx.get("orders", []):
            if digits(o) not in tried:
                out.append({"order_id": o})
        for p in ctx.get("phones", []):
            if p not in tried:
                out.append({"buyer_cellphone": p})
    elif name == "get_return_info":
        for p in ctx.get("phones", []):
            if p not in tried:
                out.append({"customer_phone_number": p})
    return out[:2]


def lookup_empty(name: str, body) -> bool:
    if not isinstance(body, dict):
        return True
    if body.get("status") in ("not_found", "invalid", "error"):
        return True
    data = body.get("data") or {}
    if name == "get_order_info":
        return not data.get("orders")
    if name == "get_return_info":
        return not (data.get("반품정보") or data.get("returns"))
    return False


# ───────────────────────── 2. 조회 결과 요약 ─────────────────────────

def _dt(v) -> datetime | None:
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?", str(v or ""))
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4) or 0), int(m.group(5) or 0))
    except ValueError:
        return None


def kdate(d: datetime, now: datetime | None = None) -> str:
    y = f"{d.year}년 " if (now is not None and d.year != now.year) else ""
    return f"{y}{d.month}월 {d.day}일({WEEK[d.weekday()]})"


def ship_day(paid: datetime) -> datetime:
    """브랜드 정보: 평일 오후 3시 이전 주문은 당일 출고, 금요일 3시 이후는 다음 주 월요일. 공휴일은 알 수 없어 주말만 본다."""
    d = paid
    if d.weekday() >= 5 or d.hour >= 15:
        d = (d + timedelta(days=1)).replace(hour=9, minute=0)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def clean_product(name: str) -> str:
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", str(name or ""))
    t = re.sub(r"\[?\s*비밀특가\s*\]?", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _option(v: str) -> str:
    t = re.sub(r"^옵션 선택=", "", str(v or "")).strip()
    return clean_product(t)


def render_tool_result(name: str, body, now: datetime) -> str:
    """모델에게 넘길 조회 결과. 사실 목록 + 원본에서 필요한 식별자만. 날짜 계산은 여기서 끝낸다."""
    if not isinstance(body, dict):
        return json.dumps(body, ensure_ascii=False)[:1500]
    status = body.get("status")
    head = f"[{name} 결과] status={status}"
    if body.get("message"):
        head += f" / {body['message']}"
    data = body.get("data") or {}
    lines = [head]
    if name == "get_order_info":
        orders = data.get("orders") or []
        if status == "ok" and not orders:
            lines.append("조회된 주문이 없다 (이 번호로는 주문이 확인되지 않음).")
        nums = list(dict.fromkeys(o.get("주문번호") for o in orders if o.get("주문번호")))
        if orders:
            lines.append(f"주문 {len(nums)}건, 상품 줄 {len(orders)}개. 통화 시각 기준으로 계산한 값이다.")
        for i, o in enumerate(orders, 1):
            od, dd, pd = _dt(o.get("주문일자")), _dt(o.get("배송완료일자")), _dt(o.get("결제일자"))
            st = o.get("배송상태")
            pay = "결제 완료" if o.get("결제여부") == "T" else ("미결제(입금 전)" if o.get("결제여부") == "F" else "결제여부 불명")
            bits = [f"{i}) 주문번호 {o.get('주문번호')} (상세주문번호 {o.get('상세주문번호')})",
                    f"제품 {clean_product(o.get('제품명'))}", f"옵션 {_option(o.get('옵션'))}", f"수량 {o.get('수량')}"]
            if od:
                days = (now.date() - od.date()).days
                when = "오늘" if days == 0 else ("어제" if days == 1 else f"{days}일 전")
                bits.append(f"주문일 {kdate(od, now)} {od:%H:%M} ({when})")
            means = ", ".join(o.get("결제수단") or []) if isinstance(o.get("결제수단"), list) else str(o.get("결제수단") or "")
            bits.append(f"결제수단 {means or '불명'}, {pay}" + (f" ({kdate(pd, now)})" if pd and o.get("결제여부") == "T" else ""))
            bits.append(f"배송상태 {st}")
            if o.get("송장번호"):
                bits.append(f"송장번호 {o['송장번호']}")
            if o.get("출고예정일자"):
                ed = _dt(o.get("출고예정일자"))
                bits.append(f"출고예정일 {kdate(ed, now) if ed else o.get('출고예정일자')}")
            elif st in ("상품준비중", "배송준비중") and od and o.get("결제여부") == "T":
                sd = ship_day(pd or od)
                bits.append(f"출고 예정 {kdate(sd, now)} (평일 오후 3시 이전 결제는 당일, 이후와 주말은 다음 영업일 출고. 받으시기까지 출고 후 영업일 2~3일)")
            if dd:
                n = (now.date() - dd.date()).days
                ok = "반품 가능 기간(수령 후 10일) 이내" if n <= 10 else "수령 후 10일이 지나 단순 변심·효과 불만족 반품 불가"
                bits.append(f"배송완료일 {kdate(dd, now)} (수령 후 {n}일 경과, {ok})")
            if o.get("주문자기본주소"):
                bits.append(f"주소 {o.get('주문자기본주소')} {o.get('주문자상세주소') or ''}".strip())
            lines.append(" / ".join(bits))
        return "\n".join(lines)
    if name == "get_return_info":
        rows = data.get("반품정보") or []
        if status == "ok" and not rows:
            lines.append("조회된 반품 건이 없다.")
        for i, r in enumerate(rows, 1):
            bits = [f"{i}) 주문번호 {r.get('주문번호')}"]
            m = _ORDER_RE.match(str(r.get("주문번호") or ""))
            od = _dt(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}") if m else None
            if od:
                bits.append(f"주문일 {kdate(od, now)}")
            bits.append(f"상품 {clean_product(r.get('상품명'))}")
            bits.append(f"현재상태 {r.get('현재상태')}")
            for k in ("송장번호", "접수일자", "회수일자", "환불일자", "환불금액"):
                if r.get(k) not in (None, ""):
                    bits.append(f"{k} {r.get(k)}")
            if r.get("휴대폰결제여부") is True:
                bits.append("휴대폰 결제")
            lines.append(" / ".join(bits))
        return "\n".join(lines)
    if name == "search_products":
        rows = data.get("결과") or []
        by: dict[str, list[str]] = {}
        for r in rows:
            if isinstance(r, dict):
                by.setdefault(clean_product(r.get("제품명")), []).append(
                    f"- {r.get('정보_그룹')}: " + re.sub(r"\s*\n\s*", " / ", str(r.get("내용") or "")))
        for prod, items in by.items():
            lines.append(f"제품 {prod}")
            lines += items[:12]
        return "\n".join(lines)[:3500]
    if data:
        lines.append(json.dumps(data, ensure_ascii=False)[:800])
    return "\n".join(lines)


# ───────────────────────── 3. 답 검증 ─────────────────────────

_KDATE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_SENT = re.compile(r"[^.!?。]+[.!?。]+[\"')\]]*|[^.!?。]+$")


def split_sentences(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()
    return [s.strip() for s in _SENT.findall(t) if s.strip()]


def allowed_dates(evidence: str, now: datetime) -> set[tuple[int, int]]:
    out = {(m.group(2), m.group(3)) for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", evidence)}
    al = {(int(a), int(b)) for a, b in out}
    al |= {(int(m.group(1)), int(m.group(2))) for m in _KDATE.finditer(evidence)}
    for d in range(-31, 32):
        x = now + timedelta(days=d)
        al.add((x.month, x.day))
    return al


_ORDERED_ON = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일(?=\s*(?:에|자로)?\s*(?:주문|구매|결제))")
_DELIVERED_ON = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일(?=\s*(?:에|자로)?\s*(?:배송\s*(?:이|까지)?\s*(?:완료|끝)|수령|도착))")


def _fix_dated(text: str, pat: re.Pattern, allowed: set[tuple[int, int]], fixes: list[str]) -> str:
    """'9월 29일에 주문하신'처럼 날짜와 사건이 붙은 표현. 근거의 날짜가 하나뿐이면 그 날짜로 고친다."""
    if not allowed:
        return text

    def sub(m: re.Match) -> str:
        md = (int(m.group(1)), int(m.group(2)))
        if md in allowed or len(allowed) != 1:
            return m.group(0)
        mo, d = next(iter(allowed))
        fixes.append(f"{m.group(0)} → {mo}월 {d}일")
        return f"{mo}월 {d}일"

    return pat.sub(sub, text)


def verify_answer(text: str, evidence: str, now: datetime) -> tuple[str, list[str]]:
    """① 주문일·배송완료일을 틀리게 말했고 근거의 날짜가 하나면 고친다 ② 근거에 없는 날짜를 말한 문장은 뺀다.
    (고친 답, 고치거나 뺀 내용)"""
    fixes: list[str] = []
    order_dates = {(int(a), int(b)) for a, b in re.findall(r"주문일 (?:\d{4}년 )?(\d{1,2})월 (\d{1,2})일", evidence)}
    done_dates = {(int(a), int(b)) for a, b in re.findall(r"배송완료일 (?:\d{4}년 )?(\d{1,2})월 (\d{1,2})일", evidence)}
    text = _fix_dated(text, _ORDERED_ON, order_dates, fixes)
    text = _fix_dated(text, _DELIVERED_ON, done_dates, fixes)
    sents = split_sentences(text)
    if not sents:
        return text, fixes
    ev_dates = {(int(m.group(2)), int(m.group(3))) for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", evidence)}
    ev_dates |= {(int(m.group(1)), int(m.group(2))) for m in _KDATE.finditer(evidence)}
    strict = bool(ev_dates)                      # 조회 결과에 날짜가 있으면 그 날짜(와 통화일 근처 설명용 날짜)만
    near = {((now + timedelta(days=d)).month, (now + timedelta(days=d)).day) for d in range(-1, 15)}
    keep, dropped = [], []
    for s in sents:
        bad = False
        for m in _KDATE.finditer(s):
            md = (int(m.group(1)), int(m.group(2)))
            if strict and md not in ev_dates and md not in near:
                bad = True
        (dropped if bad else keep).append(s)
    if not keep:                                  # 다 빼면 답이 없어진다. 그대로 둔다
        return text, fixes
    return " ".join(keep), fixes + [f"뺌: {s}" for s in dropped]


def clamp_answer(text: str, max_sentences: int = 3, max_chars: int = 120, is_bank=None) -> str:
    """전화 답변 길이 제한. 문장 경계에서 자른다. 은행 문장(미리 합성·검수된 문장)은 세지 않는다."""
    sents = split_sentences(text)
    out, n, k = [], 0, 0
    for s in sents[: max_sentences + 3]:
        if is_bank is not None and is_bank(s):
            out.append(s)
            continue
        if k >= max_sentences or (k and n + len(s) > max_chars):
            break
        out.append(s)
        n += len(s)
        k += 1
    return " ".join(out) if out else text.strip()
