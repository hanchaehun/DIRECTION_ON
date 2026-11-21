"""두뇌: 메시지 조립(캐시가 적중하도록 고정 부분을 앞에), GPT-5.4 mini 호출, 완결성 판정.

OpenAIBackend  Chat Completions 스트리밍 (httpx). 게이트웨이를 쓰면 OPENAI_BASE_URL만 바꾼다.
FakeBackend    키 없이 시험하는 규칙 기반 응답기.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import httpx

from . import facts, knowledge
from .ratelimit import LIMITER
from .config import PROMPTS, env
from .textnorm import DIGIT_WORD

log = logging.getLogger("brain")
WEEK = "월화수목금토일"
JUDGE_SYS = ("전화 상담 중이다. 고객의 말이 끝나서 상담원이 지금 답해도 되면 Y, "
             "고객이 이어서 더 말할 것 같으면 N. 한 글자만 출력한다.")


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    raw: str = ""


@dataclass
class Decision:
    kind: str                       # answer | tool | error
    text: str = ""
    tool_calls: list = field(default_factory=list)
    assistant_msg: dict | None = None
    t_start: float = 0.0
    t_first: float | None = None
    t_done: float | None = None
    usage: dict | None = None
    finish: str = ""
    error: str = ""
    status: int = 0


# ───────────────────────── 메시지 조립 ─────────────────────────

def parse_now(now) -> datetime:
    """'2026-11-04 (수) 12:39'(공식 형식), '2026-11-21 11:00', ISO 형식을 받는다."""
    if isinstance(now, datetime):
        return now
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:\D+?(\d{1,2}):(\d{2})(?::(\d{2}))?)?", str(now or ""))
    if m:
        y, mo, d, hh, mm, ss = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0))
        except ValueError:
            pass
    return datetime.now()


def _d(dt: datetime) -> str:
    # 한글 날짜로 쓴다. 게이트웨이 개인정보 필터가 '2026-11-04 (수)' 같은 ISO 날짜를 [DATE-OF-BIRTH]로 가리는 경우가 있다 (실측)
    return f"{dt.year}년 {dt.month}월 {dt.day}일({WEEK[dt.weekday()]})"


def date_table(now: datetime) -> str:
    """날짜 계산을 LLM 암산에 맡기지 않는다 (피드백 반영)."""
    rows = [f"현재(통화 시각): {_d(now)} {now.hour}시 {now.minute:02d}분"]
    for label, days in (("어제", -1), ("그제", -2), ("내일", 1), ("모레", 2), ("7일 전", -7), ("14일 전", -14), ("30일 전", -30)):
        rows.append(f"{label}: {_d(now + timedelta(days=days))}")
    monday = now - timedelta(days=now.weekday())
    rows.append(f"이번 주 월요일: {_d(monday)}, 지난주 월요일: {_d(monday - timedelta(days=7))}")
    return " | ".join(rows)


def _read(name: str) -> str:
    p = PROMPTS / name
    if not p.exists():
        return ""
    t = p.read_text(encoding="utf-8").strip()
    return "" if t.startswith("TODO:") else t


def load_static_prompt(s, phrases: dict) -> str:
    """모든 요청에서 똑같은 앞부분 (프롬프트 캐시). 브랜드 자료는 data/brand/에서 바로 읽는다."""
    st = knowledge.style()
    parts = [_read("system.md").replace("{brand}", s.brand_name)]
    if st.get("style"):
        parts.append("# 브랜드 말투 규칙\n" + str(st["style"]).strip())
    for title, body in (("상담 시나리오", knowledge.scenarios()),
                        ("업무별 정책 표", _read("policy_table.md")),
                        ("브랜드 정보", knowledge.brand_info()),
                        ("모범 응대 예시", _read("examples.md"))):
        if body:
            parts.append(f"# [{title}]\n{body}")
    from .clips import all_fixed_sentences
    fixed = all_fixed_sentences(phrases, with_products=False)
    if fixed:
        parts.append("# 정해진 문구 (이 뜻으로 말할 때는 글자 그대로 쓴다)\n" + "\n".join(f"- {x}" for x in fixed))
    if getattr(s, "pii_placeholders", False):
        parts.append(PII_NOTE)
    return "\n\n".join(p for p in parts if p)


def caller_line(caller: dict | None) -> str:
    """걸려 온 번호로 찾은 회원 정보. 다른 번호로 주문했을 수도 있다."""
    if not caller:
        return "발신자 정보: 없음 (회원 정보를 찾지 못함)"
    names = {"name": "이름", "phone": "전화번호", "email": "이메일"}
    bits = [f"{names.get(k, k)} {v}" for k, v in caller.items() if v not in (None, "")]
    return "발신자 정보(걸려 온 번호로 찾은 회원): " + (", ".join(bits) if bits else "없음")


_HIST_URL = re.compile(r"https?://\S+")
_HIST_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


def _clean_hist(t: str, limit: int = 900) -> str:
    t = _HIST_EMOJI.sub("", _HIST_URL.sub("(링크)", str(t or "")))
    t = re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", t)).strip()
    return t if len(t) <= limit else t[:limit] + "…"


def history_messages(history: list[dict], now: datetime | None = None) -> list[dict]:
    """공식 history(customer/agent/tool_call/tool_result)를 Chat Completions 메시지로.
    tool_call과 tool_result는 id로 짝지어 assistant.tool_calls + tool 메시지로 넣는다. 짝이 없으면 글로 남긴다."""
    msgs: list[dict] = []
    i, n, k = 0, len(history), 0
    while i < n:
        h = history[i]
        role = h.get("role")
        if role == "tool_call":
            calls, results = [], []
            while i < n and history[i].get("role") == "tool_call":
                calls.append(history[i])
                i += 1
            while i < n and history[i].get("role") == "tool_result":
                results.append(history[i])
                i += 1
            paired, notes, pool = [], [], list(results)
            for c in calls:
                r = next((x for x in pool if x.get("name") == c.get("name")), None)
                if r is None:
                    notes.append(f"(함수 {c.get('name')} 호출, 결과 기록 없음: {json.dumps(c.get('arguments'), ensure_ascii=False)})")
                    continue
                pool.remove(r)
                k += 1
                paired.append((f"hist_call_{k}", c, r))
            if paired:
                msgs.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": cid, "type": "function",
                     "function": {"name": c.get("name"), "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False)}}
                    for cid, c, _ in paired]})
                for cid, _, r in paired:
                    msgs.append({"role": "tool", "tool_call_id": cid,
                                 "content": (facts.render_tool_result(str(r.get("name")), r.get("result"), now) if now is not None
                                             else json.dumps(r.get("result"), ensure_ascii=False))[:4000]})
            for r in pool:
                notes.append(f"(함수 {r.get('name')} 결과: {json.dumps(r.get('result'), ensure_ascii=False)[:2000]})")
            if notes:
                msgs.append({"role": "system", "content": "[이전 함수 기록]\n" + "\n".join(notes)})
            continue
        text = h.get("text")
        if role == "customer" and text:
            msgs.append({"role": "user", "content": _clean_hist(text)})
        elif role == "agent" and text:
            msgs.append({"role": "assistant", "content": _clean_hist(text, 420)})     # 긴 채팅식 답은 줄인다 (길이를 따라 하지 않게)
        elif role == "tool_result":
            msgs.append({"role": "system", "content": f"[이전 함수 기록] (함수 {h.get('name')} 결과: "
                                                      f"{json.dumps(h.get('result'), ensure_ascii=False)[:2000]})"})
        elif role == "note" and text:
            msgs.append({"role": "system", "content": text})
        i += 1
    return msgs


# ───────────────────────── 개인정보 자리표시 ─────────────────────────
# BAZE 게이트웨이의 개인정보 필터가 전화번호·이메일을 [PHONE]·[EMAIL]로 바꿔 모델에 넘긴다 (실측).
# 그러면 모델이 번호를 몰라 함수 인자가 틀린다. 보내기 전에 우리가 {{PHONE_n}} 자리표시로 바꾸고
# 응답(답 문장, 함수 인자)에서 실제 값으로 되돌린다. 실제 번호는 모델 제공사로 가지 않는다.
_PII_PHONE = re.compile(r"(?<![\d{])(?:0\d{1,2}|\d{3})[ \t.\-]{0,3}\d{3,4}[ \t.\-]{0,3}\d{4}(?!\d)|(?<!\d)01[016789]\d{7,8}(?!\d)")
_PII_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
_PII_ACCT = re.compile(r"(?<![\d{\-])\d{2,6}-\d{2,6}-\d{2,8}(?:-\d{1,6})?(?![\d\-])")   # 하이픈 계좌번호 (필터가 일부를 전화로 가린다)
_ISO_DATE = re.compile(r"(?<![\d\-])(20\d{2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})\.?(?![\d\-])")
_ORDER_ID = re.compile(r"\d{8}-\d{7}(?:-\d{2})?")                                       # 주문번호는 필터가 건드리지 않는다
_PII_TOKEN = re.compile(r"\{\{\s*(PHONE|EMAIL|ACCT)_(\d+)\s*\}\}|\b(PHONE|EMAIL|ACCT)_(\d+)\b")


class Pseudonymizer:
    def __init__(self):
        self.fwd: dict[str, str] = {}
        self.back: dict[str, str] = {}

    def _tok(self, kind: str, value: str) -> str:
        key = (kind, re.sub(r"\D", "", value) if kind in ("PHONE", "ACCT") else value.lower())
        k = f"{key[0]}:{key[1]}"
        if k not in self.fwd:
            n = sum(1 for x in self.fwd if x.startswith(kind + ":")) + 1
            tok = f"{{{{{kind}_{n}}}}}"
            self.fwd[k] = tok
            self.back[f"{kind}_{n}"] = value
        return self.fwd[k]

    def mask(self, text):
        if not isinstance(text, str) or not text:
            return text
        text = _ISO_DATE.sub(lambda m: f"{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)   # 필터가 날짜를 가리지 않게
        text = _PII_EMAIL.sub(lambda m: self._tok("EMAIL", m.group(0)), text)
        text = _PII_PHONE.sub(lambda m: self._tok("PHONE", m.group(0)), text)
        return _PII_ACCT.sub(lambda m: m.group(0) if _ORDER_ID.fullmatch(m.group(0)) else self._tok("ACCT", m.group(0)), text)

    def unmask(self, obj):
        if isinstance(obj, str):
            return _PII_TOKEN.sub(lambda m: self.back.get(f"{m.group(1) or m.group(3)}_{m.group(2) or m.group(4)}", m.group(0)), obj)
        if isinstance(obj, list):
            return [self.unmask(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.unmask(v) for k, v in obj.items()}
        return obj

    def mask_messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            m2 = dict(m)
            if isinstance(m2.get("content"), str):
                m2["content"] = self.mask(m2["content"])
            if m2.get("tool_calls"):
                m2["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": self.mask(tc["function"].get("arguments", ""))}}
                                    for tc in m2["tool_calls"]]
            out.append(m2)
        return out


PII_NOTE = ("# 개인정보 자리표시\n전화번호, 계좌번호, 이메일은 개인정보 보호를 위해 {{PHONE_1}}, {{ACCT_1}}, {{EMAIL_1}} 같은 자리표시로 바뀌어 있다. "
            "같은 자리표시는 같은 번호다. 함수 인자에는 자리표시를 글자 그대로 넣으면 시스템이 실제 번호로 바꿔 호출한다. "
            "고객에게 번호를 말해야 할 때도 자리표시를 그대로 쓰면 실제 번호로 읽힌다. 자리표시를 숫자로 지어내지 않는다.")


# ───────────────────────── OpenAI ─────────────────────────

class OpenAIBackend:
    OPTIONAL = ("prompt_cache_key", "reasoning_effort", "parallel_tool_calls", "stream_options", "max_completion_tokens")

    def __init__(self, http: httpx.AsyncClient, s):
        self.http, self.s = http, s
        self.url = env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.key = env("OPENAI_API_KEY")
        self.drop: set[str] = set()

    def _body(self, messages, tools, effort, max_tokens) -> dict:
        b = {"model": self.s.openai_model, "messages": messages, "stream": True,
             "stream_options": {"include_usage": True}, "max_completion_tokens": max_tokens,
             "prompt_cache_key": self.s.prompt_cache_key}
        if effort:
            b["reasoning_effort"] = effort
        if tools:
            b["tools"] = tools
            b["tool_choice"] = "auto"
            b["parallel_tool_calls"] = True
        for p in self.drop:
            v = b.pop(p, None)
            if p == "max_completion_tokens" and v is not None:
                b["max_tokens"] = v
        return b

    async def complete(self, messages, tools, effort, max_tokens) -> Decision:
        if not self.key:
            return Decision("error", error="OPENAI_API_KEY 없음")
        pz = Pseudonymizer() if self.s.pii_placeholders else None
        if pz:
            messages = pz.mask_messages(messages)
        for attempt in range(4):
            d = await self._stream(self._body(messages, tools, effort, max_tokens))
            if d.kind == "error" and d.status == 400:
                bad = [p for p in self.OPTIONAL if p in d.error and p not in self.drop]
                if bad:
                    log.warning("openai: 지원하지 않는 파라미터 제외 %s", bad)
                    self.drop.update(bad)
                    continue
            if d.kind == "error" and d.status in (429, 500, 502, 503) and attempt < 3:
                log.warning("openai %s, 재시도: %s", d.status, d.error[:120])
                if d.status == 429:
                    LIMITER.note_429(1.5 * (attempt + 1))          # 분당 한도: 모든 호출을 잠시 멈춘다
                else:
                    await asyncio.sleep(0.4 * (attempt + 1))
                continue
            break
        if pz and d.kind != "error":
            d.text = pz.unmask(d.text)
            for tc in d.tool_calls:
                tc.args = pz.unmask(tc.args)
                tc.raw = json.dumps(tc.args, ensure_ascii=False)
            d.assistant_msg = pz.unmask(d.assistant_msg)
        return d

    async def _stream(self, body: dict) -> Decision:
        await LIMITER.acquire()
        d = Decision("error", t_start=time.perf_counter())
        content, calls, usage, finish = "", {}, None, ""
        try:
            async with self.http.stream("POST", self.url, json=body, timeout=httpx.Timeout(self.s.llm_timeout_s, connect=4.0),
                                        headers={"Authorization": f"Bearer {self.key}"}) as r:
                if r.status_code != 200:
                    txt = (await r.aread()).decode("utf-8", "replace")
                    d.status, d.error = r.status_code, txt[:600]
                    return d
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    if ev.get("usage"):
                        usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            d.t_first = d.t_first or time.perf_counter()
                            content += delta["content"]
                        for tc in delta.get("tool_calls") or []:
                            d.t_first = d.t_first or time.perf_counter()
                            slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["args"] += fn.get("arguments") or ""
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            d.error = repr(e)[:300]
            return d
        d.t_done, d.usage, d.finish, d.status = time.perf_counter(), usage, finish, 200
        if finish == "length":                      # 추론 토큰이 상한을 먹어 잘림
            d.error = "length"
            return d
        if calls:
            tcs = []
            for i in sorted(calls):
                c = calls[i]
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    d.error = f"bad tool args: {c['args'][:200]}"
                    return d
                tcs.append(ToolCall(c["id"] or f"call_{i}", c["name"], args, c["args"]))
            d.kind, d.tool_calls = "tool", tcs        # 함수 호출과 함께 온 글은 버린다 (첫마디와 겹침 방지)
            d.assistant_msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.raw or json.dumps(t.args, ensure_ascii=False)}}
                for t in tcs]}
            return d
        d.kind, d.text = "answer", content.strip()
        d.assistant_msg = {"role": "assistant", "content": d.text}
        return d


# ───────────────────────── 가짜 응답기 ─────────────────────────

_SUFFIX = re.compile(r"(인데요|이에요|이요|예요|입니다|요|번|이고|이구)$")


def _spoken_digits(part: str) -> str:
    out = ""
    for w in part.replace("-", " ").split():
        w = _SUFFIX.sub("", re.sub(r"[.,?!]", "", w))
        if w and all(c.isdigit() or c in DIGIT_WORD for c in w):
            out += "".join(c if c.isdigit() else DIGIT_WORD[c] for c in w)
    return out


class FakeBackend:
    def __init__(self, s):
        self.s = s

    async def complete(self, messages, tools, effort, max_tokens) -> Decision:
        d = Decision("answer", t_start=time.perf_counter())
        is_judge = messages and messages[0].get("content") == JUDGE_SYS
        await asyncio.sleep((self.s.fake_judge_latency_ms if is_judge else self.s.fake_llm_latency_ms) / 1000)
        d.t_first = d.t_done = time.perf_counter()
        if is_judge:
            said = messages[-1]["content"].split("고객 말:", 1)[-1].strip()
            d.text = "N" if re.search(r"(인데요|는데요|은데요|거든요)[.]?$", said) else "Y"
            return d
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        tool_msgs = [m for m in messages[last_user + 1:] if m["role"] == "tool"]
        if tool_msgs:
            name = ""
            for m in messages[last_user + 1:]:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    name = m["tool_calls"][-1]["function"]["name"]
            try:
                res = json.loads(tool_msgs[-1]["content"])
            except json.JSONDecodeError:
                res = {}
            body = res.get("body") if isinstance(res.get("body"), dict) else res
            status = str(body.get("status", ""))
            if status == "ok" and name == "get_order_info":
                d.text = f"주문하신 상품은 현재 {body.get('data', {}).get('shipping_status', '확인 중')}입니다."
            elif status == "ok":
                d.text = "말씀하신 내용으로 접수했습니다. 내일 오전 중에 연락드리겠습니다."
            elif status == "not_implemented":
                d.text = "지금은 조회가 어려워 담당 상담원에게 연결해 드리겠습니다."
            else:
                d.text = "말씀하신 번호로 조회되는 주문이 없습니다. 주문번호를 다시 한 번 말씀해 주시겠어요?"
            d.assistant_msg = {"role": "assistant", "content": d.text}
            return d
        text = messages[last_user]["content"]
        names = {t["function"]["name"] for t in (tools or [])}
        info = " ".join(m["content"] for m in messages if m["role"] == "system")
        phone = (re.search(r"전화번호 ([0-9\-]+)", info) or [None, None])[1]
        call = None
        if "주문" in text and "번호" in text:
            part = re.split(r"아\s*아니|아니\s*아니|아니요|아니,|정정", text)[-1]
            digits = _spoken_digits(part) or _spoken_digits(text)
            if digits and "get_order_info" in names:
                call = ("get_order_info", {"order_id": ("DEMO" if "데모" in text else "") + digits})
            elif not digits:
                d.text = "주문번호를 말씀해 주시겠어요?"
        elif ("전화" in text or "연락" in text) and ("부탁" in text or "주세요" in text) and phone and "request_callback" in names:
            call = ("request_callback", {"phone_number": phone})
        elif "상담" in text and "시간" in text:
            d.text = "상담 시간은 평일 오전 아홉 시부터 오후 여섯 시까지입니다."
        elif "취소" in text:
            d.text = "네, 배송이 시작되기 전이면 바로 취소해 드릴 수 있습니다. 취소를 진행해 드릴까요?"
        elif "반품" in text and "배송비" in text:
            d.text = "단순 변심으로 반품하시면 왕복 배송비 육천 원이 부과됩니다."
        elif any(k in text for k in ("속이", "몸이", "불편", "아파", "부작용")):
            d.text = "불편을 드려 죄송합니다. 섭취를 멈추시고 증상이 계속되면 병원 진료를 받아 보세요. 담당 상담원에게 바로 연결해 드리겠습니다."
        elif re.search(r"안녕하세요|여보세요", text) and len(text) < 14:
            d.text = "네, 안녕하세요. 무엇을 도와드릴까요?"
        elif not d.text:
            d.text = "죄송하지만 다시 한 번 말씀해 주시겠어요?"
        if call:
            tc = ToolCall("call_fake_1", call[0], call[1], json.dumps(call[1], ensure_ascii=False))
            d.kind, d.tool_calls = "tool", [tc]
            d.assistant_msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.raw}}]}
            return d
        d.assistant_msg = {"role": "assistant", "content": d.text}
        return d


# ───────────────────────── 두뇌 ─────────────────────────

class Brain:
    def __init__(self, s, backend, phrases: dict):
        self.s, self.backend = s, backend
        self.static = load_static_prompt(s, phrases)
        self.last_call = time.perf_counter()

    def build_messages(self, history: list[dict], user_text: str, caller: dict | None, now: datetime,
                       ctx: dict | None = None, spoken: str = "") -> list[dict]:
        """ctx: facts.context_ids 결과 (글로 확인된 번호). spoken: 시스템이 이미 낸 첫마디."""
        info = "[통화 정보]\n" + date_table(now) + "\n" + caller_line(caller)
        info += ("\n이전 대화 기록이 아래에 이어진다. 마지막 user 메시지가 고객이 방금 전화로 한 말(음성 받아쓰기)이다."
                 if history else "\n이번 통화에서 고객의 첫 말이다.")
        if ctx and (ctx.get("phones") or ctx.get("orders")):
            info += ("\n[글로 확인된 번호] 전화번호: " + (", ".join(ctx.get("phones", [])[:4]) or "없음")
                     + " / 주문번호: " + (", ".join(ctx.get("orders", [])[:6]) or "없음")
                     + "\n음성 받아쓰기의 숫자는 틀리게 들릴 수 있다. 고객이 말한 번호가 위 번호와 비슷하면 위 번호를 쓴다. "
                       "고객이 번호를 말하지 않았고 조회가 필요하면 위 번호로 조회한다.")
        else:
            info += "\n음성 받아쓰기의 숫자는 틀리게 들릴 수 있다. 받아쓴 번호는 그대로 함수 인자에 넣되, 번호를 지어내거나 고치지 않는다."
        msgs = [{"role": "system", "content": self.static}, {"role": "system", "content": info}]
        msgs += history_messages(history, now)
        msgs.append({"role": "user", "content": user_text})
        if spoken:
            msgs.append({"role": "system", "content": f"(상담원이 방금 먼저 \"{spoken}\"라고 말했다. 이 말을 반복하지 말고 이어서 답한다.)"})
        return msgs

    def max_tokens(self, effort: str | None) -> int:
        return self.s.max_tokens_none if effort in (None, "", "none") else self.s.max_tokens_reasoning

    async def decide(self, messages, tools, effort: str | None) -> Decision:
        self.last_call = time.perf_counter()
        mt = self.max_tokens(effort)
        for attempt in range(2):
            try:
                d = await asyncio.wait_for(self.backend.complete(messages, tools, effort, mt), self.s.llm_timeout_s)
            except asyncio.TimeoutError:
                d = Decision("error", error="timeout")
            if d.kind == "error" and d.error == "length" and attempt == 0:
                mt *= 2                              # 잘렸으면 한 번만 상한을 늘려 재시도
                continue
            return d
        return d

    async def judge(self, last_agent: str, text: str) -> str | None:
        msgs = [{"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content": f"상담원 직전 말: {last_agent or '(없음)'}\n고객 말: {text}"}]
        try:
            d = await asyncio.wait_for(self.backend.complete(msgs, None, "none", self.s.judge_max_tokens), 2.0)
        except asyncio.TimeoutError:
            return None
        if d.kind != "answer" or not d.text:
            return None
        c = d.text.strip().upper()[:1]
        return {"Y": "complete", "N": "incomplete"}.get(c)

    async def keepalive(self, tools: list | None = None) -> None:
        """프롬프트 캐시와 연결을 살려 둔다 (캐시 유지 시간 5~10분). 실제 요청과 같은 함수 정의를 넣어야 앞부분이 같다."""
        msgs = [{"role": "system", "content": self.static},
                {"role": "user", "content": "연결 확인입니다. 한 글자로만 답하세요."}]
        self.last_call = time.perf_counter()
        try:
            await asyncio.wait_for(self.backend.complete(msgs, tools, self.s.reasoning_effort, 16), 10)
        except Exception as e:
            log.warning("keepalive 실패: %r", e)
