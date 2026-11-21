"""한국어 텍스트 처리.

- for_tts: LLM이 쓴 글 → TTS가 틀리지 않게 읽을 글. 항목별로 켜고 끈다 (probe로 실제로 틀리는 것만 켠다).
- trailing_digits: 받아쓰기 끝의 숫자 표현을 숫자열로 (종료 판단의 자릿수 세기 전용, 원문은 고치지 않는다).
- normalize_tool_args / invalid_tool_arg: 함수 호출 직전의 가벼운 정규화와 최소 검증.
"""
from __future__ import annotations

import re
from datetime import datetime

SINO = "영일이삼사오육칠팔구"
DIGIT_READ = "공일이삼사오육칠팔구"           # 번호를 한 자리씩 읽을 때 (0 = 공)
DIGIT_WORD = {"공": "0", "영": "0", "빵": "0", "일": "1", "이": "2", "삼": "3", "사": "4",
              "오": "5", "육": "6", "륙": "6", "칠": "7", "팔": "8", "구": "9"}
LETTER_READ = {"A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프", "G": "지",
               "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘", "M": "엠", "N": "엔",
               "O": "오", "P": "피", "Q": "큐", "R": "알", "S": "에스", "T": "티", "U": "유",
               "V": "브이", "W": "더블유", "X": "엑스", "Y": "와이", "Z": "제트"}
MONTH_READ = {6: "유월", 10: "시월"}


# ───────────────────────── 숫자 읽기 ─────────────────────────

def _sino4(n: int) -> str:
    out = ""
    for val, name in ((1000, "천"), (100, "백"), (10, "십")):
        d, n = divmod(n, val)
        if d:
            out += ("" if d == 1 else SINO[d]) + name
    if n:
        out += SINO[n]
    return out


def sino(n: int) -> str:
    """정수를 한자어 수사로: 39800 → 삼만 구천팔백, 2026 → 이천이십육."""
    if n == 0:
        return "영"
    if n < 0:
        return "마이너스 " + sino(-n)
    units = ["", "만", "억", "조"]
    parts, g = [], 0
    while n > 0:
        n, rem = divmod(n, 10000)
        if rem:
            s = "" if (g == 1 and rem == 1) else _sino4(rem)
            parts.append(s + units[g])
        g += 1
    return " ".join(reversed(parts))


def read_digits(s: str) -> str:
    return "".join(DIGIT_READ[int(c)] for c in s if c.isdigit())


def month_read(m: int) -> str:
    return MONTH_READ.get(m, sino(m) + "월")


# ───────────────────────── TTS 입력 변환 ─────────────────────────

_PHONE = re.compile(r"(?<![0-9A-Za-z])(0\d{1,2})[-\s.]{0,3}(\d{3,4})[-\s.]{0,3}(\d{4})(?!\d)")   # 게이트웨이 필터가 숫자 전화번호를 가리므로 반드시 한글로 읽힌다
_REPNUM = re.compile(r"(?<![0-9])(1[5-9]\d{2})-(\d{4})(?![0-9])")     # 대표번호 1600-0000
_ORDER = re.compile(r"(?<![0-9A-Za-z])([A-Za-z]{2,10})[-\s]?(\d{2,12})(?![0-9])")
_ID_CTX = re.compile(r"(번호\s*(?:는|은|가|이)?\s*)(\d[\d-]{2,})(?!\d)")
_ISO = re.compile(r"(?<!\d)(\d{4})[-./](\d{1,2})[-./](\d{1,2})(?!\d)")
_TILDE = re.compile(r"(\d+)\s*~\s*(\d+)")
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_AMOUNT = re.compile(r"(?<![\d,])(\d{1,3}(?:,\d{3})+|\d+)\s*원")
_KDATE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_MARKUP = re.compile(r"[*#`_>|]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


def _phone_sub(m: re.Match) -> str:
    return " ".join(read_digits(g) for g in m.groups())


def _order_sub(readings: dict):
    def sub(m: re.Match) -> str:
        prefix, digits = m.group(1), m.group(2)
        up = prefix.upper()
        p = readings.get(up) or "".join(LETTER_READ.get(c, c) for c in up)
        return f"{p} {read_digits(digits)}"
    return sub


def _iso_sub(now: datetime | None):
    def sub(m: re.Match) -> str:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # 피드백 반영: 연도가 현재와 다르면 남긴다 (작년 주문, 유통기한)
        if now is not None and y == now.year:
            return f"{mo}월 {d}일"
        return f"{y}년 {mo}월 {d}일"
    return sub


def for_tts(text: str, s=None, now: datetime | None = None) -> str:
    """s: Settings (None이면 기본 토글). 변환한 결과는 TTS 입력 전용이다."""
    on = (lambda k, d=True: getattr(s, k, d)) if s is not None else (lambda k, d=True: d)
    t = text
    if on("tts_norm_phone"):
        t = _PHONE.sub(_phone_sub, t)
        t = _REPNUM.sub(lambda m: read_digits(m.group(1)) + " " + read_digits(m.group(2)), t)
    if on("tts_norm_order_id"):
        readings = getattr(s, "tts_prefix_readings", {"DEMO": "데모"}) if s is not None else {"DEMO": "데모"}
        t = _ORDER.sub(_order_sub(readings), t)
    if on("tts_norm_id_context"):
        t = _ID_CTX.sub(lambda m: m.group(1) + read_digits(m.group(2)), t)
    if on("tts_norm_iso_date"):
        t = _ISO.sub(_iso_sub(now), t)
    if on("tts_norm_amount", False):
        t = _AMOUNT.sub(lambda m: sino(int(m.group(1).replace(",", ""))) + " 원", t)
    if on("tts_norm_kdate", False):
        t = _KDATE.sub(lambda m: f"{month_read(int(m.group(1)))} {sino(int(m.group(2)))}일", t)
    if on("tts_norm_tilde"):
        t = _TILDE.sub(r"\1에서 \2", t).replace("~", " ")
    if on("tts_norm_percent"):
        t = _PCT.sub(r"\1 퍼센트", t)
    t = t.replace("&", " 앤 ")                 # 관절&연골 → 관절 앤 연골
    t = _EMOJI.sub("", _MARKUP.sub(" ", t))
    return re.sub(r"[ \t]+", " ", t).strip()


# ───────────────────────── 받아쓰기 분석 ─────────────────────────

_TAIL_SUFFIX = re.compile(r"(번이에요|번이요|번입니다|번요|번|이에요|이요|예요|입니다|요|이고요|이구요)$")
_PLACE = set("십백천만")


def parse_sino(s: str) -> int | None:
    """'천이백삼십사' → 1234, '삼만 구천팔백' → 39800. 실패하면 None."""
    s = s.replace(" ", "")
    if not s or any(c not in DIGIT_WORD and c not in _PLACE for c in s):
        return None
    total, section, num = 0, 0, 0
    for c in s:
        if c in DIGIT_WORD:
            num = int(DIGIT_WORD[c])
        elif c == "만":
            section += num
            total += (section or 1) * 10000
            section, num = 0, 0
        else:
            unit = {"십": 10, "백": 100, "천": 1000}[c]
            section += (num or 1) * unit
            num = 0
    return total + section + num


def trailing_digits(text: str) -> str:
    """받아쓰기 끝의 숫자 표현 → 숫자열. '공일공 일이삼사' → '0101234', '010-1234' → '0101234'.
    한글 숫자만으로 된 꼬리는 두 글자 이상일 때만 숫자로 본다 ('이 상품' 같은 오인 방지)."""
    t = re.sub(r"[\s.,?!~]+$", "", text)
    t = _TAIL_SUFFIX.sub("", t)
    i = len(t)
    while i > 0 and (t[i - 1].isdigit() or t[i - 1] in DIGIT_WORD or t[i - 1] in _PLACE or t[i - 1] in " -"):
        i -= 1
    tail = t[i:].strip(" -")
    if not tail:
        return ""
    compact = tail.replace(" ", "").replace("-", "")
    if any(c in _PLACE for c in compact):
        v = parse_sino(compact)
        return str(v) if v is not None else ""
    hangul = [c for c in compact if c in DIGIT_WORD]
    if hangul and not any(c.isdigit() for c in compact) and len(hangul) < 2:
        return ""
    return "".join(c if c.isdigit() else DIGIT_WORD[c] for c in compact)


_CMP = re.compile(r"[\s.,!?~…·'\"\-]+")


def norm_compare(text: str) -> str:
    """커밋 시 추측 받아쓰기와 확정 받아쓰기를 비교하는 형태."""
    return _CMP.sub("", text).lower()


# ───────────────────────── 함수 인자 ─────────────────────────

_PLACEHOLDER = {"", "unknown", "none", "null", "n/a", "na", "없음", "모름", "미상", "?", "-", "tbd"}


def normalize_tool_args(args: dict, s=None) -> dict:
    strip_rules = getattr(s, "tool_arg_strip", {}) if s is not None else {}
    upper = getattr(s, "tool_arg_upper", []) if s is not None else []
    out = {}
    for k, v in args.items():
        if isinstance(v, str):
            v = v.strip()
            rule = strip_rules.get(k) or strip_rules.get("*")
            if rule in ("hyphen_space", "all"):
                v = re.sub(r"[\s\-]", "", v)
            elif rule == "space":
                v = re.sub(r"\s", "", v)
            if k in upper or "*" in upper:
                v = v.upper()
        out[k] = v
    return out


def invalid_tool_arg(args: dict, s=None) -> str | None:
    """최소 검증만 한다: 빈 값, 자리 채움 값, 명백한 자릿수 미달."""
    min_digits = getattr(s, "tool_arg_min_digits", {}) if s is not None else {}
    for k, v in args.items():
        if isinstance(v, str) and v.strip().lower() in _PLACEHOLDER:
            return f"{k}: 값 없음"
        if k in min_digits and isinstance(v, str):
            if len(re.sub(r"\D", "", v)) < int(min_digits[k]):
                return f"{k}: 자릿수 부족"
    return None
