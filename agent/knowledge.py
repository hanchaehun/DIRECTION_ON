"""브랜드 자료(data/brand/)를 읽어 프롬프트·STT 컨텍스트용 글로 바꾼다.

- scenarios.md       상담 시나리오 (그대로)
- information.json   브랜드 정보 96개. 전화에서 읽을 수 없는 링크·이모지·채팅 인사말을 걷어 내고 태그와 내용만 남긴다
- style.json         말투 규칙과 첫 인사
- tools.json         함수 정의 (모의 서버에서 못 받을 때의 예비)
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .config import ROOT

BRAND = ROOT / "data" / "brand"

# 정책이 없는 항목: 날짜 비교 C 함수 두 개(정책 문맥 없음), 링크만 있는 항목
_DROP_TAGS = ("[날짜 확인 함수]", "[주문일자 비교 함수]", "[제품정보링크]")
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")
_DECOR = re.compile(r"[▼▲]|:\)|\(\s*\)")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def clean_text(t: str) -> str:
    """전화로 읽을 수 없는 것(링크, 이모지, 장식)과 채팅식 맺음말을 걷어 낸다."""
    t = _URL.sub("", t)
    t = _EMOJI.sub("", t)
    t = _DECOR.sub("", t)
    t = re.sub(r"\n\s*감사합니다\.?\s*$", "", t.strip())
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


@lru_cache(maxsize=1)
def scenarios() -> str:
    t = _read(BRAND / "scenarios.md")
    t = re.sub(r"`?information\.json`?\s*\(브랜드 정보\)", "브랜드 정보", t)
    return t.strip()


@lru_cache(maxsize=1)
def info_items() -> list[dict]:
    p = BRAND / "information.json"
    return json.loads(_read(p)) if p.exists() else []


@lru_cache(maxsize=1)
def brand_info() -> str:
    out = []
    for it in info_items():
        tag = str(it.get("tag", "")).strip()
        if tag in _DROP_TAGS:
            continue
        body = clean_text(str(it.get("content", "")))
        if body:
            out.append(f"## {tag}\n{body}")
    return "\n\n".join(out)


@lru_cache(maxsize=1)
def style() -> dict:
    p = BRAND / "style.json"
    return json.loads(_read(p)) if p.exists() else {}


@lru_cache(maxsize=1)
def local_tools() -> list:
    p = BRAND / "tools.json"
    if not p.exists():
        return []
    # tools_client와 같은 정규화(키 정렬)를 해야 프롬프트 캐시 앞부분이 같다
    out = [{"type": "function", "function": json.loads(json.dumps(t.get("function", t), ensure_ascii=False, sort_keys=True))}
           for t in json.loads(_read(p))]
    return sorted(out, key=lambda t: t["function"]["name"])


@lru_cache(maxsize=1)
def product_names() -> list[str]:
    """제품명 리스트 항목과 브랜드 정보 태그에서 상품명을 모은다 (STT terms용)."""
    names: list[str] = []
    for it in info_items():
        if "제품명 리스트" in str(it.get("tag", "")):
            m = re.search(r"\[(.*)\]", str(it.get("content", "")), re.S)
            if m:
                names += [x.strip() for x in m.group(1).split(",") if x.strip()]
    db = ROOT / "data" / "mock_db.json"
    if db.exists():
        try:
            for rec in json.loads(_read(db)).get("search_products", []):
                for x in ((rec.get("result") or {}).get("data") or {}).get("결과", []):
                    if isinstance(x, dict) and x.get("제품명"):
                        names.append(str(x["제품명"]))
        except (ValueError, AttributeError):
            pass
    seen, out = set(), []
    for n in names:
        n = re.sub(r"\s+", " ", n.replace("&", " & ")).strip()
        k = n.replace(" ", "")
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out
