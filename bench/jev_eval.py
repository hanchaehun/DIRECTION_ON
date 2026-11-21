"""Jev 판정 검증: 연습 문항의 손님 말(원문)에 판정을 돌려 '은행 직행'이 정답 응대와 맞는지 본다.

    python bench/jev_eval.py [--limit 0] [--only ...] [--concurrency 6]

정답 분류는 실제 상담 기록(reference)에서 규칙으로 뽑는다:
  함수 호출이 있으면 TOOL, 없으면 답 문구로 ASK_ID / PURCHASE / ASK_PRODUCT / ASK_DETAIL / GREETING / OTHER
결과: runtime/jev_eval/<시각>/results.jsonl, 화면 요약 (오발화 = 은행으로 답했는데 정답과 다른 경우)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from agent.config import RUNTIME, Settings  # noqa: E402
from agent.router import Router, decide_bank, pick_opener  # noqa: E402
from agent.session import normalize_history  # noqa: E402

OFFICIAL = ROOT.parent / "directon-2026-voice-cs"
BANK_CLASS = {"ask_order_or_contact": "ASK_ID", "ask_order_or_phone": "ASK_ID", "ask_name_phone_return": "ASK_ID",
              "ask_order_number": "ASK_ID", "purchase_guide": "PURCHASE", "ask_product": "ASK_PRODUCT",
              "ask_detail": "ASK_DETAIL", "greeting": "GREETING"}


def expected_class(case: dict) -> str:
    ref = case["reference"]
    if ref["tool_calls"]:
        return "TOOL:" + ref["tool_calls"][0]["name"]
    ans = re.sub(r"\s+", " ", ref.get("answer") or "")
    body = re.sub(r"^안녕하세요, 고객님!.*?하루온입니다 :\)\s*", "", ans)
    if re.search(r"제품 구매는 아래 링크", body):
        return "PURCHASE"
    if re.search(r"문의 내용을 조금 더 자세히", body):
        return "ASK_DETAIL"
    if re.search(r"(주문번호|성함|전화번호|연락처)[^.]{0,40}(말씀|알려|남겨)", body) and len(body) < 130:
        return "ASK_ID"
    if re.search(r"(어떤 제품|제품명|상품명|제품 특징)[^.]{0,50}(알려|말씀)", body) or re.search(r"안내 진행해 드려도 괜찮을까요", body):
        return "ASK_PRODUCT"
    if re.fullmatch(r"\s*(안녕하세요|여보세요)[.!?~ ]*", case["customer_utterance"]):
        return "GREETING"
    return "OTHER"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OFFICIAL / "data" / "practice_cases.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    s = Settings.load()
    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8") if l.strip()]
    if a.only:
        want = set(a.only.split(","))
        cases = [c for c in cases if c["case_id"] in want]
    if a.limit:
        cases = cases[: a.limit]
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(timeout=20.0) as http:
        router = Router(http, s)

        async def one(c):
            async with sem:
                hist = normalize_history(c.get("history"))
                text = re.sub(r"\s+", " ", c["customer_utterance"]).strip()
                r = await router.classify(hist, text, c.get("customer_profile"), timeout=8.0)
                bank = decide_bank(r, text, hist, c.get("customer_profile"), s)
                exp = expected_class(c)
                if bank is None:
                    verdict = "miss" if exp in ("ASK_ID", "PURCHASE", "ASK_PRODUCT", "ASK_DETAIL", "GREETING") else "llm"
                else:
                    verdict = "hit" if BANK_CLASS[bank] == exp else "WRONG"
                row = {"case_id": c["case_id"], "text": text, "intent": r.intent, "conf": round(r.conf, 3),
                       "complete": round(r.complete, 3), "upset": round(r.upset, 3), "ms": r.ms, "bank": bank,
                       "expected": exp, "verdict": verdict, "opener": pick_opener(r, text, s), "error": r.error,
                       "ref_answer": re.sub(r"\s+", " ", c["reference"].get("answer") or "")[:200]}
                print(f"{row['case_id']:9} {verdict:5} {r.ms:4}ms c={r.complete:.2f} u={r.upset:.2f} {r.intent}({r.conf:.2f}) "
                      f"bank={bank} exp={exp} | {text[:44]}", flush=True)
                return row

        rows = await asyncio.gather(*(one(c) for c in cases))
    out_dir = RUNTIME / "jev_eval" / time.strftime("%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ms = [r["ms"] for r in rows if not r["error"]]
    summ = {"cases": len(rows), "errors": sum(1 for r in rows if r["error"]),
            "ms_p50": statistics.median(ms) if ms else None, "ms_p90": sorted(ms)[int(len(ms) * 0.9)] if ms else None,
            "bank_fired": sum(1 for r in rows if r["bank"]), "hit": sum(1 for r in rows if r["verdict"] == "hit"),
            "WRONG": sum(1 for r in rows if r["verdict"] == "WRONG"), "miss": sum(1 for r in rows if r["verdict"] == "miss"),
            "complete_lt_0.5": sum(1 for r in rows if r["complete"] < 0.5),
            "complete_lt_0.3": sum(1 for r in rows if r["complete"] < 0.3)}
    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(summ, ensure_ascii=False))
    for r in rows:
        if r["verdict"] in ("WRONG", "miss") or r["complete"] < 0.5:
            print(f"\n[{r['verdict']}] {r['case_id']} intent={r['intent']}({r['conf']}) complete={r['complete']} bank={r['bank']} exp={r['expected']}"
                  f"\n   손님: {r['text'][:110]}\n   정답: {r['ref_answer'][:150]}")
    print(f"\n결과: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
