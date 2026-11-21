"""텍스트 전용 평가: TTS를 건너뛰고 서버와 같은 판단 흐름(은행 직행 → 첫마디 → LLM → 함수 → 답 검증)을 돌려
응대·함수 점수를 낸다. 프롬프트와 규칙을 빠르게 고치는 용도.

    python bench/text_eval.py [--stt] [--limit 20] [--only P0017-02,...] [--every 2] [--judge-model gpt-5.5] [--effort low] [--tag v1]

--stt           손님 말로 원문 대신 최근 bench/stt_eval.py 받아쓰기를 쓴다 (숫자 오류까지 포함한 실제 조건)
--judge-model   공식 채점기의 채점 지시문(JUDGE_PROMPT)을 게이트웨이의 이 모델로 돌린다. 공식 채점은 gpt-5.5.
                주지 않으면 채점하지 않고 함수 점수와 답만 낸다.
결과: runtime/text_eval/<시각>_<tag>/results.jsonl, summary.json
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from agent import aftertool, facts, knowledge  # noqa: E402
from agent.brain import Brain, OpenAIBackend, parse_now  # noqa: E402
from agent.clips import ClipBank, load_phrases  # noqa: E402
from agent.config import RUNTIME, Settings, env  # noqa: E402
from agent.router import Router, choose_bank, choose_opener  # noqa: E402
from agent.session import normalize_history  # noqa: E402
from agent.toolflow import GUARD_NOTE, claims_lookup_without_evidence, evidence_text, execute_calls  # noqa: E402
from mock_tools.server import Store  # noqa: E402

OFFICIAL = ROOT.parent / "directon-2026-voice-cs"
PTS = {"맞음": 1.0, "괜찮음": 0.7, "틀림": 0.0}


def load_score_module(path: Path):
    spec = importlib.util.spec_from_file_location("official_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def run_case(case, text: str, brain: Brain, router: Router, clips: ClipBank, store: Store, s: Settings, effort: str) -> dict:
    hist = normalize_history(case.get("history"))
    caller, now = case.get("customer_profile"), parse_now(case["now"])
    ctx = facts.context_ids(hist, caller)
    out = {"case_id": case["case_id"], "utterance": case["customer_utterance"], "heard": text, "tool_calls": [], "notes": []}
    route = await router.classify(hist, text, caller, timeout=8.0) if s.use_router else None
    out["route"] = None if route is None else {"intent": route.intent, "conf": round(route.conf, 2), "complete": round(route.complete, 2),
                                               "upset": round(route.upset, 2), "ms": route.ms, "error": route.error}
    bank, why = choose_bank(text, hist, caller, route, s)
    if bank:
        out.update(path=f"bank:{bank}({why})", said=clips.bank_text(bank), answer=clips.bank_text(bank), llm_calls=0, decide_s=0.0)
        return out
    kind = choose_opener(text, route, s)
    opener = (clips.phrases.get(kind) or [""])[0]
    tools = knowledge.local_tools()
    msgs = brain.build_messages(hist, text, caller, now, ctx=ctx, spoken=opener)
    t0 = time.perf_counter()
    d = await brain.decide(msgs, tools, effort)
    out["decide_s"] = round(time.perf_counter() - t0, 3)
    llm_calls, rounds, evid = 1, 0, ""
    for h in hist:
        if h.get("role") == "tool_result":
            evid += facts.render_tool_result(str(h.get("name")), h.get("result"), now) + "\n"

    async def call_fn(name, args):
        res, _ = store.handle(name, args) if name in store.defs else ({"status": "error", "message": "unknown function"}, False)
        return {"status_code": 200, "body": res, "ms": 0}

    guard_used = False
    while True:
        if d.kind == "answer" and not guard_used and claims_lookup_without_evidence(d.text, evid):
            guard_used = True
            out["notes"].append("조회 없이 확인했다고 답하려 함 → 다시 판단")
            msgs = msgs + [d.assistant_msg, {"role": "system", "content": GUARD_NOTE}]
            d = await brain.decide(msgs, tools, effort)
            llm_calls += 1
            continue
        if not (d.kind == "tool" and rounds < s.max_tool_rounds):
            break
        runs = await execute_calls(d.tool_calls, ctx, call_fn, s, now)
        msgs = msgs + [d.assistant_msg]
        new_hist = []
        for run in runs:
            msgs.append({"role": "tool", "tool_call_id": run.tc.id, "content": run.content})
            for (name, args, _, _) in run.calls:
                out["tool_calls"].append({"name": name, "arguments": args, "matched": (run.body or {}).get("status") == "ok"})
            if run.blocked:
                out["notes"].append(f"차단 {run.tc.name}: {run.blocked[:40]}")
            out["notes"] += run.notes
            new_hist += [{"role": "tool_call", "name": run.tc.name, "arguments": run.args},
                         {"role": "tool_result", "name": run.tc.name, "result": run.body}]
        evid += evidence_text(runs) + "\n"
        ctx = facts.context_ids(hist + new_hist, caller)
        rounds += 1
        msgs.append({"role": "system", "content": aftertool.note()})
        d = await brain.decide(msgs, tools, s.reasoning_effort_round2)
        if d.kind == "answer":
            d.text = aftertool.expand(d.text)
        llm_calls += 1
    if d.kind == "answer":
        ans = facts.clamp_answer(d.text, s.answer_max_sentences, s.answer_max_chars, is_bank=lambda x: clips.match_text(x) is not None)
        ans, fixes = facts.verify_answer(ans, evid, now)
        out["notes"] += fixes
        out["raw_answer"] = d.text
    else:
        ans = f"[오류: {d.error[:120]}]"
    out.update(path="llm", answer=ans, said=(opener + " " + ans).strip(), llm_calls=llm_calls,
               bank_sentences=sum(1 for x in facts.split_sentences(ans) if clips.match_text(x) is not None),
               sentences=len(facts.split_sentences(ans)))
    return out


async def gateway_judge(http, score, case, said, calls, ctx, model: str) -> dict:
    """공식 채점 지시문을 게이트웨이 모델로 돌린다 (공식 채점기는 api.openai.com의 gpt-5.5)."""
    ref = case["reference"]
    ref_txt = "".join(f"({t['name']} 함수 호출, 인수: {json.dumps(t['arguments'], ensure_ascii=False)})\n" for t in ref["tool_calls"])
    ref_txt += "".join(f"({t['name']} 함수 결과: {json.dumps(t.get('result'), ensure_ascii=False)[:1500]})\n" for t in ref.get("tool_results", []))
    ref_txt += ref.get("answer") or ""
    ag = "".join(f"({t['name']} 함수 호출, 인수: {json.dumps(t['arguments'], ensure_ascii=False)}, 조회 성공: {t['matched']})\n" for t in calls)
    ag += "말한 내용: " + (said or "(아무 말도 하지 않음)")
    prompt = score.JUDGE_PROMPT.format(scenarios=ctx["scenarios"], info=ctx["info"], now=case["now"],
                                       caller=json.dumps(case.get("customer_profile"), ensure_ascii=False),
                                       history=score.fmt_history(case["history"]), utterance=case["customer_utterance"],
                                       reference=ref_txt, rubric="", agent=ag)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": "low",
            "response_format": {"type": "json_schema", "json_schema": {"name": "v", "strict": True, "schema": {
                "type": "object", "additionalProperties": False, "required": ["verdict", "reason"],
                "properties": {"verdict": {"type": "string", "enum": list(PTS)}, "reason": {"type": "string"}}}}}}
    url = env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions/"
    last = ""
    for i in range(3):
        try:
            r = await http.post(url, json=body, headers={"Authorization": f"Bearer {env('OPENAI_API_KEY')}"}, timeout=120)
            if r.status_code == 200:
                return json.loads(r.json()["choices"][0]["message"]["content"])
            last = f"{r.status_code} {r.text[:120]}"
            if r.status_code == 400 and "reasoning_effort" in r.text:
                body.pop("reasoning_effort", None)
        except Exception as e:
            last = repr(e)[:120]
        await asyncio.sleep(1.5 * (i + 1))
    return {"verdict": None, "reason": f"채점 실패 {last}"}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(OFFICIAL / "data" / "practice_cases.jsonl"))
    ap.add_argument("--score", default=str(OFFICIAL / "eval" / "score.py"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="N개마다 하나만")
    ap.add_argument("--only", default="")
    ap.add_argument("--stt", action="store_true")
    ap.add_argument("--judge-model", default="")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()

    s = Settings.load()
    s.llm_backend = "openai"
    if a.no_router:
        s.use_router = False
    effort = a.effort or s.reasoning_effort
    phrases = load_phrases()
    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8") if l.strip()]
    if a.only:
        want = set(a.only.split(","))
        cases = [c for c in cases if c["case_id"] in want]
    cases = cases[:: max(1, a.every)]
    if a.limit:
        cases = cases[: a.limit]
    heard = {}
    if a.stt:
        files = sorted(glob.glob(str(RUNTIME / "stt_eval" / "*" / "results.jsonl")))
        if not files:
            print("받아쓰기 결과가 없습니다. 먼저 bench/stt_eval.py 를 돌리세요.")
            return 1
        heard = {json.loads(l)["case_id"]: json.loads(l)["final"] for l in open(files[-1], encoding="utf-8")}
    store = Store(json.loads((ROOT / "data" / "mock_db.json").read_text(encoding="utf-8")),
                  json.loads((ROOT / "data" / "brand" / "tools.json").read_text(encoding="utf-8")))
    score = load_score_module(Path(a.score))
    ctx = {"scenarios": (OFFICIAL / "data" / "brand" / "scenarios.md").read_text(encoding="utf-8"),
           "info": "\n".join(f"{i['tag']}\n{i['content']}" for i in knowledge.info_items()), "rubric": {}}
    out_dir = RUNTIME / "text_eval" / f"{time.strftime('%H%M%S')}{'_' + a.tag if a.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(a.concurrency)

    async with httpx.AsyncClient(timeout=60.0) as http:
        brain = Brain(s, OpenAIBackend(http, s), phrases)
        router = Router(http, s)
        clips = ClipBank(None, phrases, s)

        async def one(case):
            async with sem:
                text = re.sub(r"\s+", " ", heard.get(case["case_id"]) or case["customer_utterance"]).strip()
                r = await run_case(case, text, brain, router, clips, store, s, effort)
                ok, why = score.tool_match(case["reference"]["tool_calls"], r["tool_calls"], case.get("history", []))
                r.update(tool_ok=ok, tool_why=why, ref_calls=case["reference"]["tool_calls"],
                         ref_answer=re.sub(r"\s+", " ", case["reference"].get("answer") or "")[:400])
                if a.judge_model:
                    v = await gateway_judge(http, score, case, r["said"], r["tool_calls"], ctx, a.judge_model)
                    r.update(verdict=v.get("verdict"), reason=v.get("reason"))
                print(f"{r['case_id']:9} {r.get('verdict') or '-':3} 함수{'O' if ok else 'X'} {r['path'][:28]:28} | "
                      f"{' '.join(c['name'] for c in r['tool_calls']) or '-':16} | {r['said'][:64]}", flush=True)
                return r

        rows = await asyncio.gather(*(one(c) for c in cases))

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    llm = [r for r in rows if r["path"] == "llm"]
    summ = {"cases": len(rows), "input": "stt" if a.stt else "text", "effort": effort,
            "tool_score": round(statistics.mean(1.0 if r["tool_ok"] else 0.0 for r in rows) * 100, 1),
            "bank_direct": sum(1 for r in rows if r["path"].startswith("bank")),
            "decide_p50_s": round(statistics.median(r["decide_s"] for r in llm), 3) if llm else None,
            "answer_chars_p50": statistics.median(len(r["answer"]) for r in llm) if llm else None,
            "answer_chars_max": max((len(r["answer"]) for r in llm), default=0),
            "bank_sentence_share": round(sum(r.get("bank_sentences", 0) for r in llm) / max(1, sum(r.get("sentences", 0) for r in llm)), 2)}
    if a.judge_model:
        judged = [r for r in rows if r.get("verdict") in PTS]
        summ["judge_model"] = a.judge_model
        summ["response_score"] = round(statistics.mean(PTS[r["verdict"]] for r in judged) * 100, 1) if judged else None
        summ["verdicts"] = {k: sum(r.get("verdict") == k for r in rows) for k in PTS}
    (out_dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(summ, ensure_ascii=False))
    print(f"결과: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
