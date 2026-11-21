"""모의 함수 서버 (공식 SPEC.md 「모의 함수 서버」 규칙).

    python mock_tools/server.py --port 8100 --db data/mock_db.json [--tools data/brand/tools.json]

GET  /tools                 함수 목록 (OpenAI tools 형식: [{"type":"function","function":{...}}])
POST /tools/{함수명}         body = 인수 JSON, 응답 = 결과 JSON. 정의에 없는 함수는 404
GET  /calls?session_id=     (참고용) 받은 호출 기록. 채점에는 쓰지 않는다
GET  /health

데이터: {함수명: [{"arguments": {...}, "result": {...}}]}
- 조회 함수(get_order_info, get_return_info): 전화번호·주문번호가 맞는 기록의 result. 하이픈·공백 무시.
  맞는 기록이 없으면 {"status":"not_found"}. 번호가 하나도 없으면 {"status":"invalid"}.
- 접수 함수(cancel_order, return_order, return_order_item, exchange_order, request_callback):
  필수 값이 다 있으면 {"status":"ok"} (같은 번호의 기록이 있으면 그 result, 예: already_requested),
  빠졌으면 {"status":"invalid"}.
- 검색 함수(search_products): 검색어가 정확히 같지 않아도 가장 가까운 기록.
최종 측정 때는 운영진이 주는 mock_db_test.json을 --db로 받는다. 측정기 중계가 호출 기록과 지연을 맡는다.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]

LOOKUP = {"get_order_info", "get_return_info"}
ACCEPT = {"cancel_order", "return_order", "return_order_item", "exchange_order", "request_callback"}
SEARCH = {"search_products"}
ID_KEYS = ("order_id", "order_number", "buyer_cellphone", "customer_phone_number", "phone_number")
ACCEPT_MSG = {"cancel_order": "주문 취소 접수 성공", "return_order": "주문 반품 접수 성공",
              "return_order_item": "상품 반품 접수 성공", "exchange_order": "주문 교환 접수 성공",
              "request_callback": "고객 정보 접수 성공"}
NOT_FOUND_MSG = {"get_order_info": "주문 정보가 없습니다.", "get_return_info": "반품 상품이 없습니다."}


def digits(v) -> str:
    return re.sub(r"\D", "", str(v)) if v not in (None, "") else ""


def empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip() or v.strip().lower() in ("null", "none", "unknown")
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


class Store:
    def __init__(self, db: dict, tools: list):
        self.db = db
        self.tools = [t.get("function", t) for t in tools]
        self.defs = {t["name"]: t for t in self.tools}
        self.calls: list = []
        self.lock = threading.Lock()

    # ── 공통 ──
    def required_missing(self, name: str, args: dict) -> list[str]:
        params = self.defs[name].get("parameters") or {}
        miss = [k for k in params.get("required", []) if empty(args.get(k))]
        if name == "return_order_item" and not miss:
            items = args.get("return_items")
            if not isinstance(items, list) or not all(
                    isinstance(x, dict) and not empty(x.get("order_item_code")) and not empty(x.get("quantity")) for x in items):
                miss.append("return_items")
        return miss

    @staticmethod
    def ids_of(args: dict) -> set[str]:
        out = {digits(args.get(k)) for k in ID_KEYS}
        for it in args.get("return_items") or []:
            if isinstance(it, dict):
                out.add(digits(it.get("order_item_code")))
        return {x for x in out if len(x) >= 4}

    @staticmethod
    def latest(rec: dict) -> str:
        data = (rec.get("result") or {}).get("data") or {}
        rows = data.get("orders") or data.get("반품정보") or []
        dates = [str(r.get("주문일자") or r.get("주문번호") or "") for r in rows if isinstance(r, dict)]
        return max(dates) if dates else ""

    # ── 조회 ──
    def lookup(self, name: str, args: dict) -> tuple[dict, bool]:
        want = self.ids_of(args)
        if not want:
            return {"status": "invalid", "message": "전화번호 또는 주문번호가 필요합니다."}, False
        # 맞는 번호가 가장 많은 기록. 같으면 파일에서 먼저 나온 기록 (같은 손님의 기록이 여러 개일 때 정답 기록과 맞았다)
        best, best_n = None, 0
        for rec in self.db.get(name, []):
            n = len(want & self.ids_of(rec.get("arguments") or {}))
            if n > best_n:
                best, best_n = rec, n
        if best is not None:
            return best["result"], best["result"].get("status") == "ok"
        # 주문번호가 다른 기록(전화번호 조회 결과) 안에만 있는 경우: 그 주문만 돌려준다
        if name == "get_order_info":
            for rec in sorted(self.db.get(name, []), key=self.latest, reverse=True):
                res = rec.get("result") or {}
                orders = (res.get("data") or {}).get("orders") or []
                hit = [o for o in orders if digits(o.get("주문번호")) in want or digits(o.get("상세주문번호")) in want]
                if hit:
                    return {**res, "data": {**res["data"], "orders": hit}}, True
        return {"status": "not_found", "message": NOT_FOUND_MSG.get(name, "기록이 없습니다.")}, False

    # ── 접수 ──
    def accept(self, name: str, args: dict) -> tuple[dict, bool]:
        miss = self.required_missing(name, args)
        if miss:
            return {"status": "invalid", "message": f"필수 값 누락: {', '.join(miss)}"}, False
        want = self.ids_of(args)
        for rec in self.db.get(name, []):
            if want & self.ids_of(rec.get("arguments") or {}):
                return rec["result"], True
        body = {"status": "ok", "message": ACCEPT_MSG.get(name, "접수 성공")}
        if name == "request_callback":
            body["data"] = {"성공여부": True}
        return body, True

    # ── 검색 ──
    @staticmethod
    def _norm_q(s: str) -> str:
        return re.sub(r"[\s\-_,.?!]+", "", str(s or "")).replace("하루온", "").lower()

    def search(self, name: str, args: dict) -> tuple[dict, bool]:
        q = self._norm_q(args.get("query"))
        recs = self.db.get(name, [])
        if not q:
            return {"status": "invalid", "message": "검색어가 필요합니다."}, False
        if not recs:
            return {"status": "not_found", "message": "검색 결과가 없습니다."}, False

        def score(rec):
            rq = self._norm_q((rec.get("arguments") or {}).get("query"))
            names = " ".join({str(x.get("제품명", "")) for x in ((rec.get("result") or {}).get("data") or {}).get("결과", [])
                              if isinstance(x, dict)})
            nn = self._norm_q(names)
            grams = lambda s: {s[i:i + 2] for i in range(max(1, len(s) - 1))}
            gq = grams(q)
            jac = len(gq & grams(rq)) / max(1, len(gq | grams(rq)))
            name_hit = len(gq & grams(nn)) / max(1, len(gq))
            return jac + 0.6 * name_hit + 0.3 * difflib.SequenceMatcher(None, q, rq).ratio()

        best = max(recs, key=score)
        return best["result"], True

    def handle(self, name: str, args: dict) -> tuple[dict, bool]:
        if name in LOOKUP:
            return self.lookup(name, args)
        if name in SEARCH:
            return self.search(name, args)
        if name in ACCEPT or name in self.defs:
            return self.accept(name, args)
        return {"status": "error", "message": "unknown"}, False


def make_handler(store: Store, quiet: bool):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"        # 연결 재사용

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path.rstrip("/") == "/tools":
                return self._send(200, [{"type": "function", "function": t} for t in store.tools])
            if u.path == "/calls":
                sid = parse_qs(u.query).get("session_id", [None])[0]
                with store.lock:
                    return self._send(200, [c for c in store.calls if sid is None or c["session_id"] == sid])
            if u.path == "/health":
                return self._send(200, {"ok": True, "tools": len(store.tools),
                                        "records": {k: len(v) for k, v in store.db.items()}})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            m = re.fullmatch(r"/tools/(\w+)/?", urlparse(self.path).path)
            if not m or m.group(1) not in store.defs:
                return self._send(404, {"error": "unknown function"})
            name = m.group(1)
            try:
                args = json.loads(raw or b"{}")
            except ValueError:
                return self._send(400, {"status": "invalid", "message": "body must be JSON"})
            if not isinstance(args, dict):
                args = {}
            t0 = time.time()
            try:
                result, matched = store.handle(name, args)
            except Exception as e:           # 어떤 입력에도 서버가 죽지 않게
                result, matched = {"status": "error", "message": repr(e)[:200]}, False
            sid = self.headers.get("X-Session-Id")
            with store.lock:
                store.calls.append({"session_id": sid, "name": name, "arguments": args, "matched": matched,
                                    "status": result.get("status"), "t_request": t0, "t_response": time.time()})
            if not quiet:
                print(f"[call] sid={sid} {name}({json.dumps(args, ensure_ascii=False)}) -> {result.get('status')}", flush=True)
            self._send(200, result)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--db", "--data", dest="db", default=str(ROOT / "data" / "mock_db.json"))
    ap.add_argument("--tools", default=str(ROOT / "data" / "brand" / "tools.json"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    db_path = Path(a.db)
    if not db_path.exists():
        print(f"데이터 파일 없음: {db_path}", file=sys.stderr)
        sys.exit(2)
    store = Store(json.loads(db_path.read_text(encoding="utf-8")),
                  json.loads(Path(a.tools).read_text(encoding="utf-8")))
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(store, a.quiet))
    srv.daemon_threads = True
    print(f"READY mock http://{a.host}:{a.port} tools={len(store.tools)} "
          f"records={sum(map(len, store.db.values()))} db={db_path}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
