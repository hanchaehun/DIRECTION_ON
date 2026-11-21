"""모의 함수 서버 호출. 주소는 항상 session.start의 tools_url (측정기 중계)."""
from __future__ import annotations

import json
import time

import httpx


def _canon(x):
    return json.loads(json.dumps(x, ensure_ascii=False, sort_keys=True))


class ToolsClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.cache: dict[str, list] = {}

    async def tools(self, url: str) -> list:
        """함수 정의를 이름순·키순으로 정렬해 둔다 (순서가 바뀌면 프롬프트 캐시가 깨진다)."""
        if url in self.cache:
            return self.cache[url]
        r = await self.http.get(url.rstrip("/") + "/tools", timeout=4.0)
        r.raise_for_status()
        raw = r.json()
        items = raw.get("tools", raw) if isinstance(raw, dict) else raw
        out = []
        for t in items:
            fn = t.get("function", t) if isinstance(t, dict) else None
            if fn and fn.get("name"):
                out.append({"type": "function", "function": _canon(fn)})
        out.sort(key=lambda t: t["function"]["name"])
        self.cache[url] = out
        return out

    async def call(self, url: str, session_id: str, name: str, args: dict, timeout: float) -> dict:
        t0 = time.perf_counter()
        try:
            r = await self.http.post(f"{url.rstrip('/')}/tools/{name}", json=args,
                                     headers={"X-Session-Id": session_id}, timeout=timeout)
            try:
                body = r.json()
            except ValueError:
                body = {"raw": r.text[:500]}
            return {"ok": r.status_code == 200, "status_code": r.status_code, "body": body,
                    "t0": t0, "t1": time.perf_counter()}
        except httpx.HTTPError as e:
            return {"ok": False, "status_code": 0, "body": {"error": repr(e)[:200]}, "t0": t0, "t1": time.perf_counter()}
