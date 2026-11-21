"""측정기로 연습 문항을 돌리고 우리 기록과 합쳐 보여 준다 (설계 8.2절).

    python bench/replay.py --cases <연습 문항 jsonl> --audio <음성 폴더> \
        --server ws://127.0.0.1:9000 --mock http://127.0.0.1:8100 [--eval-script <공식 측정기>] [--limit 20]

측정기 실행 방법이 공식 측정기와 다르면 --eval-script와 build_cmd()만 고친다.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT_EVAL = ROOT.parent / "DIRECT_ON 2026 Starter Kit_V2" / "DIRECT_ON 2026 Starter Kit" / "eval" / "run_eval.py"
sys.path.insert(0, str(ROOT))
from bench import report  # noqa: E402


def build_cmd(a, out: Path) -> list[str]:
    cmd = [sys.executable, a.eval_script, "--server", a.server, "--cases", a.cases, "--audio", a.audio,
           "--mock", a.mock, "--out", str(out), "--concurrency", str(a.concurrency)]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--server", default="ws://127.0.0.1:9000")
    ap.add_argument("--mock", default="http://127.0.0.1:8100")
    ap.add_argument("--eval-script", default=str(KIT_EVAL))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = ROOT / "runtime" / "replay" / (time.strftime("%H%M%S") + (f"_{a.tag}" if a.tag else "")) / "eval"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run(build_cmd(a, out), cwd=Path(a.eval_script).parent, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(f"측정기 종료 코드 {r.returncode}, {time.time() - t0:.0f}초")
    res = out / "results.jsonl"
    if not res.exists():
        return 1
    return report.main(["--eval", str(res), "--html", str(out.parent / "report.html")])


if __name__ == "__main__":
    sys.exit(main())
