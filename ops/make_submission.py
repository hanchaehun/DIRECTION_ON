"""제출 ZIP 만들기 (공식 README 7장).

    python ops/make_submission.py [--out ../submission/haruon_voice_agent.zip] [--no-env]

ZIP 맨 위에 start_server.sh, start_mock.sh, requirements.txt, .env 가 오도록 묶는다.
- 넣는 것: agent/, mock_tools/, data/(brand, mock_db.json, examples.json), demo/, bench/, 시작 스크립트, README_SUBMIT.md
- 미리 합성한 첫마디 음성: runtime/tts_cache/<현재 TTS 설정 폴더>/ (운영진 노트북에서 시작할 때 다시 합성하지 않게)
- 빼는 것: __pycache__, runtime/의 로그·기록, data/legacy_starter, tests/, .venv
- .sh 파일은 LF 줄바꿈인지 확인한다 (CRLF면 리눅스 bash에서 실패)
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INCLUDE_DIRS = ["agent", "mock_tools", "demo", "bench"]
INCLUDE_FILES = ["start_server.sh", "start_mock.sh", "requirements.txt", "README_SUBMIT.md", ".env.example",
                 "data/mock_db.json", "data/examples.json"]
SKIP_PARTS = {"__pycache__", ".venv", "venv", ".git"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "submission" / "haruon_voice_agent.zip"))
    ap.add_argument("--no-env", action="store_true", help=".env를 넣지 않는다 (시험 묶음용)")
    a = ap.parse_args()

    from agent.config import Settings
    s = Settings.load()

    for sh in ("start_server.sh", "start_mock.sh"):
        if b"\r\n" in (ROOT / sh).read_bytes():
            print(f"오류: {sh} 가 CRLF 줄바꿈입니다. LF로 바꾸세요.")
            return 1
    env_path = ROOT / ".env"
    if not a.no_env:
        if not env_path.exists():
            print("오류: .env 가 없습니다 (API 키를 넣어 함께 제출해야 합니다).")
            return 1
        env = env_path.read_text(encoding="utf-8")
        missing = [k for k in ("SONIOX_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
                   if not any(ln.startswith(k + "=") and ln.split("=", 1)[1].strip() for ln in env.splitlines())]
        if missing:
            print(f"경고: .env 에 비어 있는 키 {missing}")
        for k in ("AGENT_STT_BACKEND", "AGENT_LLM_BACKEND", "AGENT_TTS_BACKEND"):
            if k + "=" not in env:
                print(f"경고: .env 에 {k} 가 없습니다 (기본값 fake로 뜹니다)")
        for k in ("AGENT_RECORD_STT=1", "AGENT_RECORD_STT=true", "AGENT_LOG_TEXT=1", "AGENT_LOG_TEXT=true"):
            if k in env:
                print(f"경고: .env 에 {k} — 최종 측정에서 시험 문항 내용을 디스크에 남깁니다. 끄세요.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files: list[tuple[Path, str]] = []
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not (SKIP_PARTS & set(p.parts)) and p.suffix != ".pyc":
                files.append((p, p.relative_to(ROOT).as_posix()))
    for p in sorted((ROOT / "data" / "brand").glob("*")):
        files.append((p, p.relative_to(ROOT).as_posix()))
    for f in INCLUDE_FILES:
        if (ROOT / f).exists():
            files.append((ROOT / f, f))
    env_text = None
    if not a.no_env:
        # 제출용 .env: 로컬 개발 전용 설정(아래 '로컬 개발 전용' 표시 아래의 줄, DECISIONS_*)은 넣지 않는다 → 제출본은 게이트웨이만 쓴다
        keep, dev = [], False
        for ln in env_path.read_text(encoding="utf-8").splitlines():
            if ln.strip().startswith("# 로컬 개발 전용"):
                dev = True
            if dev or ln.split("=", 1)[0].strip().startswith("DECISIONS_"):
                continue
            keep.append(ln)
        env_text = "\n".join(keep).rstrip("\n") + "\n"
    voice = s.gemini_voice if s.tts_backend == "gemini" else "fake"
    cache = ROOT / "runtime" / "tts_cache" / f"{s.tts_backend if s.tts_backend == 'gemini' else 'fake'}_{s.gemini_tts_model}_{voice}"
    n_cache = 0
    if cache.exists():
        for p in sorted(cache.glob("*.pcm")):
            files.append((p, p.relative_to(ROOT).as_posix()))
            n_cache += 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in files:
            z.write(p, arc)
        if env_text is not None:
            z.writestr(".env", env_text)
            files.append((env_path, ".env"))
    size = out.stat().st_size / 1e6
    print(f"{out}  ({len(files)}개 파일, {size:.1f}MB, 첫마디 캐시 {n_cache}개: {cache.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
