#!/usr/bin/env bash
# 상담 서버 시작 (최종 측정 때 운영진이 그대로 실행). 9000번, 0.0.0.0
# 키와 백엔드 설정은 같은 폴더의 .env (SONIOX_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, AGENT_*_BACKEND)
cd "$(dirname "$0")"
exec python3 -m agent.server --host 0.0.0.0 --port 9000
