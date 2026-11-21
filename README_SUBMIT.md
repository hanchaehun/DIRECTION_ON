# 하루온 실시간 음성 상담 AI (DIRECT:ON 2026 제출본)

## 실행

```bash
pip install -r requirements.txt
bash start_mock.sh data/mock_db_test.json     # 모의 함수 서버 :8100 (0.0.0.0)
bash start_server.sh                          # 상담 서버 :9000 (0.0.0.0, WebSocket)
```

- 환경: Python 3.11 이상, 리눅스. GPU 불필요. 인터넷 필요(API 호출).
- 패키지: websockets, httpx, numpy, scipy (`requirements.txt`).
- 상담 서버는 `session.start`의 `tools_url`(측정기 함수 중계)로만 함수를 부르고 `X-Session-Id`에 세션 ID를 넣는다.

## 환경 변수 (`.env`)

| 이름 | 뜻 |
|---|---|
| `SONIOX_API_KEY` | Soniox `stt-rt-v6` 실시간 받아쓰기 |
| `OPENAI_API_KEY` | GPT-5.4 mini |
| `GEMINI_API_KEY` | Gemini 3.8 Flash-Lite TTS (`gemini-3.8-flash-lite-tts`) |
| `AGENT_STT_BACKEND=soniox`, `AGENT_LLM_BACKEND=openai`, `AGENT_TTS_BACKEND=gemini` | 실제 모델 사용 (없으면 가짜 응답기) |
| `OPENAI_BASE_URL`, `GEMINI_BASE_URL`, `SONIOX_STT_URL` | (선택) 게이트웨이·리전 주소 |

그 밖의 설정은 `agent/config.py`의 기본값이며 `AGENT_<이름 대문자>` 환경 변수로 바꿀 수 있다.

## 구성

```
agent/            상담 서버 (WebSocket :9000)
  server.py       진입점, 공용 자원, 첫마디 음성 예열
  session.py      한 문항: 듣기 → 말 끝 판단 → 추측 실행 검증 → 첫마디 → 함수 호출 → 답 → turn.end
  turn.py         말 끝 판단 (무음 + Soniox <end> + 완결성 규칙 + LLM 판정)
  stt.py          Soniox 실시간 스트림
  brain.py        GPT-5.4 mini 호출, 프롬프트·대화 이력 조립
  tts.py          Gemini 3.8 Flash-Lite TTS 스트리밍, 24k→16k 변환
  knowledge.py    data/brand/ 자료를 프롬프트로
  prompts/        시스템 프롬프트, 업무별 정책 표, 예시, 첫마디 문구
mock_tools/server.py  모의 함수 서버 (:8100)
data/brand/       공식 브랜드 자료 (시나리오, 브랜드 정보, 함수 정의, 말투)
runtime/tts_cache/    미리 합성한 첫마디·정형 문장 음성
```

시험 문항의 음성이나 받아쓰기는 디스크에 저장하지 않는다 (`AGENT_RECORD_STT`, `AGENT_LOG_TEXT` 기본 꺼짐).
