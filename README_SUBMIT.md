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
| `GEMINI_API_KEY` | Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`, 게이트웨이 `/audio/speech/`) |
| `AGENT_STT_BACKEND=soniox`, `AGENT_LLM_BACKEND=openai`, `AGENT_TTS_BACKEND=gemini` | 실제 모델 사용 (없으면 가짜 응답기) |
| `OPENAI_BASE_URL`, `GEMINI_BASE_URL`, `SONIOX_STT_URL` | (선택) 게이트웨이·리전 주소 |

그 밖의 설정은 `agent/config.py`의 기본값이며 `AGENT_<이름 대문자>` 환경 변수로 바꿀 수 있다.

## 구성

```
agent/            상담 서버 (WebSocket :9000)
  server.py       진입점, 공용 자원, 미리 합성한 음성 예열 (포트를 먼저 연다)
  session.py      한 통화: 듣기 → 말 끝 판단 → 첫마디·고정 답 → 판단 → 함수 → 답 검증 → 답 → turn.end
                  측정 모드(채점, 질문 하나)와 시연 모드(여러 턴, 끼어들기)를 session.start의 mode로 나눈다
  stt.py          Soniox 실시간 받아쓰기
  router.py       판정 모델 Jev (말 끝·불만·용건), 고정 답 직행 판단, 첫마디 고르기
  turn.py         인사·서두 판별 등 말 끝 판단 보조 규칙
  pace.py         말하는 사람의 속도·쉼에 맞춘 말 끝 대기 조절 (이전 턴 기록이 있을 때만, 채점에서는 발동하지 않음)
  brain.py        GPT-5.4 mini 호출, 프롬프트·대화 이력 조립, 개인정보 자리표시
  toolflow.py     함수 실행 전후 검사 (번호 형식·맥락 보정, 접수 차단, 재조회, 답 가드)
  facts.py        번호 보정, 조회 결과 요약·날짜 계산, 답 날짜 검증, 길이 제한
  aftertool.py    조회 뒤 답의 첫 문장 목록 (번호로 고르면 미리 합성한 음성으로 바로 나간다)
  directlib.py    직접 답 첫 문장 후보 (미리 합성, 조건 위반 가드)
  clips.py        미리 합성한 문장 창고와 대조 규칙 (숫자·부정어가 다르면 다른 문장)
  tts.py          Gemini 3.8 Flash TTS, 디스크 캐시, 24k→16k 변환
  ratelimit.py    게이트웨이 분당 호출 한도 관리
  knowledge.py    data/brand/ 자료를 프롬프트로
  prompts/        시스템 프롬프트, 업무별 정책 표, 예시, 첫마디·고정 문구
mock_tools/server.py  모의 함수 서버 (:8100)
data/brand/       공식 브랜드 자료 (시나리오, 브랜드 정보, 함수 정의, 말투)
runtime/tts_cache/    미리 합성한 첫마디·정형 문장 음성
```

시험 문항의 음성이나 받아쓰기는 디스크에 저장하지 않는다 (`AGENT_RECORD_STT`, `AGENT_LOG_TEXT` 기본 꺼짐).
