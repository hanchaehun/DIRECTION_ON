# 하루온 실시간 음성 상담원 (DIRECT:ON 2026)

측정기가 보내는 손님 음성(16kHz PCM, WebSocket)을 받아 말 끝을 판단하고, 1.5초 안에 첫 소리를 내고,
매뉴얼과 조회 결과에 맞게 답하는 상담 서버와 모의 함수 서버. 모든 모델은 행사 크레딧 게이트웨이(BAZE)로 쓴다.

| 역할 | 모델 | 게이트웨이 경로 | 실측 (11/21) |
|---|---|---|---|
| STT | Soniox 실시간 | `wss://…/gateway/soniox/transcribe-websocket` | 연결 70ms |
| LLM | GPT-5.4 mini | `/chat/completions/` | 첫 토큰 1.3~1.7초 |
| TTS | Gemini 3.8 Flash TTS | `/audio/speech/` | 스트리밍 없음, 25~60자 2.8~3.3초 |
| 판정 | Jev (운영진 허용) | `/decisions/` | 0.4~0.6초 |

## 실행

```bash
pip install -r requirements.txt
cp .env.example .env          # 키 입력 (BAZE 게이트웨이 키 하나로 세 모델 모두)
bash start_mock.sh data/mock_db.json     # 모의 함수 서버 :8100
bash start_server.sh                     # 상담 서버 :9000
python demo/serve.py                     # 시연 화면 http://localhost:8080 (마이크 권한 때문에 localhost)
```

**데이터:** 운영진 제공 데이터는 공개 저장소에 올리지 않는다 (대회 규칙). 공식 자료 폴더 `directon-2026-voice-cs/`에서
`data/brand/*`(scenarios.md, information.json, tools.json, style.json), `data/mock_db.json`, `data/examples.json`을
이 폴더의 `data/` 아래 같은 경로로 복사한다. 자세한 것은 `data/README.md`.

## 한 턴의 흐름

1. **듣기**: 음성을 Soniox에 넣고, 무음 0.45초에 Jev로 "말이 끝났는가·의도·불만"을 판정한다.
2. **말 끝 확정**: 무음 1.1초 (Jev가 끝났다고 보면 1.0초, 덜 끝났다고 보면 1.6~2.0초). Soniox `<end>`는 말 도중에 나와 쓰지 않는다.
3. **첫 소리 (약 1.0~1.1초)**: 미리 합성한 음성만 1.5초 안에 낼 수 있다.
   - 의도가 분명하고 답이 정해져 있으면(번호 요청, 구매 안내, 인사) 키워드+Jev가 함께 동의할 때 고정 답으로 바로 답한다.
   - 아니면 맥락에 맞는 첫마디(공감/확인/안내)를 내고 LLM을 부른다.
4. **판단**: GPT-5.4 mini. 함수 인자의 번호는 발신자 번호·이전 대화의 번호에 맞춘다 (손님 음성의 숫자가 틀리게 들리는 경우가 많다).
   접수 함수는 번호가 온전할 때만 실행하고, 조회가 비면 맥락의 다른 번호로 다시 조회한다.
5. **답**: 조회 뒤 첫 문장은 번호로 고르게 해 미리 합성한 문장이 바로 나간다 (조회 뒤 답 1.3~1.6초). 나머지는 문장 단위로 동시에 합성한다.
   조회 결과는 날짜·상태·경과일을 코드로 계산한 사실 목록으로 넘기고, 답의 날짜는 근거와 대조해 고친다.

## 게이트웨이 대응

- 조직에 개인정보 필터가 켜져 있어 전화번호·이메일·일부 날짜가 `[PHONE]`, `[DATE-OF-BIRTH]`로 가려진다.
  LLM에는 번호를 `{{PHONE_n}}` 자리표시로 바꿔 보내고 응답에서 되돌린다. 날짜는 한글로 바꿔 보낸다. TTS에는 번호를 한글로 읽힌다.
- 키당 분당 120건 한도: 모든 호출이 공용 속도 조절(`agent/ratelimit.py`, 기본 100건)을 거치고 429는 재시도한다.

## 측정과 평가

```bash
python bench/stt_eval.py                 # 연습 음성 → Soniox 받아쓰기, 숫자 정확도, <end> 시점
python bench/jev_eval.py                 # Jev 판정과 고정 답 직행의 정확도
python bench/text_eval.py --stt --judge-model gpt-6-luna   # TTS 없이 응대·함수 점수 (받아쓰기 기준)
# 공식 측정기: directon-2026-voice-cs/eval/run_eval.py, score.py
python -m unittest tests.test_units -v
```

## 구조

```
agent/server.py      진입점, 공용 자원, 첫마디·문장 음성 예열(포트를 먼저 연다)
agent/session.py     한 연결의 턴 처리 (측정 모드 / 시연 모드: 여러 턴, 끼어들기)
agent/router.py      Jev 판정, 키워드 규칙, 고정 답·첫마디 선택
agent/brain.py       메시지 조립, GPT-5.4 mini, 개인정보 자리표시
agent/facts.py       번호 맥락 보정, 조회 결과 요약, 답 검증·길이 제한
agent/toolflow.py    함수 실행 흐름 (보정 → 차단 → 재조회 → 요약)
agent/aftertool.py   조회 뒤 첫 문장 번호 선택
agent/clips.py       미리 합성한 음성 (첫마디, 문장 은행, 제품별 첫 문장)
agent/tts.py         Gemini TTS (게이트웨이 /audio/speech/), 24k→16k
agent/stt.py         Soniox 실시간 / 가짜 STT
agent/ratelimit.py   게이트웨이 분당 한도
agent/prompts/       시스템 프롬프트, 업무별 정책 표, 예시, 첫마디·문장 목록
mock_tools/server.py 모의 함수 서버 (SPEC.md 규칙)
bench/               실측·평가 도구
ops/make_submission.py  제출 ZIP
docs/                대회 정리, 설계기획서
```
