"""설정. 기본값 → config.json → 환경변수(AGENT_<이름 대문자>) 순서로 덮어쓴다.

API 키와 주소는 .env 또는 환경변수로 준다 (SONIOX_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY ...).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # voice_agent/
RUNTIME = ROOT / "runtime"                            # 로그, 캐시, 측정 결과 (git 제외)
PROMPTS = Path(__file__).resolve().parent / "prompts"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


@dataclass
class Settings:
    # ── 사용할 구현 (키 없이 시험할 때는 fake) ──
    stt_backend: str = "fake"            # fake | soniox
    llm_backend: str = "fake"            # fake | openai
    tts_backend: str = "fake"            # fake | gemini

    # ── 무음과 종료 판단 (5.3절) ──
    silence_rms: float = 300.0           # 측정기와 같은 기준
    adaptive_noise: bool = True          # 말 시작 전 소음 바닥이 높으면 임계값을 올림
    noise_factor: float = 2.5
    speech_start_ms: int = 60            # 말소리가 이만큼 이어져야 말 시작으로 봄
    spec_pause_ms: int = 450             # 완결·미정일 때 추측 시작 (문장 안 쉼 400ms대가 많아 헛된 추측 호출을 줄인다)
    spec_pause_slow_ms: int = 800        # 미완결·서두일 때 추측 시작
    r1_end_silence_ms: int = 300
    r2_complete_silence_ms: int = 900
    r3_incomplete_end_silence_ms: int = 1600
    r4_preamble_silence_ms: int = 2400
    r5_max_silence_ms: int = 2400
    undecided_wait_ms: int = 1200
    # ── 말 끝 판단 (무음 길이 + Jev) ──
    commit_silence_ms: int = 1100        # Jev 판정이 없을 때: 무음이 이만큼이면 말 끝 (첫 소리 약 1.13초, 만점 한도 1.5초)
    commit_comma_ms: int = 2400          # 쉼표·말줄임으로 끝난 말 (이어서 말하는 중)
    commit_noun_ms: int = 2000          # 서술어 없이 명사로 끝난 짧은 말 (이어서 말할 수 있다)
    commit_preamble_ms: int = 1600      # 인사·용건 예고뿐일 때 (뒤에 본론이 이어질 수 있다)
    commit_complete_ms: int = 1000      # Jev가 끝난 말로 볼 때 (측정 모드)
    commit_incomplete_ms: int = 1600     # Jev가 덜 끝난 말로 볼 때 (0.25~0.5)
    commit_very_incomplete_ms: int = 2000  # Jev가 말하다 멈춘 것으로 볼 때 (0.25 미만)
    no_text_commit_ms: int = 1800        # 말소리는 있었는데 받아쓰기가 비어 있을 때
    live_commit_fast_ms: int = 600       # 시연 모드: Jev 확신(0.6 이상)이면 빨리 받아친다
    live_commit_mid_ms: int = 900
    live_router_pause_ms: int = 250      # 시연 모드: Jev를 더 일찍 부른다
    spec_live: bool = True               # 시연 모드 추측 실행 (답까지의 시간 단축)
    spec_measure: bool = False           # 측정 모드에서는 끈다 (첫 소리는 미리 합성한 음성이라 점수에 영향 없음, 호출만 는다)
    tts_unit_chars: int = 70             # 한 번에 합성하는 글자 수 상한 (긴 문장은 TTS가 7초를 넘긴 실측)
    sentence_gap_ms: int = 150           # 문장 사이 숨
    # ── 판정 모델 Jev (운영진 허용, 0.4~0.6초) ──
    use_router: bool = True              # 말 끝·의도 판정에 Jev 사용
    jev_model: str = "jev-1.14.0"          # 고정: 게이트웨이 jev-latest는 1.13, 직접 호출은 1.14로 연결돼 판정이 달랐다
    jev_commit_wait_ms: int = 200        # 말 끝 확정 시점에 판정이 오는 중이면 이만큼 더 기다린다 (확정 기준 1.1초 이하일 때만)
    jev_commit_wait_max_thr_ms: int = 1100
    router_extra_delay_ms: int = 0       # 시험용: 판정 응답에 지연을 더해 느린 경로를 흉내 낸다 (기본 0)
    jev_timeout_s: float = 1.2
    router_pause_ms: int = 450           # 이만큼 조용하면 Jev를 부른다 (결과는 약 0.5초 뒤)
    route_reuse_ratio: float = 0.88      # 받아쓰기가 이만큼 같으면 앞서 받은 Jev 판정을 그대로 쓴다
    bank_min_conf: float = 0.9          # 은행 직행에 필요한 의도 확신도
    bank_agree_conf: float = 0.55        # 키워드 후보가 있을 때 Jev가 같은 판단이라고 보는 최소 확신도
    bank_fuzzy_ratio: float = 0.92      # LLM 문장이 은행 문장과 이만큼 같으면 은행 음성으로 말한다
    upset_min_prob: float = 0.8
    complete_min_prob: float = 0.5       # 이 이상이면 말이 끝난 것으로 본다
    llm_judge: bool = False             # 애매한 끝맺음에 LLM 완결성 판정 (게이트웨이 호출당 마스킹 5크레딧·분당 120건 한도로 기본 끔)
    noise_commit_token_idle_ms: int = 800
    noise_commit_min_voiced_ms: int = 3000   # 소음 경로는 말소리가 이만큼 끊김 없이 이어질 때만
    noise_commit_token_idle_slow_ms: int = 2400
    finalize_wait_ms: int = 300          # 커밋 시 확정 받아쓰기 대기 상한 (4.4절 1~2단계 합계)
    order_id_digit_lengths: list = field(default_factory=lambda: [15])   # 하루온 주문번호 20261101-0001234 (8+7자리)
    phone_digit_lengths: list = field(default_factory=lambda: [10, 11])

    # ── 응답 시각 (4.4절, 피드백 반영: 두 한도 분리) ──
    decision_deadline_s: float = 1.2     # 말 끝 → LLM 결정 대기 한도. 넘으면 중립 첫마디
    answer_filler_deadline_s: float = 1.35       # 측정 모드: 답 음성이 이 시각까지 없으면 중립 첫마디 (1.45는 e2e에서 1.47초 → 여유 부족)
    live_answer_filler_deadline_s: float = 1.8   # 시연 모드
    filler_tail_silence_ms: int = 200    # 첫마디 뒤 숨 쉬는 틈
    no_speech_timeout_s: float = 10.0
    response_watchdog_s: float = 20.0
    tool_timeout_s: float = 5.0
    tool_gap_after_filler_ms: int = 150  # 측정 모드: 첫마디가 측정기에 다 도착한 뒤 함수 호출 (조회 뒤 답 측정 왜곡 방지, 실측으로 확인)
    max_tool_rounds: int = 2
    out_chunk_ms: int = 40               # 측정 모드 전송 단위
    live_lead_ms: int = 300              # 시연 모드: 재생보다 앞서 보내는 양
    barge_in_ms: int = 180               # 시연 모드: 말소리가 이만큼 이어지면 끼어들기
    barge_in_chars: int = 2

    # ── LLM (5.4절) ──
    openai_model: str = "gpt-5.4-mini"
    reasoning_effort: str = "none"       # 1차 판단
    reasoning_effort_round2: str = "none"  # 함수 결과 요약
    max_tokens_none: int = 400           # 피드백 반영: 추론 토큰 포함 상한
    max_tokens_reasoning: int = 1024
    judge_max_tokens: int = 16
    prompt_cache_key: str = "directon-agent-v1"
    cache_keepalive_s: float = 270.0
    llm_timeout_s: float = 8.0
    answer_max_sentences: int = 3        # 전화 답변 길이 제한 (코드로 강제)
    answer_max_chars: int = 130
    gateway_rpm: int = 115               # 게이트웨이 분당 호출 상한 (BAZE 한도 120보다 낮게)
    gateway_jev_reserve: int = 25        # 그중 Jev 몫으로 떼어 두는 수 (일반 호출은 90건까지)
    pii_placeholders: bool = True        # 게이트웨이 개인정보 필터 대응: 번호를 {{PHONE_n}}로 바꿔 보내고 응답에서 되돌린다
    brand_name: str = "하루온"

    # ── TTS (5.6절) ──
    gemini_tts_model: str = "gemini-3.8-flash-tts"   # 공식 지정은 3.8 Flash-Lite이나 행사 게이트웨이(BAZE)에는 3.8 Flash TTS만 있다 (Lite는 404)
    gemini_tts_api: str = "speech"        # speech (BAZE /audio/speech/, 원시 PCM 24k) | interactions (Google 직접) | generate
    gemini_voice: str = "Kore"
    tts_style: str = "차분하고 친절한 고객센터 상담원, 또박또박 자연스러운 속도"   # 3.8: speech_metadata.style (본문과 분리)
    tts_style_prefix: str = ""           # generate 경로에서만: 본문 앞에 붙이는 지시문
    tts_sample_rate: int = 24000         # interactions 출력 요청값. 문서 기본 24k (16k도 요청 가능하나 probe로 확인 후)
    tts_concurrency: int = 4
    tts_first_audio_timeout_s: float = 6.0   # 게이트웨이 TTS는 스트리밍 없이 약 3초 뒤 한 번에 온다
    tts_retries: int = 1
    spec_tts: bool = False               # 추측 합성. 기본 끔 (7.4절 Plan A일 때만)
    spec_tts_after_ms: int = 500
    tts_gain: float = 1.0

    # ── TTS 입력 변환 (피드백 반영: 실측으로 틀리는 것만 켠다) ──
    tts_norm_phone: bool = True
    tts_norm_order_id: bool = True
    tts_norm_id_context: bool = True
    tts_norm_iso_date: bool = True
    tts_norm_tilde: bool = True
    tts_norm_percent: bool = True
    tts_norm_amount: bool = False
    tts_norm_kdate: bool = False
    tts_prefix_readings: dict = field(default_factory=dict)

    # ── 함수 인자 (피드백 반영: 하이픈 제거는 09:00 데이터 확인 후 켠다) ──
    tool_arg_strip: dict = field(default_factory=dict)   # 예: {"phone_number": "hyphen_space"}
    tool_arg_upper: list = field(default_factory=list)   # 예: ["order_id"]
    tool_arg_min_digits: dict = field(default_factory=dict)  # 예: {"phone_number": 10}

    # ── STT (5.2절) ──
    soniox_url: str = "wss://stt-rt.soniox.com/transcribe-websocket"
    soniox_model: str = "stt-rt-v6"
    endpoint_latency_adjustment_level: int = 1
    endpoint_sensitivity: float = 0.0
    max_endpoint_delay_ms: int = 3000
    stt_use_context: bool = True
    record_stt: bool = False             # 종료 정책 재생 실험용 토큰 기록 (연습 때만 켠다. 시험 문항 내용을 디스크에 남기지 않는다)
    log_text: bool = False               # 세션 로그에 받아쓰기·답 문장을 남길지 (연습 때만 켠다)
    persist_dynamic_tts: bool = False    # 실시간 답 음성을 디스크 캐시에 저장할지 (첫마디·정형 문장은 항상 저장)
    live_agent_text: bool = False        # 시연 모드 상담원 텍스트 입력 (실험)
    agent_text_lead_ms: int = 700

    # ── 가짜 구현 (키 없이 시험) ──
    fake_cases: str = ""                 # case_id → customer_utterance (jsonl)
    fake_audio_dir: str = ""             # case_id.wav
    fake_stt_adversarial: bool = True    # 쉼마다 종료 신호를 내는 공격적인 가짜 STT
    fake_stt_lag_ms: int = 120
    fake_endpoint_ms: int = 450
    fake_llm_latency_ms: int = 550
    fake_judge_latency_ms: int = 250
    fake_tts_ttfb_ms: int = 700

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        load_dotenv()
        s = cls()
        cfg = Path(path) if path else ROOT / "config.json"
        if cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k.startswith("_"):
                    continue
                if not hasattr(s, k):
                    raise KeyError(f"{cfg.name}: 알 수 없는 설정 '{k}'")
                setattr(s, k, v)
        for f in fields(s):
            raw = os.environ.get("AGENT_" + f.name.upper())
            if raw is None:
                continue
            cur = getattr(s, f.name)
            if isinstance(cur, bool):
                val = raw.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            elif isinstance(cur, (dict, list)):
                val = json.loads(raw)
            else:
                val = raw
            setattr(s, f.name, val)
        return s

    def public(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
