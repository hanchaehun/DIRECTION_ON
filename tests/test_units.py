"""단위 시험.  python -m unittest tests.test_units -v"""
import unittest
from datetime import datetime

import numpy as np

from agent.audio import LeadTrimmer, StreamResampler, rms
from agent.config import Settings
from agent.textnorm import (for_tts, invalid_tool_arg, norm_compare, normalize_tool_args, parse_sino, sino,
                            trailing_digits)
from agent.turn import TurnView, classify, decide_commit

S = Settings()


class TestTextnorm(unittest.TestCase):
    def test_sino(self):
        self.assertEqual(sino(39800), "삼만 구천팔백")
        self.assertEqual(sino(2026), "이천이십육")
        self.assertEqual(sino(10000), "만")
        self.assertEqual(sino(110000), "십일만")
        self.assertEqual(sino(21), "이십일")

    def test_parse_sino(self):
        self.assertEqual(parse_sino("천이백삼십사"), 1234)
        self.assertEqual(parse_sino("삼만 구천팔백"), 39800)
        self.assertEqual(parse_sino("만"), 10000)

    def test_phone(self):
        self.assertEqual(for_tts("연락처는 010-1234-5678입니다."), "연락처는 공일공 일이삼사 오육칠팔입니다.")
        self.assertEqual(for_tts("01000001234로 연락드릴게요"), "공일공 공공공공 일이삼사로 연락드릴게요")

    def test_order_id(self):
        self.assertEqual(for_tts("주문번호 DEMO1001 확인했습니다."), "주문번호 데모 일공공일 확인했습니다.")
        self.assertEqual(for_tts("ORD2026001 건"), "오알디 이공이육공공일 건")

    def test_iso_date_keeps_other_year(self):
        now = datetime(2026, 11, 21)
        self.assertEqual(for_tts("2026-11-21 출고", now=now), "11월 21일 출고")
        self.assertEqual(for_tts("2025-06-10 주문", now=now), "2025년 6월 10일 주문")

    def test_kdate_off_by_default_and_months(self):
        self.assertEqual(for_tts("6월 10일"), "6월 10일")
        s = Settings()
        s.tts_norm_kdate = True
        self.assertEqual(for_tts("6월 10일, 10월 3일", s), "유월 십일, 시월 삼일")

    def test_tilde_percent_amount(self):
        self.assertEqual(for_tts("3~5일 소요"), "3에서 5일 소요")
        self.assertEqual(for_tts("10% 할인"), "10 퍼센트 할인")
        s = Settings()
        s.tts_norm_amount = True
        self.assertEqual(for_tts("39,800원입니다", s), "삼만 구천팔백 원입니다")
        self.assertEqual(for_tts("39,800원입니다"), "39,800원입니다")

    def test_trailing_digits(self):
        self.assertEqual(trailing_digits("공일공 일이삼사"), "0101234")
        self.assertEqual(trailing_digits("010-1234-5678"), "01012345678")
        self.assertEqual(trailing_digits("1234번이요."), "1234")
        self.assertEqual(trailing_digits("천이백삼십사"), "1234")
        self.assertEqual(trailing_digits("주문번호가 데모 일 공 공 일인데요,"), "")
        self.assertEqual(trailing_digits("안녕하세요"), "")
        self.assertEqual(trailing_digits("이 상품"), "")

    def test_compare(self):
        self.assertEqual(norm_compare("주문번호가 데모 일인데요,"), norm_compare("주문번호가 데모 일인데요."))

    def test_tool_args(self):
        self.assertEqual(normalize_tool_args({"phone_number": " 010-1234-5678 "}), {"phone_number": "010-1234-5678"})
        s = Settings()
        s.tool_arg_strip = {"phone_number": "hyphen_space"}
        s.tool_arg_upper = ["order_id"]
        self.assertEqual(normalize_tool_args({"phone_number": "010-1234-5678", "order_id": "demo1001"}, s),
                         {"phone_number": "01012345678", "order_id": "DEMO1001"})
        self.assertIsNotNone(invalid_tool_arg({"order_id": "unknown"}))
        self.assertIsNone(invalid_tool_arg({"order_id": "ORD-2026-001"}))


class TestClassify(unittest.TestCase):
    def test_basic(self):
        c = lambda t, a="": classify(t, a, S)
        self.assertEqual(c("안녕하세요."), "preamble")
        self.assertEqual(c("여보세요?"), "preamble")
        self.assertEqual(c("네, 안녕하세요. 문의 좀 드리려고요."), "preamble")
        self.assertEqual(c("안녕하세요, 고객센터 상담 시간이 어떻게 되나요?"), "complete")
        self.assertEqual(c("주문번호가 데모 일 공 공 일인데요,"), "incomplete")
        self.assertEqual(c("주문번호가 데모 일 공 공 일인데요."), "undecided")
        self.assertEqual(c("택배가 지금 어디쯤 왔는지 알 수 있을까요"), "complete")
        self.assertEqual(c("내일 오전에 다시 전화 좀 부탁드려요."), "complete")
        self.assertEqual(c("어제 취소했는데"), "incomplete")
        self.assertEqual(c("그러니까 어…"), "incomplete")
        self.assertEqual(c(""), "empty")

    def test_context(self):
        ask_phone = "연락받으실 전화번호를 말씀해 주시겠어요?"
        self.assertEqual(classify("공일공 일이삼사", ask_phone, S), "incomplete")
        self.assertEqual(classify("공일공 일이삼사 오육칠팔", ask_phone, S), "complete")
        self.assertEqual(classify("네", "취소를 진행해 드릴까요?", S), "complete")
        self.assertEqual(classify("아니요.", "더 궁금하신 점 있으실까요?", S), "complete")


def view(**kw):
    base = dict(speech_started=True, has_text=True, silence_ms=0, voiced_now=False, end_seen=False,
                verdict="complete", judge=None, token_idle_ms=0)
    base.update(kw)
    return TurnView(**base)


class TestCommit(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(decide_commit(view(end_seen=True, silence_ms=320), S), "r1")
        self.assertIsNone(decide_commit(view(end_seen=False, silence_ms=600), S))
        self.assertEqual(decide_commit(view(end_seen=False, silence_ms=920), S), "r2")
        self.assertIsNone(decide_commit(view(verdict="incomplete", end_seen=True, silence_ms=1200), S))
        self.assertEqual(decide_commit(view(verdict="incomplete", end_seen=True, silence_ms=1620), S), "r3")
        self.assertIsNone(decide_commit(view(verdict="preamble", end_seen=True, silence_ms=1500), S))
        self.assertEqual(decide_commit(view(verdict="preamble", end_seen=True, silence_ms=2400), S), "r4")
        self.assertEqual(decide_commit(view(verdict="incomplete", silence_ms=2400), S), "r5")

    def test_undecided(self):
        self.assertIsNone(decide_commit(view(verdict="undecided", end_seen=True, silence_ms=800), S))
        self.assertEqual(decide_commit(view(verdict="undecided", judge="complete", end_seen=True, silence_ms=800), S), "r1")
        self.assertIsNone(decide_commit(view(verdict="undecided", end_seen=True, silence_ms=1300), S))
        self.assertEqual(decide_commit(view(verdict="undecided", end_seen=True, silence_ms=1700), S), "r3")

    def test_noise(self):
        self.assertIsNone(decide_commit(view(voiced_now=True, end_seen=True, token_idle_ms=500), S))
        self.assertEqual(decide_commit(view(voiced_now=True, end_seen=True, token_idle_ms=900), S), "noise")


class TestAudio(unittest.TestCase):
    def test_resampler(self):
        t = np.arange(24000) / 24000
        x = (np.sin(2 * np.pi * 1000 * t) * 8000).astype(np.int16).tobytes()
        rs = StreamResampler(24000, 16000)
        out = b"".join(rs.process(x[i:i + 1001]) for i in range(0, len(x), 1001))   # 홀수 바이트 조각
        y = np.frombuffer(out, np.int16).astype(np.float64)
        self.assertAlmostEqual(len(y), 16000, delta=2)
        spec = np.abs(np.fft.rfft(y[2000:14000]))
        peak = np.argmax(spec) * 16000 / 12000
        self.assertAlmostEqual(peak, 1000, delta=5)
        self.assertAlmostEqual(np.sqrt(np.mean(y[2000:14000] ** 2)), 8000 / np.sqrt(2), delta=300)

    def test_trimmer_preroll(self):
        sil = np.zeros(4800, np.int16)
        tone = (np.sin(2 * np.pi * 300 * np.arange(3200) / 16000) * 6000).astype(np.int16)
        pcm = np.concatenate([sil, tone]).tobytes()
        tr = LeadTrimmer(thr=300, preroll_ms=25)
        out = tr.process(pcm[:3000]) + tr.process(pcm[3000:])
        n = len(out) // 2
        self.assertGreater(n, 3200)                 # 말소리 앞 25ms가 살아 있다
        self.assertLess(n, 3200 + 16000 * 0.03)
        self.assertLess(abs(np.frombuffer(out[:2], np.int16)[0]), 50)   # 페이드인
        self.assertGreater(rms(out), 300)



class FactsTest(unittest.TestCase):
    """번호 맥락 우선, 조회 결과 요약, 답 검증 (agent/facts.py)."""

    def setUp(self):
        from datetime import datetime
        self.now = datetime(2026, 9, 29, 14, 55)
        self.order = {"status": "ok", "data": {"orders": [{
            "주문번호": "20261104-0007777", "상세주문번호": "20261104-0007777-01", "주문일자": "2026-09-28 08:25:08",
            "제품명": "🔒비밀특가🔒 하루온 맑은 오메가3", "옵션": "옵션 선택=하루온 맑은 오메가3 5BOX", "수량": 1,
            "결제수단": ["신용카드"], "결제여부": "T", "결제일자": "2026-09-28 08:25:08", "배송상태": "배송중",
            "송장번호": "994268494700", "배송완료일자": None}]}}

    def test_render_dates_are_precomputed(self):
        from agent.facts import render_tool_result
        t = render_tool_result("get_order_info", self.order, self.now)
        self.assertIn("주문일 9월 28일(월)", t)
        self.assertIn("(어제)", t)
        self.assertIn("결제 완료", t)
        self.assertNotIn("비밀특가", t)

    def test_verify_fixes_wrong_order_date(self):
        from agent.facts import render_tool_result, verify_answer
        ev = render_tool_result("get_order_info", self.order, self.now)
        out, fixes = verify_answer("확인해 보니 9월 29일에 주문하신 맑은 오메가3는 현재 배송 중입니다.", ev, self.now)
        self.assertIn("9월 28일에 주문하신", out)
        self.assertTrue(fixes)
        same, f2 = verify_answer("확인해 보니 9월 28일에 주문하신 맑은 오메가3는 현재 배송 중입니다.", ev, self.now)
        self.assertEqual(f2, [])

    def test_snap_to_context_phone(self):
        from agent.facts import context_ids, resolve_ids
        hist = [{"role": "customer", "text": "연락처ㅡ 010 6666 7777"}, {"role": "agent", "text": "성함도 남겨 주시겠어요?"}]
        ctx = context_ids(hist, None)
        out, notes = resolve_ids("request_callback", {"phone_number": "01066676777", "customer_name": "김하루"}, ctx)
        self.assertEqual(out["phone_number"], "01066667777")
        far, _ = resolve_ids("request_callback", {"phone_number": "01099998888", "customer_name": "김하루"}, ctx)
        self.assertEqual(far["phone_number"], "01099998888")          # 다른 번호는 건드리지 않는다

    def test_caller_phone_backup_for_garbled_ids(self):
        from agent.facts import context_ids, resolve_ids
        ctx = context_ids([], {"name": "이온새", "phone": "01088889999"})
        out, _ = resolve_ids("get_order_info", {"order_id": "20611058008888"}, ctx)      # 깨져 들린 주문번호
        self.assertEqual(out.get("buyer_cellphone"), "01088889999")
        ok, _ = resolve_ids("get_order_info", {"order_id": "20261105-0008888"}, ctx)     # 멀쩡한 주문번호
        self.assertNotIn("buyer_cellphone", ok)
        out2, _ = resolve_ids("get_order_info", {"buyer_cellphone": "520101811222"}, ctx)
        self.assertEqual(out2["buyer_cellphone"], "01088889999")

    def test_clamp_answer(self):
        from agent.facts import clamp_answer
        t = clamp_answer("배송비는 3,000원이에요. 5만 원 이상이면 무료입니다. 제품은 함께 드셔도 됩니다. 감사합니다.", 2, 90)
        self.assertEqual(t, "배송비는 3,000원이에요. 5만 원 이상이면 무료입니다.")

    def test_pseudonymizer_roundtrip(self):
        from agent.brain import Pseudonymizer
        pz = Pseudonymizer()
        s = "전화 010-4444-5555, 계좌 우리은행 625-862-293658, 주문 20261102-0004321"
        m = pz.mask(s)
        self.assertNotIn("5550", m)
        self.assertIn("20261102-0004321", m)                           # 주문번호는 필터가 건드리지 않으므로 그대로
        self.assertEqual(pz.unmask(m), s)
        self.assertEqual(pz.unmask({"buyer_cellphone": "{{PHONE_1}}"})["buyer_cellphone"], "010-4444-5555")


class RateLimitTest(unittest.TestCase):
    """게이트웨이 분당 한도: 일반 호출은 limit-reserve 까지, Jev는 limit 까지."""

    def test_jev_reserve(self):
        import asyncio
        from agent.ratelimit import RateLimiter

        async def run():
            rl = RateLimiter(per_minute=10, reserve=3)
            got = [await rl.acquire(optional=True) for _ in range(9)]
            self.assertEqual(sum(got), 7)                               # 일반 호출은 7건에서 막힌다
            jev = [await rl.acquire(optional=True, kind="jev") for _ in range(5)]
            self.assertEqual(sum(jev), 3)                               # Jev는 떼어 둔 3건을 더 쓴다
            self.assertFalse(await rl.acquire(optional=True, kind="jev"))
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
