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


class GuardTest(unittest.TestCase):
    def test_fixed_answer_guard(self):
        from agent.router import needs_llm
        self.assertEqual(needs_llm("배송은 언제 오는지랑 배송비가 얼마인지 알려 주세요."), "compound")
        self.assertEqual(needs_llm("간 영양제 효과가 어떤지 알고 싶고 주문은 어떻게 하는지도 알려 주세요."), "compound")
        self.assertEqual(needs_llm("제품을 먹고 나서 두드러기가 났어요. 환불해 주세요."), "symptom")
        # 용건이 하나인 말은 그대로 고정 답으로 간다 (효과가 없다는 말은 반품 사유)
        for t in ("택배가 아직도 안 왔는데 어떻게 된 건가요?", "반품하고 싶은데요.", "눈 영양제 하나 사려고요.",
                  "3일째 먹는데 아직도 변을 못봤어요 환불신청합니다", "먹어도 효과가 없어서 환불해 주세요.",
                  "환불이 아직 안 들어왔는데 언제 되나요?"):
            self.assertEqual(needs_llm(t), "", t)

    def test_search_guard(self):
        from agent.toolflow import product_mentioned
        for q in ("조인플렉스 2개", "하루온혈관캐어 2개", "밀크씨슬 성분", "루테인 하루 몇 알", "유산균 추천", "간 영양제 효과"):
            self.assertTrue(product_mentioned(q), q)
        for q in ("선박스 효과", "3박스 효과", "효과 있나요", "하루에 몇 알 먹어요", "언제 먹는 게 좋아요", "제품 가격", ""):
            self.assertFalse(product_mentioned(q), q)

    def test_phone_validation_and_example_phone_guard(self):
        from agent.facts import context_ids, resolve_ids, valid_phone
        from agent.toolflow import block_reason, reask_kind
        # 010은 11자리만 통과 (P0193-02 실측 '010-3710-35' 차단)
        self.assertFalse(valid_phone("010-3710-35"))
        self.assertFalse(valid_phone("010-3710-351"))
        self.assertTrue(valid_phone("010-3572-6035"))
        self.assertTrue(valid_phone("02-1234-5678"))
        # 프롬프트·예시 번호(policy_table, examples.md) 차단 + reask_phone 유도
        empty_ctx = {"phones": [], "orders": []}
        for bad in ("010-1234-5678", "010-4444-5555", "01022223333", "010-3710-35"):
            blk = block_reason("get_return_info", {"customer_phone_number": bad}, empty_ctx)
            self.assertTrue(blk, bad)
            self.assertEqual(reask_kind("get_return_info", {"customer_phone_number": bad}, blk), "reask_phone")
        # 발신자 번호가 있으면 9자리 010이나 예시 번호도 발신자 번호로 보정된다
        ctx = context_ids([], {"name": "홍길동", "phone": "01088889999"})
        out1, _ = resolve_ids("get_return_info", {"customer_phone_number": "010-3710-35"}, ctx)
        self.assertEqual(out1["customer_phone_number"], "01088889999")
        out2, _ = resolve_ids("get_order_info", {"buyer_cellphone": "010-1234-5678"}, ctx)
        self.assertEqual(out2["buyer_cellphone"], "01088889999")


class FaqDirectTest(unittest.TestCase):
    """시연용 단문 FAQ 키워드 직행: 정상 단문만 발동하고, 예외·복합·문맥 문항은 차단한다."""

    def test_true_short_faqs_in_live_mode(self):
        from agent.router import choose_bank
        cases = [
            ("배송비 얼마예요?", "faq_shipping_fee"),
            ("택배사가 어디예요?", "faq_courier"),
            ("고객센터 운영시간이 몇 시까지예요?", "faq_hours"),
            ("적립금은 얼마 이상부터 쓸 수 있어요?", "faq_points"),
            ("쿠폰 중복 사용 되나요?", "faq_coupon_stack"),
            ("고등학생이 먹어도 되나요?", "faq_minor"),
            ("술 마신 날 먹어도 되나요?", "faq_alcohol"),
            ("하루온 영양제 두 가지 같이 먹어도 되나요?", "faq_together"),
        ]
        for text, want in cases:
            k, why = choose_bank(text, [], None, None, S, mode="live")
            self.assertEqual((k, why), (want, "faq-keyword"), text)
            # 기본 설정(faq_direct_live_only=True)에서는 측정 모드에서 발동하지 않는다
            k_m, _ = choose_bank(text, [], None, None, S, mode="measure")
            self.assertIsNone(k_m, f"measure mode should not fire for {text}")

    def test_trap_and_context_cases_blocked(self):
        from agent.router import choose_bank
        traps = [
            ("오전에 눈 관련 1개 주문했는데 오후에 간 제품 1개 주문하려는데 배송비 추가 내야 하나요?", []),
            ("제주도 배송비는 얼마예요?", []),
            ("반품 배송비가 얼마예요?", []),
            ("조인플렉스 관절 영양제 주문하고 싶은데 한 박스만 사면 배송비 얼마예요?", []),
            ("CJ택배 송장 조회가 안 돼요.", []),
            ("고혈압약이랑 같이 먹어도 괜찮을까요?", []),
            ("타사 비타민이랑 하루온 같이 먹어도 되나요?", []),
            ("고등학생 아들이 먹어도 되나요? 그리고 남편은 술 마신 날 먹어도 돼요?", []),
            ("적립금이랑 할인 쿠폰 같이 쓸 수 있나요? 그리고 적립금은 얼마 이상 써요?", []),
            ("만료된 쿠폰 복구해 주세요.", []),
            ("아이가 먹고 두드러기가 났어요.", []),
            ("배송비 얼마예요?", [{"role": "customer", "text": "주문 취소하려는데"}]),
            ("배송비 얼마예요? 010-1234-5678 확인해 주세요.", []),
        ]
        for text, hist in traps:
            k, why = choose_bank(text, hist, None, None, S, mode="live")
            self.assertIsNone(k, f"trap should be blocked: {text} -> {k}({why})")


class UnfinishedTailTest(unittest.TestCase):
    def test_unfinished_tail(self):
        from agent.session import _unfinished_tail
        from agent.router import needs_llm
        for t in ("010 9901.", "20261003-00.", "어제 주문했는데.", "환불 받았고."):
            self.assertTrue(_unfinished_tail(t), t)
        for t in ("010-1234-5678.", "20261003-0084222.", "1600-0000.", "Il 600, 000.", "조인플렉스 2개.",
                  "반품하고 싶은데요.", "배송이 안 왔어요.", "어제 주문한 제품이요."):
            self.assertEqual(_unfinished_tail(t), "", t)
        # 섭취 질문 + 구매 의사는 복합 의도 (블라인드 B0017-01)
        self.assertEqual(needs_llm("멜라토닌 제품 하루에 언제 먹어야 해요? 괜찮으면 바로 구입하려고요."), "compound")


if __name__ == "__main__":
    unittest.main()
