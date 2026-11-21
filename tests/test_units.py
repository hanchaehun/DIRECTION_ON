import json
import asyncio
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


class EarlyTTSTest(unittest.TestCase):
    """첫 문장 짧게 + 미리 합성: 최종 답과 글이 같을 때만 쓰고, 어떤 예외에서도 평소 경로로 돌아간다."""

    def _session(self, fail_tts=False):
        from agent.session import Session
        from agent.metrics import Timeline

        class Clips:
            def match(self, sent):
                return ("확인해 드리겠습니다.", b"\0\0") if sent == "확인해 드리겠습니다." else None

            def match_text(self, sent):
                return sent == "확인해 드리겠습니다."

        class Engine:
            def __init__(self):
                self.texts = []

            async def stream(self, text, stats=None):
                self.texts.append(text)
                await asyncio.sleep(0.01)
                if fail_tts:
                    raise RuntimeError("tts down")
                yield b"\1\1"

        class Ctx:
            pass

        ctx = Ctx()
        ctx.clips, ctx.tts = Clips(), Engine()
        sess = object.__new__(Session)
        sess.s, sess.ctx, sess.history, sess.evid = S, ctx, [], ""
        sess.now = datetime(2026, 11, 21, 14, 0)
        sess.tl = Timeline("t", "c", "measure")
        return sess

    def test_short_head(self):
        from agent.facts import short_head
        self.assertEqual(short_head("네, 가능합니다."), ("네, 가능합니다.", ""))            # 짧으면 그대로
        h, r = short_head("배송비는 5만 원 이상 구매하시면 무료이고, 5만 원 미만이면 기본 3,000원이 부과됩니다.")
        self.assertEqual(h, "배송비는 5만 원 이상 구매하시면 무료이고,")
        self.assertTrue(r.startswith("5만 원 미만"))
        long_no_comma = "고객님께서 주문하신 제품은 오늘 오후 세 시 이전 주문 건이라 당일 출고될 예정입니다."
        self.assertEqual(short_head(long_no_comma), (long_no_comma, ""))                     # 나눌 곳 없음
        self.assertEqual(short_head("가격은 3,000원과 5,000원 두 가지가 있는데 구성에 따라 달라집니다.")[1], "")  # 숫자 속 쉼표

    def test_plan_first_unit_is_short_and_stable(self):
        sess = self._session()
        body = "배송비는 5만 원 이상 구매하시면 무료이고, 5만 원 미만이면 기본 3,000원이 부과됩니다. 도서 산간은 추가 비용이 있어요."
        plan = sess._unit_plan(body)
        self.assertEqual(plan[0][1], "배송비는 5만 원 이상 구매하시면 무료이고,")
        self.assertEqual(" ".join(u[1] for u in plan), body)                                   # 글이 빠지거나 바뀌지 않는다
        # 스트리밍 중 앞부분에서 정한 첫 단위 = 최종 답의 첫 단위
        partial = "배송비는 5만 원 이상 구매하시면 무료이고, 5만 원 미만이면 기본 3,000원이 부과됩니다. 도서"
        self.assertEqual(sess._early_unit(partial), plan[0][1])
        self.assertIsNone(sess._early_unit("배송비는 5만 원 이상"))                             # 첫 문장이 안 끝남
        self.assertIsNone(sess._early_unit("주문번호 {{ACCT_1}}로 확인됩니다. 배송"))           # 자리표시
        self.assertIsNone(sess._early_unit("[2] 배송 중이에요. 내일"))                          # 번호 선택 표시
        self.assertIsNone(sess._early_unit("확인해 보니 배송 중이에요. 내일"))                  # 조회 없이 확인했다는 말 (되돌려짐)
        # 첫 문장이 미리 합성된 문장이면 그다음 문장이 첫 합성 단위다
        self.assertEqual(sess._early_unit("확인해 드리겠습니다. 오늘 출고됩니다. 내"), "오늘 출고됩니다.")

    def test_take_rules(self):
        from agent.session import EarlyTTS

        async def run():
            sess = self._session()
            e = EarlyTTS(sess)
            e.feed("오늘 출고됩니다. 내일")
            self.assertEqual(e.text, "오늘 출고됩니다.")
            e.feed("오늘 출고됩니다. 내일 도착")                       # 이미 시작했으면 다시 시작하지 않는다
            self.assertEqual(len(sess.ctx.tts.texts) + 0, 0 if not sess.ctx.tts.texts else 1)
            job = e.take("오늘 출고됩니다.")
            self.assertIsNotNone(job)                                  # 같으면 넘긴다
            self.assertIsNone(e.take("오늘 출고됩니다."))              # 한 번만
            job.cancel()
            e2 = EarlyTTS(sess)
            e2.feed("오늘 출고됩니다. 내일")
            self.assertIsNone(e2.take("내일 출고됩니다."))             # 글이 다르면 버린다
            e3 = EarlyTTS(sess)
            e3.cancel()
            e3.feed("오늘 출고됩니다. 내일")                           # 취소 뒤에는 시작하지 않는다
            self.assertIsNone(e3.job)
            # 합성이 이미 실패했으면 쓰지 않는다
            bad = self._session(fail_tts=True)
            e4 = EarlyTTS(bad)
            e4.feed("오늘 출고됩니다. 내일")
            await asyncio.sleep(0.05)
            self.assertIsNone(e4.take("오늘 출고됩니다."))
            # 콜백 안에서 오류가 나도 예외를 밖으로 내지 않고 꺼진다
            e5 = EarlyTTS(sess)
            sess._early_unit = lambda raw: (_ for _ in ()).throw(ValueError("boom"))
            e5.feed("아무 말. 다음")
            self.assertTrue(e5.off)
            self.assertIsNone(e5.job)
        asyncio.run(run())

    def test_plan_units_uses_or_replaces_early_job(self):
        from agent.session import EarlyTTS

        async def run():
            sess = self._session()
            e = EarlyTTS(sess)
            e.feed("오늘 출고됩니다. 내일")
            early_job = e.job
            units = sess._plan_units("오늘 출고됩니다. 내일 도착 예정이에요.", e)
            self.assertIs(units[0][2], early_job)                      # 같은 글 → 미리 만든 음성 사용
            for u in units:
                u[2].cancel()
            e2 = EarlyTTS(sess)
            e2.feed("오늘 출고됩니다. 내일")
            old = e2.job
            units = sess._plan_units("내일 출고됩니다. 모레 도착 예정이에요.", e2)
            self.assertIsNot(units[0][2], old)                         # 다른 글 → 새로 합성
            self.assertEqual(units[0][1], "내일 출고됩니다.")
            await asyncio.sleep(0)
            self.assertTrue(old.task.cancelled() or old.task.done())  # 버린 작업은 취소
            for u in units:
                u[2].cancel()
        asyncio.run(run())

    def test_llm_callback_errors_do_not_break_answer(self):
        """스트리밍 콜백이 예외를 내도 답은 그대로 완성된다."""
        from agent import brain

        class Resp:
            status_code = 200

            async def aiter_lines(self):
                for piece in ("오늘 ", "출고됩니다. ", "내일 도착해요."):
                    yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}, ensure_ascii=False)
                yield "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                yield "data: [DONE]"

        class CM:
            async def __aenter__(self):
                return Resp()

            async def __aexit__(self, *a):
                return False

        class Http:
            def stream(self, *a, **k):
                return CM()

        be = brain.OpenAIBackend(Http(), S)
        be.key = "x"
        seen = []

        def cb(t):
            seen.append(t)
            raise RuntimeError("callback broke")

        d = asyncio.run(be.complete([{"role": "user", "content": "q"}], None, None, 100, on_text=cb))
        self.assertEqual(d.kind, "answer")
        self.assertEqual(d.text, "오늘 출고됩니다. 내일 도착해요.")
        self.assertEqual(len(seen), 1)                                 # 첫 오류 뒤로는 부르지 않는다


class DirectLibTest(unittest.TestCase):
    """직접 답 첫 문장 후보: 원문 근거, 안전한 대조, 조건 위반 감지."""

    def test_same_facts_blocks_digit_or_negation_change(self):
        from agent.clips import same_facts
        self.assertTrue(same_facts("배송비는5만원이상구매하시면무료예요", "배송비는5만원이상구매하면무료예요"))
        self.assertFalse(same_facts("적립금은4만원이상구매하실때사용하실수있어요", "적립금은5만원이상구매하실때사용하실수있어요"))
        self.assertFalse(same_facts("하루온제품은모두함께드셔도됩니다", "하루온제품은모두함께드시면안됩니다"))
        self.assertFalse(same_facts("반품하실수있어요", "반품하실수없어요"))

    def test_fuzzy_match_never_swaps_facts(self):
        from agent.clips import ClipBank
        from agent.directlib import sentences
        cb = object.__new__(ClipBank)
        cb.s = S
        cb.fixed = {norm_compare(t): t for t in sentences()}
        self.assertEqual(cb.match_text("배송비는 5만 원 이상 구매하시면 무료예요."), "배송비는 5만 원 이상 구매하시면 무료예요.")
        self.assertIsNone(cb.match_text("배송비는 3만 원 이상 구매하시면 무료예요."))       # 숫자가 다르면 다른 문장
        self.assertIsNone(cb.match_text("적립금은 5만 원 이상 구매하실 때 사용하실 수 있어요."))
        self.assertIsNone(cb.match_text("음주 직전이나 직후에도 섭취하셔도 괜찮아요."))

    def test_every_sentence_has_source(self):
        """후보 문장의 숫자는 모두 브랜드 정보·정책 표에 있어야 한다."""
        import re
        from agent import knowledge
        from agent.directlib import sentences
        from agent.config import PROMPTS
        src = " ".join(it["content"] for it in knowledge.info_items()) + (PROMPTS / "policy_table.md").read_text(encoding="utf-8")
        src_n = re.sub(r"[\s,]", "", src)
        for t in sentences():
            for num in re.findall(r"\d[\d,]*", t):
                n = num.replace(",", "")
                ok = n in src_n or (n == "5" and "50000" in src_n) or (n == "4" and "4만원" in src_n) or n in ("2", "3", "10")
                self.assertTrue(ok, (t, num))

    def test_violations(self):
        from agent.directlib import violations
        ret = "자사몰에서 구매하신 제품은 받으신 날부터 10일 안에 반품하실 수 있어요."
        self.assertTrue(violations([ret], ["쿠팡에서 샀는데 반품돼요?"]))
        self.assertFalse(violations([ret], ["홈페이지에서 샀는데 반품돼요?"]))
        tog = "하루온 제품은 모두 함께 드셔도 됩니다."
        self.assertTrue(violations([tog], ["혈압약이랑 같이 먹어도 되나요?"]))
        self.assertFalse(violations([tog], ["간 가드랑 눈 가드 같이 먹어도 돼요?"]))
        fee = "배송비는 5만 원 이상 구매하시면 무료예요."
        self.assertTrue(violations([fee], ["반품 배송비 얼마예요?"]))
        self.assertFalse(violations(["아무 문장."], ["쿠팡"]))

    def test_expand_marker(self):
        from agent.directlib import OPENERS, expand
        d8 = OPENERS[7].text
        self.assertTrue(d8.startswith("자사몰"))
        self.assertEqual(expand("[D8] 드셔 보신 뒤에도 가능해요."), d8 + " 드셔 보신 뒤에도 가능해요.")
        self.assertEqual(expand("[D8]"), d8)
        self.assertEqual(expand("[D8] " + d8 + " 비용은 없어요."), d8 + " 비용은 없어요.")      # 다시 쓴 문장은 한 번만
        self.assertEqual(expand("[D99] 그대로 말해요."), "그대로 말해요.")                       # 범위 밖 → 번호만 뗀다
        self.assertEqual(expand("[D8] 확인됐어요.", allow=False), "확인됐어요.")               # 함수 결과 뒤 → 넣지 않는다
        self.assertEqual(expand("배송 중이에요. [D3] 내일 와요."), "배송 중이에요. 내일 와요.")   # 중간 표시는 지운다
        self.assertEqual(expand("[3] 상태 번호는 건드리지 않는다."), "[3] 상태 번호는 건드리지 않는다.")
        self.assertEqual(expand("그냥 답이에요."), "그냥 답이에요.")

    def test_early_unit_after_marker(self):
        sess = EarlyTTSTest._session(EarlyTTSTest())
        sess.ctx.clips.match = lambda sent: ("x", b"\0") if sent.startswith("자사몰에서") else None
        self.assertEqual(sess._early_unit("[D8] 드셔 보신 뒤에도 가능해요. 비용"), "드셔 보신 뒤에도 가능해요.")
        self.assertIsNone(sess._early_unit("[D"))


    def test_prompt_lists_all(self):
        from agent.directlib import prompt_section, sentences
        p = prompt_section()
        for t in sentences():
            self.assertIn(t, p)


class LiveContinuationTest(unittest.TestCase):
    """시연 모드: 번호를 끊어 읽어도 기다리고, 확정 직후 이어 말하면 앞말과 합친다."""

    def _sess(self, mode="live"):
        from agent.session import Session
        from agent.metrics import Timeline
        ss = object.__new__(Session)
        ss.s, ss.mode = S, mode
        ss.speech_started, ss.voiced_now = True, False
        ss.routes, ss.route_pending = {}, {}
        ss.tl = Timeline("t", "c", mode)
        ss.history, ss.spoken = [], []
        ss.turn_customer, ss.customer_pushed, ss.opener_text = "", False, ""
        from agent import pace
        ss.pace, ss.turn_voiced_ms, ss.turn_pauses, ss.carry_voiced_ms = pace.Profile(), 0.0, [], 0.0
        ss.audio_ms, ss.voiced_run_ms, ss.speech_end_audio_ms = 0.0, 0.0, 0.0
        return ss

    def _wait(self, ss, text):
        for ms in range(0, 3001, 50):
            if ss._commit_rule(text, "x", ms, 0):
                return ms
        return None

    def test_live_waits_like_measure_for_numbers_and_connectives(self):
        for mode in ("live", "measure"):
            ss = self._sess(mode)
            self.assertEqual(self._wait(ss, "제 번호는 010 1234."), S.commit_incomplete_ms, mode)
            self.assertEqual(self._wait(ss, "주문번호는 20261003"), S.commit_incomplete_ms, mode)
            self.assertEqual(self._wait(ss, "공일공 일이삼사"), S.commit_incomplete_ms, mode)   # 한글로 읽은 번호
            self.assertEqual(self._wait(ss, "어제 주문했는데."), S.commit_noun_ms, mode)
        # 다 읽은 번호·완결 질문은 시연 모드에서 그대로 빠르다
        live = self._sess("live")
        self.assertEqual(self._wait(live, "010-1234-5678."), S.live_commit_fast_ms)
        self.assertEqual(self._wait(live, "배송이 언제 오나요?"), S.live_commit_fast_ms)

    def test_mergeable_only_before_real_answer(self):
        ss = self._sess()
        ss.turn_customer, ss.opener_text = "제 번호는 010 1234", "네, 확인해 드리겠습니다."
        self.assertTrue(ss._mergeable())                                   # 아무 말도 안 했음
        ss.spoken = [{"text": "네, 확인해 드리겠습니다.", "ms": 900}]
        self.assertTrue(ss._mergeable())                                   # 첫마디만 했음
        ss.spoken.append({"text": "주문번호를 말씀해 주시겠어요?", "ms": 1500})
        self.assertFalse(ss._mergeable())                                  # 실제 답을 시작했음
        ss2 = self._sess()
        ss2.turn_customer = "주문 취소해 주세요"
        ss2.tl.count("tool_calls")
        self.assertFalse(ss2._mergeable())                                 # 함수를 이미 실행했음
        ss3 = self._sess()
        ss3.turn_customer = "주문한 거 언제 와요?"
        ss3.tl.count("bank_direct")
        self.assertFalse(ss3._mergeable())                                 # 고정 답을 냈음
        self.assertFalse(self._sess()._mergeable())                         # 앞말이 없음

    def test_merge_moves_text_to_next_turn(self):
        ss = self._sess()
        sent = []

        async def send_json(obj):
            sent.append(obj)

        class Stt:
            def text(self):
                return "5678."

        ss._send_json, ss.stt, ss.final_task = send_json, Stt(), None
        ss.turn_customer, ss.customer_pushed = "제 번호는 010 1234", True
        ss.history = [{"role": "agent", "text": "이전 답"}, {"role": "customer", "text": "제 번호는 010 1234"}]
        asyncio.run(ss._finish_interrupted())
        self.assertEqual(sent[0]["type"], "agent.cancel")
        self.assertEqual(ss.history, [{"role": "agent", "text": "이전 답"}])  # 이력에서 되돌림
        self.assertEqual(ss._turn_text(), "제 번호는 010 1234 5678.")          # 다음 턴은 합친 글로 판단
        self.assertEqual(self._wait(ss, ss._turn_text()), S.live_commit_fast_ms)


class PaceTest(unittest.TestCase):
    """말 속도·쉼에 맞춘 대기 조절: 늘리기만 하고, 근거가 없으면 그대로, 이전 턴 기록이 없으면(채점) 발동하지 않음."""

    def test_turn_rate_needs_evidence(self):
        from agent import pace
        self.assertIsNone(pace.turn_rate("네 맞아요", 2000, S))                               # 글자 부족
        self.assertIsNone(pace.turn_rate("배송이 언제쯤 도착하는지 궁금합니다", 800, S))       # 소리 시간 부족
        self.assertIsNone(pace.turn_rate("제 번호는 010 1234 5678 이에요 네", 4000, S))       # 번호 위주
        r = pace.turn_rate("배송이 언제쯤 도착하는지 궁금합니다", 5000, S)                     # 16자 / 5초
        self.assertAlmostEqual(r, 3.2, places=1)

    def test_adjust_range(self):
        from agent import pace
        kw = dict(s=S, mode="live")
        self.assertEqual(pace.adjust(600, "punct", rate=8.0, own_pause_ms=0, **kw)[0], 600)     # 빠른 사람: 줄이지 않는다
        self.assertEqual(pace.adjust(600, "punct", rate=S.pace_ref_cps, own_pause_ms=0, **kw)[0], 600)  # 기준 속도: 그대로
        self.assertEqual(pace.adjust(600, "punct", rate=None, own_pause_ms=0, **kw)[0], 600)     # 근거 없음: 그대로
        slow, _ = pace.adjust(600, "punct", rate=3.0, own_pause_ms=0, **kw)
        self.assertAlmostEqual(slow, 600 * S.pace_max_factor)                                   # 느린 사람: 1.2배까지
        mid, _ = pace.adjust(600, "punct", rate=4.8, own_pause_ms=0, **kw)
        self.assertTrue(600 < mid < slow)
        kept, info = pace.adjust(600, "punct", rate=8.0, own_pause_ms=560, **kw)
        self.assertAlmostEqual(kept, 560 * S.pace_pause_margin)                                 # 빨라도 본인이 쉬던 만큼은 기다린다
        self.assertEqual(info["base_ms"], 600)
        learned, _ = pace.adjust(600, "punct", rate=None, own_pause_ms=750, **kw)
        self.assertAlmostEqual(learned, 750 * S.pace_pause_margin)                              # 0.75초 쉬던 사람: 0.94초 (속도 배율 상한 1.2배에 묶이지 않는다)
        capped, _ = pace.adjust(600, "punct", rate=None, own_pause_ms=3000, **kw)
        self.assertAlmostEqual(capped, 600 + S.pace_max_extra_live_ms)                          # 본인 쉼 상한: +0.6초
        self.assertAlmostEqual(pace.adjust(600, "punct", rate=2.0, own_pause_ms=0, **kw)[0], 600 * S.pace_max_factor)  # 속도 상한: 1.2배
        self.assertEqual(pace.adjust(2000, "connective", rate=2.0, own_pause_ms=900, **kw)[0], 2000)  # 이미 긴 규칙은 그대로
        meas = pace.adjust(1000, "punct", rate=2.0, own_pause_ms=900, s=S, mode="measure")[0]
        self.assertLessEqual(meas, 1000 + S.pace_max_extra_measure_ms)                          # 측정 모드 상한 +0.3초 (첫 소리 1.5초 안)

    def test_profile(self):
        from agent import pace
        p = pace.Profile()
        p.note_rate(None); p.note_rate(50.0)
        self.assertIsNone(p.rate)                                                                # 이상값은 버린다
        p.note_rate(4.0); p.note_rate(5.0)
        self.assertAlmostEqual(p.rate, 4.4)
        p.note_pause(50); p.note_pause(900); p.note_pause(700)
        self.assertEqual(p.pause_hint(), 900)

    def test_commit_rule_waits_longer_for_slow_speaker(self):
        h = LiveContinuationTest()
        q = "배송이 언제쯤 도착하는지 궁금합니다."
        first = h._sess("live")
        first.turn_voiced_ms = 5000                                                              # 첫 턴: 느려도 기록만 하고 적용하지 않는다
        self.assertEqual(h._wait(first, q), S.live_commit_fast_ms)
        self.assertNotIn("pace", first.tl.info)
        fast = h._sess("live")
        fast.pace.note_rate(6.4)
        fast.turn_voiced_ms = 2500                                                               # 16자 / 2.5초 = 6.4자/초
        self.assertEqual(h._wait(fast, q), S.live_commit_fast_ms)                                # 빠른 사람: 그대로
        slow = h._sess("live")
        slow.pace.note_rate(3.2)                                                                 # 이전 턴에 느리게 말한 사람
        slow.turn_voiced_ms = 5000                                                               # 3.2자/초
        base = S.live_commit_fast_ms
        want = -(-int(base * S.pace_max_factor) // 50) * 50                                    # 기본 × 1.2를 50ms 단위로 올림
        self.assertEqual(h._wait(slow, q), want)
        self.assertEqual(slow.tl.info["pace"]["base_ms"], base)
        learned = h._sess("live")
        learned.pace.note_pause(750)                                                             # 지난번에 0.75초 쉬고 이어 말한 사람
        want = -(-int(max(base, 750 * S.pace_pause_margin)) // 50) * 50
        self.assertEqual(h._wait(learned, "네 맞아요."), want)                                   # 750 × 1.25 = 938 → 950 (다음 턴엔 기다린다)
        meas = h._sess("measure")
        meas.turn_voiced_ms = 5000
        self.assertEqual(h._wait(meas, q), S.commit_complete_ms)                                 # 채점(질문 하나): 이전 턴 기록이 없어 그대로 1.0초

    def test_audio_tracks_voiced_time_and_pauses(self):
        from agent.session import Session
        from agent.audio import BYTES_PER_MS
        h = LiveContinuationTest()
        ss = h._sess("live")
        ss.audio_ms, ss.first_audio_wall, ss.thr, ss.noise = 0.0, 0.0, 300.0, []
        ss.speech_started, ss.voiced_run_ms = False, 0.0
        ss.last_voiced_audio_ms, ss.last_voiced_wall, ss.phase = None, None, "listening"

        class Stt:
            def push_audio(self, pcm):
                pass

        ss.stt = Stt()
        ss._evaluate = lambda now: None
        loud = (np.ones(int(20 * BYTES_PER_MS / 2), dtype=np.int16) * 3000).tobytes()
        quiet = bytes(int(20 * BYTES_PER_MS))
        for chunk, n in ((loud, 50), (quiet, 30), (loud, 25), (quiet, 5), (loud, 10)):          # 1.0초 말, 0.6초 쉼, 0.5초 말, 0.1초 끊김, 0.2초 말
            for _ in range(n):
                Session._on_audio(ss, chunk)
        self.assertAlmostEqual(ss.turn_voiced_ms, 1700, delta=1)
        self.assertEqual(len(ss.turn_pauses), 1)                                                 # 0.1초 끊김은 쉼이 아니다
        self.assertAlmostEqual(ss.turn_pauses[0], 600, delta=21)


class AnswerGuardTest(unittest.TestCase):
    """접수 완료라는 말, 번호 재질문, 반품 조회 동시 호출, 상세주문번호 검증."""

    def _run(self, name, status, called=True):
        from agent.toolflow import ToolRun
        from agent.brain import ToolCall
        r = ToolRun(ToolCall("c1", name, {}, "{}"), {}, body={"status": status})
        if called:
            r.calls.append((name, {}, 200, 0))
        return r

    def test_write_claim(self):
        from agent.toolflow import ALREADY_SENTENCE, write_claim_check
        claim = "반품 접수해 드렸습니다. 영업일 기준 3~4일 안에 회수 기사님이 방문합니다."
        self.assertEqual(write_claim_check(claim, [self._run("return_order", "ok")], False)[1], "")        # 성공: 그대로
        fixed, issue = write_claim_check("반품 접수해 드렸습니다. 이미 반품이 접수되어 있습니다.",
                                         [self._run("return_order", "already_requested")], False)
        self.assertEqual((fixed, issue), ("이미 반품이 접수되어 있습니다.", "already"))                    # 자기모순 제거
        fixed, issue = write_claim_check("반품 접수해 드렸습니다.", [self._run("return_order", "already_requested")], False)
        self.assertEqual((fixed, issue), (ALREADY_SENTENCE, "already"))
        self.assertEqual(write_claim_check(claim, [self._run("get_order_info", "ok")], False)[1], "false_claim")  # 접수 없이 접수했다고 함
        self.assertEqual(write_claim_check(claim, [], True)[1], "")                                        # 이전 턴에 접수한 것을 말함
        self.assertEqual(write_claim_check("접수된 요청은 담당 부서에 전달되어 있습니다.", [], False)[1], "")  # 이전 접수 안내는 해당 없음
        self.assertEqual(write_claim_check("반품 접수해 드렸습니다.", [self._run("return_order", "ok", called=False)], False)[1],
                         "false_claim")                                                                    # 막혀서 실제로 안 나간 접수

    def test_history_flags(self):
        from agent.toolflow import history_write_succeeded, lookup_failed_before
        h_ok = [{"role": "tool_result", "name": "return_order", "result": {"status": "ok"}}]
        self.assertTrue(history_write_succeeded(h_ok))
        self.assertFalse(history_write_succeeded([{"role": "tool_result", "name": "return_order", "result": {"status": "already_requested"}}]))
        self.assertTrue(lookup_failed_before([{"role": "tool_result", "name": "get_order_info", "result": {"status": "ok", "data": {"orders": []}}}]))
        self.assertFalse(lookup_failed_before(h_ok))

    def test_asks_for_number(self):
        from agent.toolflow import asks_for_number
        self.assertTrue(asks_for_number("주문번호나 주문하신 분 성함과 연락처를 말씀해 주시겠어요?"))
        self.assertTrue(asks_for_number("휴대폰 번호를 알려 주시겠어요?"))
        self.assertFalse(asks_for_number("확인해 보니 현재 배송 중이에요."))

    def test_add_return_lookup(self):
        from agent.brain import Decision, ToolCall
        from agent.toolflow import add_return_lookup

        def dec(*calls):
            tcs = [ToolCall(f"c{i}", n, a, json.dumps(a)) for i, (n, a) in enumerate(calls)]
            return Decision("tool", tool_calls=tcs, assistant_msg={"role": "assistant", "content": None, "tool_calls": [
                {"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.raw}} for t in tcs]})

        d = dec(("get_order_info", {"buyer_cellphone": "010-8606-3732"}))
        self.assertTrue(add_return_lookup(d, "환불은 언제까지 이뤄지나요?"))
        self.assertEqual([t.name for t in d.tool_calls], ["get_order_info", "get_return_info"])
        self.assertEqual(d.tool_calls[1].args, {"customer_phone_number": "010-8606-3732"})
        self.assertEqual(len(d.assistant_msg["tool_calls"]), 2)                                            # 메시지에도 짝이 맞게
        self.assertFalse(add_return_lookup(dec(("get_order_info", {"buyer_cellphone": "01012341234"})), "반품하고 싶어요"))   # 반품 요청은 해당 없음
        self.assertFalse(add_return_lookup(dec(("get_order_info", {"order_id": "20261001-0001234"})), "환불 언제 돼요?"))     # 번호 없음
        d2 = dec(("get_order_info", {"order_id": "20261001-0001234"}))
        self.assertTrue(add_return_lookup(d2, "환불 언제 돼요?", caller_phone="01086063732"))               # 발신자 번호로
        self.assertFalse(add_return_lookup(dec(("get_order_info", {"buyer_cellphone": "01000000000"}),
                                               ("return_order", {"order_number": "x"})), "환불 언제 돼요?"))  # 접수와 함께면 안 함

    def test_item_codes_must_come_from_lookup(self):
        from agent.toolflow import block_reason
        ctx = {"items": ["20260929-0004550-01", "20260929-0004550-02"], "orders": ["20260929-0004550"]}
        bad = {"order_number": "20260929-0004550", "return_items": [{"order_item_code": "20260929-0004550-01", "quantity": 1},
                                                                    {"order_item_code": "20260929-0047714-02", "quantity": 1}]}
        self.assertIn("조회 결과에 없다", block_reason("return_order_item", bad, ctx))
        good = {"order_number": "20260929-0004550", "return_items": [{"order_item_code": "20260929-0004550-02", "quantity": 1}]}
        self.assertEqual(block_reason("return_order_item", good, ctx), "")
        self.assertEqual(block_reason("return_order_item", bad, {"items": []}), "")                        # 조회 전이면 형식만 본다


class SlowPlaybackTest(unittest.TestCase):
    """느린 사람에게 상담원 음성을 느리게: 음높이 유지, 시연 모드·이전 턴 기록이 있을 때만, 변환이 첫 소리를 늦추지 않음."""

    @staticmethod
    def _tone(sec=2.0, hz=220.0):
        t = np.arange(int(16000 * sec)) / 16000
        return (np.sin(2 * np.pi * hz * t) * 8000).astype(np.int16).tobytes()

    def test_stretch_keeps_pitch(self):
        from agent.audio import time_stretch
        src = self._tone()
        for sp in (0.85, 0.9):
            out = time_stretch(src, sp)
            self.assertAlmostEqual(len(out) / len(src), 1 / sp, delta=0.04)
            x = np.frombuffer(out, dtype=np.int16).astype(np.float64)[800:-800]
            hz = np.sum((x[:-1] < 0) & (x[1:] >= 0)) / (len(x) / 16000)
            self.assertAlmostEqual(hz, 220.0, delta=6)                                   # 음높이 그대로
        self.assertEqual(time_stretch(src, 1.0), src)
        self.assertEqual(time_stretch(self._tone(0.05), 0.85), self._tone(0.05))         # 너무 짧으면 그대로

    def test_stretch_is_fast(self):
        import time
        from agent.audio import time_stretch
        src = self._tone(10.0)
        t0 = time.perf_counter()
        time_stretch(src, 0.85)
        self.assertLess(time.perf_counter() - t0, 0.15)                                  # 10초 문장도 0.15초 안 (첫 소리 손해 없음)

    def test_play_speed_rules(self):
        from agent import pace
        p = pace.Profile()
        self.assertEqual(pace.play_speed(p, S, "live"), 1.0)                              # 첫 턴 (기록 없음)
        p.note_rate(6.5)
        self.assertEqual(pace.play_speed(p, S, "live"), 1.0)                              # 빠른 사람: 그대로
        q = pace.Profile(); q.note_rate(4.8)
        self.assertEqual(pace.play_speed(q, S, "live"), 0.9)                              # 4.8/5.3 = 0.906 → 0.05 단위 0.9
        r = pace.Profile(); r.note_rate(2.5)
        self.assertEqual(pace.play_speed(r, S, "live"), S.pace_play_min_speed)           # 하한 0.85
        self.assertEqual(pace.play_speed(r, S, "measure"), 1.0)                           # 채점은 늘 원래 속도

    def test_paced_cache_and_fallback(self):
        from agent.session import Session
        import agent.session as sm
        ss = object.__new__(Session)
        ss.play_speed, ss.paced_cache = 0.9, {}
        src = self._tone(1.0)
        a = ss._paced(src)
        self.assertIs(ss._paced(src), a)                                                  # 같은 문장은 한 번만 변환
        ss.play_speed = 1.0
        self.assertIs(ss._paced(src), src)
        ss.play_speed, ss.paced_cache = 0.85, {}
        orig = sm.time_stretch
        sm.time_stretch = lambda *x, **k: (_ for _ in ()).throw(ValueError("boom"))
        try:
            self.assertEqual(ss._paced(src), src)                                         # 변환 실패 → 원래 음성
        finally:
            sm.time_stretch = orig


class Feedback2Test(unittest.TestCase):
    """두 번째 피드백 반영: 변환 길이·홀수 바이트, 없는 날짜 만들지 않기, 반품 진행 문의 가드."""

    def test_stretch_length_and_odd_bytes(self):
        from agent.audio import time_stretch
        t = np.arange(16000 * 3) / 16000
        src = (np.sin(2 * np.pi * 200 * t) * 6000).astype(np.int16).tobytes()
        for sp in (0.85, 0.9):
            out = time_stretch(src, sp)
            self.assertEqual(len(out) // 2, int(round(len(src) // 2 / sp)))              # 끝까지 (잘림 없음)
        odd = time_stretch(src + b"\x01", 0.85)                                          # 홀수 바이트도 오류 없이
        self.assertEqual(len(odd) % 2, 0)

    def test_no_invented_dates(self):
        from agent import facts
        now = datetime(2026, 10, 6, 21, 0)
        body = {"status": "ok", "data": {"orders": [{"주문번호": "20261006-0094618", "상세주문번호": "20261006-0094618-01",
                "주문일자": "2026-10-06 19:01:44", "결제일자": "2026-10-06 19:01:50", "제품명": "하루온 위 가드", "옵션": "x",
                "수량": 1, "결제수단": ["계좌이체"], "결제여부": "T", "배송상태": "상품준비중"}]}}
        txt = facts.render_tool_result("get_order_info", body, now)
        self.assertIn("출고예정일은 조회 결과에 없음", txt)
        self.assertIn("평일 오후 3시 이후 결제", txt)
        self.assertNotIn("출고 예정 10월", txt)
        ret = {"status": "ok", "data": {"반품정보": [{"주문번호": "20260915-0067173", "상품명": "하루온 이눌린 배변 가드",
                                                   "현재상태": "반품처리중[수거전]", "송장번호": "203554713824"}]}}
        rtxt = facts.render_tool_result("get_return_info", ret, now)
        self.assertNotIn("9월 15일", rtxt)                                               # 주문번호 앞자리로 주문일을 만들지 않는다
        self.assertIn("주문일은 반품 조회 결과에 없다", rtxt)
        self.assertIn("20260915-0067173", rtxt)

    def test_order_date_removed_without_evidence(self):
        from agent import facts
        now = datetime(2026, 10, 1, 12)
        ev_ret = "주문일은 반품 조회 결과에 없다.\n1) 주문번호 20260915-0067173 / 현재상태 반품처리중[수거전]"
        out, fixes = facts.verify_answer("9월 15일에 주문하신 이눌린 배변 가드는 반품처리중입니다.", ev_ret, now)
        self.assertNotIn("9월 15일", out)
        self.assertIn("근거에 없는 주문일 제거", fixes)
        ev_ord = "1) 주문번호 20260928-0000001 / 주문일 9월 28일 08:25 (3일 전) / 배송상태 배송중"
        self.assertEqual(facts.verify_answer("9월 28일에 주문하신 제품은 배송 중이에요.", ev_ord, now)[0],
                         "9월 28일에 주문하신 제품은 배송 중이에요.")                               # 근거가 있으면 그대로

    def test_cancelled_order_marked(self):
        from agent import facts
        body = {"status": "ok", "data": {"orders": [{"주문번호": "20260924-0076068", "상세주문번호": "20260924-0076068-01",
                "주문일자": "2026-09-24 10:00:00", "제품명": "x", "옵션": "x", "수량": 1, "결제여부": "T", "배송상태": "취소완료"}]}}
        self.assertIn("반품·교환 대상 아님", facts.render_tool_result("get_order_info", body, datetime(2026, 10, 1, 12)))

    def test_needs_return_lookup(self):
        from agent.toolflow import needs_return_lookup
        self.assertTrue(needs_return_lookup("환불은 언제까지 이뤄지나요?", [], True))
        self.assertFalse(needs_return_lookup("환불은 언제까지 이뤄지나요?", [], False))   # 전화번호 없음
        self.assertFalse(needs_return_lookup("반품하고 싶어요", [], True))                 # 진행 문의가 아님
        done = [{"role": "tool_result", "name": "get_return_info", "result": {"status": "ok"}}]
        self.assertFalse(needs_return_lookup("환불 아직 안 들어왔어요", done, True))       # 이미 조회함


if __name__ == "__main__":
    unittest.main()
