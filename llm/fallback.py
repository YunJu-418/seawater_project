"""
폴백 템플릿 문장.

네트워크 장애, sLLM 오류, 응답 검증 실패 시 즉시 사용되는 고정 안내 문장.
노인 의사소통 교재 원칙(한 문장 한 행동, 테스트식 질문 금지, 지시대명사 금지,
유도형 표현, 안심시키는 어조)을 지켜 사람이 직접 작성한다.

새 문장을 추가·수정하면 scripts/generate_fallback_audio.py 로
오프라인 폴백 mp3를 다시 생성해 둘 것.
"""

from llm.events import GuidanceEvent

FALLBACK_SENTENCES = {
    # ---- 약 복용 ----
    GuidanceEvent.MEDICATION_DONE: "약을 잘 드셨어요. 오늘도 건강하게 보내세요.",
    GuidanceEvent.DUPLICATE_MEDICATION: "약은 조금 전에 드셨어요. 안심하고 내려놓으셔도 돼요.",
    GuidanceEvent.MEDICATION_OFF_SCHEDULE: "약은 정해진 시간에 드시는 게 좋아요. 시간이 되면 알려드릴게요.",
    GuidanceEvent.MEDICATION_CHECK_NEEDED: "약이 손에 있는지 한번 봐 주세요. 바닥도 한번 살펴봐 주세요.",
    GuidanceEvent.MEDICATION_TIME: "지금 약 드실 시간이에요. 약을 드셔 보세요.",
    GuidanceEvent.MEDICATION_MISSED: "약 드실 시간이 지나가고 있어요. 지금 약을 드셔 보세요.",
    # ---- 식사 ----
    GuidanceEvent.MEAL_START: "식사를 시작하셨네요. 천천히 맛있게 드세요.",
    GuidanceEvent.MEAL_DONE: "식사를 잘 하셨어요. 참 잘하셨어요.",
    GuidanceEvent.DUPLICATE_MEAL: "조금 전에 식사를 하셨어요. 편안히 쉬셔도 돼요.",
    GuidanceEvent.MEAL_TIME: "식사하실 시간이에요. 맛있게 드셔 보세요.",
    GuidanceEvent.MEAL_MISSED: "식사하실 시간이 지나가고 있어요. 지금 식사를 해 보세요.",
    # ---- 외출 ----
    GuidanceEvent.GOING_OUT_DETECTED: "외출하시나 봐요. 겉옷을 챙겨 보세요.",
    GuidanceEvent.RETURN_HOME: "집에 잘 오셨어요. 편히 쉬세요.",
}


def get_fallback(event: GuidanceEvent) -> str:
    return FALLBACK_SENTENCES[event]
