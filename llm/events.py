"""
FSM 판단 결과 → LLM 안내 모듈로 전달되는 이벤트 정의.

각 이벤트와 실제 발생원의 연결은 llm/adapters.py(비전 이벤트)와
schedule_monitor.py(시간 기반 이벤트)가 담당한다.
비전/FSM 쪽은 이 모듈 내부를 몰라도 되고, 우리 쪽도 비전 내부를 모른다 —
서로 이벤트 하나만 주고받는 인터페이스 계약이 모듈화의 실체다.
"""

from enum import Enum


class GuidanceEvent(Enum):
    # ---- 약 복용 (medication-detector 브랜치) ----
    MEDICATION_DONE = "medication_done"                  # 복용 완료 (log_state=LOGGED)
    DUPLICATE_MEDICATION = "duplicate_medication"        # 이미 복용한 시간대에 복용 시도(GRABBED)
    MEDICATION_OFF_SCHEDULE = "medication_off_schedule"  # 복약 시간대 밖 복용 시도(GRABBED)
    MEDICATION_CHECK_NEEDED = "medication_check_needed"  # 복용 확인 필요(약 분실/미복용 의심)
    MEDICATION_MISSED = "medication_missed"              # 시간대 종료 임박, 미복용 → 재안내

    # ---- 식사 (feature/meal-fsm 브랜치) ----
    MEAL_START = "meal_start"                            # 식사 시작 확정 (SESSION_STARTED)
    MEAL_DONE = "meal_done"                              # 식사 종료 확정 (SESSION_FINISHED)
    DUPLICATE_MEAL = "duplicate_meal"                    # 이미 식사한 시간대에 새 식사 시작
    MEAL_MISSED = "meal_missed"                          # 시간대 종료 임박, 미식사 → 재안내

    # ---- 외출 (main 브랜치 exit_system) ----
    GOING_OUT_DETECTED = "going_out_detected"            # 현관 밖 방향 이동 감지 (OUTWARD_CROSSING)

    # ---- 정의만 (발생원 미연결) ----
    MEDICATION_TIME = "medication_time"                  # 시간대 시작 시 복용 유도
    MEAL_TIME = "meal_time"                              # 시간대 시작 시 식사 안내
    RETURN_HOME = "return_home"                          # 귀가 감지 (귀가 방향 FSM 추가 시)


# 실제 발생원(비전 어댑터 또는 스케줄 모니터)과 연결되는 이벤트
ACTIVE_EVENTS = {
    GuidanceEvent.MEDICATION_DONE,
    GuidanceEvent.DUPLICATE_MEDICATION,
    GuidanceEvent.MEDICATION_OFF_SCHEDULE,
    GuidanceEvent.MEDICATION_CHECK_NEEDED,
    GuidanceEvent.MEDICATION_MISSED,
    GuidanceEvent.MEAL_START,
    GuidanceEvent.MEAL_DONE,
    GuidanceEvent.DUPLICATE_MEAL,
    GuidanceEvent.MEAL_MISSED,
    GuidanceEvent.GOING_OUT_DETECTED,
}
