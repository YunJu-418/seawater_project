"""
팀원 브랜치의 이벤트 → GuidanceEvent 번역 계층 (어댑터).

원칙: 우리 파트는 비전/FSM 코드를 import 하지 않는다.
각 브랜치가 이미 만들어 내는 결과물(상태 문자열, 딕셔너리, 이벤트 객체)의
'모양'만 보고 duck typing 으로 번역한다. 팀원 코드가 바뀌어도 이 파일만 고치면 된다.

발생원 대응표
    medication-detector : medication_result["state"] + log_if_taken() 반환 dict
                          → MedicationAdapter.translate()
    feature/meal-fsm    : MealFSM.update(observation).events (MealEvent 리스트)
                          → from_meal_events()
    main (exit_system)  : ExitFSM.update() 반환 FSMEvent 리스트
                          → from_exit_events()
    (시간 기반)          : schedule_monitor.ScheduleMonitor.tick()
"""

from datetime import datetime
from typing import Any, Iterable, Optional

from llm.events import GuidanceEvent


def _get(obj: Any, key: str, default=None):
    """dict / dataclass / 임의 객체를 같은 방식으로 읽는다."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ======================================================================
# 1) 약 복용 — medication-detector 브랜치
# ======================================================================

_MEDICATION_LOG_STATE_TO_EVENT = {
    "LOGGED": GuidanceEvent.MEDICATION_DONE,
    "ALREADY_LOGGED": GuidanceEvent.DUPLICATE_MEDICATION,
    "NOT_MEDICATION_TIME": GuidanceEvent.MEDICATION_OFF_SCHEDULE,
}


class MedicationAdapter:
    """
    비전 상태(매 프레임)와 복용 기록 결과를 받아 '전이 순간'만 안내 이벤트로 번역.

    핵심 설계 — 경고는 삼키기 전에:
      복용 '시도'(GRABBED) 시점에 시간대를 확인해서
        · 시간대 밖이면          → MEDICATION_OFF_SCHEDULE (미리 안내)
        · 이미 복용한 시간대면    → DUPLICATE_MEDICATION   (미리 안내)
      를 내보낸다. 완료(DONE) 시점에야 알리면 이미 삼킨 뒤라 늦기 때문.
      GRABBED 시점에 경고가 나갔으면, 완료 시점의 같은 경고는 생략한다.

    상태 전이 규칙 (edge trigger — 같은 상태의 연속 프레임은 무시):
        대기 → GRABBED                : 시간대 검사 후 필요 시 경고
        →   MEDICATION_CHECK_NEEDED   : 약 분실/미복용 의심 → 살펴보기 안내 (동작당 1회)
        log_state NOT_DONE → LOGGED   : 복용 완료 칭찬
    """

    IDLE_STATES = frozenset({None, "WAITING", "NO_MEDICATION_DETECTED"})

    def __init__(self, medication_logger=None, monitor=None):
        """
        medication_logger: 팀원 MedicationLogger 객체 (권장 — 시간표 단일 기준).
        monitor: schedule_monitor.ScheduleMonitor (logger 대신 슬롯 조회용).
        둘 다 없으면 기본 ScheduleMonitor를 내부 생성한다.
        """
        self._logger = medication_logger
        self._monitor = monitor
        self._prev_vision: Optional[str] = None
        self._prev_log: Optional[str] = "NOT_DONE"
        self._warned_this_episode = False   # GRABBED 시점 경고를 이미 했는가
        self._checked_this_episode = False  # CHECK_NEEDED 안내를 이미 했는가

    def translate(self, vision_state=None, log_result=None, now=None) -> list:
        """
        매 프레임 호출. 안내할 것이 있으면 [(GuidanceEvent, context), ...] 반환.

        사용 예 (medication-detector main.py 루프 안):
            pairs = adapter.translate(
                vision_state=medication_result["state"],
                log_result=medication_log_result,
            )
            guidance.notify_all(pairs)
        """
        # 하위호환: 예전 방식 translate(log_dict) 한 개 인자 호출 지원
        if isinstance(vision_state, dict) and log_result is None:
            vision_state, log_result = None, vision_state

        now = now or datetime.now()
        t = now.strftime("%H:%M")
        out = []

        # ---- 1) 비전 상태 edge: 복용 '시도' 시점의 조기 안내 ----
        if vision_state is not None:
            prev = self._prev_vision
            self._prev_vision = vision_state

            entering = prev in self.IDLE_STATES and vision_state not in self.IDLE_STATES
            if entering:
                # 새 복용 동작 시작 → 에피소드 플래그 초기화 후 시간대 검사
                self._warned_this_episode = False
                self._checked_this_episode = False

                slot_key, label, taken = self._slot_status(now)
                if slot_key is None:
                    ctx = {"time": t}
                    next_label, next_start = self._next_slot(now)
                    if next_label:
                        ctx["next_slot_label"] = next_label
                        ctx["next_slot_starts_at"] = next_start
                    out.append((GuidanceEvent.MEDICATION_OFF_SCHEDULE, ctx))
                    self._warned_this_episode = True
                elif taken:
                    out.append((
                        GuidanceEvent.DUPLICATE_MEDICATION,
                        {"meal_label": label, "time": t},
                    ))
                    self._warned_this_episode = True

            if (
                vision_state == "MEDICATION_CHECK_NEEDED"
                and prev != "MEDICATION_CHECK_NEEDED"
                and not self._checked_this_episode
            ):
                self._checked_this_episode = True
                out.append((GuidanceEvent.MEDICATION_CHECK_NEEDED, {"time": t}))

            if vision_state in self.IDLE_STATES:
                self._warned_this_episode = False
                self._checked_this_episode = False

        # ---- 2) 복용 기록 edge: 완료 시점 ----
        if log_result is not None:
            state = _get(log_result, "log_state")
            prev_log = self._prev_log
            self._prev_log = state

            if state != prev_log and prev_log in (None, "NOT_DONE"):
                event = _MEDICATION_LOG_STATE_TO_EVENT.get(state)
                if event is GuidanceEvent.MEDICATION_DONE:
                    context = {"time": t}
                    for key in ("meal_label", "taken_time"):
                        value = _get(log_result, key)
                        if value:
                            context[key] = value
                    out.append((event, context))
                elif event is not None and not self._warned_this_episode:
                    # GRABBED 시점 조기 경고가 없었던 경우에만 완료 시점 경고 (이중 안내 방지)
                    context = {"time": t}
                    label = _get(log_result, "meal_label")
                    if label:
                        context["meal_label"] = label
                    out.append((event, context))
                    self._warned_this_episode = True

        return out

    def reset(self):
        """MedicationDetector.reset() 과 함께 호출하면 상태가 같이 초기화된다."""
        self._prev_vision = None
        self._prev_log = "NOT_DONE"
        self._warned_this_episode = False
        self._checked_this_episode = False

    # ---------------- internal ----------------

    def _slot_status(self, now):
        """(slot_key, 라벨, 복용 여부) — logger > monitor > 기본 monitor 순으로 조회."""
        if self._logger is not None:
            try:
                key = self._logger.get_current_meal(now)
                if key is None:
                    return None, None, False
                label = self._logger.meal_times[key]["label"]
                return key, label, bool(self._logger.has_taken_today(key, now))
            except Exception:
                pass  # 팀원 로거 조회 실패 시 모니터로 폴백
        if self._monitor is None:
            from schedule_monitor import ScheduleMonitor
            self._monitor = ScheduleMonitor()
        return self._monitor.slot_status(now)

    def _next_slot(self, now):
        """(다음 시간대 라벨, 시작 시각 문자열) — 시각으로 추측하지 않도록 정답을 제공."""
        if self._logger is not None:
            try:
                t = now.time()
                items = list(self._logger.meal_times.items())
                for _key, info in items:
                    if t < info["start"]:
                        return info["label"], info["start"].strftime("%H:%M")
                info = items[0][1]
                return info["label"], info["start"].strftime("%H:%M")  # 내일 첫 시간대
            except Exception:
                pass
        if self._monitor is None:
            from schedule_monitor import ScheduleMonitor
            self._monitor = ScheduleMonitor()
        return self._monitor.next_slot(now)


# 하위호환 별칭 (이전 통합 가이드에서 쓰던 이름)
MedicationLogAdapter = MedicationAdapter


# ======================================================================
# 2) 식사 — feature/meal-fsm 브랜치
# ======================================================================

_MEAL_EVENT_TYPE_TO_EVENT = {
    "session_started": GuidanceEvent.MEAL_START,
    "session_finished": GuidanceEvent.MEAL_DONE,
}


def from_meal_events(events: Iterable[Any], monitor=None, now=None) -> list:
    """
    MealFSM.update(observation).events → [(GuidanceEvent, context), ...]

    monitor(ScheduleMonitor)를 넘기면 중복 식사 방지가 켜진다:
      - SESSION_STARTED 인데 현재 시간대에 이미 식사 기록이 있으면
        MEAL_START 대신 DUPLICATE_MEAL ("조금 전에 식사하셨어요") 로 번역
      - SESSION_FINISHED 는 monitor.record_meal_done() 으로 기록
    식사 자체는 시간대 제한 없음 — 시간대 밖 식사는 그대로 MEAL_START.

    식사 '시작 확정'과 '종료 확정'만 안내 대상. BITE/DRINK/REST 등 순간 이벤트는
    식사 내내 소음이 되므로 의도적으로 제외 (_MEAL_EVENT_TYPE_TO_EVENT 에 추가로 확장 가능).

    사용 예 (main_meal.py):
        self.guidance.notify_all(from_meal_events(result.events, monitor=self.monitor))
    """
    now = now or datetime.now()
    out = []
    for ev in events or []:
        event_type = _get(ev, "event_type")
        value = getattr(event_type, "value", event_type)  # str Enum / 문자열 모두 대응
        if value is None:
            continue
        event = _MEAL_EVENT_TYPE_TO_EVENT.get(str(value).lower())
        if event is None:
            continue

        context = {"time": now.strftime("%H:%M")}
        # 시간대 이름을 모델이 시각으로 추측하지 않도록, 지금 시간대 라벨을 정답으로 제공
        from schedule_monitor import current_slot
        _skey, slot_label, _send = current_slot(now)
        if slot_label:
            context["meal_label"] = slot_label
        for key in ("bite_count", "drink_count"):
            v = _get(ev, key)
            if v:
                context[key] = v

        if event is GuidanceEvent.MEAL_START and monitor is not None:
            if monitor.meal_done_in_current_slot(now):
                event = GuidanceEvent.DUPLICATE_MEAL
        elif event is GuidanceEvent.MEAL_DONE and monitor is not None:
            monitor.record_meal_done(now)

        out.append((event, context))
    return out


# ======================================================================
# 3) 외출 — main 브랜치 exit_system
# ======================================================================

_EXIT_NAME_TO_EVENT = {
    "OUTWARD_CROSSING": GuidanceEvent.GOING_OUT_DETECTED,
}


def from_exit_events(events: Iterable[Any]) -> list:
    """
    ExitFSM.update() 반환 FSMEvent 리스트 → [(GuidanceEvent, context), ...]

    OUTWARD_CROSSING(실외선 통과) 순간이 아직 사람이 현관에 있는 시점이라,
    "겉옷을 챙겨 보세요"를 말할 유일한 타이밍이다.
    EXIT_CONFIRMED 는 사람이 사라진 뒤 확정되는 이벤트 → 들을 사람이 없으므로
    음성 안내 대상이 아니고, 보호자 카카오 알림(run_exit.py 기존 로직) 담당이다.
    """
    out = []
    for ev in events or []:
        name = _get(ev, "name")
        event = _EXIT_NAME_TO_EVENT.get(name)
        if event is None:
            continue
        out.append((event, {}))
    return out
