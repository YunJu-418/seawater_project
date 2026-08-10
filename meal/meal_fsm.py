"""
식사 행동 인식 시스템 V3의 FSM.

역할
----
- vision/meal.py가 생성한 BiteSessionObservation과
  DrinkSessionObservation을 입력으로 받는다.
- Bite / Drink 완료 신호의 상승 순간만 1회 카운트한다.
- WAITING / EATING / REST / OTHER / END_CANDIDATE / FINISHED
  식사 상태를 관리한다.
- 상태 변화와 확정 이벤트를 MealEvent로 반환한다.

설계 원칙
---------
- 점수 기반 후보 판단을 하지 않는다.
- Bite와 Drink의 세부 동작 판단은 vision/meal.py의 세션 관측값을 사용한다.
- 동일한 completed=True가 여러 프레임 유지되어도 한 번만 카운트한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
import json
import time

from meal.meal_state import (
    BiteState,
    DrinkState,
    HandSide,
    EventType,
    MealAction,
    MealEvent,
    MealObservation,
    MealSessionSnapshot,
    MealState,
)


@dataclass
class FSMUpdateResult:
    """한 프레임 FSM 업데이트 결과."""

    snapshot: MealSessionSnapshot
    events: List[MealEvent] = field(default_factory=list)


class MealFSM:
    """V3 식사 행동 상태기계."""

    def __init__(
        self,
        config_path: str | Path = "meal/meal_config.json",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        self.session_id = self._new_session_id()
        self.started_at = time.monotonic()

        self._snapshot = MealSessionSnapshot(
            session_id=self.session_id,
            started_at=self.started_at,
            updated_at=self.started_at,
        )

        self._previous_bite_completed = False
        self._previous_drink_completed = False

        self._waiting_bite_times: List[float] = []
        self._waiting_drink_times: List[float] = []

        self._meal_candidate_score: float = 0.0
        self._meal_candidate_started_at: Optional[float] = None
        self._last_candidate_update_at: Optional[float] = None
        self._last_evidence_bite_at: Optional[float] = None
        self._candidate_bite_times: List[float] = []
        self._candidate_context_bite_times: List[float] = []

        # 원시 Bite Count와 Meal Bite를 분리한다.
        # 확정 Bite는 잠시 보류한 뒤 객체 기억과 행동 문맥으로 채택한다.
        self._pending_meal_bites: List[Dict[str, Any]] = []
        self._accepted_meal_bite_times: List[float] = []
        self._accepted_context_bite_times: List[float] = []
        self._rejected_meal_bite_count: int = 0
        self._last_meal_evidence_decision: Dict[str, Any] = {
            "object": "none",
            "label": "",
            "behavior_context": False,
            "drink_motion_evidence": False,
            "decision": "none",
            "reason": "no_decision_yet",
        }

        self._strong_context_first_seen_at: Optional[float] = None
        self._last_strong_context_seen_at: Optional[float] = None
        self._last_weak_context_seen_at: Optional[float] = None
        self._strong_context_active: bool = False
        self._weak_context_active: bool = False

        # 기존 디버그/호환 코드가 참조할 수 있도록 유지한다.
        self._object_context_started_at: Optional[float] = None
        self._object_context_active: bool = False

        self._rest_started_at: Optional[float] = None
        self._end_candidate_started_at: Optional[float] = None
        self._other_started_at: Optional[float] = None

        self._last_frame_index = 0

    @property
    def snapshot(self) -> MealSessionSnapshot:
        return self._copy_snapshot()

    def reset(self) -> None:
        self.session_id = self._new_session_id()
        self.started_at = time.monotonic()

        self._snapshot = MealSessionSnapshot(
            session_id=self.session_id,
            started_at=self.started_at,
            updated_at=self.started_at,
        )

        self._previous_bite_completed = False
        self._previous_drink_completed = False
        self._waiting_bite_times = []
        self._waiting_drink_times = []

        self._meal_candidate_score = 0.0
        self._meal_candidate_started_at = None
        self._last_candidate_update_at = None
        self._last_evidence_bite_at = None
        self._candidate_bite_times = []
        self._candidate_context_bite_times = []

        self._pending_meal_bites = []
        self._accepted_meal_bite_times = []
        self._accepted_context_bite_times = []
        self._rejected_meal_bite_count = 0
        self._last_meal_evidence_decision = {
            "object": "none",
            "label": "",
            "behavior_context": False,
            "drink_motion_evidence": False,
            "decision": "none",
            "reason": "no_decision_yet",
        }

        self._strong_context_first_seen_at = None
        self._last_strong_context_seen_at = None
        self._last_weak_context_seen_at = None
        self._strong_context_active = False
        self._weak_context_active = False

        self._object_context_started_at = None
        self._object_context_active = False

        self._rest_started_at = None
        self._end_candidate_started_at = None
        self._other_started_at = None
        self._last_frame_index = 0

    def update(
        self,
        observation: MealObservation,
    ) -> FSMUpdateResult:
        timestamp = float(observation.timestamp)
        frame_index = int(observation.frame_index)
        self._last_frame_index = frame_index

        events: List[MealEvent] = []

        bite_completed = bool(
            observation.bite_session.session_completed
        )
        drink_completed = bool(
            observation.drink_session.session_completed
        )

        bite_rising = (
            bite_completed
            and not self._previous_bite_completed
        )
        drink_rising = (
            drink_completed
            and not self._previous_drink_completed
        )

        self._previous_bite_completed = bite_completed
        self._previous_drink_completed = drink_completed

        previous_bite_state = self._snapshot.bite_state
        previous_drink_state = self._snapshot.drink_state
        previous_meal_state = self._snapshot.meal_state

        self._snapshot.bite_state = (
            observation.bite_session.state
        )
        self._snapshot.drink_state = (
            observation.drink_session.state
        )
        self._snapshot.updated_at = timestamp

        if (
            self._snapshot.bite_state
            != previous_bite_state
        ):
            events.append(
                self._event(
                    event_type=EventType.BITE_STATE_CHANGED,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    action=MealAction.BITE_CANDIDATE,
                    message=(
                        f"Bite state: "
                        f"{previous_bite_state.value} -> "
                        f"{self._snapshot.bite_state.value}"
                    ),
                )
            )

        if (
            self._snapshot.drink_state
            != previous_drink_state
        ):
            events.append(
                self._event(
                    event_type=EventType.DRINK_STATE_CHANGED,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    action=MealAction.DRINK_CANDIDATE,
                    message=(
                        f"Drink state: "
                        f"{previous_drink_state.value} -> "
                        f"{self._snapshot.drink_state.value}"
                    ),
                )
            )

        if bite_rising:
            self._confirm_bite(
                observation=observation,
                events=events,
            )

        if drink_rising:
            self._confirm_drink(
                observation=observation,
                events=events,
            )

        self._update_current_action(
            observation=observation,
            bite_rising=bite_rising,
            drink_rising=drink_rising,
        )

        self._update_meal_candidate(
            observation=observation,
            bite_rising=bite_rising,
            drink_rising=drink_rising,
        )

        self._update_meal_state(
            observation=observation,
            bite_rising=bite_rising,
            drink_rising=drink_rising,
            events=events,
        )

        if (
            self._snapshot.meal_state
            != previous_meal_state
        ):
            events.append(
                self._event(
                    event_type=EventType.MEAL_STATE_CHANGED,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    action=self._snapshot.current_action,
                    message=(
                        f"Meal state: "
                        f"{previous_meal_state.value} -> "
                        f"{self._snapshot.meal_state.value}"
                    ),
                )
            )

        self._snapshot.clamp()
        return FSMUpdateResult(
            snapshot=self._copy_snapshot(),
            events=events,
        )

    # ------------------------------------------------------------------
    # Bite / Drink 확정
    # ------------------------------------------------------------------

    def _confirm_bite(
        self,
        observation: MealObservation,
        events: List[MealEvent],
    ) -> None:
        timestamp = float(observation.timestamp)
        frame_index = int(observation.frame_index)

        minimum_interval = self._cfg(
            "bite_fsm",
            "minimum_bite_interval_seconds",
            default=1.1,
        )

        last_time = self._snapshot.last_bite_time
        if (
            last_time is not None
            and timestamp - last_time
            < minimum_interval
        ):
            return

        self._snapshot.bite_count += 1
        self._snapshot.last_bite_time = timestamp
        self._snapshot.last_activity_time = timestamp
        self._snapshot.current_action = MealAction.BITE

        self._waiting_bite_times.append(timestamp)
        self._trim_recent_times(
            self._waiting_bite_times,
            timestamp,
            self._cfg(
                "meal_fsm",
                "meal_start_bite_window_seconds",
                default=90.0,
            ),
        )

        events.append(
            self._event(
                event_type=EventType.BITE_CONFIRMED,
                timestamp=timestamp,
                frame_index=frame_index,
                action=MealAction.BITE,
                confidence=1.0,
                message="Bite confirmed.",
                metadata={
                    "active_hand": (
                        observation.bite_session
                        .active_hand.value
                    ),
                    "dwell_duration": (
                        observation.bite_session
                        .dwell_duration
                    ),
                    "total_duration": (
                        observation.bite_session
                        .total_duration
                    ),
                },
            )
        )

    def _confirm_drink(
        self,
        observation: MealObservation,
        events: List[MealEvent],
    ) -> None:
        timestamp = float(observation.timestamp)
        frame_index = int(observation.frame_index)

        minimum_interval = self._cfg(
            "drink_fsm",
            "minimum_drink_interval_seconds",
            default=1.5,
        )

        last_time = self._snapshot.last_drink_time
        if (
            last_time is not None
            and timestamp - last_time
            < minimum_interval
        ):
            return

        self._snapshot.drink_count += 1
        self._snapshot.last_drink_time = timestamp
        self._snapshot.last_activity_time = timestamp
        self._snapshot.current_action = MealAction.DRINK

        self._waiting_drink_times.append(timestamp)
        self._trim_recent_times(
            self._waiting_drink_times,
            timestamp,
            self._cfg(
                "meal_fsm",
                "meal_start_drink_window_seconds",
                default=120.0,
            ),
        )

        events.append(
            self._event(
                event_type=EventType.DRINK_CONFIRMED,
                timestamp=timestamp,
                frame_index=frame_index,
                action=MealAction.DRINK,
                confidence=1.0,
                message="Drink confirmed.",
                metadata={
                    "contact_hand": (
                        observation.drink_session
                        .contact_hand.value
                    ),
                    "vessel_label": (
                        observation.drink_session
                        .vessel_label
                    ),
                    "session_duration": (
                        observation.drink_session
                        .session_duration
                    ),
                },
            )
        )

    # ------------------------------------------------------------------
    # Meal 상태
    # ------------------------------------------------------------------

    def _update_current_action(
        self,
        observation: MealObservation,
        bite_rising: bool,
        drink_rising: bool,
    ) -> None:
        if drink_rising:
            self._snapshot.current_action = MealAction.DRINK
            return

        if bite_rising:
            self._snapshot.current_action = MealAction.BITE
            return

        if observation.drink_session.session_active:
            self._snapshot.current_action = (
                MealAction.DRINK_CANDIDATE
            )
            return

        if observation.bite_session.active:
            self._snapshot.current_action = (
                MealAction.BITE_CANDIDATE
            )
            return

        if (
            observation.suggested_action
            != MealAction.NONE
        ):
            self._snapshot.current_action = (
                observation.suggested_action
            )
            return

        self._snapshot.current_action = MealAction.NONE

    def _update_meal_state(
        self,
        observation: MealObservation,
        bite_rising: bool,
        drink_rising: bool,
        events: List[MealEvent],
    ) -> None:
        timestamp = float(observation.timestamp)
        frame_index = int(observation.frame_index)

        activity_now = bool(
            bite_rising
            or drink_rising
            or observation.bite_session.active
            or observation.drink_session.session_active
            or observation.suggested_action
            == MealAction.UTENSILING
        )

        if activity_now:
            self._snapshot.last_activity_time = timestamp

        state = self._snapshot.meal_state

        if state == MealState.WAITING:
            if self._meal_start_condition(
                observation=observation,
                timestamp=timestamp,
            ):
                self._snapshot.meal_state = MealState.EATING
                self._snapshot.last_activity_time = timestamp
                self._rest_started_at = None
                self._end_candidate_started_at = None

                events.append(
                    self._event(
                        event_type=EventType.SESSION_STARTED,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        action=self._snapshot.current_action,
                        message="Meal session started.",
                        metadata={
                            "candidate_score": (
                                self._meal_candidate_score
                            ),
                            "object_context_active": (
                                self._object_context_active
                            ),
                            "strong_context_active": (
                                self._strong_context_active
                            ),
                            "weak_context_active": (
                                self._weak_context_active
                            ),
                            "recent_bite_evidence": len(
                                self._candidate_bite_times
                            ),
                            "context_bite_evidence": len(
                                self._accepted_context_bite_times
                            ),
                            "meal_bite_evidence": len(
                                self._accepted_meal_bite_times
                            ),
                            "rejected_meal_bites": (
                                self._rejected_meal_bite_count
                            ),
                            "recent_drink_evidence": len(
                                self._waiting_drink_times
                            ),
                        },
                    )
                )
            return

        if state == MealState.FINISHED:
            return

        last_activity = (
            self._snapshot.last_activity_time
            if self._snapshot.last_activity_time
            is not None
            else timestamp
        )
        inactive_duration = max(
            0.0,
            timestamp - last_activity,
        )

        if activity_now:
            self._rest_started_at = None
            self._end_candidate_started_at = None
            self._other_started_at = None
            self._snapshot.rest_duration = 0.0
            self._snapshot.end_candidate_duration = 0.0

            if self._snapshot.meal_state in {
                MealState.REST,
                MealState.OTHER,
                MealState.END_CANDIDATE,
            }:
                self._snapshot.meal_state = MealState.EATING
            return

        rest_enter = self._cfg(
            "meal_fsm",
            "rest_enter_seconds",
            default=30.0,
        )
        end_candidate = self._cfg(
            "meal_fsm",
            "end_candidate_seconds",
            default=180.0,
        )
        finish_seconds = self._cfg(
            "meal_fsm",
            "finish_seconds",
            default=300.0,
        )

        if inactive_duration >= finish_seconds:
            self._snapshot.meal_state = MealState.FINISHED
            self._snapshot.finished = True

            events.append(
                self._event(
                    event_type=EventType.SESSION_FINISHED,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    action=MealAction.NONE,
                    message="Meal session finished.",
                )
            )
            return

        if inactive_duration >= end_candidate:
            if (
                self._snapshot.meal_state
                != MealState.END_CANDIDATE
            ):
                self._end_candidate_started_at = timestamp
                events.append(
                    self._event(
                        event_type=(
                            EventType.END_CANDIDATE_STARTED
                        ),
                        timestamp=timestamp,
                        frame_index=frame_index,
                        action=MealAction.NONE,
                        message="Meal end candidate started.",
                    )
                )

            self._snapshot.meal_state = (
                MealState.END_CANDIDATE
            )
            self._snapshot.end_candidate_duration = (
                inactive_duration
            )
            self._snapshot.rest_duration = inactive_duration
            return

        if inactive_duration >= rest_enter:
            if self._snapshot.meal_state != MealState.REST:
                self._rest_started_at = timestamp
                events.append(
                    self._event(
                        event_type=EventType.REST_STARTED,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        action=MealAction.REST,
                        message="Rest started.",
                    )
                )

            self._snapshot.meal_state = MealState.REST
            self._snapshot.current_action = MealAction.REST
            self._snapshot.rest_duration = inactive_duration
            return

        self._snapshot.meal_state = MealState.EATING
        self._snapshot.rest_duration = 0.0
        self._snapshot.end_candidate_duration = 0.0

    def _update_meal_candidate(
        self,
        observation: MealObservation,
        bite_rising: bool,
        drink_rising: bool,
    ) -> None:
        """
        식사 시작 후보는 Bite만으로 구성한다.

        Drink, 컵, 물병은 WAITING 상태의 식사 시작 증거로 사용하지 않는다.
        확정 Bite 중 low-zone / plate / utensil / handheld 증거가 있는
        Bite를 별도의 '식사 문맥 Bite'로 기록한다.
        """
        timestamp = float(observation.timestamp)
        config = self.config.get("meal_candidate", {})

        # 점수는 디버그 표시와 후보 생존 정도에만 사용한다.
        # 실제 EATING 진입은 횟수·문맥 횟수·시간 분산으로 결정한다.
        if self._last_candidate_update_at is not None:
            elapsed = max(
                0.0,
                timestamp - self._last_candidate_update_at,
            )
            decay_per_second = float(
                config.get("score_decay_per_second", 0.02)
            )
            self._meal_candidate_score = max(
                0.0,
                self._meal_candidate_score
                - elapsed * decay_per_second,
            )
        self._last_candidate_update_at = timestamp

        detected_labels = {
            detection.label.strip().lower()
            for detection in observation.detections
        }

        strong_labels = {
            str(label).strip().lower()
            for label in config.get(
                "strong_context_labels",
                [
                    "bowl",
                    "plate",
                    "dish",
                    "sandwich",
                    "apple",
                    "banana",
                    "orange",
                    "broccoli",
                    "carrot",
                    "hot dog",
                    "pizza",
                    "donut",
                    "cake",
                    "food",
                ],
            )
        }

        strong_seen_now = bool(
            detected_labels & strong_labels
        )
        context_gap_tolerance = float(
            config.get(
                "object_context_gap_tolerance_seconds",
                0.75,
            )
        )

        if strong_seen_now:
            if (
                self._strong_context_first_seen_at is None
                or (
                    self._last_strong_context_seen_at is not None
                    and timestamp - self._last_strong_context_seen_at
                    > context_gap_tolerance
                )
            ):
                self._strong_context_first_seen_at = timestamp

            self._last_strong_context_seen_at = timestamp
        elif (
            self._last_strong_context_seen_at is None
            or timestamp - self._last_strong_context_seen_at
            > context_gap_tolerance
        ):
            self._strong_context_first_seen_at = None

        minimum_context_seconds = float(
            config.get(
                "object_context_min_seconds",
                2.0,
            )
        )
        context_memory_seconds = float(
            config.get(
                "object_context_memory_seconds",
                5.0,
            )
        )

        context_stable = bool(
            self._strong_context_first_seen_at is not None
            and timestamp - self._strong_context_first_seen_at
            >= minimum_context_seconds
        )
        self._strong_context_active = bool(
            context_stable
            and self._last_strong_context_seen_at is not None
            and timestamp - self._last_strong_context_seen_at
            <= context_memory_seconds
        )

        # 컵·물병 문맥은 식사 시작에 사용하지 않는다.
        self._weak_context_active = False

        # 이전 필드와 로그 호환.
        self._object_context_active = self._strong_context_active
        if self._strong_context_active:
            if self._object_context_started_at is None:
                self._object_context_started_at = timestamp
        else:
            self._object_context_started_at = None

        candidate_window = float(
            config.get("candidate_window_seconds", 300.0)
        )
        self._trim_recent_times(
            self._candidate_bite_times,
            timestamp,
            candidate_window,
        )
        self._trim_recent_times(
            self._candidate_context_bite_times,
            timestamp,
            candidate_window,
        )
        self._trim_recent_times(
            self._accepted_meal_bite_times,
            timestamp,
            candidate_window,
        )
        self._trim_recent_times(
            self._accepted_context_bite_times,
            timestamp,
            candidate_window,
        )

        self._resolve_pending_meal_bites(
            observation=observation,
            timestamp=timestamp,
        )

        if bite_rising:
            minimum_gap = float(
                config.get(
                    "minimum_distinct_bite_gap_seconds",
                    6.0,
                )
            )
            previous_evidence_time = self._last_evidence_bite_at
            distinct_bite = bool(
                previous_evidence_time is None
                or timestamp - previous_evidence_time
                >= minimum_gap
            )

            if distinct_bite:
                self._last_evidence_bite_at = timestamp

                bite_debug = observation.bite_session.debug
                contact_source = str(
                    bite_debug.get(
                        "bite_contact_source",
                        "",
                    )
                ).lower()

                behavior_context = bool(
                    observation.bite_session.started_near_plate
                    or bite_debug.get(
                        "low_zone_context_active",
                        False,
                    )
                    or bite_debug.get(
                        "handheld_context_active",
                        False,
                    )
                    or contact_source
                    in {
                        "index_tip_utensil",
                        "virtual_utensil_tip",
                    }
                    or "utensil" in contact_source
                )

                carried_kind = str(
                    bite_debug.get(
                        "carried_context_kind",
                        "unknown",
                    )
                ).lower()
                carried_label = str(
                    bite_debug.get(
                        "carried_context_label",
                        "",
                    )
                ).lower()

                bite_hand = observation.bite_session.active_hand
                drink = observation.drink_session
                same_hand_drink = bool(
                    drink.contact_hand == bite_hand
                    or drink.contact_hand == HandSide.UNKNOWN
                )
                drink_motion_evidence = bool(
                    same_hand_drink
                    and (
                        drink.mouth_reached
                        or drink.near_mouth
                        or drink.session_completed
                    )
                )

                self._pending_meal_bites.append(
                    {
                        "timestamp": timestamp,
                        "hand": bite_hand,
                        "behavior_context": behavior_context,
                        "contact_source": contact_source,
                        "carried_kind": carried_kind,
                        "carried_label": carried_label,
                        "drink_motion_evidence": (
                            drink_motion_evidence
                        ),
                    }
                )

        # 의도적으로 drink_rising은 사용하지 않는다.
        _ = drink_rising

        if (
            self._accepted_meal_bite_times
            and self._meal_candidate_started_at is None
        ):
            self._meal_candidate_started_at = (
                self._accepted_meal_bite_times[0]
            )

        candidate_timeout = float(
            config.get("candidate_window_seconds", 300.0)
        )
        last_candidate_bite = (
            self._accepted_meal_bite_times[-1]
            if self._accepted_meal_bite_times
            else None
        )

        if (
            self._meal_candidate_started_at is not None
            and (
                last_candidate_bite is None
                or timestamp - last_candidate_bite
                > candidate_timeout
            )
        ):
            self._meal_candidate_score = 0.0
            self._meal_candidate_started_at = None
            self._last_evidence_bite_at = None
            self._candidate_bite_times = []
            self._candidate_context_bite_times = []
            self._pending_meal_bites = []
            self._accepted_meal_bite_times = []
            self._accepted_context_bite_times = []

    def _resolve_pending_meal_bites(
        self,
        observation: MealObservation,
        timestamp: float,
    ) -> None:
        """
        Confirmed Bite는 그대로 유지하고 Meal Bite 여부만 결정한다.

        Meal Evidence Filter는 DrinkFSM과 역할을 분리한다.

        - 같은 손의 phone 이력: Meal Bite 제외
        - food / utensil 이력: Meal Bite 인정
        - vessel은 중립 처리:
          물병·컵이 가까이 있다는 이유만으로 인정하거나 제외하지 않는다.
          기존 식사 행동 문맥이 있으면 인정한다.
        - unknown도 식사 행동 문맥이 있으면 인정한다.
        - 식사 문맥이 전혀 없으면 제외한다.

        Drink Count와 DrinkFSM 판정에는 영향을 주지 않는다.
        """
        config = self.config.get("meal_evidence", {})
        delay_seconds = float(
            config.get("decision_delay_seconds", 1.0)
        )

        current_hand = observation.active_hand
        current_debug = observation.bite_session.debug
        current_kind = str(
            current_debug.get(
                "carried_context_kind",
                "unknown",
            )
        ).lower()
        current_label = str(
            current_debug.get(
                "carried_context_label",
                "",
            )
        ).lower()

        remaining: List[Dict[str, Any]] = []

        for pending in self._pending_meal_bites:
            pending_hand = pending.get(
                "hand",
                HandSide.UNKNOWN,
            )
            age = timestamp - float(pending["timestamp"])

            if age < delay_seconds:
                # Bite를 수행한 동일 손의 객체 정보만 갱신한다.
                # 반대손의 휴대폰은 현재 Bite를 차단하지 않는다.
                if pending_hand == current_hand:
                    pending_kind = str(
                        pending.get(
                            "carried_kind",
                            "unknown",
                        )
                    ).lower()

                    if current_kind == "phone":
                        pending["carried_kind"] = "phone"
                        pending["carried_label"] = current_label

                    elif (
                        current_kind in {
                            "food",
                            "utensil",
                        }
                        and pending_kind != "phone"
                    ):
                        pending["carried_kind"] = current_kind
                        pending["carried_label"] = current_label

                    elif (
                        current_kind == "vessel"
                        and pending_kind == "unknown"
                    ):
                        # vessel은 객체 이력으로는 저장하지만
                        # 그 자체로 Meal Bite를 차단하지 않는다.
                        pending["carried_kind"] = "vessel"
                        pending["carried_label"] = current_label

                remaining.append(pending)
                continue

            carried_kind = str(
                pending.get(
                    "carried_kind",
                    "unknown",
                )
            ).lower()
            behavior_context = bool(
                pending.get(
                    "behavior_context",
                    False,
                )
            )

            if carried_kind == "phone":
                accepted = False
                reason = "phone_object"

            elif carried_kind == "food":
                accepted = True
                reason = "food_object"

            elif carried_kind == "utensil":
                accepted = True
                reason = "utensil_object"

            elif behavior_context:
                # vessel/unknown 여부와 관계없이 실제 식사 행동 문맥이
                # 확인되면 Meal Bite로 인정한다.
                accepted = True
                if carried_kind == "vessel":
                    reason = "vessel_with_meal_context"
                else:
                    reason = "behavior_context"

            else:
                accepted = False
                if carried_kind == "vessel":
                    reason = "vessel_without_meal_context"
                else:
                    reason = "no_meal_context"

            self._last_meal_evidence_decision = {
                "object": carried_kind,
                "label": str(
                    pending.get(
                        "carried_label",
                        "",
                    )
                ),
                "behavior_context": behavior_context,
                # 오버레이/기존 코드 호환을 위해 필드는 유지하되
                # Meal Evidence 판단에는 사용하지 않는다.
                "drink_motion_evidence": False,
                "decision": (
                    "accept"
                    if accepted
                    else "reject"
                ),
                "reason": reason,
            }

            if accepted:
                event_time = float(pending["timestamp"])
                self._accepted_meal_bite_times.append(
                    event_time
                )

                if behavior_context:
                    self._accepted_context_bite_times.append(
                        event_time
                    )

                self._meal_candidate_score += float(
                    config.get(
                        "accepted_meal_bite_score",
                        1.0,
                    )
                )

                if behavior_context:
                    self._meal_candidate_score += float(
                        config.get(
                            "context_meal_bite_bonus",
                            1.0,
                        )
                    )
            else:
                self._rejected_meal_bite_count += 1

        self._pending_meal_bites = remaining

    @property
    def meal_evidence_debug(self) -> Dict[str, Any]:
        """
        main_meal.py 디버그 오버레이에서 사용할 읽기 전용 요약.

        confirmed_bites는 기존 snapshot.bite_count를 사용하고,
        여기서는 Meal Evidence Filter 내부 수치만 제공한다.
        """
        return {
            "meal_bites": len(
                self._accepted_meal_bite_times
            ),
            "context_meal_bites": len(
                self._accepted_context_bite_times
            ),
            "rejected_bites": (
                self._rejected_meal_bite_count
            ),
            "pending_bites": len(
                self._pending_meal_bites
            ),
            "last_object": str(
                self._last_meal_evidence_decision.get(
                    "object",
                    "none",
                )
            ),
            "last_label": str(
                self._last_meal_evidence_decision.get(
                    "label",
                    "",
                )
            ),
            "last_behavior_context": bool(
                self._last_meal_evidence_decision.get(
                    "behavior_context",
                    False,
                )
            ),
            "last_drink_motion_evidence": bool(
                self._last_meal_evidence_decision.get(
                    "drink_motion_evidence",
                    False,
                )
            ),
            "last_decision": str(
                self._last_meal_evidence_decision.get(
                    "decision",
                    "none",
                )
            ),
            "last_reason": str(
                self._last_meal_evidence_decision.get(
                    "reason",
                    "no_decision_yet",
                )
            ),
        }

    def _meal_start_condition(
        self,
        observation: MealObservation,
        timestamp: float,
    ) -> bool:
        """
        음식/그릇 문맥이 있어도 총 Bite 5회를 요구한다.

        안정적 객체 문맥 있음:
        - Bite 5회
        - 식사 문맥 Bite 2회
        - 첫 Bite~마지막 Bite 30초 이상

        객체 문맥 없음:
        - Bite 5회
        - 식사 문맥 Bite 3회
        - 첫 Bite~마지막 Bite 45초 이상

        Drink는 조건에 포함하지 않는다.
        """
        _ = observation
        config = self.config.get("meal_candidate", {})

        candidate_window = float(
            config.get("candidate_window_seconds", 300.0)
        )
        self._trim_recent_times(
            self._accepted_meal_bite_times,
            timestamp,
            candidate_window,
        )
        self._trim_recent_times(
            self._accepted_context_bite_times,
            timestamp,
            candidate_window,
        )

        total_bites = len(self._accepted_meal_bite_times)
        context_bites = len(
            self._accepted_context_bite_times
        )

        required_total_bites = int(
            config.get("minimum_total_bites", 5)
        )
        if total_bites < required_total_bites:
            return False

        first_bite = self._accepted_meal_bite_times[0]
        last_bite = self._accepted_meal_bite_times[-1]
        bite_span = max(0.0, last_bite - first_bite)

        if self._strong_context_active:
            required_context_bites = int(
                config.get(
                    "minimum_context_bites_with_object",
                    2,
                )
            )
            minimum_span = float(
                config.get(
                    "minimum_span_with_object_seconds",
                    30.0,
                )
            )
        else:
            required_context_bites = int(
                config.get(
                    "minimum_context_bites_without_object",
                    3,
                )
            )
            minimum_span = float(
                config.get(
                    "minimum_span_without_object_seconds",
                    45.0,
                )
            )

        return bool(
            context_bites >= required_context_bites
            and bite_span >= minimum_span
        )

    # ------------------------------------------------------------------
    # Event / snapshot / config
    # ------------------------------------------------------------------

    def _event(
        self,
        event_type: EventType,
        timestamp: float,
        frame_index: int,
        action: MealAction = MealAction.NONE,
        confidence: float = 0.0,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MealEvent:
        event = MealEvent(
            event_type=event_type,
            timestamp=timestamp,
            frame_index=frame_index,
            action=action,
            meal_state=self._snapshot.meal_state,
            bite_state=self._snapshot.bite_state,
            drink_state=self._snapshot.drink_state,
            bite_count=self._snapshot.bite_count,
            drink_count=self._snapshot.drink_count,
            confidence=confidence,
            message=message,
            metadata=metadata or {},
        )
        event.clamp()
        return event

    def _copy_snapshot(self) -> MealSessionSnapshot:
        return MealSessionSnapshot(
            session_id=self._snapshot.session_id,
            started_at=self._snapshot.started_at,
            updated_at=self._snapshot.updated_at,
            meal_state=self._snapshot.meal_state,
            bite_state=self._snapshot.bite_state,
            drink_state=self._snapshot.drink_state,
            current_action=self._snapshot.current_action,
            bite_count=self._snapshot.bite_count,
            drink_count=self._snapshot.drink_count,
            last_bite_time=self._snapshot.last_bite_time,
            last_drink_time=self._snapshot.last_drink_time,
            last_activity_time=(
                self._snapshot.last_activity_time
            ),
            rest_duration=self._snapshot.rest_duration,
            end_candidate_duration=(
                self._snapshot.end_candidate_duration
            ),
            finished=self._snapshot.finished,
        )

    @staticmethod
    def _trim_recent_times(
        values: List[float],
        timestamp: float,
        window_seconds: float,
    ) -> None:
        cutoff = timestamp - max(
            0.0,
            float(window_seconds),
        )
        values[:] = [
            value
            for value in values
            if value >= cutoff
        ]

    def _cfg(
        self,
        section: str,
        key: str,
        default: float,
    ) -> float:
        return float(
            self.config.get(section, {}).get(
                key,
                default,
            )
        )

    @staticmethod
    def _new_session_id() -> str:
        return (
            f"meal_"
            f"{int(time.time())}_"
            f"{uuid4().hex[:8]}"
        )

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Meal config file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            value = json.load(file)

        if not isinstance(value, dict):
            raise ValueError(
                "meal_config.json 최상위 값은 객체여야 합니다."
            )

        return value


__all__ = [
    "FSMUpdateResult",
    "MealFSM",
]