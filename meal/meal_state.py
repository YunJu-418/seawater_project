"""
식사 행동 인식 시스템 V3의 공통 상태와 데이터 구조.

설계 원칙
---------
- vision/meal.py는 관측값만 생성한다.
- meal/meal_fsm.py는 Bite / Drink / Meal 상태 전이를 판단한다.
- 점수 가중합, EvidenceScores, TemporalEvidence는 사용하지 않는다.
- Bite와 Drink는 순서 기반 세션으로 표현한다.
- 기존 vision/holding.py, main_meal.py, meal_logger.py와의
  호환에 필요한 공통 타입은 유지한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


Point2D = Tuple[float, float]
BoxXYXY = Tuple[float, float, float, float]


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class HandPose(str, Enum):
    PINCH = "pinch"
    GRIP = "grip"
    OPEN = "open"
    UNKNOWN = "unknown"


class HoldingKind(str, Enum):
    NONE = "none"
    UTENSIL = "utensil"
    CUP = "cup"
    BOTTLE = "bottle"
    FOOD = "food"
    PHONE = "phone"
    OTHER = "other"
    UNKNOWN = "unknown"


class BiteState(str, Enum):
    IDLE = "idle"
    APPROACHING = "approaching"
    NEAR_MOUTH = "near_mouth"
    LEAVING = "leaving"
    CONFIRMED = "confirmed"
    COOLDOWN = "cooldown"
    CANCELLED = "cancelled"


class DrinkState(str, Enum):
    IDLE = "idle"
    VESSEL_AVAILABLE = "vessel_available"
    HAND_NEAR_VESSEL = "hand_near_vessel"
    PICKUP_SESSION = "pickup_session"
    MANIPULATING = "manipulating"
    APPROACHING_MOUTH = "approaching_mouth"
    NEAR_MOUTH = "near_mouth"
    LEFT_MOUTH = "left_mouth"
    AWAITING_RETURN = "awaiting_return"
    RETURN_CANDIDATE = "return_candidate"
    CONFIRMED = "confirmed"
    COOLDOWN = "cooldown"
    EXPIRED = "expired"


class MealState(str, Enum):
    WAITING = "waiting"
    EATING = "eating"
    REST = "rest"
    OTHER = "other"
    END_CANDIDATE = "end_candidate"
    FINISHED = "finished"


class MealAction(str, Enum):
    NONE = "none"
    BITE_CANDIDATE = "bite_candidate"
    BITE = "bite"
    DRINK_CANDIDATE = "drink_candidate"
    DRINK = "drink"
    UTENSILING = "utensiling"
    REST = "rest"
    OTHER = "other"


class EventType(str, Enum):
    SESSION_STARTED = "session_started"
    SESSION_FINISHED = "session_finished"
    MEAL_STATE_CHANGED = "meal_state_changed"
    BITE_STATE_CHANGED = "bite_state_changed"
    DRINK_STATE_CHANGED = "drink_state_changed"
    BITE_CONFIRMED = "bite_confirmed"
    DRINK_CONFIRMED = "drink_confirmed"
    REST_STARTED = "rest_started"
    END_CANDIDATE_STARTED = "end_candidate_started"
    TRACK_LOST = "track_lost"
    LOW_QUALITY_FRAME = "low_quality_frame"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: BoxXYXY
    track_id: Optional[int] = None

    @property
    def center(self) -> Point2D:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.box
        return max(0.0, x2 - x1)

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.box
        return max(0.0, y2 - y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FrameQuality:
    frame_valid: bool = True
    brightness: float = 0.0
    blur_score: float = 0.0
    face_detected: bool = False
    mouth_detected: bool = False
    hand_count: int = 0
    object_detection_valid: bool = True
    low_light: bool = False
    blurry: bool = False
    tracking_lost: bool = False
    missing_duration: float = 0.0
    quality_score: float = 1.0
    reasons: List[str] = field(default_factory=list)

    def clamp(self) -> None:
        self.hand_count = max(0, int(self.hand_count))
        self.brightness = max(0.0, float(self.brightness))
        self.blur_score = max(0.0, float(self.blur_score))
        self.missing_duration = max(0.0, float(self.missing_duration))
        self.quality_score = _clamp01(self.quality_score)


@dataclass
class HandObservation:
    side: HandSide = HandSide.UNKNOWN
    detected: bool = False
    tracking_id: Optional[int] = None

    wrist: Optional[Point2D] = None
    palm_center: Optional[Point2D] = None
    index_tip: Optional[Point2D] = None
    thumb_tip: Optional[Point2D] = None
    smoothed_wrist: Optional[Point2D] = None
    smoothed_palm_center: Optional[Point2D] = None

    velocity: Point2D = (0.0, 0.0)
    speed: float = 0.0
    motion_direction_to_mouth: float = 0.0

    hand_pose: HandPose = HandPose.UNKNOWN
    pose_confidence: float = 0.0

    holding: bool = False
    holding_kind: HoldingKind = HoldingKind.NONE
    holding_label: Optional[str] = None
    holding_confidence: float = 0.0
    holding_score: float = 0.0
    spatial_stability: float = 0.0
    object_comotion: float = 0.0

    started_near_plate: bool = False
    near_mouth: bool = False
    inside_mouth_roi: bool = False
    mouth_occluded: bool = False

    hand_mouth_distance: Optional[float] = None
    normalized_hand_mouth_distance: Optional[float] = None
    distance_delta: float = 0.0
    approaching_mouth: bool = False
    leaving_mouth: bool = False
    dwell_time_near_mouth: float = 0.0
    missing_duration: float = 0.0

    landmarks: Sequence[Point2D] = field(default_factory=tuple)
    debug: Dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> None:
        self.pose_confidence = _clamp01(self.pose_confidence)
        self.holding_confidence = _clamp01(self.holding_confidence)
        self.holding_score = _clamp01(self.holding_score)
        self.spatial_stability = _clamp01(self.spatial_stability)
        self.object_comotion = _clamp01(self.object_comotion)
        self.motion_direction_to_mouth = min(
            1.0,
            max(-1.0, float(self.motion_direction_to_mouth)),
        )
        self.speed = max(0.0, float(self.speed))
        self.dwell_time_near_mouth = max(
            0.0,
            float(self.dwell_time_near_mouth),
        )
        self.missing_duration = max(
            0.0,
            float(self.missing_duration),
        )


@dataclass
class MouthObservation:
    detected: bool = False
    center: Optional[Point2D] = None
    smoothed_center: Optional[Point2D] = None
    roi: Optional[BoxXYXY] = None

    openness: float = 0.0
    openness_smoothed: float = 0.0
    is_open: bool = False
    open_started_at: Optional[float] = None
    open_duration: float = 0.0

    occluded: bool = False
    occlusion_score: float = 0.0
    missing_duration: float = 0.0

    debug: Dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> None:
        self.openness = max(0.0, float(self.openness))
        self.openness_smoothed = max(
            0.0,
            float(self.openness_smoothed),
        )
        self.open_duration = max(0.0, float(self.open_duration))
        self.occlusion_score = _clamp01(self.occlusion_score)
        self.missing_duration = max(
            0.0,
            float(self.missing_duration),
        )


@dataclass
class VirtualPlateObservation:
    available: bool = False
    center: Optional[Point2D] = None
    roi: Optional[BoxXYXY] = None
    confidence: float = 0.0
    source: str = "none"
    stable_duration: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> None:
        self.confidence = _clamp01(self.confidence)
        self.stable_duration = max(
            0.0,
            float(self.stable_duration),
        )


@dataclass
class VesselObservation:
    """
    컵·물병·wine glass 등을 하나의 음용 용기(Vessel)로 표현한다.

    detected는 현재 프레임 검출 여부이며,
    remembered는 짧은 검출 누락을 메모리로 유지했는지를 나타낸다.
    """

    detected: bool = False
    label: Optional[str] = None
    normalized_label: Optional[str] = None
    confidence: float = 0.0
    box: Optional[BoxXYXY] = None
    center: Optional[Point2D] = None

    source: str = "none"
    remembered: bool = False
    missing_duration: float = 0.0

    hand_near: bool = False
    nearest_hand: HandSide = HandSide.UNKNOWN
    disappeared_after_contact: bool = False
    returned_near_origin: bool = False

    origin_box: Optional[BoxXYXY] = None
    origin_center: Optional[Point2D] = None
    return_visible_duration: float = 0.0

    debug: Dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> None:
        self.confidence = _clamp01(self.confidence)
        self.missing_duration = max(
            0.0,
            float(self.missing_duration),
        )
        self.return_visible_duration = max(
            0.0,
            float(self.return_visible_duration),
        )


# 기존 이름을 참조하는 코드와의 임시 호환용 별칭.
CupObservation = VesselObservation


@dataclass
class BiteSessionObservation:
    """
    한 번의 Bite 왕복 동작을 표현한다.

    접근 → 입 근처 도달 → 최소 체류 → 이탈 → 충분히 멀어짐
    순서를 완료했을 때 session_completed가 True가 된다.
    """

    state: BiteState = BiteState.IDLE
    active: bool = False
    active_hand: HandSide = HandSide.UNKNOWN

    approaching: bool = False
    near_mouth: bool = False
    leaving: bool = False

    approach_duration: float = 0.0
    dwell_duration: float = 0.0
    leave_duration: float = 0.0
    total_duration: float = 0.0

    mouth_reached: bool = False
    round_trip_completed: bool = False
    session_completed: bool = False

    hand_pose: HandPose = HandPose.UNKNOWN
    pose_context_active: bool = False
    started_near_plate: bool = False
    phone_blocked: bool = False
    blocked_by_drink: bool = False

    cancel_reason: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> None:
        self.approach_duration = max(
            0.0,
            float(self.approach_duration),
        )
        self.dwell_duration = max(
            0.0,
            float(self.dwell_duration),
        )
        self.leave_duration = max(
            0.0,
            float(self.leave_duration),
        )
        self.total_duration = max(
            0.0,
            float(self.total_duration),
        )


@dataclass
class DrinkSessionObservation:
    """
    용기 검출 → 손 접촉 후 소실 → 입 도달 → 이탈 → 원위치 반환
    흐름을 하나의 Drink Session으로 표현한다.
    """

    state: DrinkState = DrinkState.IDLE
    session_active: bool = False
    contact_hand: HandSide = HandSide.UNKNOWN

    vessel_visible: bool = False
    vessel_label: Optional[str] = None
    hand_near_vessel: bool = False
    vessel_disappeared_after_contact: bool = False

    approaching_mouth: bool = False
    near_mouth: bool = False
    mouth_reached: bool = False
    left_mouth: bool = False

    return_candidate: bool = False
    return_confirmed: bool = False
    session_completed: bool = False
    session_expired: bool = False

    session_duration: float = 0.0
    return_visible_duration: float = 0.0
    completed_at: Optional[float] = None

    debug: Dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> None:
        self.session_duration = max(
            0.0,
            float(self.session_duration),
        )
        self.return_visible_duration = max(
            0.0,
            float(self.return_visible_duration),
        )


@dataclass
class MealObservation:
    timestamp: float
    frame_index: int
    delta_time: float
    frame_size: Tuple[int, int] = (0, 0)

    quality: FrameQuality = field(default_factory=FrameQuality)
    mouth: MouthObservation = field(default_factory=MouthObservation)
    hands: Dict[HandSide, HandObservation] = field(default_factory=dict)
    vessel: VesselObservation = field(default_factory=VesselObservation)
    virtual_plate: VirtualPlateObservation = field(
        default_factory=VirtualPlateObservation
    )
    detections: List[Detection] = field(default_factory=list)

    active_hand: HandSide = HandSide.UNKNOWN
    bite_session: BiteSessionObservation = field(
        default_factory=BiteSessionObservation
    )
    drink_session: DrinkSessionObservation = field(
        default_factory=DrinkSessionObservation
    )

    suggested_action: MealAction = MealAction.NONE
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def cup(self) -> VesselObservation:
        """
        V2 main/debug 코드와의 임시 호환을 위한 별칭.

        V3에서는 cup 대신 vessel이라는 이름을 사용한다.
        """
        return self.vessel

    @property
    def bite_candidate(self) -> bool:
        return self.bite_session.active

    @property
    def drink_candidate(self) -> bool:
        return self.drink_session.session_active

    def clamp(self) -> None:
        self.delta_time = max(0.0, float(self.delta_time))
        self.quality.clamp()
        self.mouth.clamp()
        self.vessel.clamp()
        self.virtual_plate.clamp()
        self.bite_session.clamp()
        self.drink_session.clamp()

        for hand in self.hands.values():
            hand.clamp()

    def get_hand(
        self,
        side: HandSide,
    ) -> Optional[HandObservation]:
        return self.hands.get(side)

    def active_hand_observation(
        self,
    ) -> Optional[HandObservation]:
        return self.hands.get(self.active_hand)

    def as_summary(self) -> Dict[str, Any]:
        active = self.active_hand_observation()
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "delta_time": self.delta_time,
            "active_hand": self.active_hand.value,
            "bite_state": self.bite_session.state.value,
            "bite_active": self.bite_session.active,
            "bite_completed": (
                self.bite_session.session_completed
            ),
            "drink_state": self.drink_session.state.value,
            "drink_active": (
                self.drink_session.session_active
            ),
            "drink_completed": (
                self.drink_session.session_completed
            ),
            "suggested_action": self.suggested_action.value,
            "mouth_open": self.mouth.is_open,
            "mouth_occluded": self.mouth.occluded,
            "hand_pose": (
                active.hand_pose.value
                if active is not None
                else None
            ),
            "holding_kind": (
                active.holding_kind.value
                if active is not None
                else None
            ),
            "vessel_detected": self.vessel.detected,
            "vessel_label": self.vessel.normalized_label,
            "frame_quality": self.quality.quality_score,
        }


@dataclass
class MealEvent:
    event_type: EventType
    timestamp: float
    frame_index: int

    action: MealAction = MealAction.NONE
    meal_state: Optional[MealState] = None
    bite_state: Optional[BiteState] = None
    drink_state: Optional[DrinkState] = None

    bite_count: int = 0
    drink_count: int = 0
    confidence: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> None:
        self.bite_count = max(0, int(self.bite_count))
        self.drink_count = max(0, int(self.drink_count))
        self.confidence = _clamp01(self.confidence)


@dataclass
class MealSessionSnapshot:
    session_id: str
    started_at: float
    updated_at: float

    meal_state: MealState = MealState.WAITING
    bite_state: BiteState = BiteState.IDLE
    drink_state: DrinkState = DrinkState.IDLE
    current_action: MealAction = MealAction.NONE

    bite_count: int = 0
    drink_count: int = 0

    last_bite_time: Optional[float] = None
    last_drink_time: Optional[float] = None
    last_activity_time: Optional[float] = None

    rest_duration: float = 0.0
    end_candidate_duration: float = 0.0
    finished: bool = False

    def clamp(self) -> None:
        self.bite_count = max(0, int(self.bite_count))
        self.drink_count = max(0, int(self.drink_count))
        self.rest_duration = max(
            0.0,
            float(self.rest_duration),
        )
        self.end_candidate_duration = max(
            0.0,
            float(self.end_candidate_duration),
        )


def enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def dataclass_to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: dataclass_to_dict(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {
            str(enum_value(key)): dataclass_to_dict(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            dataclass_to_dict(item)
            for item in value
        ]

    return value


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "BiteSessionObservation",
    "BiteState",
    "BoxXYXY",
    "CupObservation",
    "Detection",
    "DrinkSessionObservation",
    "DrinkState",
    "EventType",
    "FrameQuality",
    "HandObservation",
    "HandPose",
    "HandSide",
    "HoldingKind",
    "MealAction",
    "MealEvent",
    "MealObservation",
    "MealSessionSnapshot",
    "MealState",
    "MouthObservation",
    "Point2D",
    "VesselObservation",
    "VirtualPlateObservation",
    "dataclass_to_dict",
    "enum_value",
]
