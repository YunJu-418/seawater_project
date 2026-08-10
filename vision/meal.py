"""
식사 행동 인식용 비전 통합 처리 모듈 V3.

역할
----
- detector.py, hand.py, face_mesh.py 결과를 공통 입력 형식으로 변환한다.
- HoldingAnalyzer 결과를 HandObservation에 반영한다.
- 손-입 거리, 접근, 입 근처 체류, 이탈을 계산한다.
- Bite를 접근 → 입 근처 → 체류 → 이탈 순서의 세션으로 관측한다.
- Drink를 용기 검출 → 손 접근 → 용기 소실 → 입 도달 → 이탈
  → 원위치 반환 순서의 세션으로 관측한다.
- 점수 가중합, EvidenceScores, TemporalEvidence는 사용하지 않는다.
- 최종 판단 카운트는 meal/meal_fsm.py가 담당한다.

주의
----
이 모듈은 프레임 간 상태를 유지하므로 하나의 MealVisionProcessor 인스턴스를
영상 전체에서 계속 사용해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json

from meal.meal_state import (
    BiteSessionObservation,
    BiteState,
    BoxXYXY,
    Detection,
    DrinkSessionObservation,
    DrinkState,
    FrameQuality,
    HandObservation,
    HandPose,
    HandSide,
    HoldingKind,
    MealAction,
    MealObservation,
    MouthObservation,
    Point2D,
    VesselObservation,
    VirtualPlateObservation,
)
from vision.holding import (
    HoldingAnalyzer,
    HoldingConfig,
    HoldingInput,
)


_EPSILON = 1e-9


@dataclass(frozen=True)
class RawHandInput:
    """hand.py에서 받은 한 손의 표준 입력."""

    side: HandSide
    landmarks: Sequence[Point2D]
    tracking_id: Optional[int] = None
    confidence: float = 1.0


@dataclass(frozen=True)
class RawMouthInput:
    """face_mesh.py에서 받은 입 정보의 표준 입력."""

    detected: bool = False
    center: Optional[Point2D] = None
    face_box: Optional[BoxXYXY] = None
    mouth_landmarks: Sequence[Point2D] = field(default_factory=tuple)
    openness: Optional[float] = None


@dataclass(frozen=True)
class MealVisionInput:
    """한 프레임의 모든 비전 입력."""

    timestamp: float
    frame_index: int
    frame_size: Tuple[int, int]
    hands: Sequence[RawHandInput] = field(default_factory=tuple)
    mouth: RawMouthInput = field(default_factory=RawMouthInput)
    detections: Sequence[Detection] = field(default_factory=tuple)

    brightness: Optional[float] = None
    blur_score: Optional[float] = None
    frame_valid: bool = True
    object_detection_valid: bool = True


@dataclass
class _HandMotionState:
    previous_smoothed_center: Optional[Point2D] = None
    previous_contact_point: Optional[Point2D] = None
    previous_distance: Optional[float] = None
    near_started_at: Optional[float] = None
    leave_candidate_started_at: Optional[float] = None
    last_seen_at: Optional[float] = None
    reached_mouth: bool = False
    started_near_plate: bool = False
    plate_context_until: float = 0.0
    pose_context_until: float = 0.0

    low_zone_started_at: Optional[float] = None
    low_zone_context_until: float = 0.0
    low_zone_reference_distance: Optional[float] = None
    low_zone_reference_y: Optional[float] = None
    low_zone_qualified: bool = False


@dataclass
class _MouthState:
    smoothed_center: Optional[Point2D] = None
    last_roi: Optional[BoxXYXY] = None
    openness_smoothed: float = 0.0
    open_started_at: Optional[float] = None
    last_seen_at: Optional[float] = None


@dataclass
class _BiteTracker:
    state: BiteState = BiteState.IDLE
    active_hand: HandSide = HandSide.UNKNOWN

    started_at: Optional[float] = None
    approach_started_at: Optional[float] = None
    near_started_at: Optional[float] = None
    leave_started_at: Optional[float] = None
    confirmed_at: Optional[float] = None
    cooldown_until: float = 0.0

    start_distance: Optional[float] = None
    minimum_distance: Optional[float] = None
    low_zone_qualified: bool = False
    returned_to_low_zone: bool = False
    direct_near_entry: bool = False

    mouth_reached: bool = False
    completed_latch: bool = False
    cancel_reason: Optional[str] = None


@dataclass
class _DrinkTracker:
    state: DrinkState = DrinkState.IDLE

    session_active: bool = False
    contact_hand: HandSide = HandSide.UNKNOWN
    vessel_label: Optional[str] = None

    origin_box: Optional[BoxXYXY] = None
    latest_box: Optional[BoxXYXY] = None

    contact_started_at: Optional[float] = None
    contact_last_seen_at: Optional[float] = None
    contact_hand_origin: Optional[Point2D] = None
    hand_movement_ratio: float = 0.0
    strong_contact_confirmed: bool = False
    pickup_verified: bool = False
    movement_ratio: float = 0.0

    contact_time: Optional[float] = None
    disappeared_time: Optional[float] = None
    session_start: Optional[float] = None
    return_visible_since: Optional[float] = None
    completed_at: Optional[float] = None
    cooldown_until: float = 0.0

    mouth_reached: bool = False
    left_mouth: bool = False
    completed_latch: bool = False
    expired_latch: bool = False


class MealVisionProcessor:
    """비전 신호를 V3 MealObservation으로 변환한다."""

    VESSEL_LABELS = {
        "cup",
        "mug",
        "glass",
        "wine glass",
        "bottle",
        "water bottle",
    }
    PLATE_LABELS = {
        "bowl",
        "plate",
        "dish",
        "dining table",
        "table",
    }
    FOOD_LABELS = {
        "apple",
        "banana",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "food",
    }

    def __init__(
        self,
        config_path: str | Path = "meal/meal_config.json",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        self._last_timestamp: Optional[float] = None

        self._hand_states: Dict[HandSide, _HandMotionState] = {
            HandSide.LEFT: _HandMotionState(),
            HandSide.RIGHT: _HandMotionState(),
            HandSide.UNKNOWN: _HandMotionState(),
        }
        self._mouth_state = _MouthState()
        self._bite = _BiteTracker()
        self._drink = _DrinkTracker()

        self._last_vessel_seen_at: Optional[float] = None
        self._last_vessel: Optional[Detection] = None

        self._handheld_until: Dict[HandSide, float] = {
            HandSide.LEFT: 0.0,
            HandSide.RIGHT: 0.0,
            HandSide.UNKNOWN: 0.0,
        }
        self._handheld_rearmed: Dict[HandSide, bool] = {
            HandSide.LEFT: False,
            HandSide.RIGHT: False,
            HandSide.UNKNOWN: False,
        }

        # Bite 확정 자체에는 사용하지 않고, Meal Evidence 필터가
        # 직전/직후 손에 든 객체 문맥을 확인할 수 있도록 유지한다.
        self._carried_context: Dict[HandSide, Dict[str, Any]] = {
            HandSide.LEFT: {
                "kind": "unknown",
                "label": "",
                "until": 0.0,
                "last_seen_at": None,
            },
            HandSide.RIGHT: {
                "kind": "unknown",
                "label": "",
                "until": 0.0,
                "last_seen_at": None,
            },
            HandSide.UNKNOWN: {
                "kind": "unknown",
                "label": "",
                "until": 0.0,
                "last_seen_at": None,
            },
        }

        self.holding_analyzer = HoldingAnalyzer(
            self._build_holding_config()
        )

    def reset(self) -> None:
        self.holding_analyzer.reset()
        self._last_timestamp = None
        self._hand_states = {
            HandSide.LEFT: _HandMotionState(),
            HandSide.RIGHT: _HandMotionState(),
            HandSide.UNKNOWN: _HandMotionState(),
        }
        self._mouth_state = _MouthState()
        self._bite = _BiteTracker()
        self._drink = _DrinkTracker()
        self._last_vessel_seen_at = None
        self._last_vessel = None
        self._handheld_until = {
            HandSide.LEFT: 0.0,
            HandSide.RIGHT: 0.0,
            HandSide.UNKNOWN: 0.0,
        }
        self._handheld_rearmed = {
            HandSide.LEFT: False,
            HandSide.RIGHT: False,
            HandSide.UNKNOWN: False,
        }
        self._carried_context = {
            HandSide.LEFT: {
                "kind": "unknown",
                "label": "",
                "until": 0.0,
                "last_seen_at": None,
            },
            HandSide.RIGHT: {
                "kind": "unknown",
                "label": "",
                "until": 0.0,
                "last_seen_at": None,
            },
            HandSide.UNKNOWN: {
                "kind": "unknown",
                "label": "",
                "until": 0.0,
                "last_seen_at": None,
            },
        }

    def create_observation(
        self,
        value: MealVisionInput,
    ) -> MealObservation:
        timestamp = float(value.timestamp)
        frame_size = _normalize_frame_size(value.frame_size)
        delta_time = self._calculate_delta_time(timestamp)

        detections = [
            detection
            for detection in (
                _coerce_detection(item)
                for item in value.detections
            )
            if detection is not None
        ]

        quality = self._build_frame_quality(
            value=value,
            hand_count=len(value.hands),
        )
        mouth = self._build_mouth_observation(
            raw=value.mouth,
            timestamp=timestamp,
            frame_size=frame_size,
            detections=detections,
        )
        virtual_plate = self._build_virtual_plate(
            detections=detections,
            frame_size=frame_size,
        )
        hands = self._build_hand_observations(
            raw_hands=value.hands,
            detections=detections,
            mouth=mouth,
            virtual_plate=virtual_plate,
            timestamp=timestamp,
            delta_time=delta_time,
            frame_size=frame_size,
        )

        self._update_carried_object_memory(
            detections=detections,
            hands=hands,
            timestamp=timestamp,
        )

        bite_session_in_progress = bool(
            self._bite.state
            in {
                BiteState.APPROACHING,
                BiteState.NEAR_MOUTH,
                BiteState.LEAVING,
            }
            and self._bite.active_hand
            in {
                HandSide.LEFT,
                HandSide.RIGHT,
            }
        )

        if bite_session_in_progress:
            locked_hand = hands.get(self._bite.active_hand)
            if locked_hand is not None and locked_hand.detected:
                active_hand_side = self._bite.active_hand
            else:
                active_hand_side = self._select_active_hand(hands)
        else:
            active_hand_side = self._select_active_hand(hands)

        active_hand = hands.get(active_hand_side)

        vessel = self._build_vessel_observation(
            detections=detections,
            hands=hands,
            timestamp=timestamp,
        )

        drink_session = self._update_drink_session(
            vessel=vessel,
            hands=hands,
            mouth=mouth,
            timestamp=timestamp,
        )

        bite_session = self._update_bite_session(
            active_hand=active_hand,
            active_hand_side=active_hand_side,
            drink_session=drink_session,
            timestamp=timestamp,
        )

        suggested_action = self._suggest_action(
            bite_session=bite_session,
            drink_session=drink_session,
            active_hand=active_hand,
        )

        observation = MealObservation(
            timestamp=timestamp,
            frame_index=int(value.frame_index),
            delta_time=delta_time,
            frame_size=frame_size,
            quality=quality,
            mouth=mouth,
            hands=hands,
            vessel=vessel,
            virtual_plate=virtual_plate,
            detections=detections,
            active_hand=active_hand_side,
            bite_session=bite_session,
            drink_session=drink_session,
            suggested_action=suggested_action,
            debug={
                "config_version": self.config.get("version", "3"),
                "active_hand_reason": self._active_hand_reason(
                    active_hand
                ),
            },
        )
        observation.clamp()
        return observation

    def _update_carried_object_memory(
        self,
        detections: Sequence[Detection],
        hands: Dict[HandSide, HandObservation],
        timestamp: float,
    ) -> None:
        """
        손과 가까이 검출된 객체를 손별로 짧게 기억한다.

        우선순위:
        phone > vessel > utensil > food > unknown

        이 메모리는 Bite를 취소하지 않는다. MealFSM이 확정 Bite를
        식사 증거로 채택할지 판단할 때만 사용한다.
        """
        phone_labels = {
            str(label).strip().lower()
            for label in self.config.get(
                "meal_evidence",
                {},
            ).get(
                "phone_labels",
                ["cell phone", "phone", "mobile phone"],
            )
        }
        vessel_labels = {
            str(label).strip().lower()
            for label in self.config.get(
                "meal_evidence",
                {},
            ).get(
                "vessel_labels",
                [
                    "cup",
                    "mug",
                    "glass",
                    "wine glass",
                    "bottle",
                    "water bottle",
                ],
            )
        }
        utensil_labels = {
            str(label).strip().lower()
            for label in self.config.get(
                "meal_evidence",
                {},
            ).get(
                "utensil_labels",
                ["fork", "knife", "spoon", "chopsticks"],
            )
        }
        food_labels = {
            str(label).strip().lower()
            for label in self.config.get(
                "meal_evidence",
                {},
            ).get(
                "food_labels",
                list(self.FOOD_LABELS),
            )
        }

        memory_seconds = self._cfg(
            "meal_evidence",
            "object_memory_seconds",
            default=3.0,
        )
        association_ratio = self._cfg(
            "meal_evidence",
            "hand_object_association_ratio",
            default=0.85,
        )
        minimum_confidence = self._cfg(
            "meal_evidence",
            "minimum_object_confidence",
            default=0.20,
        )

        priority = {
            "unknown": 0,
            "food": 1,
            "utensil": 2,
            "vessel": 3,
            "phone": 4,
        }

        frame_candidates: Dict[
            HandSide,
            Tuple[str, str, float],
        ] = {}

        for detection in detections:
            label = detection.label.strip().lower()
            if detection.confidence < minimum_confidence:
                continue

            if label in phone_labels:
                kind = "phone"
            elif label in vessel_labels:
                kind = "vessel"
            elif label in utensil_labels:
                kind = "utensil"
            elif label in food_labels:
                kind = "food"
            else:
                continue

            for side, hand in hands.items():
                if (
                    side == HandSide.UNKNOWN
                    or not hand.detected
                    or hand.smoothed_palm_center is None
                ):
                    continue

                normalized_distance = (
                    _point_box_distance(
                        hand.smoothed_palm_center,
                        detection.box,
                    )
                    / max(
                        _box_diagonal(detection.box),
                        1.0,
                    )
                )
                if normalized_distance > association_ratio:
                    continue

                old = frame_candidates.get(side)
                candidate = (
                    kind,
                    label,
                    normalized_distance,
                )
                if old is None:
                    frame_candidates[side] = candidate
                    continue

                old_kind, _, old_distance = old
                if (
                    priority[kind] > priority[old_kind]
                    or (
                        priority[kind] == priority[old_kind]
                        and normalized_distance < old_distance
                    )
                ):
                    frame_candidates[side] = candidate

        for side in (HandSide.LEFT, HandSide.RIGHT):
            candidate = frame_candidates.get(side)
            if candidate is not None:
                kind, label, normalized_distance = candidate
                self._carried_context[side] = {
                    "kind": kind,
                    "label": label,
                    "until": timestamp + memory_seconds,
                    "last_seen_at": timestamp,
                    "association_distance": normalized_distance,
                }
                continue

            current = self._carried_context[side]
            if timestamp > float(current.get("until", 0.0)):
                self._carried_context[side] = {
                    "kind": "unknown",
                    "label": "",
                    "until": 0.0,
                    "last_seen_at": None,
                }

    # ------------------------------------------------------------------
    # 공통 관측
    # ------------------------------------------------------------------

    def _calculate_delta_time(self, timestamp: float) -> float:
        if self._last_timestamp is None:
            delta_time = 0.0
        else:
            delta_time = max(
                0.0,
                timestamp - self._last_timestamp,
            )
            delta_time = min(
                delta_time,
                self._cfg(
                    "camera",
                    "max_frame_gap_seconds",
                    default=0.25,
                ),
            )

        self._last_timestamp = timestamp
        return delta_time

    def _build_frame_quality(
        self,
        value: MealVisionInput,
        hand_count: int,
    ) -> FrameQuality:
        brightness = (
            float(value.brightness)
            if value.brightness is not None
            else self._cfg(
                "preprocessing",
                "brightness_target",
                default=115.0,
            )
        )
        blur_score = (
            float(value.blur_score)
            if value.blur_score is not None
            else 100.0
        )

        low_threshold = self._cfg(
            "preprocessing",
            "brightness_low_threshold",
            default=65.0,
        )
        high_threshold = self._cfg(
            "preprocessing",
            "brightness_high_threshold",
            default=205.0,
        )
        blur_threshold = self._cfg(
            "preprocessing",
            "minimum_blur_score",
            default=35.0,
        )

        reasons: List[str] = []
        score = 1.0

        low_light = brightness < low_threshold
        too_bright = brightness > high_threshold
        blurry = blur_score < blur_threshold

        if not value.frame_valid:
            score -= 0.65
            reasons.append("invalid_frame")
        if low_light:
            score -= 0.22
            reasons.append("low_light")
        if too_bright:
            score -= 0.15
            reasons.append("overexposed")
        if blurry:
            score -= 0.22
            reasons.append("blurry")
        if not value.object_detection_valid:
            score -= 0.20
            reasons.append("object_detection_invalid")

        quality = FrameQuality(
            frame_valid=bool(value.frame_valid),
            brightness=brightness,
            blur_score=blur_score,
            face_detected=bool(value.mouth.detected),
            mouth_detected=bool(value.mouth.detected),
            hand_count=max(0, int(hand_count)),
            object_detection_valid=bool(
                value.object_detection_valid
            ),
            low_light=low_light,
            blurry=blurry,
            tracking_lost=(
                not value.mouth.detected
                and hand_count == 0
            ),
            quality_score=_clamp01(score),
            reasons=reasons,
        )
        quality.clamp()
        return quality

    def _build_mouth_observation(
        self,
        raw: RawMouthInput,
        timestamp: float,
        frame_size: Tuple[int, int],
        detections: Sequence[Detection],
    ) -> MouthObservation:
        detected = bool(
            raw.detected
            and raw.center is not None
        )
        center = (
            _coerce_point(raw.center)
            if raw.center is not None
            else None
        )

        if detected and center is not None:
            self._mouth_state.smoothed_center = _ema_point(
                self._mouth_state.smoothed_center,
                center,
                self._cfg(
                    "ema",
                    "mouth_position_alpha",
                    default=0.40,
                ),
            )
            self._mouth_state.last_seen_at = timestamp

            openness = self._calculate_mouth_openness(raw)
            self._mouth_state.openness_smoothed = _ema_scalar(
                self._mouth_state.openness_smoothed,
                openness,
                self._cfg(
                    "ema",
                    "distance_alpha",
                    default=0.35,
                ),
            )
            is_open = (
                self._mouth_state.openness_smoothed
                >= self._cfg(
                    "mouth",
                    "minimum_open_ratio",
                    default=0.055,
                )
            )

            if is_open:
                if self._mouth_state.open_started_at is None:
                    self._mouth_state.open_started_at = timestamp
            else:
                self._mouth_state.open_started_at = None

            roi = self._mouth_roi(
                center=self._mouth_state.smoothed_center,
                face_box=raw.face_box,
                frame_size=frame_size,
            )
            self._mouth_state.last_roi = roi
            occlusion_score = self._mouth_occlusion_score(
                roi=roi,
                detections=detections,
            )
            occluded = (
                occlusion_score
                >= self._cfg(
                    "mouth",
                    "occlusion_threshold",
                    default=0.45,
                )
            )

            open_duration = (
                timestamp
                - self._mouth_state.open_started_at
                if self._mouth_state.open_started_at is not None
                else 0.0
            )
            missing_duration = 0.0
        else:
            openness = 0.0
            is_open = False
            open_duration = 0.0
            missing_duration = (
                0.0
                if self._mouth_state.last_seen_at is None
                else max(
                    0.0,
                    timestamp
                    - self._mouth_state.last_seen_at,
                )
            )
            memory_active = (
                self._mouth_state.smoothed_center is not None
                and missing_duration
                <= self._cfg(
                    "mouth",
                    "occlusion_memory_seconds",
                    default=1.0,
                )
            )
            roi = (
                self._mouth_state.last_roi
                if memory_active
                else None
            )
            occlusion_score = 1.0 if memory_active else 0.0
            occluded = memory_active

        result = MouthObservation(
            detected=detected,
            center=center,
            smoothed_center=self._mouth_state.smoothed_center,
            roi=roi,
            openness=openness,
            openness_smoothed=(
                self._mouth_state.openness_smoothed
            ),
            is_open=is_open,
            open_started_at=(
                self._mouth_state.open_started_at
            ),
            open_duration=open_duration,
            occluded=occluded,
            occlusion_score=occlusion_score,
            missing_duration=missing_duration,
            debug={
                "raw_landmark_count": len(
                    raw.mouth_landmarks
                ),
                "memory_active": bool(
                    not detected
                    and roi is not None
                ),
            },
        )
        result.clamp()
        return result

    def _build_virtual_plate(
        self,
        detections: Sequence[Detection],
        frame_size: Tuple[int, int],
    ) -> VirtualPlateObservation:
        width, height = frame_size
        config = self.config.get("virtual_plate", {})

        initial = config.get(
            "initial_roi",
            {
                "x_min_ratio": 0.12,
                "y_min_ratio": 0.54,
                "x_max_ratio": 0.88,
                "y_max_ratio": 0.98,
            },
        )
        fallback_roi = (
            width * float(initial.get("x_min_ratio", 0.12)),
            height * float(initial.get("y_min_ratio", 0.54)),
            width * float(initial.get("x_max_ratio", 0.88)),
            height * float(initial.get("y_max_ratio", 0.98)),
        )

        candidates = [
            item
            for item in detections
            if item.label.strip().lower()
            in self.PLATE_LABELS | self.FOOD_LABELS
        ]

        if not candidates:
            return VirtualPlateObservation(
                available=True,
                center=_box_center(fallback_roi),
                roi=fallback_roi,
                confidence=0.30,
                source="fallback_roi",
            )

        union = _union_boxes(
            [item.box for item in candidates]
        )
        expanded = _expand_box(
            union,
            float(config.get("roi_expand_ratio", 0.18)),
            frame_size,
        )
        confidence = _clamp01(
            sum(item.confidence for item in candidates)
            / len(candidates)
        )
        return VirtualPlateObservation(
            available=True,
            center=_box_center(expanded),
            roi=expanded,
            confidence=confidence,
            source="detected_objects",
            debug={
                "candidate_count": len(candidates),
            },
        )

    def _build_hand_observations(
        self,
        raw_hands: Sequence[RawHandInput],
        detections: Sequence[Detection],
        mouth: MouthObservation,
        virtual_plate: VirtualPlateObservation,
        timestamp: float,
        delta_time: float,
        frame_size: Tuple[int, int],
    ) -> Dict[HandSide, HandObservation]:
        raw_by_side = {
            _coerce_hand_side(item.side): item
            for item in raw_hands
        }
        hands: Dict[HandSide, HandObservation] = {}

        for side in (
            HandSide.LEFT,
            HandSide.RIGHT,
        ):
            raw = raw_by_side.get(side)

            if raw is None:
                holding_result = self.holding_analyzer.update(
                    HoldingInput(
                        side=side,
                        timestamp=timestamp,
                        frame_size=frame_size,
                        landmarks=(),
                        detections=detections,
                    )
                )
                hand = holding_result.apply_to(
                    HandObservation(side=side)
                )
            else:
                landmarks = _normalize_landmarks(
                    raw.landmarks
                )
                holding_result = self.holding_analyzer.update(
                    HoldingInput(
                        side=side,
                        timestamp=timestamp,
                        frame_size=frame_size,
                        landmarks=landmarks,
                        detections=detections,
                        tracking_id=raw.tracking_id,
                    )
                )
                hand = holding_result.apply_to(
                    HandObservation(
                        side=side,
                        tracking_id=raw.tracking_id,
                        landmarks=landmarks,
                    )
                )

            self._enrich_hand_motion(
                hand=hand,
                mouth=mouth,
                virtual_plate=virtual_plate,
                timestamp=timestamp,
                delta_time=delta_time,
                frame_size=frame_size,
            )
            hands[side] = hand

        return hands

    def _enrich_hand_motion(
        self,
        hand: HandObservation,
        mouth: MouthObservation,
        virtual_plate: VirtualPlateObservation,
        timestamp: float,
        delta_time: float,
        frame_size: Tuple[int, int],
    ) -> None:
        state = self._hand_states[hand.side]

        if (
            not hand.detected
            or hand.smoothed_palm_center is None
        ):
            hand.missing_duration = (
                0.0
                if state.last_seen_at is None
                else max(
                    0.0,
                    timestamp - state.last_seen_at,
                )
            )
            return

        palm_center = hand.smoothed_palm_center
        state.last_seen_at = timestamp

        contact_point, contact_source = self._select_bite_contact_point(
            hand=hand,
            mouth=mouth,
        )

        if contact_point is None:
            contact_point = palm_center
            contact_source = "palm_fallback"

        if (
            state.previous_contact_point is not None
            and delta_time > 0.0
        ):
            velocity = (
                (
                    contact_point[0]
                    - state.previous_contact_point[0]
                )
                / delta_time,
                (
                    contact_point[1]
                    - state.previous_contact_point[1]
                )
                / delta_time,
            )
        else:
            velocity = (0.0, 0.0)

        hand.velocity = velocity
        hand.speed = hypot(*velocity)
        hand.debug["bite_contact_point"] = contact_point
        hand.debug["bite_contact_source"] = contact_source

        if (
            mouth.smoothed_center is not None
            and (
                mouth.detected
                or mouth.missing_duration
                <= self._cfg(
                    "mouth",
                    "occlusion_memory_seconds",
                    default=1.0,
                )
            )
        ):
            mouth_center = mouth.smoothed_center
            distance = _distance(contact_point, mouth_center)
            face_scale = self._face_scale(
                mouth=mouth,
                frame_size=frame_size,
            )
            normalized_distance = (
                distance / max(face_scale, _EPSILON)
            )

            if state.previous_distance is None:
                distance_delta = 0.0
            else:
                distance_delta = (
                    state.previous_distance - distance
                )

            mouth_vector = (
                mouth_center[0] - contact_point[0],
                mouth_center[1] - contact_point[1],
            )
            direction = _direction_alignment(
                velocity,
                mouth_vector,
            )

            utensil_context = bool(
                hand.hand_pose in {
                    HandPose.PINCH,
                    HandPose.GRIP,
                }
                or hand.holding_kind
                == HoldingKind.UTENSIL
            )
            enter_ratio = self._cfg(
                "mouth",
                (
                    "utensil_near_mouth_enter_ratio"
                    if utensil_context
                    else "near_mouth_enter_ratio"
                ),
                default=0.48 if utensil_context else 0.42,
            )
            exit_ratio = self._cfg(
                "mouth",
                "near_mouth_exit_ratio",
                default=0.56,
            )

            previous_near = (
                state.near_started_at is not None
            )
            near_mouth = (
                normalized_distance <= (
                    exit_ratio
                    if previous_near
                    else enter_ratio
                )
            )
            inside_roi = (
                mouth.roi is not None
                and _point_in_box(
                    contact_point,
                    mouth.roi,
                )
            )

            minimum_change = (
                face_scale
                * self._cfg(
                    "relative_motion",
                    "minimum_distance_change_ratio",
                    default=0.015,
                )
            )
            approaching = (
                distance_delta > minimum_change
                and direction
                >= self._cfg(
                    "relative_motion",
                    "approach_direction_threshold",
                    default=0.20,
                )
            )

            raw_leaving = (
                distance_delta < -minimum_change
                or direction
                <= self._cfg(
                    "relative_motion",
                    "leave_direction_threshold",
                    default=-0.15,
                )
            )
            mouth_memory_active = bool(
                not mouth.detected
                and mouth.occluded
            )
            still_near_during_occlusion = bool(
                mouth_memory_active
                and normalized_distance <= exit_ratio
            )

            if near_mouth or inside_roi:
                if state.near_started_at is None:
                    state.near_started_at = timestamp
                state.reached_mouth = True
            elif not still_near_during_occlusion:
                state.near_started_at = None

            leave_candidate = bool(
                raw_leaving
                and normalized_distance >= exit_ratio
                and not still_near_during_occlusion
            )
            if leave_candidate:
                if state.leave_candidate_started_at is None:
                    state.leave_candidate_started_at = timestamp
            else:
                state.leave_candidate_started_at = None

            leave_candidate_duration = (
                timestamp - state.leave_candidate_started_at
                if state.leave_candidate_started_at is not None
                else 0.0
            )
            leaving = (
                leave_candidate_duration
                >= self._cfg(
                    "bite_session",
                    "leave_signal_hold_seconds",
                    default=0.06,
                )
            )

            hand.hand_mouth_distance = distance
            hand.normalized_hand_mouth_distance = (
                normalized_distance
            )
            hand.distance_delta = distance_delta
            hand.motion_direction_to_mouth = direction
            hand.approaching_mouth = approaching
            hand.leaving_mouth = leaving
            hand.near_mouth = bool(
                near_mouth
                or inside_roi
                or still_near_during_occlusion
            )
            hand.debug["near_enter_ratio"] = enter_ratio
            hand.debug["near_exit_ratio"] = exit_ratio
            hand.debug["utensil_context"] = utensil_context
            hand.debug["raw_leaving"] = raw_leaving
            hand.debug["leave_candidate_duration"] = (
                leave_candidate_duration
            )
            hand.debug["occlusion_hold_near"] = (
                still_near_during_occlusion
            )

            # ----------------------------------------------------------
            # 그릇 검출에 의존하지 않는 가상 LOW FOOD ZONE
            # ----------------------------------------------------------
            low_zone_y_threshold = (
                mouth_center[1]
                + face_scale
                * self._cfg(
                    "bite_session",
                    "low_zone_y_offset_ratio",
                    default=0.85,
                )
            )
            low_zone_distance_threshold = self._cfg(
                "bite_session",
                "low_zone_min_distance_ratio",
                default=1.20,
            )
            in_low_zone = bool(
                palm_center[1] >= low_zone_y_threshold
                and normalized_distance
                >= low_zone_distance_threshold
            )

            if in_low_zone:
                if state.low_zone_started_at is None:
                    state.low_zone_started_at = timestamp
                    state.low_zone_reference_distance = (
                        normalized_distance
                    )
                    state.low_zone_reference_y = (
                        palm_center[1]
                    )
                    state.low_zone_qualified = False
                else:
                    state.low_zone_reference_distance = max(
                        float(
                            state.low_zone_reference_distance
                            or normalized_distance
                        ),
                        normalized_distance,
                    )

                low_zone_dwell = (
                    timestamp - state.low_zone_started_at
                )
                if (
                    low_zone_dwell
                    >= self._cfg(
                        "bite_session",
                        "minimum_low_zone_seconds",
                        default=0.35,
                    )
                ):
                    state.low_zone_qualified = True
                    state.low_zone_context_until = (
                        timestamp
                        + self._cfg(
                            "bite_session",
                            "low_zone_context_seconds",
                            default=3.0,
                        )
                    )
            else:
                low_zone_dwell = (
                    timestamp - state.low_zone_started_at
                    if state.low_zone_started_at is not None
                    else 0.0
                )
                state.low_zone_started_at = None

            if timestamp > state.low_zone_context_until:
                state.low_zone_qualified = False
                state.low_zone_reference_distance = None
                state.low_zone_reference_y = None

            low_reference_distance = (
                state.low_zone_reference_distance
            )
            distance_reduction_ratio = (
                (
                    low_reference_distance
                    - normalized_distance
                )
                / max(
                    low_reference_distance,
                    _EPSILON,
                )
                if low_reference_distance is not None
                else 0.0
            )
            vertical_rise_ratio = (
                (
                    state.low_zone_reference_y
                    - palm_center[1]
                )
                / max(face_scale, _EPSILON)
                if state.low_zone_reference_y is not None
                else 0.0
            )
            low_zone_context_active = bool(
                state.low_zone_qualified
                and timestamp
                <= state.low_zone_context_until
            )

            hand.debug["in_low_food_zone"] = in_low_zone
            hand.debug["low_zone_dwell"] = low_zone_dwell
            hand.debug["low_zone_context_active"] = (
                low_zone_context_active
            )
            hand.debug["low_reference_distance"] = (
                low_reference_distance
            )
            hand.debug["distance_reduction_ratio"] = (
                distance_reduction_ratio
            )
            hand.debug["vertical_rise_ratio"] = (
                vertical_rise_ratio
            )

            hand.inside_mouth_roi = inside_roi
            hand.mouth_occluded = (
                mouth.occluded
                and (near_mouth or inside_roi)
            )
            hand.debug["mouth_currently_visible"] = bool(
                mouth.detected
            )
            hand.debug["mouth_memory_active"] = bool(
                not mouth.detected
                and mouth.occluded
            )
            hand.debug["mouth_missing_duration"] = float(
                mouth.missing_duration
            )
            hand.dwell_time_near_mouth = (
                timestamp - state.near_started_at
                if state.near_started_at is not None
                else 0.0
            )
            state.previous_distance = distance

        if (
            virtual_plate.available
            and virtual_plate.roi is not None
            and _point_in_box(
                palm_center,
                virtual_plate.roi,
            )
        ):
            state.started_near_plate = True
            state.low_zone_qualified = True
            state.low_zone_context_until = (
                timestamp
                + self._cfg(
                    "bite_session",
                    "low_zone_context_seconds",
                    default=3.0,
                )
            )
            if (
                hand.normalized_hand_mouth_distance
                is not None
            ):
                state.low_zone_reference_distance = max(
                    float(
                        state.low_zone_reference_distance
                        or 0.0
                    ),
                    hand.normalized_hand_mouth_distance,
                )
            state.low_zone_reference_y = palm_center[1]
            state.plate_context_until = (
                timestamp
                + self._cfg(
                    "bite_session",
                    "plate_context_seconds",
                    default=2.5,
                )
            )

        if timestamp > state.plate_context_until:
            state.started_near_plate = False

        if hand.hand_pose in {
            HandPose.PINCH,
            HandPose.GRIP,
        }:
            state.pose_context_until = (
                timestamp
                + self._cfg(
                    "bite_session",
                    "pose_context_seconds",
                    default=1.0,
                )
            )

        hand.started_near_plate = (
            state.started_near_plate
        )
        state.previous_smoothed_center = palm_center
        state.previous_contact_point = contact_point
        hand.clamp()

    def _select_bite_contact_point(
        self,
        hand: HandObservation,
        mouth: MouthObservation,
    ) -> Tuple[Optional[Point2D], str]:
        """
        Bite 판단에 사용할 손의 대표 접촉점을 선택한다.

        - PINCH / GRIP / UTENSIL 맥락에서는 검지 끝 우선
        - 그 외에는 검지 끝과 손바닥 중심 중 입에 더 가까운 점
        - 완전한 Mouth ROI 접촉은 요구하지 않고 거리 기준으로 사용한다.
        """
        palm = hand.smoothed_palm_center
        index_tip = hand.index_tip
        mouth_center = mouth.smoothed_center

        utensil_context = bool(
            hand.hand_pose in {
                HandPose.PINCH,
                HandPose.GRIP,
            }
            or hand.holding_kind
            == HoldingKind.UTENSIL
        )

        if utensil_context and index_tip is not None:
            # 긴 숟가락·젓가락에서는 실제 음식 위치가 검지 끝보다
            # 손가락 진행 방향의 앞쪽에 있을 수 있다.
            # Holding 검출에만 의존하지 않고 PINCH/GRIP에서도 적용한다.
            virtual_tip: Optional[Point2D] = None
            if len(hand.landmarks) > 8:
                wrist = hand.landmarks[0]
                direction_x = index_tip[0] - wrist[0]
                direction_y = index_tip[1] - wrist[1]
                extension_ratio = self._cfg(
                    "bite_session",
                    "virtual_utensil_extension_ratio",
                    default=0.28,
                )
                virtual_tip = (
                    index_tip[0] + direction_x * extension_ratio,
                    index_tip[1] + direction_y * extension_ratio,
                )

            if (
                virtual_tip is not None
                and mouth_center is not None
                and _distance(virtual_tip, mouth_center)
                < _distance(index_tip, mouth_center)
            ):
                return virtual_tip, "virtual_utensil_tip"

            return index_tip, "index_tip_utensil"

        candidates: List[Tuple[Point2D, str]] = []
        if palm is not None:
            candidates.append((palm, "palm_center"))
        if index_tip is not None:
            candidates.append((index_tip, "index_tip"))

        if not candidates:
            return None, "none"

        if mouth_center is None:
            return candidates[0]

        return min(
            candidates,
            key=lambda item: _distance(
                item[0],
                mouth_center,
            ),
        )

    # ------------------------------------------------------------------
    # Vessel 관측 및 Drink Session
    # ------------------------------------------------------------------

    def _build_vessel_observation(
        self,
        detections: Sequence[Detection],
        hands: Dict[HandSide, HandObservation],
        timestamp: float,
    ) -> VesselObservation:
        candidates = [
            item
            for item in detections
            if item.label.strip().lower()
            in self.VESSEL_LABELS
        ]
        candidate = max(
            candidates,
            key=lambda item: item.confidence,
            default=None,
        )

        remembered = False
        missing_duration = 0.0

        if candidate is not None:
            self._last_vessel = candidate
            self._last_vessel_seen_at = timestamp
        elif (
            self._last_vessel is not None
            and self._last_vessel_seen_at is not None
        ):
            age = timestamp - self._last_vessel_seen_at
            if age <= self._cfg(
                "drink_session",
                "vessel_memory_seconds",
                default=1.2,
            ):
                remembered = True
                missing_duration = age
                candidate = self._last_vessel

        if candidate is None:
            return VesselObservation(
                detected=False,
                remembered=False,
                missing_duration=missing_duration,
            )

        nearest_side = HandSide.UNKNOWN
        hand_near = False
        nearest_ratio = 999.0

        for side, hand in hands.items():
            if (
                not hand.detected
                or hand.smoothed_palm_center is None
            ):
                continue

            ratio = _point_box_distance(
                hand.smoothed_palm_center,
                candidate.box,
            ) / max(
                _box_diagonal(candidate.box),
                1.0,
            )
            if ratio < nearest_ratio:
                nearest_ratio = ratio
                nearest_side = side

        if nearest_ratio <= self._cfg(
            "drink_session",
            "vessel_hand_near_ratio",
            default=0.85,
        ):
            hand_near = True

        normalized_label = _normalize_vessel_label(
            candidate.label
        )

        return VesselObservation(
            detected=not remembered,
            label=candidate.label,
            normalized_label=normalized_label,
            confidence=(
                candidate.confidence
                if not remembered
                else _clamp01(
                    1.0
                    - missing_duration
                    / max(
                        self._cfg(
                            "drink_session",
                            "vessel_memory_seconds",
                            default=1.2,
                        ),
                        _EPSILON,
                    )
                )
            ),
            box=candidate.box,
            center=candidate.center,
            remembered=remembered,
            missing_duration=missing_duration,
            hand_near=hand_near,
            nearest_hand=nearest_side,
            origin_box=self._drink.origin_box,
            origin_center=(
                _box_center(self._drink.origin_box)
                if self._drink.origin_box is not None
                else None
            ),
            debug={
                "nearest_hand_ratio": nearest_ratio,
                "candidate_count": len(candidates),
            },
        )

    def _update_drink_session(
        self,
        vessel: VesselObservation,
        hands: Dict[HandSide, HandObservation],
        mouth: MouthObservation,
        timestamp: float,
    ) -> DrinkSessionObservation:
        tracker = self._drink

        # 완료 신호는 한 번만 유지하고, 쿨다운 종료 후에는
        # 이전 contact/origin 상태까지 완전히 제거한다.
        if tracker.completed_latch:
            if timestamp < tracker.cooldown_until:
                return self._drink_observation(
                    vessel=vessel,
                    timestamp=timestamp,
                    state=DrinkState.COOLDOWN,
                    session_completed=False,
                )

            self._reset_drink_session(
                preserve_cooldown=False
            )
            tracker = self._drink

        expired_now = tracker.expired_latch
        tracker.expired_latch = False

        visible_now = bool(
            vessel.detected
            and vessel.box is not None
        )
        nearest_ratio = float(
            vessel.debug.get(
                "nearest_hand_ratio",
                999.0,
            )
        )
        strong_contact = bool(
            visible_now
            and vessel.hand_near
            and vessel.nearest_hand
            != HandSide.UNKNOWN
            and nearest_ratio
            <= self._cfg(
                "drink_session",
                "strong_contact_ratio",
                default=0.30,
            )
        )

        # --------------------------------------------------------------
        # 아직 Pickup이 검증되지 않은 상태
        # --------------------------------------------------------------
        if not tracker.session_active:
            if visible_now and vessel.box is not None:
                tracker.latest_box = vessel.box
                tracker.vessel_label = (
                    vessel.normalized_label
                )

                # 새로운 후보의 기준 위치는 강한 접촉 전 정지 위치로 잡는다.
                if (
                    tracker.origin_box is None
                    or (
                        not tracker.strong_contact_confirmed
                        and not strong_contact
                    )
                ):
                    tracker.origin_box = vessel.box

                if strong_contact:
                    if (
                        tracker.contact_hand
                        != vessel.nearest_hand
                    ):
                        tracker.contact_started_at = timestamp
                        tracker.strong_contact_confirmed = False
                        tracker.contact_hand_origin = None
                        tracker.hand_movement_ratio = 0.0

                    if tracker.contact_started_at is None:
                        tracker.contact_started_at = timestamp

                    tracker.contact_hand = (
                        vessel.nearest_hand
                    )
                    contact_hand_observation = hands.get(
                        tracker.contact_hand
                    )
                    if (
                        tracker.contact_hand_origin is None
                        and contact_hand_observation is not None
                        and contact_hand_observation.detected
                        and contact_hand_observation.smoothed_palm_center
                        is not None
                    ):
                        tracker.contact_hand_origin = (
                            contact_hand_observation.smoothed_palm_center
                        )
                    tracker.contact_last_seen_at = timestamp
                    tracker.contact_time = timestamp
                    tracker.state = (
                        DrinkState.HAND_NEAR_VESSEL
                    )

                    contact_duration = (
                        timestamp
                        - tracker.contact_started_at
                    )
                    if (
                        contact_duration
                        >= self._cfg(
                            "drink_session",
                            "minimum_strong_contact_seconds",
                            default=0.20,
                        )
                    ):
                        tracker.strong_contact_confirmed = True

                elif not tracker.strong_contact_confirmed:
                    tracker.contact_started_at = None
                    tracker.contact_last_seen_at = None
                    tracker.contact_time = None
                    tracker.contact_hand = HandSide.UNKNOWN
                    tracker.contact_hand_origin = None
                    tracker.hand_movement_ratio = 0.0
                    tracker.state = (
                        DrinkState.VESSEL_AVAILABLE
                    )

                # 강한 접촉 뒤, 같은 손이 입 방향으로 움직이면서
                # 용기도 함께 이동한 경우에만 Pickup으로 인정한다.
                if (
                    tracker.strong_contact_confirmed
                    and tracker.origin_box is not None
                ):
                    tracker.movement_ratio = (
                        _distance(
                            _box_center(vessel.box),
                            _box_center(
                                tracker.origin_box
                            ),
                        )
                        / max(
                            _box_diagonal(
                                tracker.origin_box
                            ),
                            1.0,
                        )
                    )

                    contact_hand_observation = hands.get(
                        tracker.contact_hand
                    )
                    same_hand_approaching = bool(
                        contact_hand_observation is not None
                        and contact_hand_observation.detected
                        and contact_hand_observation.approaching_mouth
                    )
                    same_hand_still_near = bool(
                        vessel.nearest_hand
                        == tracker.contact_hand
                        and vessel.hand_near
                    )

                    if (
                        contact_hand_observation is not None
                        and contact_hand_observation.detected
                        and contact_hand_observation.smoothed_palm_center
                        is not None
                        and tracker.contact_hand_origin is not None
                    ):
                        tracker.hand_movement_ratio = (
                            _distance(
                                contact_hand_observation.smoothed_palm_center,
                                tracker.contact_hand_origin,
                            )
                            / max(
                                _box_diagonal(
                                    tracker.origin_box
                                ),
                                1.0,
                            )
                        )

                    if (
                        tracker.movement_ratio
                        >= self._cfg(
                            "drink_session",
                            "pickup_movement_ratio",
                            default=0.12,
                        )
                        and tracker.hand_movement_ratio
                        >= self._cfg(
                            "drink_session",
                            "pickup_hand_movement_ratio",
                            default=0.18,
                        )
                        and same_hand_approaching
                        and same_hand_still_near
                    ):
                        tracker.pickup_verified = True
                        tracker.session_active = True
                        tracker.session_start = timestamp
                        tracker.state = (
                            DrinkState.PICKUP_SESSION
                        )
                        tracker.mouth_reached = False
                        tracker.left_mouth = False
                        tracker.return_visible_since = None
                        tracker.completed_at = None

            else:
                # 용기 소실만으로 Pickup을 인정하지 않는다.
                # 직전에 '강한 접촉이 일정 시간 유지'되었어야 한다.
                recent_strong_contact = bool(
                    tracker.strong_contact_confirmed
                    and tracker.contact_last_seen_at
                    is not None
                    and timestamp
                    - tracker.contact_last_seen_at
                    <= self._cfg(
                        "drink_session",
                        "strong_contact_to_disappear_seconds",
                        default=0.50,
                    )
                )

                contact_hand_observation = hands.get(
                    tracker.contact_hand
                )
                same_hand_approaching = bool(
                    contact_hand_observation is not None
                    and contact_hand_observation.detected
                    and contact_hand_observation.approaching_mouth
                )

                if (
                    contact_hand_observation is not None
                    and contact_hand_observation.detected
                    and contact_hand_observation.smoothed_palm_center
                    is not None
                    and tracker.contact_hand_origin is not None
                    and tracker.origin_box is not None
                ):
                    tracker.hand_movement_ratio = (
                        _distance(
                            contact_hand_observation.smoothed_palm_center,
                            tracker.contact_hand_origin,
                        )
                        / max(
                            _box_diagonal(
                                tracker.origin_box
                            ),
                            1.0,
                        )
                    )

                if (
                    recent_strong_contact
                    and same_hand_approaching
                    and tracker.hand_movement_ratio
                    >= self._cfg(
                        "drink_session",
                        "pickup_hand_movement_ratio",
                        default=0.18,
                    )
                ):
                    tracker.pickup_verified = True
                    tracker.session_active = True
                    tracker.session_start = timestamp
                    tracker.disappeared_time = timestamp
                    tracker.state = (
                        DrinkState.PICKUP_SESSION
                    )
                    tracker.mouth_reached = False
                    tracker.left_mouth = False
                    tracker.return_visible_since = None
                    tracker.completed_at = None
                else:
                    # 오래된 접촉 후보가 다음 Bite까지 살아남지 않게 제거.
                    stale_contact = bool(
                        tracker.contact_last_seen_at
                        is not None
                        and timestamp
                        - tracker.contact_last_seen_at
                        > self._cfg(
                            "drink_session",
                            "contact_candidate_timeout_seconds",
                            default=0.70,
                        )
                    )
                    if stale_contact:
                        self._reset_drink_session(
                            preserve_cooldown=False
                        )
                        tracker = self._drink
                    else:
                        tracker.state = DrinkState.IDLE

            return self._drink_observation(
                vessel=vessel,
                timestamp=timestamp,
                state=tracker.state,
                session_expired=expired_now,
            )

        # Pickup 검증 없이 활성화된 세션은 즉시 폐기한다.
        if not tracker.pickup_verified:
            self._reset_drink_session(
                preserve_cooldown=False
            )
            return self._drink_observation(
                vessel=vessel,
                timestamp=timestamp,
                state=DrinkState.IDLE,
            )

        # 전체 세션 시간 초과
        if (
            tracker.session_start is not None
            and timestamp - tracker.session_start
            > self._cfg(
                "drink_session",
                "session_timeout_seconds",
                default=15.0,
            )
        ):
            result = self._drink_observation(
                vessel=vessel,
                timestamp=timestamp,
                state=DrinkState.EXPIRED,
                session_expired=True,
            )
            self._reset_drink_session(
                preserve_cooldown=False
            )
            return result

        # 반드시 처음 접촉했던 동일한 손만 사용한다.
        hand = hands.get(tracker.contact_hand)
        if hand is not None and not hand.detected:
            hand = None

        approaching = bool(
            hand is not None
            and hand.approaching_mouth
        )
        near_mouth = bool(
            hand is not None
            and (
                hand.near_mouth
                or hand.inside_mouth_roi
            )
        )
        leaving = bool(
            hand is not None
            and hand.leaving_mouth
        )

        if near_mouth:
            tracker.mouth_reached = True
            tracker.state = DrinkState.NEAR_MOUTH
        elif (
            tracker.mouth_reached
            and (
                leaving
                or (
                    hand is not None
                    and hand.normalized_hand_mouth_distance
                    is not None
                    and hand.normalized_hand_mouth_distance
                    >= self._cfg(
                        "mouth",
                        "near_mouth_exit_ratio",
                        default=0.56,
                    )
                )
            )
        ):
            tracker.left_mouth = True
            tracker.state = DrinkState.LEFT_MOUTH
        elif approaching and not tracker.mouth_reached:
            tracker.state = (
                DrinkState.APPROACHING_MOUTH
            )
        elif tracker.mouth_reached and tracker.left_mouth:
            tracker.state = (
                DrinkState.AWAITING_RETURN
            )
        else:
            tracker.state = DrinkState.MANIPULATING

        eligible_for_return = bool(
            tracker.pickup_verified
            and tracker.mouth_reached
            and tracker.left_mouth
        )

        return_candidate = False
        return_confirmed = False

        if (
            eligible_for_return
            and visible_now
            and vessel.box is not None
            and _boxes_near(
                vessel.box,
                tracker.origin_box,
                self._cfg(
                    "drink_session",
                    "return_position_ratio",
                    default=1.10,
                ),
            )
        ):
            return_candidate = True
            tracker.state = (
                DrinkState.RETURN_CANDIDATE
            )

            if tracker.return_visible_since is None:
                tracker.return_visible_since = timestamp

            if (
                timestamp
                - tracker.return_visible_since
                >= self._cfg(
                    "drink_session",
                    "return_confirm_seconds",
                    default=0.65,
                )
            ):
                return_confirmed = True
                tracker.state = DrinkState.CONFIRMED
                tracker.session_active = False
                tracker.completed_at = timestamp
                tracker.completed_latch = True
                tracker.cooldown_until = (
                    timestamp
                    + self._cfg(
                        "drink_session",
                        "cooldown_seconds",
                        default=3.0,
                    )
                )
        else:
            tracker.return_visible_since = None

        result = self._drink_observation(
            vessel=vessel,
            timestamp=timestamp,
            state=tracker.state,
            approaching=approaching,
            near_mouth=near_mouth,
            return_candidate=return_candidate,
            return_confirmed=return_confirmed,
            session_completed=return_confirmed,
        )

        if return_confirmed:
            self._bite = _BiteTracker()

        return result

    def _drink_observation(
        self,
        vessel: VesselObservation,
        timestamp: float,
        state: DrinkState,
        approaching: bool = False,
        near_mouth: bool = False,
        return_candidate: bool = False,
        return_confirmed: bool = False,
        session_completed: bool = False,
        session_expired: bool = False,
    ) -> DrinkSessionObservation:
        tracker = self._drink
        duration = (
            timestamp - tracker.session_start
            if tracker.session_start is not None
            else 0.0
        )
        return_duration = (
            timestamp - tracker.return_visible_since
            if tracker.return_visible_since is not None
            else 0.0
        )

        return DrinkSessionObservation(
            state=state,
            session_active=tracker.session_active,
            contact_hand=tracker.contact_hand,
            vessel_visible=bool(vessel.detected),
            vessel_label=(
                vessel.normalized_label
                or tracker.vessel_label
            ),
            hand_near_vessel=vessel.hand_near,
            vessel_disappeared_after_contact=(
                tracker.disappeared_time is not None
            ),
            approaching_mouth=approaching,
            near_mouth=near_mouth,
            mouth_reached=tracker.mouth_reached,
            left_mouth=tracker.left_mouth,
            return_candidate=return_candidate,
            return_confirmed=return_confirmed,
            session_completed=session_completed,
            session_expired=session_expired,
            session_duration=max(0.0, duration),
            return_visible_duration=max(
                0.0,
                return_duration,
            ),
            completed_at=tracker.completed_at,
            debug={
                "origin_box": tracker.origin_box,
                "latest_box": tracker.latest_box,
                "strong_contact_confirmed": (
                    tracker.strong_contact_confirmed
                ),
                "pickup_verified": (
                    tracker.pickup_verified
                ),
                "movement_ratio": (
                    tracker.movement_ratio
                ),
                "hand_movement_ratio": (
                    tracker.hand_movement_ratio
                ),
                "contact_hand_origin": (
                    tracker.contact_hand_origin
                ),
                "contact_started_at": (
                    tracker.contact_started_at
                ),
                "contact_last_seen_at": (
                    tracker.contact_last_seen_at
                ),
            },
        )

    # ------------------------------------------------------------------
    # Bite Session
    # ------------------------------------------------------------------

    def _update_bite_session(
        self,
        active_hand: Optional[HandObservation],
        active_hand_side: HandSide,
        drink_session: DrinkSessionObservation,
        timestamp: float,
    ) -> BiteSessionObservation:
        tracker = self._bite

        # Pickup이 검증된 Drink는 해당 손의 입 왕복을 독점한다.
        pickup_verified = bool(
            drink_session.debug.get(
                "pickup_verified",
                False,
            )
        )

        bite_in_progress = bool(
            tracker.state
            in {
                BiteState.APPROACHING,
                BiteState.NEAR_MOUTH,
                BiteState.LEAVING,
            }
            and tracker.started_at is not None
        )

        # 진행 중 Bite는 단순 Pickup 후보 때문에 취소하지 않는다.
        # Drink가 실제 입 근처에 도달했거나 반환 단계로 넘어간 경우만
        # Drink가 우선권을 가진다.
        drink_has_reached_mouth = bool(
            drink_session.mouth_reached
            or drink_session.state
            in {
                DrinkState.NEAR_MOUTH,
                DrinkState.LEFT_MOUTH,
                DrinkState.RETURN_CANDIDATE,
                DrinkState.CONFIRMED,
                DrinkState.COOLDOWN,
            }
        )

        # 새 Bite는 검증된 용기 Pickup 동안 시작하지 않되,
        # 이미 시작된 Bite는 실제 Drink 입 도달 전까지 계속 추적한다.
        blocked_by_drink = bool(
            drink_has_reached_mouth
            or (
                not bite_in_progress
                and (
                    pickup_verified
                    or drink_session.session_active
                    or drink_session.state
                    == DrinkState.PICKUP_SESSION
                )
            )
        )

        if blocked_by_drink:
            self._reset_bite_tracker(
                cancel_reason="blocked_by_actual_drink"
            )
            return BiteSessionObservation(
                state=BiteState.IDLE,
                active=False,
                active_hand=active_hand_side,
                blocked_by_drink=True,
                cancel_reason="blocked_by_actual_drink",
            )

        if tracker.completed_latch:
            tracker.completed_latch = False

        # 고정 시간만 기다리지 않고, 손이 다시 낮은 식사 영역으로
        # 돌아오면 다음 Bite를 바로 시작할 수 있게 재활성화한다.
        if timestamp < tracker.cooldown_until:
            cooldown_rearmed = bool(
                active_hand is not None
                and active_hand.detected
                and (
                    active_hand.debug.get(
                        "in_low_food_zone",
                        False,
                    )
                    or (
                        active_hand.debug.get(
                            "low_zone_context_active",
                            False,
                        )
                        and active_hand.normalized_hand_mouth_distance
                        is not None
                        and active_hand.normalized_hand_mouth_distance
                        >= self._cfg(
                            "bite_session",
                            "cooldown_rearm_distance_ratio",
                            default=1.20,
                        )
                    )
                )
            )

            if cooldown_rearmed:
                self._bite = _BiteTracker()
                tracker = self._bite
            else:
                tracker.state = BiteState.COOLDOWN
                return self._bite_observation(
                    active_hand=active_hand,
                    timestamp=timestamp,
                    session_completed=False,
                )

        if active_hand is None or not active_hand.detected:
            if (
                tracker.started_at is not None
                and timestamp - tracker.started_at
                > self._cfg(
                    "bite_session",
                    "tracking_grace_seconds",
                    default=0.45,
                )
            ):
                self._reset_bite_tracker(
                    cancel_reason="hand_lost"
                )
            return self._bite_observation(
                active_hand=active_hand,
                timestamp=timestamp,
            )

        if active_hand.holding_kind == HoldingKind.PHONE:
            self._reset_bite_tracker(
                cancel_reason="phone_holding"
            )
            return BiteSessionObservation(
                state=BiteState.IDLE,
                active=False,
                active_hand=active_hand.side,
                hand_pose=active_hand.hand_pose,
                phone_blocked=True,
                cancel_reason="phone_holding",
            )

        side_state = self._hand_states[active_hand.side]
        pose_context_active = bool(
            timestamp <= side_state.pose_context_until
        )

        distance = (
            active_hand.normalized_hand_mouth_distance
        )
        near_now = bool(
            active_hand.near_mouth
            or active_hand.inside_mouth_roi
        )
        approaching_now = bool(
            active_hand.approaching_mouth
        )
        leaving_now = bool(
            active_hand.leaving_mouth
        )
        mouth_visible = bool(
            active_hand.debug.get(
                "mouth_currently_visible",
                False,
            )
        )
        low_context = bool(
            active_hand.debug.get(
                "low_zone_context_active",
                False,
            )
            or active_hand.started_near_plate
        )
        in_low_zone = bool(
            active_hand.debug.get(
                "in_low_food_zone",
                False,
            )
        )
        reduction_ratio = float(
            active_hand.debug.get(
                "distance_reduction_ratio",
                0.0,
            )
        )
        low_reference_distance = (
            active_hand.debug.get(
                "low_reference_distance"
            )
        )

        handheld_context_active = bool(
            timestamp
            <= self._handheld_until.get(
                active_hand.side,
                0.0,
            )
        )

        if (
            handheld_context_active
            and distance is not None
            and not near_now
            and distance
            >= self._cfg(
                "bite_session",
                "handheld_rearm_distance_ratio",
                default=0.90,
            )
        ):
            self._handheld_rearmed[
                active_hand.side
            ] = True

        handheld_ready = bool(
            handheld_context_active
            and self._handheld_rearmed.get(
                active_hand.side,
                False,
            )
        )

        absolute_distance_drop = (
            (
                float(low_reference_distance)
                - float(distance)
            )
            if (
                low_reference_distance is not None
                and distance is not None
            )
            else 0.0
        )

        reduction_ready = bool(
            low_context
            and (
                reduction_ratio
                >= self._cfg(
                    "bite_session",
                    "minimum_distance_reduction_ratio",
                    default=0.35,
                )
                or absolute_distance_drop
                >= self._cfg(
                    "bite_session",
                    "minimum_absolute_distance_drop",
                    default=0.45,
                )
            )
        )

        # 큰 수저처럼 손끝이 입까지 가까워지지 않아도,
        # 낮은 식사 영역에서 출발해 거리가 충분히 크게 줄었다면
        # 입 근처 도달로 인정한다. 수저 검출 자체에는 의존하지 않는다.
        adaptive_near = bool(
            low_context
            and distance is not None
            and (
                reduction_ratio
                >= self._cfg(
                    "bite_session",
                    "adaptive_near_reduction_ratio",
                    default=0.58,
                )
                or absolute_distance_drop
                >= self._cfg(
                    "bite_session",
                    "adaptive_near_absolute_drop",
                    default=0.65,
                )
            )
            and distance
            <= self._cfg(
                "bite_session",
                "adaptive_near_max_distance_ratio",
                default=1.20,
            )
        )

        # 세션 중 최솟값 갱신
        if tracker.started_at is not None and distance is not None:
            tracker.minimum_distance = min(
                float(
                    tracker.minimum_distance
                    if tracker.minimum_distance is not None
                    else distance
                ),
                distance,
            )

        if tracker.state in {
            BiteState.IDLE,
            BiteState.CANCELLED,
            BiteState.CONFIRMED,
        }:
            tracker.state = BiteState.IDLE
            tracker.cancel_reason = None

            can_start = bool(
                mouth_visible
                and (
                    low_context
                    or handheld_ready
                )
                and (
                    approaching_now
                    or reduction_ready
                    or near_now
                    or adaptive_near
                )
            )

            if can_start:
                tracker.active_hand = active_hand.side
                tracker.started_at = timestamp
                tracker.approach_started_at = timestamp
                tracker.start_distance = float(
                    low_reference_distance
                    if low_reference_distance is not None
                    else (
                        distance
                        if distance is not None
                        else 0.0
                    )
                )
                tracker.minimum_distance = distance
                tracker.low_zone_qualified = low_context
                tracker.returned_to_low_zone = False
                tracker.leave_started_at = None
                tracker.mouth_reached = False

                if handheld_ready:
                    self._handheld_rearmed[
                        active_hand.side
                    ] = False

                # 빠른 동작은 APPROACHING 프레임을 건너뛰고
                # 바로 Near에 들어갈 수 있으므로 직접 진입 허용.
                if near_now or adaptive_near:
                    tracker.state = BiteState.NEAR_MOUTH
                    tracker.near_started_at = timestamp
                    tracker.mouth_reached = True
                    tracker.direct_near_entry = True
                else:
                    tracker.state = BiteState.APPROACHING
                    tracker.near_started_at = None
                    tracker.direct_near_entry = False

        elif tracker.state == BiteState.APPROACHING:
            mouth_missing_duration = float(
                active_hand.debug.get(
                    "mouth_missing_duration",
                    0.0,
                )
            )
            mouth_memory_active = bool(
                active_hand.debug.get(
                    "mouth_memory_active",
                    False,
                )
            )

            if tracker.active_hand != active_hand.side:
                self._reset_bite_tracker(
                    cancel_reason="active_hand_changed"
                )
            elif (
                mouth_memory_active
                and mouth_missing_duration
                > self._cfg(
                    "mouth",
                    "occlusion_memory_seconds",
                    default=1.0,
                )
            ):
                self._reset_bite_tracker(
                    cancel_reason="mouth_occlusion_timeout"
                )
            elif near_now or adaptive_near:
                tracker.state = BiteState.NEAR_MOUTH
                tracker.near_started_at = timestamp
                tracker.mouth_reached = True
            elif (
                leaving_now
                and (
                    reduction_ratio
                    >= self._cfg(
                        "bite_session",
                        "fast_bite_reduction_ratio",
                        default=0.45,
                    )
                    or absolute_distance_drop
                    >= self._cfg(
                        "bite_session",
                        "fast_bite_absolute_drop",
                        default=0.60,
                    )
                )
                and tracker.minimum_distance is not None
                and tracker.minimum_distance
                <= self._cfg(
                    "bite_session",
                    "fast_bite_max_distance_ratio",
                    default=0.95,
                )
            ):
                # 처리 FPS가 낮거나 동작이 빠르면 NEAR_MOUTH가
                # 화면상 한 프레임도 유지되지 않을 수 있다.
                # 낮은 영역 출발 + 큰 거리 감소 + 다시 이탈이
                # 확인되면 APPROACHING에서 LEAVING으로 연결한다.
                tracker.state = BiteState.LEAVING
                tracker.leave_started_at = timestamp
                tracker.mouth_reached = True
                tracker.direct_near_entry = True
            elif (
                tracker.started_at is not None
                and timestamp - tracker.started_at
                > self._cfg(
                    "bite_session",
                    "maximum_approach_seconds",
                    default=3.0,
                )
            ):
                self._reset_bite_tracker(
                    cancel_reason="approach_timeout"
                )
            elif (
                not approaching_now
                and not reduction_ready
                and distance is not None
                and tracker.start_distance is not None
                and distance
                >= tracker.start_distance
                * self._cfg(
                    "bite_session",
                    "approach_cancel_return_ratio",
                    default=0.92,
                )
            ):
                self._reset_bite_tracker(
                    cancel_reason="approach_returned_without_near"
                )

        elif tracker.state == BiteState.NEAR_MOUTH:
            mouth_missing_duration = float(
                active_hand.debug.get(
                    "mouth_missing_duration",
                    0.0,
                )
            )
            mouth_memory_active = bool(
                active_hand.debug.get(
                    "mouth_memory_active",
                    False,
                )
            )

            if (
                mouth_memory_active
                and mouth_missing_duration
                > self._cfg(
                    "mouth",
                    "occlusion_memory_seconds",
                    default=1.0,
                )
            ):
                self._reset_bite_tracker(
                    cancel_reason="mouth_occlusion_timeout"
                )
                return self._bite_observation(
                    active_hand=active_hand,
                    timestamp=timestamp,
                )

            dwell = (
                timestamp - tracker.near_started_at
                if tracker.near_started_at is not None
                else 0.0
            )
            minimum_dwell = self._cfg(
                "bite_session",
                "minimum_near_mouth_seconds",
                default=0.08,
            )
            maximum_dwell = self._cfg(
                "bite_session",
                "maximum_near_mouth_seconds",
                default=2.0,
            )

            if dwell > maximum_dwell:
                self._reset_bite_tracker(
                    cancel_reason="near_mouth_too_long"
                )
            else:
                minimum_distance = tracker.minimum_distance
                distance_increase_from_min = (
                    distance - minimum_distance
                    if (
                        distance is not None
                        and minimum_distance is not None
                    )
                    else 0.0
                )
                # adaptive_near는 손을 내리기 시작해도 잠시 True로
                # 남을 수 있으므로 Near 해제를 기다리지 않는다.
                # 최소거리 이후 거리 증가 또는 낮은 영역 복귀가 보이면
                # 실제 이탈 동작으로 판단한다.
                return_motion_detected = bool(
                    tracker.mouth_reached
                    and (
                        leaving_now
                        or in_low_zone
                        or distance_increase_from_min
                        >= self._cfg(
                            "bite_session",
                            "near_to_leave_distance_increase",
                            default=0.14,
                        )
                    )
                )

                if (
                    dwell >= minimum_dwell
                    and return_motion_detected
                ):
                    tracker.state = BiteState.LEAVING
                    tracker.leave_started_at = timestamp
                elif (
                    dwell < minimum_dwell
                    and distance_increase_from_min
                    >= self._cfg(
                        "bite_session",
                        "fast_leave_distance_increase",
                        default=0.18,
                    )
                ):
                    tracker.state = BiteState.LEAVING
                    tracker.leave_started_at = timestamp
                elif (
                    not (near_now or adaptive_near)
                    and dwell < minimum_dwell
                ):
                    # Near가 한 프레임만 잡힌 빠른 동작의 안전 경로.
                    fast_near_valid = bool(
                        (
                            reduction_ratio
                            >= self._cfg(
                                "bite_session",
                                "fast_bite_reduction_ratio",
                                default=0.42,
                            )
                            or absolute_distance_drop
                            >= self._cfg(
                                "bite_session",
                                "fast_bite_absolute_drop",
                                default=0.60,
                            )
                        )
                        and tracker.minimum_distance is not None
                        and tracker.minimum_distance
                        <= self._cfg(
                            "bite_session",
                            "fast_bite_max_distance_ratio",
                            default=1.20,
                        )
                    )

                    if fast_near_valid:
                        tracker.state = BiteState.LEAVING
                        tracker.leave_started_at = timestamp
                        tracker.mouth_reached = True
                    else:
                        self._reset_bite_tracker(
                            cancel_reason="near_mouth_too_short"
                        )

        elif tracker.state == BiteState.LEAVING:
            leave_duration = (
                timestamp - tracker.leave_started_at
                if tracker.leave_started_at is not None
                else 0.0
            )

            if in_low_zone:
                tracker.returned_to_low_zone = True

            minimum_distance = tracker.minimum_distance
            distance_increase = (
                distance - minimum_distance
                if (
                    distance is not None
                    and minimum_distance is not None
                )
                else 0.0
            )

            returned_far_enough = bool(
                tracker.returned_to_low_zone
                or (
                    distance is not None
                    and (
                        distance_increase
                        >= self._cfg(
                            "bite_session",
                            "minimum_return_distance_increase",
                            default=0.18,
                        )
                    )
                    and distance
                    >= self._cfg(
                        "bite_session",
                        "minimum_return_distance_ratio",
                        default=0.90,
                    )
                )
            )
            long_enough = bool(
                leave_duration
                >= self._cfg(
                    "bite_session",
                    "leave_confirm_seconds",
                    default=0.03,
                )
            )

            if returned_far_enough and long_enough:
                tracker.state = BiteState.CONFIRMED
                tracker.confirmed_at = timestamp
                tracker.completed_latch = True

                if tracker.active_hand in {
                    HandSide.LEFT,
                    HandSide.RIGHT,
                }:
                    self._handheld_until[
                        tracker.active_hand
                    ] = (
                        timestamp
                        + self._cfg(
                            "bite_session",
                            "handheld_context_seconds",
                            default=180.0,
                        )
                    )
                    self._handheld_rearmed[
                        tracker.active_hand
                    ] = False

                tracker.cooldown_until = (
                    timestamp
                    + self._cfg(
                        "bite_session",
                        "cooldown_seconds",
                        default=0.25,
                    )
                )
                return self._bite_observation(
                    active_hand=active_hand,
                    timestamp=timestamp,
                    session_completed=True,
                )

            if (
                tracker.started_at is not None
                and timestamp - tracker.started_at
                > self._cfg(
                    "bite_session",
                    "maximum_total_seconds",
                    default=7.0,
                )
            ):
                self._reset_bite_tracker(
                    cancel_reason="total_timeout"
                )

        return self._bite_observation(
            active_hand=active_hand,
            timestamp=timestamp,
            session_completed=False,
            pose_context_active=pose_context_active,
        )

    def _bite_observation(
        self,
        active_hand: Optional[HandObservation],
        timestamp: float,
        session_completed: bool = False,
        pose_context_active: Optional[bool] = None,
    ) -> BiteSessionObservation:
        tracker = self._bite

        approach_duration = (
            timestamp - tracker.approach_started_at
            if tracker.approach_started_at is not None
            else 0.0
        )
        dwell_duration = (
            timestamp - tracker.near_started_at
            if tracker.near_started_at is not None
            else 0.0
        )
        leave_duration = (
            timestamp - tracker.leave_started_at
            if tracker.leave_started_at is not None
            else 0.0
        )
        total_duration = (
            timestamp - tracker.started_at
            if tracker.started_at is not None
            else 0.0
        )

        if pose_context_active is None:
            pose_context_active = False
            if active_hand is not None:
                pose_context_active = (
                    timestamp
                    <= self._hand_states[
                        active_hand.side
                    ].pose_context_until
                )

        return BiteSessionObservation(
            state=tracker.state,
            active=tracker.state
            in {
                BiteState.APPROACHING,
                BiteState.NEAR_MOUTH,
                BiteState.LEAVING,
            },
            active_hand=tracker.active_hand,
            approaching=bool(
                active_hand is not None
                and active_hand.approaching_mouth
            ),
            near_mouth=bool(
                active_hand is not None
                and (
                    active_hand.near_mouth
                    or active_hand.inside_mouth_roi
                )
            ),
            leaving=bool(
                active_hand is not None
                and active_hand.leaving_mouth
            ),
            approach_duration=approach_duration,
            dwell_duration=dwell_duration,
            leave_duration=leave_duration,
            total_duration=total_duration,
            mouth_reached=tracker.mouth_reached,
            round_trip_completed=session_completed,
            session_completed=session_completed,
            hand_pose=(
                active_hand.hand_pose
                if active_hand is not None
                else HandPose.UNKNOWN
            ),
            pose_context_active=bool(
                pose_context_active
            ),
            started_near_plate=bool(
                active_hand is not None
                and active_hand.started_near_plate
            ),
            cancel_reason=tracker.cancel_reason,
            debug={
                "cooldown_until": tracker.cooldown_until,
                "cooldown_remaining": max(
                    0.0,
                    tracker.cooldown_until - timestamp,
                ),
                "cooldown_rearm_ready": bool(
                    active_hand is not None
                    and active_hand.detected
                    and active_hand.debug.get(
                        "in_low_food_zone",
                        False,
                    )
                ),
                "start_distance": tracker.start_distance,
                "minimum_distance": tracker.minimum_distance,
                "low_zone_qualified": tracker.low_zone_qualified,
                "returned_to_low_zone": (
                    tracker.returned_to_low_zone
                ),
                "direct_near_entry": tracker.direct_near_entry,
                "bite_contact_source": (
                    active_hand.debug.get(
                        "bite_contact_source",
                        "none",
                    )
                    if active_hand is not None
                    else "none"
                ),
                "carried_context_kind": (
                    str(
                        self._carried_context.get(
                            tracker.active_hand,
                            {},
                        ).get(
                            "kind",
                            "unknown",
                        )
                    )
                ),
                "carried_context_label": (
                    str(
                        self._carried_context.get(
                            tracker.active_hand,
                            {},
                        ).get(
                            "label",
                            "",
                        )
                    )
                ),
                "carried_context_remaining": (
                    max(
                        0.0,
                        float(
                            self._carried_context.get(
                                tracker.active_hand,
                                {},
                            ).get(
                                "until",
                                0.0,
                            )
                        )
                        - timestamp,
                    )
                ),
                "distance_increase_from_min": (
                    (
                        float(
                            active_hand.normalized_hand_mouth_distance
                        )
                        - float(tracker.minimum_distance)
                    )
                    if (
                        active_hand is not None
                        and active_hand.normalized_hand_mouth_distance
                        is not None
                        and tracker.minimum_distance is not None
                    )
                    else 0.0
                ),
                "distance_reduction_ratio": (
                    float(
                        active_hand.debug.get(
                            "distance_reduction_ratio",
                            0.0,
                        )
                    )
                    if active_hand is not None
                    else 0.0
                ),
                "raw_leaving": bool(
                    active_hand is not None
                    and active_hand.debug.get(
                        "raw_leaving",
                        False,
                    )
                ),
                "leave_candidate_duration": (
                    float(
                        active_hand.debug.get(
                            "leave_candidate_duration",
                            0.0,
                        )
                    )
                    if active_hand is not None
                    else 0.0
                ),
                "blocked_by_drink": False,
                "absolute_distance_drop": (
                    (
                        float(
                            active_hand.debug.get(
                                "low_reference_distance",
                                0.0,
                            )
                        )
                        - float(
                            active_hand.normalized_hand_mouth_distance
                        )
                    )
                    if (
                        active_hand is not None
                        and active_hand.normalized_hand_mouth_distance
                        is not None
                        and active_hand.debug.get(
                            "low_reference_distance"
                        )
                        is not None
                    )
                    else 0.0
                ),
                "adaptive_near": bool(
                    active_hand is not None
                    and active_hand.debug.get(
                        "low_zone_context_active",
                        False,
                    )
                    and active_hand.normalized_hand_mouth_distance
                    is not None
                    and float(
                        active_hand.debug.get(
                            "distance_reduction_ratio",
                            0.0,
                        )
                    )
                    >= self._cfg(
                        "bite_session",
                        "adaptive_near_reduction_ratio",
                        default=0.58,
                    )
                    and active_hand.normalized_hand_mouth_distance
                    <= self._cfg(
                        "bite_session",
                        "adaptive_near_max_distance_ratio",
                        default=1.15,
                    )
                ),
                "in_low_food_zone": bool(
                    active_hand is not None
                    and active_hand.debug.get(
                        "in_low_food_zone",
                        False,
                    )
                ),
                "low_zone_context_active": bool(
                    active_hand is not None
                    and active_hand.debug.get(
                        "low_zone_context_active",
                        False,
                    )
                ),
                "handheld_context_active": bool(
                    active_hand is not None
                    and timestamp
                    <= self._handheld_until.get(
                        active_hand.side,
                        0.0,
                    )
                ),
                "handheld_rearmed": bool(
                    active_hand is not None
                    and self._handheld_rearmed.get(
                        active_hand.side,
                        False,
                    )
                ),
                "handheld_remaining": (
                    max(
                        0.0,
                        self._handheld_until.get(
                            active_hand.side,
                            0.0,
                        )
                        - timestamp,
                    )
                    if active_hand is not None
                    else 0.0
                ),
                "mouth_memory_active": bool(
                    active_hand is not None
                    and active_hand.debug.get(
                        "mouth_memory_active",
                        False,
                    )
                ),
                "mouth_missing_duration": (
                    float(
                        active_hand.debug.get(
                            "mouth_missing_duration",
                            0.0,
                        )
                    )
                    if active_hand is not None
                    else 0.0
                ),
            },
        )

    # ------------------------------------------------------------------
    # 선택 / 보조
    # ------------------------------------------------------------------

    def _select_active_hand(
        self,
        hands: Dict[HandSide, HandObservation],
    ) -> HandSide:
        valid = [
            hand
            for hand in hands.values()
            if hand.detected
        ]
        if not valid:
            return HandSide.UNKNOWN

        def score(hand: HandObservation) -> float:
            value = 0.0

            low_context = bool(
                hand.started_near_plate
                or hand.debug.get(
                    "low_zone_context_active",
                    False,
                )
            )
            reduction_ratio = float(
                hand.debug.get(
                    "distance_reduction_ratio",
                    0.0,
                )
            )

            if low_context:
                value += 0.40
            if hand.approaching_mouth:
                value += 0.35
            value += 0.20 * _clamp01(
                reduction_ratio / 0.60
            )

            if hand.near_mouth:
                if low_context or hand.approaching_mouth:
                    value += 0.18
                else:
                    value -= 0.12

            if hand.leaving_mouth:
                value += 0.10
            if hand.hand_pose in {
                HandPose.PINCH,
                HandPose.GRIP,
            }:
                value += 0.06

            value += 0.05 * _clamp01(
                hand.speed / 150.0
            )
            return value

        return max(valid, key=score).side

    @staticmethod
    def _best_hand_for_mouth(
        hands: Dict[HandSide, HandObservation],
    ) -> Optional[HandObservation]:
        valid = [
            hand
            for hand in hands.values()
            if hand.detected
        ]
        if not valid:
            return None

        return max(
            valid,
            key=lambda hand: (
                int(hand.near_mouth),
                int(hand.approaching_mouth),
                int(hand.leaving_mouth),
                -float(
                    hand.normalized_hand_mouth_distance
                    if hand.normalized_hand_mouth_distance
                    is not None
                    else 999.0
                ),
            ),
        )

    @staticmethod
    def _suggest_action(
        bite_session: BiteSessionObservation,
        drink_session: DrinkSessionObservation,
        active_hand: Optional[HandObservation],
    ) -> MealAction:
        if drink_session.session_completed:
            return MealAction.DRINK
        if bite_session.session_completed:
            return MealAction.BITE
        if drink_session.session_active:
            return MealAction.DRINK_CANDIDATE
        if bite_session.active:
            return MealAction.BITE_CANDIDATE
        if (
            active_hand is not None
            and active_hand.holding_kind
            == HoldingKind.UTENSIL
        ):
            return MealAction.UTENSILING
        return MealAction.NONE

    @staticmethod
    def _active_hand_reason(
        hand: Optional[HandObservation],
    ) -> Dict[str, Any]:
        if hand is None:
            return {"detected": False}
        return {
            "detected": hand.detected,
            "near_mouth": hand.near_mouth,
            "approaching_mouth": (
                hand.approaching_mouth
            ),
            "leaving_mouth": hand.leaving_mouth,
            "started_near_plate": (
                hand.started_near_plate
            ),
            "hand_pose": hand.hand_pose.value,
            "speed": hand.speed,
        }

    def _reset_bite_tracker(
        self,
        cancel_reason: Optional[str] = None,
    ) -> None:
        cooldown_until = self._bite.cooldown_until
        self._bite = _BiteTracker(
            cancel_reason=cancel_reason,
            cooldown_until=cooldown_until,
        )

    def _reset_drink_session(
        self,
        preserve_cooldown: bool,
    ) -> None:
        cooldown_until = (
            self._drink.cooldown_until
            if preserve_cooldown
            else 0.0
        )
        completed_latch = (
            self._drink.completed_latch
            if preserve_cooldown
            else False
        )
        self._drink = _DrinkTracker(
            cooldown_until=cooldown_until,
            completed_latch=completed_latch,
        )

    # ------------------------------------------------------------------
    # Mouth / config helpers
    # ------------------------------------------------------------------

    def _calculate_mouth_openness(
        self,
        raw: RawMouthInput,
    ) -> float:
        if raw.openness is not None:
            return max(0.0, float(raw.openness))

        landmarks = _normalize_landmarks(
            raw.mouth_landmarks
        )
        if len(landmarks) < 4:
            return 0.0

        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return height / max(width, _EPSILON)

    def _mouth_roi(
        self,
        center: Point2D,
        face_box: Optional[BoxXYXY],
        frame_size: Tuple[int, int],
    ) -> BoxXYXY:
        width, height = frame_size

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            face_width = max(x2 - x1, 1.0)
            face_height = max(y2 - y1, 1.0)
        else:
            face_width = width * 0.22
            face_height = height * 0.32

        roi_width = (
            face_width
            * self._cfg(
                "mouth",
                "roi_width_face_ratio",
                default=0.38,
            )
        )
        roi_height = (
            face_height
            * self._cfg(
                "mouth",
                "roi_height_face_ratio",
                default=0.30,
            )
        )

        return _clip_box(
            (
                center[0] - roi_width * 0.5,
                center[1] - roi_height * 0.5,
                center[0] + roi_width * 0.5,
                center[1] + roi_height * 0.5,
            ),
            frame_size,
        )

    def _mouth_occlusion_score(
        self,
        roi: BoxXYXY,
        detections: Sequence[Detection],
    ) -> float:
        score = 0.0
        relevant = (
            self.VESSEL_LABELS
            | HoldingAnalyzer.UTENSIL_LABELS
            | self.FOOD_LABELS
        )
        for detection in detections:
            if (
                detection.label.strip().lower()
                not in relevant
            ):
                continue
            score = max(
                score,
                _intersection_over_smaller_area(
                    roi,
                    detection.box,
                )
                * _clamp01(detection.confidence),
            )
        return _clamp01(score)

    def _face_scale(
        self,
        mouth: MouthObservation,
        frame_size: Tuple[int, int],
    ) -> float:
        if mouth.roi is not None:
            x1, _, x2, _ = mouth.roi
            roi_width = max(x2 - x1, 1.0)
            return (
                roi_width
                / max(
                    self._cfg(
                        "mouth",
                        "roi_width_face_ratio",
                        default=0.38,
                    ),
                    _EPSILON,
                )
            )
        return frame_size[0] * 0.22

    def _build_holding_config(self) -> HoldingConfig:
        holding = self.config.get("holding", {})
        hand_pose = self.config.get("hand_pose", {})
        ema = self.config.get("ema", {})
        weights = holding.get("weights", {})

        return HoldingConfig(
            ema_alpha_position=float(
                ema.get("hand_position_alpha", 0.45)
            ),
            ema_alpha_score=float(
                ema.get("score_alpha", 0.35)
            ),
            hand_box_expand_ratio=float(
                holding.get(
                    "hand_box_expand_ratio",
                    0.55,
                )
            ),
            max_center_distance_ratio=float(
                holding.get(
                    "max_center_distance_ratio",
                    1.10,
                )
            ),
            near_center_distance_ratio=float(
                holding.get(
                    "near_center_distance_ratio",
                    0.55,
                )
            ),
            overlap_weight=float(
                weights.get("overlap", 0.30)
            ),
            distance_weight=float(
                weights.get("distance", 0.25)
            ),
            hand_pose_weight=float(
                weights.get("hand_pose", 0.20)
            ),
            stability_weight=float(
                weights.get(
                    "spatial_stability",
                    0.15,
                )
            ),
            comotion_weight=float(
                weights.get("comotion", 0.10)
            ),
            holding_on_threshold=float(
                holding.get(
                    "holding_on_threshold",
                    0.58,
                )
            ),
            holding_off_threshold=float(
                holding.get(
                    "holding_off_threshold",
                    0.36,
                )
            ),
            minimum_object_confidence=float(
                holding.get(
                    "minimum_object_confidence",
                    0.18,
                )
            ),
            object_memory_seconds=float(
                holding.get(
                    "object_memory_seconds",
                    0.45,
                )
            ),
            hand_memory_seconds=float(
                holding.get(
                    "hand_memory_seconds",
                    0.30,
                )
            ),
            association_memory_seconds=float(
                holding.get(
                    "association_memory_seconds",
                    0.65,
                )
            ),
            stable_distance_ratio=float(
                holding.get(
                    "stable_distance_ratio",
                    0.28,
                )
            ),
            stable_angle_degrees=float(
                holding.get(
                    "stable_angle_degrees",
                    38.0,
                )
            ),
            minimum_motion_for_comotion=float(
                holding.get(
                    "minimum_motion_for_comotion",
                    1.5,
                )
            ),
            pinch_ratio_threshold=float(
                hand_pose.get(
                    "pinch_ratio_threshold",
                    0.34,
                )
            ),
            grip_finger_ratio_threshold=float(
                hand_pose.get(
                    "grip_finger_ratio_threshold",
                    0.78,
                )
            ),
            open_finger_ratio_threshold=float(
                hand_pose.get(
                    "open_finger_ratio_threshold",
                    1.15,
                )
            ),
            maximum_history=int(
                holding.get("maximum_history", 10)
            ),
        )

    def _cfg(
        self,
        section: str,
        key: str,
        default: float,
    ) -> float:
        value = (
            self.config.get(section, {})
            .get(key, default)
        )
        return float(value)

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Meal config file not found: {path}"
            )
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError(
                "meal_config.json 최상위 값은 객체여야 합니다."
            )
        return value


# ----------------------------------------------------------------------
# 순수 helper 함수
# ----------------------------------------------------------------------

def _coerce_hand_side(value: Any) -> HandSide:
    if isinstance(value, HandSide):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"left", "l"}:
        return HandSide.LEFT
    if normalized in {"right", "r"}:
        return HandSide.RIGHT
    return HandSide.UNKNOWN


def _coerce_detection(
    value: Any,
) -> Optional[Detection]:
    if isinstance(value, Detection):
        return value

    if isinstance(value, Mapping):
        label = value.get(
            "label",
            value.get(
                "class_name",
                value.get("name"),
            ),
        )
        confidence = value.get(
            "confidence",
            value.get(
                "conf",
                value.get("score", 0.0),
            ),
        )
        box = value.get(
            "box",
            value.get(
                "bbox",
                value.get("xyxy"),
            ),
        )
        track_id = value.get(
            "track_id",
            value.get("id"),
        )
    else:
        label = getattr(
            value,
            "label",
            getattr(
                value,
                "class_name",
                getattr(value, "name", None),
            ),
        )
        confidence = getattr(
            value,
            "confidence",
            getattr(
                value,
                "conf",
                getattr(value, "score", 0.0),
            ),
        )
        box = getattr(
            value,
            "box",
            getattr(
                value,
                "bbox",
                getattr(value, "xyxy", None),
            ),
        )
        track_id = getattr(
            value,
            "track_id",
            getattr(value, "id", None),
        )

    if label is None or box is None:
        return None

    try:
        x1, y1, x2, y2 = box
        return Detection(
            label=str(label),
            confidence=float(confidence),
            box=(
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ),
            track_id=(
                int(track_id)
                if track_id is not None
                else None
            ),
        )
    except (TypeError, ValueError):
        return None


def _normalize_landmarks(
    landmarks: Sequence[Any],
) -> Tuple[Point2D, ...]:
    points: List[Point2D] = []

    for value in landmarks:
        if isinstance(value, Mapping):
            x = value.get("x")
            y = value.get("y")
        elif hasattr(value, "x") and hasattr(value, "y"):
            x = getattr(value, "x")
            y = getattr(value, "y")
        else:
            try:
                x, y = value[:2]
            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                return tuple()

        try:
            points.append((float(x), float(y)))
        except (TypeError, ValueError):
            return tuple()

    return tuple(points)


def _coerce_point(value: Any) -> Point2D:
    if isinstance(value, Mapping):
        return float(value["x"]), float(value["y"])
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    x, y = value[:2]
    return float(x), float(y)


def _normalize_frame_size(
    frame_size: Tuple[int, int],
) -> Tuple[int, int]:
    try:
        width, height = frame_size
        return max(int(width), 1), max(int(height), 1)
    except (TypeError, ValueError):
        return 1, 1


def _ema_point(
    previous: Optional[Point2D],
    current: Point2D,
    alpha: float,
) -> Point2D:
    if previous is None:
        return current
    return (
        alpha * current[0]
        + (1.0 - alpha) * previous[0],
        alpha * current[1]
        + (1.0 - alpha) * previous[1],
    )


def _ema_scalar(
    previous: float,
    current: float,
    alpha: float,
) -> float:
    return (
        alpha * current
        + (1.0 - alpha) * previous
    )


def _distance(
    first: Point2D,
    second: Point2D,
) -> float:
    return hypot(
        first[0] - second[0],
        first[1] - second[1],
    )


def _direction_alignment(
    velocity: Point2D,
    target_vector: Point2D,
) -> float:
    velocity_norm = hypot(*velocity)
    target_norm = hypot(*target_vector)

    if (
        velocity_norm <= _EPSILON
        or target_norm <= _EPSILON
    ):
        return 0.0

    value = (
        velocity[0] * target_vector[0]
        + velocity[1] * target_vector[1]
    ) / (velocity_norm * target_norm)

    return min(1.0, max(-1.0, value))


def _point_in_box(
    point: Point2D,
    box: BoxXYXY,
) -> bool:
    x1, y1, x2, y2 = box
    return (
        x1 <= point[0] <= x2
        and y1 <= point[1] <= y2
    )


def _point_box_distance(
    point: Point2D,
    box: BoxXYXY,
) -> float:
    x, y = point
    x1, y1, x2, y2 = box

    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return hypot(dx, dy)


def _box_center(box: BoxXYXY) -> Point2D:
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) * 0.5,
        (y1 + y2) * 0.5,
    )


def _box_diagonal(box: BoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return hypot(
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    )


def _boxes_near(
    current: BoxXYXY,
    origin: Optional[BoxXYXY],
    ratio: float,
) -> bool:
    if origin is None:
        return False
    distance = _distance(
        _box_center(current),
        _box_center(origin),
    )
    return (
        distance
        / max(_box_diagonal(origin), 1.0)
        <= ratio
    )


def _clip_box(
    box: BoxXYXY,
    frame_size: Tuple[int, int],
) -> BoxXYXY:
    width, height = frame_size
    x1, y1, x2, y2 = box
    return (
        min(max(0.0, x1), float(width)),
        min(max(0.0, y1), float(height)),
        min(max(0.0, x2), float(width)),
        min(max(0.0, y2), float(height)),
    )


def _expand_box(
    box: BoxXYXY,
    ratio: float,
    frame_size: Tuple[int, int],
) -> BoxXYXY:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return _clip_box(
        (
            x1 - width * ratio,
            y1 - height * ratio,
            x2 + width * ratio,
            y2 + height * ratio,
        ),
        frame_size,
    )


def _union_boxes(
    boxes: Sequence[BoxXYXY],
) -> BoxXYXY:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _intersection_over_smaller_area(
    first: BoxXYXY,
    second: BoxXYXY,
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    width = max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1),
    )
    height = max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    intersection = width * height

    first_area = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )
    second_area = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    return _clamp01(
        intersection
        / max(
            min(first_area, second_area),
            _EPSILON,
        )
    )


def _normalize_vessel_label(
    label: str,
) -> str:
    normalized = str(label).strip().lower()
    if normalized in {
        "bottle",
        "water bottle",
    }:
        return "bottle"
    if normalized in {
        "cup",
        "mug",
        "glass",
        "wine glass",
    }:
        return "cup_or_glass"
    return normalized


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "MealVisionInput",
    "MealVisionProcessor",
    "RawHandInput",
    "RawMouthInput",
]
