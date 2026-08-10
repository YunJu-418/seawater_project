"""
손에 객체를 들고 있는지 판단하는 Holding 분석 모듈.

주요 기능
---------
- 손 랜드마크 기반 Hand Pose(PINCH / GRIP / OPEN / UNKNOWN) 판정
- 손 중심과 객체 바운딩 박스의 거리 및 겹침 계산
- EMA 기반 좌표와 Holding Score 안정화
- 손과 객체의 동반 이동(comotion) 분석
- 손-객체 상대 위치의 공간 안정성(spatial stability) 분석
- YOLO 객체 검출 누락 시 짧은 시간 동안 객체 기억
- ON/OFF 임계값을 분리한 히스테리시스 Holding 판정
- 양손별 독립 상태 추적
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, hypot
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from meal.meal_state import (
    BoxXYXY,
    Detection,
    HandObservation,
    HandPose,
    HandSide,
    HoldingKind,
    Point2D,
)

_EPSILON = 1e-9


@dataclass(frozen=True)
class HoldingConfig:
    ema_alpha_position: float = 0.45
    ema_alpha_score: float = 0.35
    hand_box_expand_ratio: float = 0.55
    max_center_distance_ratio: float = 1.10
    near_center_distance_ratio: float = 0.55
    overlap_weight: float = 0.30
    distance_weight: float = 0.25
    hand_pose_weight: float = 0.20
    stability_weight: float = 0.15
    comotion_weight: float = 0.10
    holding_on_threshold: float = 0.58
    holding_off_threshold: float = 0.36
    minimum_object_confidence: float = 0.18
    object_memory_seconds: float = 0.45
    hand_memory_seconds: float = 0.30
    association_memory_seconds: float = 0.65
    stable_distance_ratio: float = 0.28
    stable_angle_degrees: float = 38.0
    minimum_motion_for_comotion: float = 1.5
    pinch_ratio_threshold: float = 0.34
    grip_finger_ratio_threshold: float = 0.78
    open_finger_ratio_threshold: float = 1.15
    maximum_history: int = 10

    def validate(self) -> None:
        names = (
            "ema_alpha_position", "ema_alpha_score", "overlap_weight",
            "distance_weight", "hand_pose_weight", "stability_weight",
            "comotion_weight", "holding_on_threshold",
            "holding_off_threshold", "minimum_object_confidence",
        )
        for name in names:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        if self.holding_off_threshold >= self.holding_on_threshold:
            raise ValueError(
                "holding_off_threshold must be lower than holding_on_threshold"
            )

        total = (
            self.overlap_weight + self.distance_weight
            + self.hand_pose_weight + self.stability_weight
            + self.comotion_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Holding score weights must sum to 1.0")

        if self.maximum_history < 2:
            raise ValueError("maximum_history must be at least 2")


@dataclass(frozen=True)
class HoldingInput:
    side: HandSide
    timestamp: float
    frame_size: Tuple[int, int]
    landmarks: Sequence[Point2D] = field(default_factory=tuple)
    detections: Sequence[Detection] = field(default_factory=tuple)
    tracking_id: Optional[int] = None


@dataclass
class HoldingCandidate:
    detection: Detection
    kind: HoldingKind
    center_distance: float
    normalized_distance: float
    overlap_score: float
    distance_score: float
    remembered: bool = False

    @property
    def base_association_score(self) -> float:
        return _clamp01(
            0.58 * self.overlap_score + 0.42 * self.distance_score
        )


@dataclass
class HoldingResult:
    side: HandSide
    timestamp: float
    detected: bool
    hand_pose: HandPose
    pose_confidence: float
    holding: bool
    holding_kind: HoldingKind
    holding_label: Optional[str]
    holding_confidence: float
    holding_score: float
    palm_center: Optional[Point2D]
    smoothed_palm_center: Optional[Point2D]
    wrist: Optional[Point2D]
    index_tip: Optional[Point2D]
    thumb_tip: Optional[Point2D]
    associated_box: Optional[BoxXYXY]
    associated_object_center: Optional[Point2D]
    spatial_stability: float
    object_comotion: float
    remembered_object: bool
    missing_duration: float
    debug: Dict[str, Any] = field(default_factory=dict)

    def apply_to(self, observation: HandObservation) -> HandObservation:
        observation.side = self.side
        observation.detected = self.detected
        observation.wrist = self.wrist
        observation.palm_center = self.palm_center
        observation.smoothed_palm_center = self.smoothed_palm_center
        observation.index_tip = self.index_tip
        observation.thumb_tip = self.thumb_tip
        observation.hand_pose = self.hand_pose
        observation.pose_confidence = self.pose_confidence
        observation.holding = self.holding
        observation.holding_kind = self.holding_kind
        observation.holding_label = self.holding_label
        observation.holding_confidence = self.holding_confidence
        observation.holding_score = self.holding_score
        observation.spatial_stability = self.spatial_stability
        observation.object_comotion = self.object_comotion
        observation.missing_duration = self.missing_duration
        observation.debug["holding"] = dict(self.debug)
        observation.clamp()
        return observation


@dataclass
class _ObjectMemory:
    detection: Detection
    kind: HoldingKind
    last_seen_at: float


@dataclass
class _HandTracker:
    last_seen_at: Optional[float] = None
    smoothed_palm: Optional[Point2D] = None
    previous_palm: Optional[Point2D] = None
    previous_object_center: Optional[Point2D] = None
    score_ema: float = 0.0
    holding: bool = False
    associated_label: Optional[str] = None
    associated_kind: HoldingKind = HoldingKind.NONE
    associated_box: Optional[BoxXYXY] = None
    associated_object_center: Optional[Point2D] = None
    associated_last_seen_at: Optional[float] = None
    object_memory: Dict[str, _ObjectMemory] = field(default_factory=dict)
    relative_history: List[Point2D] = field(default_factory=list)


class HoldingAnalyzer:
    UTENSIL_LABELS = {
        "spoon", "fork", "knife", "chopsticks", "toothbrush", "utensil"
    }
    CUP_LABELS = {"cup", "mug", "glass", "wine glass"}
    BOTTLE_LABELS = {"bottle", "water bottle"}
    FOOD_LABELS = {
        "apple", "banana", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "food"
    }
    PHONE_LABELS = {"cell phone", "phone", "mobile phone"}

    def __init__(self, config: Optional[HoldingConfig] = None) -> None:
        self.config = config or HoldingConfig()
        self.config.validate()
        self._trackers: Dict[HandSide, _HandTracker] = {
            HandSide.LEFT: _HandTracker(),
            HandSide.RIGHT: _HandTracker(),
            HandSide.UNKNOWN: _HandTracker(),
        }

    def reset(self, side: Optional[HandSide] = None) -> None:
        if side is None:
            for key in list(self._trackers):
                self._trackers[key] = _HandTracker()
        else:
            self._trackers[side] = _HandTracker()

    def update(self, value: HoldingInput) -> HoldingResult:
        tracker = self._trackers.setdefault(value.side, _HandTracker())
        timestamp = float(value.timestamp)
        frame_width, frame_height = _normalize_frame_size(value.frame_size)
        landmarks = _normalize_landmarks(value.landmarks)
        detected = len(landmarks) >= 21

        if not detected:
            missing_duration = (
                0.0 if tracker.last_seen_at is None
                else max(0.0, timestamp - tracker.last_seen_at)
            )
            return self._handle_missing_hand(
                value.side, timestamp, tracker, missing_duration
            )

        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        palm_center = _mean_point(
            [landmarks[index] for index in (0, 5, 9, 13, 17)]
        )

        previous_smoothed = tracker.smoothed_palm
        smoothed_palm = _ema_point(
            tracker.smoothed_palm,
            palm_center,
            self.config.ema_alpha_position,
        )
        tracker.previous_palm = previous_smoothed
        tracker.smoothed_palm = smoothed_palm
        tracker.last_seen_at = timestamp

        hand_pose, pose_confidence, pose_debug = self.classify_hand_pose(
            landmarks
        )

        hand_box = _landmark_box(landmarks)
        expanded_hand_box = _expand_box(
            hand_box,
            self.config.hand_box_expand_ratio,
            frame_width,
            frame_height,
        )
        hand_scale = max(
            _box_diagonal(hand_box),
            hypot(frame_width, frame_height) * 0.025,
        )

        detections = self._update_object_memory(
            tracker, value.detections, timestamp
        )
        candidates = self._build_candidates(
            detections=detections,
            hand_center=smoothed_palm,
            expanded_hand_box=expanded_hand_box,
            hand_scale=hand_scale,
            landmarks=landmarks,
        )
        candidate = max(
            candidates,
            key=lambda item: item.base_association_score,
            default=None,
        )

        stability = 0.0
        comotion = 0.0
        associated_center = None
        associated_box = None
        associated_label = None
        associated_kind = HoldingKind.NONE
        remembered_object = False
        overlap_score = 0.0
        distance_score = 0.0
        object_confidence = 0.0

        if candidate is not None:
            associated_center = candidate.detection.center
            associated_box = candidate.detection.box
            associated_label = candidate.detection.label
            associated_kind = candidate.kind
            remembered_object = candidate.remembered
            overlap_score = candidate.overlap_score
            distance_score = candidate.distance_score
            object_confidence = candidate.detection.confidence

            relative_vector = (
                associated_center[0] - smoothed_palm[0],
                associated_center[1] - smoothed_palm[1],
            )
            stability = self._spatial_stability(
                tracker, relative_vector, hand_scale
            )
            comotion = self._object_comotion(
                tracker, smoothed_palm, associated_center
            )

            tracker.associated_label = associated_label
            tracker.associated_kind = associated_kind
            tracker.associated_box = associated_box
            tracker.associated_object_center = associated_center
            tracker.associated_last_seen_at = timestamp
            tracker.previous_object_center = associated_center
        else:
            (
                associated_label,
                associated_kind,
                associated_box,
                associated_center,
                remembered_object,
            ) = self._recover_previous_association(tracker, timestamp)

        pose_score = self._pose_holding_score(
            hand_pose, pose_confidence
        )
        raw_score = (
            self.config.overlap_weight * overlap_score
            + self.config.distance_weight * distance_score
            + self.config.hand_pose_weight * pose_score
            + self.config.stability_weight * stability
            + self.config.comotion_weight * comotion
        )

        if candidate is None and remembered_object:
            raw_score = max(raw_score, tracker.score_ema * 0.72)
        elif candidate is not None:
            confidence_factor = _clamp01(
                (object_confidence - self.config.minimum_object_confidence)
                / max(
                    1.0 - self.config.minimum_object_confidence,
                    _EPSILON,
                )
            )
            raw_score *= 0.72 + 0.28 * confidence_factor

            # 손가락 또는 손바닥이 객체 박스와 실제로 닿아 있으면
            # 객체 중심이 손에서 멀더라도 Holding 후보로 인정한다.
            strong_contact = overlap_score >= 0.18
            close_contact = distance_score >= 0.72

            if strong_contact:
                raw_score = max(
                    raw_score,
                    0.54 + 0.10 * pose_score,
                )
            elif close_contact and hand_pose != HandPose.OPEN:
                raw_score = max(
                    raw_score,
                    0.49 + 0.10 * pose_score,
                )

            if (
                associated_kind
                in {
                    HoldingKind.CUP,
                    HoldingKind.BOTTLE,
                    HoldingKind.UTENSIL,
                    HoldingKind.FOOD,
                }
                and hand_pose in {HandPose.PINCH, HandPose.GRIP}
                and (strong_contact or close_contact)
            ):
                raw_score += 0.06

        if associated_kind == HoldingKind.PHONE:
            raw_score *= 0.90
        elif associated_kind == HoldingKind.OTHER:
            raw_score *= 0.82

        raw_score = _clamp01(raw_score)

        # 첫 프레임부터 0에서 EMA를 시작하면 실제로 물체를 잡아도
        # Holding ON까지 지나치게 오래 걸린다. 새 객체 후보가 분명하면
        # 해당 점수로 EMA를 시작하고 이후 프레임부터 평활화한다.
        if candidate is not None and tracker.score_ema <= _EPSILON:
            tracker.score_ema = raw_score
        else:
            tracker.score_ema = _ema_scalar(
                tracker.score_ema,
                raw_score,
                self.config.ema_alpha_score,
            )

        if tracker.holding:
            tracker.holding = (
                tracker.score_ema >= self.config.holding_off_threshold
            )
        else:
            tracker.holding = (
                tracker.score_ema >= self.config.holding_on_threshold
            )

        result = HoldingResult(
            side=value.side,
            timestamp=timestamp,
            detected=True,
            hand_pose=hand_pose,
            pose_confidence=pose_confidence,
            holding=tracker.holding,
            holding_kind=(
                associated_kind if tracker.holding else HoldingKind.NONE
            ),
            holding_label=associated_label if tracker.holding else None,
            holding_confidence=tracker.score_ema,
            holding_score=tracker.score_ema,
            palm_center=palm_center,
            smoothed_palm_center=smoothed_palm,
            wrist=wrist,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            associated_box=associated_box,
            associated_object_center=associated_center,
            spatial_stability=stability,
            object_comotion=comotion,
            remembered_object=remembered_object,
            missing_duration=0.0,
            debug={
                "raw_score": raw_score,
                "score_ema": tracker.score_ema,
                "overlap_score": overlap_score,
                "distance_score": distance_score,
                "pose_score": pose_score,
                "object_confidence": object_confidence,
                "candidate_count": len(candidates),
                "hand_box": hand_box,
                "expanded_hand_box": expanded_hand_box,
                "pose": pose_debug,
            },
        )
        return result

    def classify_hand_pose(
        self,
        landmarks: Sequence[Point2D],
    ) -> Tuple[HandPose, float, Dict[str, float]]:
        points = _normalize_landmarks(landmarks)
        if len(points) < 21:
            return HandPose.UNKNOWN, 0.0, {"reason": 0.0}

        wrist = points[0]
        palm_scale = max(_distance(wrist, points[9]), _EPSILON)
        pinch_ratio = _distance(points[4], points[8]) / palm_scale

        finger_ratios = []
        for tip_index, pip_index in ((8, 6), (12, 10), (16, 14), (20, 18)):
            finger_ratios.append(
                _distance(wrist, points[tip_index])
                / max(_distance(wrist, points[pip_index]), _EPSILON)
            )

        extended_count = sum(
            ratio >= self.config.open_finger_ratio_threshold
            for ratio in finger_ratios
        )
        curled_count = sum(
            ratio <= self.config.grip_finger_ratio_threshold
            for ratio in finger_ratios
        )

        pinch_strength = _clamp01(
            1.0
            - pinch_ratio
            / max(self.config.pinch_ratio_threshold, _EPSILON)
        )
        grip_strength = _clamp01(curled_count / 4.0)
        open_strength = _clamp01(extended_count / 4.0)

        if pinch_ratio <= self.config.pinch_ratio_threshold:
            pose = HandPose.PINCH
            confidence = max(pinch_strength, 0.55)
        elif curled_count >= 3:
            pose = HandPose.GRIP
            confidence = max(grip_strength, 0.55)
        elif extended_count >= 3:
            pose = HandPose.OPEN
            confidence = max(open_strength, 0.55)
        else:
            pose = HandPose.UNKNOWN
            confidence = max(
                pinch_strength, grip_strength, open_strength
            ) * 0.55

        return pose, _clamp01(confidence), {
            "pinch_ratio": pinch_ratio,
            "pinch_strength": pinch_strength,
            "grip_strength": grip_strength,
            "open_strength": open_strength,
            "extended_count": float(extended_count),
            "curled_count": float(curled_count),
        }

    def _handle_missing_hand(
        self,
        side: HandSide,
        timestamp: float,
        tracker: _HandTracker,
        missing_duration: float,
    ) -> HoldingResult:
        if missing_duration > self.config.hand_memory_seconds:
            tracker.holding = False
            tracker.score_ema = 0.0

        remembered = (
            tracker.associated_last_seen_at is not None
            and timestamp - tracker.associated_last_seen_at
            <= self.config.association_memory_seconds
        )

        return HoldingResult(
            side=side,
            timestamp=timestamp,
            detected=False,
            hand_pose=HandPose.UNKNOWN,
            pose_confidence=0.0,
            holding=tracker.holding if remembered else False,
            holding_kind=(
                tracker.associated_kind
                if tracker.holding and remembered
                else HoldingKind.NONE
            ),
            holding_label=(
                tracker.associated_label
                if tracker.holding and remembered
                else None
            ),
            holding_confidence=tracker.score_ema if remembered else 0.0,
            holding_score=tracker.score_ema if remembered else 0.0,
            palm_center=None,
            smoothed_palm_center=tracker.smoothed_palm,
            wrist=None,
            index_tip=None,
            thumb_tip=None,
            associated_box=tracker.associated_box if remembered else None,
            associated_object_center=(
                tracker.associated_object_center if remembered else None
            ),
            spatial_stability=0.0,
            object_comotion=0.0,
            remembered_object=remembered,
            missing_duration=missing_duration,
            debug={"reason": "hand_missing"},
        )

    def _update_object_memory(
        self,
        tracker: _HandTracker,
        detections: Sequence[Detection],
        timestamp: float,
    ) -> List[Tuple[Detection, bool]]:
        valid: List[Tuple[Detection, bool]] = []
        seen = set()

        for value in detections:
            detection = _coerce_detection(value)
            if detection is None:
                continue
            if detection.confidence < self.config.minimum_object_confidence:
                continue

            key = _memory_key(detection)
            seen.add(key)
            tracker.object_memory[key] = _ObjectMemory(
                detection=detection,
                kind=self.classify_holding_kind(detection.label),
                last_seen_at=timestamp,
            )
            valid.append((detection, False))

        expired = []
        for key, memory in tracker.object_memory.items():
            age = timestamp - memory.last_seen_at
            if age > self.config.object_memory_seconds:
                expired.append(key)
            elif key not in seen:
                valid.append((memory.detection, True))

        for key in expired:
            del tracker.object_memory[key]

        return valid

    def _build_candidates(
        self,
        detections: Sequence[Tuple[Detection, bool]],
        hand_center: Point2D,
        expanded_hand_box: BoxXYXY,
        hand_scale: float,
        landmarks: Sequence[Point2D],
    ) -> List[HoldingCandidate]:
        """
        손과 객체의 연관 후보를 만든다.

        기존에는 손바닥 중심과 객체 바운딩박스 중심 사이의 거리를
        사용했다. 컵 손잡이, 물병 몸통, 숟가락 끝처럼 사용자가 객체의
        중심이 아닌 부분을 잡으면 이 방식은 실제 Holding을 누락한다.

        수정된 방식은 다음 신호를 함께 사용한다.
        - 손바닥에서 객체 박스까지의 최단거리
        - 주요 손가락 랜드마크에서 객체 박스까지의 최단거리
        - 확장 손 박스와 객체 박스의 겹침
        - 손가락 랜드마크가 객체 박스 내부에 들어간 비율
        """
        candidates: List[HoldingCandidate] = []

        normalized_landmarks = _normalize_landmarks(landmarks)
        key_indices = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
        key_points = [
            normalized_landmarks[index]
            for index in key_indices
            if index < len(normalized_landmarks)
        ]
        if not key_points:
            key_points = [hand_center]

        for detection, remembered in detections:
            kind = self.classify_holding_kind(detection.label)
            if kind == HoldingKind.NONE:
                continue

            object_box = detection.box
            expanded_object_box = _expand_box_by_pixels(
                object_box,
                max(3.0, hand_scale * 0.10),
            )

            palm_box_distance = _point_to_box_distance(
                hand_center,
                expanded_object_box,
            )
            landmark_box_distance = min(
                (
                    _point_to_box_distance(
                        point,
                        expanded_object_box,
                    )
                    for point in key_points
                ),
                default=palm_box_distance,
            )
            association_distance = min(
                palm_box_distance,
                landmark_box_distance,
            )
            normalized_distance = association_distance / max(
                hand_scale,
                _EPSILON,
            )

            box_overlap = _intersection_over_smaller_area(
                expanded_hand_box,
                object_box,
            )
            landmark_contact = _landmark_box_contact_score(
                key_points,
                expanded_object_box,
            )
            overlap_score = max(box_overlap, landmark_contact)

            # 큰 컵·물병 또는 길쭉한 수저는 객체 중심이 손과 멀 수 있으므로
            # 종류에 따라 허용 거리를 조금 다르게 둔다.
            kind_distance_multiplier = {
                HoldingKind.CUP: 1.55,
                HoldingKind.BOTTLE: 1.75,
                HoldingKind.UTENSIL: 1.65,
                HoldingKind.FOOD: 1.35,
                HoldingKind.PHONE: 1.30,
                HoldingKind.OTHER: 1.20,
            }.get(kind, 1.20)

            maximum_distance_ratio = (
                self.config.max_center_distance_ratio
                * kind_distance_multiplier
            )
            distance_score = _clamp01(
                1.0
                - normalized_distance
                / max(maximum_distance_ratio, _EPSILON)
            )

            intersects = _boxes_intersect(
                expanded_hand_box,
                object_box,
            )
            has_landmark_contact = landmark_contact > 0.0

            if (
                not intersects
                and not has_landmark_contact
                and normalized_distance > maximum_distance_ratio
            ):
                continue

            candidates.append(
                HoldingCandidate(
                    detection=detection,
                    kind=kind,
                    center_distance=association_distance,
                    normalized_distance=normalized_distance,
                    overlap_score=overlap_score,
                    distance_score=distance_score,
                    remembered=remembered,
                )
            )

        return candidates

    def _spatial_stability(
        self,
        tracker: _HandTracker,
        relative_vector: Point2D,
        hand_scale: float,
    ) -> float:
        tracker.relative_history.append(relative_vector)
        if len(tracker.relative_history) > self.config.maximum_history:
            tracker.relative_history.pop(0)

        if len(tracker.relative_history) < 2:
            return 0.5

        latest = tracker.relative_history[-1]
        changes = [
            _distance(latest, previous) / max(hand_scale, _EPSILON)
            for previous in tracker.relative_history[:-1]
        ]
        mean_change = sum(changes) / max(len(changes), 1)
        return _clamp01(
            1.0
            - mean_change
            / max(self.config.stable_distance_ratio, _EPSILON)
        )

    def _object_comotion(
        self,
        tracker: _HandTracker,
        hand_center: Point2D,
        object_center: Point2D,
    ) -> float:
        if (
            tracker.previous_palm is None
            or tracker.previous_object_center is None
        ):
            return 0.5

        hand_motion = (
            hand_center[0] - tracker.previous_palm[0],
            hand_center[1] - tracker.previous_palm[1],
        )
        object_motion = (
            object_center[0] - tracker.previous_object_center[0],
            object_center[1] - tracker.previous_object_center[1],
        )

        hand_speed = hypot(*hand_motion)
        object_speed = hypot(*object_motion)

        if (
            hand_speed < self.config.minimum_motion_for_comotion
            and object_speed < self.config.minimum_motion_for_comotion
        ):
            return 0.55

        angle = _angle_between(hand_motion, object_motion)
        direction_score = (
            _clamp01(1.0 - angle / 90.0)
            if angle is not None else 0.0
        )
        magnitude_score = _clamp01(
            1.0
            - abs(hand_speed - object_speed)
            / max(hand_speed, object_speed, _EPSILON)
        )
        return _clamp01(
            0.65 * direction_score + 0.35 * magnitude_score
        )

    def _recover_previous_association(
        self,
        tracker: _HandTracker,
        timestamp: float,
    ):
        if tracker.associated_last_seen_at is None:
            return None, HoldingKind.NONE, None, None, False

        age = timestamp - tracker.associated_last_seen_at
        if age > self.config.association_memory_seconds:
            return None, HoldingKind.NONE, None, None, False

        return (
            tracker.associated_label,
            tracker.associated_kind,
            tracker.associated_box,
            tracker.associated_object_center,
            True,
        )

    @classmethod
    def classify_holding_kind(cls, label: str) -> HoldingKind:
        normalized = str(label).strip().lower()
        if normalized in cls.UTENSIL_LABELS:
            return HoldingKind.UTENSIL
        if normalized in cls.CUP_LABELS:
            return HoldingKind.CUP
        if normalized in cls.BOTTLE_LABELS:
            return HoldingKind.BOTTLE
        if normalized in cls.FOOD_LABELS:
            return HoldingKind.FOOD
        if normalized in cls.PHONE_LABELS:
            return HoldingKind.PHONE
        if normalized in {"person", "chair", "dining table", "table"}:
            return HoldingKind.NONE
        if not normalized:
            return HoldingKind.UNKNOWN
        return HoldingKind.OTHER

    @staticmethod
    def _pose_holding_score(
        hand_pose: HandPose,
        confidence: float,
    ) -> float:
        confidence = _clamp01(confidence)
        if hand_pose == HandPose.PINCH:
            return 0.95 * confidence
        if hand_pose == HandPose.GRIP:
            return confidence
        if hand_pose == HandPose.OPEN:
            return 0.12 * confidence
        return 0.30 * confidence


def _coerce_detection(value: Any) -> Optional[Detection]:
    if isinstance(value, Detection):
        return value

    if isinstance(value, Mapping):
        label = value.get("label", value.get("class_name", value.get("name")))
        confidence = value.get(
            "confidence", value.get("conf", value.get("score", 0.0))
        )
        box = value.get("box", value.get("bbox", value.get("xyxy")))
        track_id = value.get("track_id", value.get("id"))
    else:
        label = getattr(
            value, "label",
            getattr(value, "class_name", getattr(value, "name", None))
        )
        confidence = getattr(
            value, "confidence",
            getattr(value, "conf", getattr(value, "score", 0.0))
        )
        box = getattr(
            value, "box",
            getattr(value, "bbox", getattr(value, "xyxy", None))
        )
        track_id = getattr(value, "track_id", getattr(value, "id", None))

    if label is None or box is None:
        return None

    try:
        x1, y1, x2, y2 = box
        return Detection(
            label=str(label),
            confidence=float(confidence),
            box=(float(x1), float(y1), float(x2), float(y2)),
            track_id=int(track_id) if track_id is not None else None,
        )
    except (TypeError, ValueError):
        return None


def _normalize_landmarks(landmarks: Sequence[Any]) -> Tuple[Point2D, ...]:
    points = []

    for value in landmarks:
        if isinstance(value, Mapping):
            x, y = value.get("x"), value.get("y")
        elif hasattr(value, "x") and hasattr(value, "y"):
            x, y = getattr(value, "x"), getattr(value, "y")
        else:
            try:
                x, y = value[:2]
            except (TypeError, ValueError, IndexError):
                return tuple()

        try:
            points.append((float(x), float(y)))
        except (TypeError, ValueError):
            return tuple()

    return tuple(points)


def _normalize_frame_size(frame_size: Tuple[int, int]) -> Tuple[int, int]:
    try:
        width, height = frame_size
        return max(int(width), 1), max(int(height), 1)
    except (TypeError, ValueError):
        return 1, 1


def _memory_key(detection: Detection) -> str:
    x1, y1, x2, y2 = detection.box
    return (
        f"{detection.label.lower()}_"
        f"{round(x1 / 16)}_{round(y1 / 16)}_"
        f"{round(x2 / 16)}_{round(y2 / 16)}"
    )


def _mean_point(points: Sequence[Point2D]) -> Point2D:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _ema_point(
    previous: Optional[Point2D],
    current: Point2D,
    alpha: float,
) -> Point2D:
    if previous is None:
        return current
    return (
        alpha * current[0] + (1.0 - alpha) * previous[0],
        alpha * current[1] + (1.0 - alpha) * previous[1],
    )


def _ema_scalar(previous: float, current: float, alpha: float) -> float:
    return alpha * current + (1.0 - alpha) * previous


def _distance(first: Point2D, second: Point2D) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def _angle_between(
    first: Point2D,
    second: Point2D,
) -> Optional[float]:
    first_norm = hypot(*first)
    second_norm = hypot(*second)
    if first_norm <= _EPSILON or second_norm <= _EPSILON:
        return None

    cosine = (
        first[0] * second[0] + first[1] * second[1]
    ) / (first_norm * second_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return acos(cosine) * 180.0 / 3.141592653589793


def _landmark_box(landmarks: Sequence[Point2D]) -> BoxXYXY:
    xs = [point[0] for point in landmarks]
    ys = [point[1] for point in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def _expand_box(
    box: BoxXYXY,
    ratio: float,
    frame_width: int,
    frame_height: int,
) -> BoxXYXY:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return (
        max(0.0, x1 - width * ratio),
        max(0.0, y1 - height * ratio),
        min(float(frame_width), x2 + width * ratio),
        min(float(frame_height), y2 + height * ratio),
    )



def _expand_box_by_pixels(
    box: BoxXYXY,
    padding: float,
) -> BoxXYXY:
    """프레임 경계를 알 수 없는 상황에서 픽셀 단위로 박스를 확장한다."""
    x1, y1, x2, y2 = box
    padding = max(0.0, float(padding))
    return (
        x1 - padding,
        y1 - padding,
        x2 + padding,
        y2 + padding,
    )


def _point_to_box_distance(
    point: Point2D,
    box: BoxXYXY,
) -> float:
    """점이 박스 안에 있으면 0, 밖에 있으면 박스까지의 최단거리."""
    x, y = point
    x1, y1, x2, y2 = box

    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return hypot(dx, dy)


def _point_in_box(
    point: Point2D,
    box: BoxXYXY,
) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _landmark_box_contact_score(
    points: Sequence[Point2D],
    box: BoxXYXY,
) -> float:
    """
    주요 손 랜드마크가 객체 박스 안에 들어간 정도를 반환한다.

    한 점만 닿아도 약한 접촉 신호를 주고, 여러 점이 들어가면
    빠르게 높은 점수가 되도록 구성한다.
    """
    if not points:
        return 0.0

    contact_count = sum(
        1 for point in points if _point_in_box(point, box)
    )
    if contact_count <= 0:
        return 0.0

    ratio = contact_count / len(points)
    return _clamp01(0.35 + 1.65 * ratio)



def _box_diagonal(box: BoxXYXY) -> float:
    x1, y1, x2, y2 = box
    return hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))


def _boxes_intersect(first: BoxXYXY, second: BoxXYXY) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return not (
        ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1
    )


def _intersection_over_smaller_area(
    first: BoxXYXY,
    second: BoxXYXY,
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height

    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return _clamp01(
        intersection / max(min(first_area, second_area), _EPSILON)
    )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "HoldingAnalyzer",
    "HoldingCandidate",
    "HoldingConfig",
    "HoldingInput",
    "HoldingResult",
]