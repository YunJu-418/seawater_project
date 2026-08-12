"""
식사 행동 인식 기능 V3 최종 실행 파일.

파이프라인
----------
카메라/영상
→ 전체 프레임 YOLO
→ 사람 ROI 계산 및 확대
→ ROI YOLO 재검출
→ ROI 손 검출
→ ROI 얼굴/입 검출
→ MealVisionProcessor V3
→ MealFSM V3
→ MealLogger V3
→ OpenCV Debug Overlay

실행
----
python main_meal.py
python main_meal.py --source sample.mp4
python main_meal.py --camera 0
python main_meal.py --no-display
python main_meal.py --config meal/meal_config.json

키 입력
-------
q / ESC : 종료
r       : FSM, 비전, Logger 세션 초기화
s       : 현재 화면 저장
d       : Debug Overlay 켜기/끄기
p       : 일시정지/재개
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2

from meal.meal_fsm import FSMUpdateResult, MealFSM
from utils.logger import MealLogger
from meal.meal_state import (
    BoxXYXY,
    Detection,
    HandSide,
    MealObservation,
    MealSessionSnapshot,
)
from vision.meal import (
    MealVisionInput,
    MealVisionProcessor,
    RawHandInput,
    RawMouthInput,
)


Point2D = Tuple[float, float]


@dataclass
class RuntimeOptions:
    source: str
    camera_index: int
    config_path: Path
    model_path: Optional[str]
    display: bool
    debug_overlay: bool
    save_output: bool
    output_path: Optional[Path]
    maximum_frames: Optional[int]
    target_fps: Optional[float]


@dataclass
class RuntimeComponents:
    detector: Any
    hand_detector: Any
    face_detector: Any


class ComponentLoadError(RuntimeError):
    """기존 비전 모듈을 불러오지 못한 경우의 예외."""


class MealApplication:
    """식사 기능 V3 실행 애플리케이션."""

    VESSEL_LABELS = {
        "cup",
        "mug",
        "glass",
        "wine glass",
        "bottle",
        "water bottle",
    }

    def __init__(self, options: RuntimeOptions) -> None:
        self.options = options
        self.config = self._load_config(
            options.config_path
        )

        self.components = RuntimeComponents(
            detector=self._create_detector(
                options.model_path
            ),
            hand_detector=self._create_hand_detector(),
            face_detector=self._create_face_detector(),
        )

        self.vision_processor = MealVisionProcessor(
            options.config_path
        )
        self.meal_fsm = MealFSM(options.config_path)
        self.logger = self._create_logger()

        self.capture: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None

        self.frame_index = 0
        self.debug_overlay = options.debug_overlay
        self.test_mode = True
        self.paused = False
        self._event_banner_text = ""
        self._event_banner_until = 0.0
        self._previous_overlay_bite_count = 0
        self._previous_overlay_drink_count = 0

        self.last_observation: Optional[
            MealObservation
        ] = None
        self.last_result: Optional[
            FSMUpdateResult
        ] = None
        self.last_output_frame: Optional[Any] = None

        self.previous_frame_time: Optional[float] = None
        self.current_fps = 0.0
        self.output_directory = Path(
            "logs/meal/screenshots"
        )
        self.window_name = "Meal Recognition V3"
        self.is_video_file = bool(options.source)
        self.video_duration_ms = 0.0
        self._trackbar_updating = False
        self._last_video_timestamp = 0.0

        roi_config = self.config.get(
            "runtime_roi",
            {},
        )
        self.person_roi_scale = max(
            1.0,
            float(
                roi_config.get(
                    "person_roi_scale",
                    2.2,
                )
            ),
        )
        self.person_roi_expand_ratio = max(
            0.0,
            float(
                roi_config.get(
                    "person_roi_expand_ratio",
                    0.18,
                )
            ),
        )
        self.person_memory_seconds = max(
            0.0,
            float(
                roi_config.get(
                    "person_memory_seconds",
                    1.0,
                )
            ),
        )
        self.face_roi_scale = max(
            1.0,
            float(
                roi_config.get(
                    "face_roi_scale",
                    3.4,
                )
            ),
        )
        self.face_roi_height_ratio = min(
            0.85,
            max(
                0.30,
                float(
                    roi_config.get(
                        "face_roi_height_ratio",
                        0.58,
                    )
                ),
            ),
        )
        self.face_roi_side_margin_ratio = min(
            0.30,
            max(
                0.0,
                float(
                    roi_config.get(
                        "face_roi_side_margin_ratio",
                        0.08,
                    )
                ),
            ),
        )
        self.face_person_fallback_scale = max(
            1.0,
            float(
                roi_config.get(
                    "face_person_fallback_scale",
                    3.0,
                )
            ),
        )

        self._last_person_roi: Optional[
            Tuple[int, int, int, int]
        ] = None
        self._last_person_seen_at: Optional[float] = None

    def run(self) -> int:
        """메인 루프를 실행한다."""

        try:
            self.capture = self._open_capture()
            self._prepare_writer()
            self._prepare_video_controls()

            while True:
                if self.paused:
                    key = self._show_paused_frame()
                    if self._handle_key(key):
                        break
                    continue

                ok, frame = self.capture.read()
                if not ok or frame is None:
                    print("[Meal] 영상 입력이 종료되었습니다.")
                    break

                runtime_timestamp = time.monotonic()
                timestamp = self._resolve_frame_timestamp(
                    runtime_timestamp
                )
                self._update_fps(runtime_timestamp)

                observation = self._process_frame(
                    frame=frame,
                    timestamp=timestamp,
                )
                result = self.meal_fsm.update(observation)

                self.logger.log_update(
                    observation=observation,
                    events=result.events,
                    snapshot=result.snapshot,
                )

                self.last_observation = observation
                self.last_result = result

                output_frame = frame.copy()
                if self.debug_overlay:
                    self._update_event_banner(
                        snapshot=result.snapshot,
                        timestamp=timestamp,
                    )
                    if self.test_mode:
                        self._draw_test_overlay(
                            frame=output_frame,
                            observation=observation,
                            snapshot=result.snapshot,
                            timestamp=timestamp,
                        )
                    else:
                        self._draw_debug_overlay(
                            frame=output_frame,
                            observation=observation,
                            snapshot=result.snapshot,
                        )

                self.last_output_frame = output_frame.copy()

                if self.writer is not None:
                    self.writer.write(output_frame)

                if self.options.display:
                    self._update_video_trackbar()
                    cv2.imshow(
                        self.window_name,
                        output_frame,
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if self._handle_key(key):
                        break

                self.frame_index += 1

                if (
                    self.options.maximum_frames
                    is not None
                    and self.frame_index
                    >= self.options.maximum_frames
                ):
                    break

                if result.snapshot.finished:
                    print("[Meal] 식사 세션이 종료되었습니다.")
                    break

                self._limit_fps()

            return 0

        except KeyboardInterrupt:
            print(
                "\n[Meal] 사용자에 의해 중단되었습니다."
            )
            return 0

        except Exception as error:
            print(
                "[Meal] 실행 중 오류가 발생했습니다: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            return 1

        finally:
            self.close()

    def close(self) -> None:
        """자원을 안전하게 닫는다."""

        if self.last_result is not None:
            try:
                self.logger.finalize(
                    snapshot=self.last_result.snapshot,
                    extra_summary={
                        "terminated_by_runtime": (
                            not self.last_result
                            .snapshot.finished
                        ),
                        "processed_frames": (
                            self.frame_index
                        ),
                        "runtime_fps": self.current_fps,
                    },
                )
            except Exception as error:
                print(
                    f"[Meal] 최종 요약 저장 실패: {error}",
                    file=sys.stderr,
                )

        self.logger.close()

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self._close_component(
            self.components.detector
        )
        self._close_component(
            self.components.hand_detector
        )
        self._close_component(
            self.components.face_detector
        )

        if self.options.display:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Frame pipeline
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame: Any,
        timestamp: float,
    ) -> MealObservation:
        height, width = frame.shape[:2]
        frame_size = (width, height)

        raw_full_detections = self._run_detector(frame)
        full_detections = self._adapt_detections(
            raw_full_detections
        )

        person_roi = self._resolve_person_roi(
            detections=full_detections,
            frame_size=frame_size,
            timestamp=timestamp,
        )

        crop, crop_scale = self._crop_and_resize(
            frame=frame,
            roi=person_roi,
            desired_scale=self.person_roi_scale,
        )

        raw_roi_detections = self._run_detector(crop)
        roi_detections = self._adapt_detections(
            raw_roi_detections
        )
        roi_detections = (
            self._restore_detections_to_full_frame(
                detections=roi_detections,
                roi=person_roi,
                scale=crop_scale,
            )
        )

        detections = self._merge_detections(
            full_detections,
            roi_detections,
        )

        raw_hands = self._run_hand_detector(crop)
        hands_in_crop = self._adapt_hands(
            raw=raw_hands,
            frame_size=(
                crop.shape[1],
                crop.shape[0],
            ),
        )
        hands = self._restore_hands_to_full_frame(
            hands=hands_in_crop,
            roi=person_roi,
            scale=crop_scale,
        )

        (
            mouth,
            face_roi,
            face_detection_source,
        ) = self._detect_mouth_with_fallback(
            frame=frame,
            person_roi=person_roi,
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )
        brightness = float(gray.mean())
        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        value = MealVisionInput(
            timestamp=timestamp,
            frame_index=self.frame_index,
            frame_size=frame_size,
            hands=hands,
            mouth=mouth,
            detections=detections,
            brightness=brightness,
            blur_score=blur_score,
            frame_valid=True,
            object_detection_valid=True,
        )
        observation = (
            self.vision_processor.create_observation(
                value
            )
        )
        observation.debug["person_roi"] = person_roi
        observation.debug["person_roi_scale"] = (
            crop_scale
        )
        observation.debug["face_roi"] = face_roi
        observation.debug["face_detection_source"] = (
            face_detection_source
        )
        return observation

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------

    def _resolve_person_roi(
        self,
        detections: Sequence[Detection],
        frame_size: Tuple[int, int],
        timestamp: float,
    ) -> Tuple[int, int, int, int]:
        people = [
            item
            for item in detections
            if item.label.strip().lower() == "person"
        ]

        if people:
            person = max(
                people,
                key=lambda item: (
                    item.confidence,
                    item.area,
                ),
            )
            roi = self._expand_and_clip_roi(
                person.box,
                frame_size=frame_size,
                expand_ratio=(
                    self.person_roi_expand_ratio
                ),
            )
            self._last_person_roi = roi
            self._last_person_seen_at = timestamp
            return roi

        if (
            self._last_person_roi is not None
            and self._last_person_seen_at is not None
            and timestamp - self._last_person_seen_at
            <= self.person_memory_seconds
        ):
            return self._last_person_roi

        width, height = frame_size
        return (0, 0, width, height)

    def _detect_mouth_with_fallback(
        self,
        frame: Any,
        person_roi: Tuple[int, int, int, int],
    ) -> Tuple[
        RawMouthInput,
        Tuple[int, int, int, int],
        str,
    ]:
        """
        얼굴 전용 ROI에서 FaceMesh를 우선 실행한다.

        1차: Person ROI 상단의 얼굴 전용 ROI를 크게 확대
        2차: 전체 Person ROI를 얼굴용 배율로 재확대
        3차: 전체 프레임에서 마지막 재시도
        """
        height, width = frame.shape[:2]
        frame_size = (width, height)
        face_roi = self._make_face_roi(
            person_roi=person_roi,
            frame_size=frame_size,
        )

        attempts = (
            (
                face_roi,
                self.face_roi_scale,
                "face_roi",
            ),
            (
                person_roi,
                self.face_person_fallback_scale,
                "person_roi_fallback",
            ),
            (
                (0, 0, width, height),
                1.0,
                "full_frame_fallback",
            ),
        )

        last_mouth = RawMouthInput()
        last_roi = face_roi
        last_source = "not_detected"

        for roi, scale, source_name in attempts:
            crop, applied_scale = self._crop_and_resize(
                frame=frame,
                roi=roi,
                desired_scale=scale,
            )
            raw_face = self._run_face_detector(crop)
            mouth_in_crop = self._adapt_mouth(
                raw=raw_face,
                frame_size=(
                    crop.shape[1],
                    crop.shape[0],
                ),
            )
            mouth = self._restore_mouth_to_full_frame(
                mouth=mouth_in_crop,
                roi=roi,
                scale=applied_scale,
            )

            last_mouth = mouth
            last_roi = roi
            last_source = source_name

            if mouth.detected and mouth.center is not None:
                return mouth, roi, source_name

        return last_mouth, last_roi, last_source

    def _make_face_roi(
        self,
        person_roi: Tuple[int, int, int, int],
        frame_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Person ROI의 상단을 얼굴 후보 영역으로 만든다.

        몸과 식탁을 함께 포함한 Person ROI 전체보다 FaceMesh에
        얼굴이 크게 들어오도록 상단 영역만 잘라 사용한다.
        """
        frame_width, frame_height = frame_size
        x1, y1, x2, y2 = person_roi

        person_width = max(1, x2 - x1)
        person_height = max(1, y2 - y1)

        side_margin = int(
            round(
                person_width
                * self.face_roi_side_margin_ratio
            )
        )
        face_bottom = y1 + int(
            round(
                person_height
                * self.face_roi_height_ratio
            )
        )

        return (
            max(0, x1 + side_margin),
            max(0, y1),
            min(frame_width, x2 - side_margin),
            min(frame_height, max(y1 + 1, face_bottom)),
        )

    @staticmethod
    def _expand_and_clip_roi(
        box: BoxXYXY,
        frame_size: Tuple[int, int],
        expand_ratio: float,
    ) -> Tuple[int, int, int, int]:
        width, height = frame_size
        x1, y1, x2, y2 = [
            float(value)
            for value in box
        ]

        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)

        # 식탁 위 용기까지 포함하도록 아래쪽을 더 넓힌다.
        left = x1 - box_width * expand_ratio
        right = x2 + box_width * expand_ratio
        top = y1 - box_height * expand_ratio * 0.55
        bottom = y2 + box_height * (
            expand_ratio + 0.30
        )

        return (
            max(0, int(round(left))),
            max(0, int(round(top))),
            min(width, int(round(right))),
            min(height, int(round(bottom))),
        )

    @staticmethod
    def _crop_and_resize(
        frame: Any,
        roi: Tuple[int, int, int, int],
        desired_scale: float,
    ) -> Tuple[Any, float]:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return frame, 1.0

        applied_scale = max(
            1.0,
            float(desired_scale),
        )
        if applied_scale <= 1.01:
            return crop, 1.0

        resized = cv2.resize(
            crop,
            None,
            fx=applied_scale,
            fy=applied_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        return resized, applied_scale

    @staticmethod
    def _restore_detections_to_full_frame(
        detections: Sequence[Detection],
        roi: Tuple[int, int, int, int],
        scale: float,
    ) -> List[Detection]:
        x_offset, y_offset, _, _ = roi
        scale = max(float(scale), 1e-9)

        result: List[Detection] = []
        for item in detections:
            x1, y1, x2, y2 = item.box
            result.append(
                Detection(
                    label=item.label,
                    confidence=item.confidence,
                    box=(
                        x1 / scale + x_offset,
                        y1 / scale + y_offset,
                        x2 / scale + x_offset,
                        y2 / scale + y_offset,
                    ),
                    track_id=item.track_id,
                )
            )
        return result

    @staticmethod
    def _restore_hands_to_full_frame(
        hands: Sequence[RawHandInput],
        roi: Tuple[int, int, int, int],
        scale: float,
    ) -> List[RawHandInput]:
        x_offset, y_offset, _, _ = roi
        scale = max(float(scale), 1e-9)

        result: List[RawHandInput] = []
        for hand in hands:
            landmarks = [
                (
                    point[0] / scale + x_offset,
                    point[1] / scale + y_offset,
                )
                for point in hand.landmarks
            ]
            result.append(
                RawHandInput(
                    side=hand.side,
                    landmarks=landmarks,
                    tracking_id=hand.tracking_id,
                    confidence=hand.confidence,
                )
            )
        return result

    @staticmethod
    def _restore_mouth_to_full_frame(
        mouth: RawMouthInput,
        roi: Tuple[int, int, int, int],
        scale: float,
    ) -> RawMouthInput:
        x_offset, y_offset, _, _ = roi
        scale = max(float(scale), 1e-9)

        def restore_point(
            point: Optional[Point2D],
        ) -> Optional[Point2D]:
            if point is None:
                return None
            return (
                point[0] / scale + x_offset,
                point[1] / scale + y_offset,
            )

        def restore_box(
            box: Optional[BoxXYXY],
        ) -> Optional[BoxXYXY]:
            if box is None:
                return None
            x1, y1, x2, y2 = box
            return (
                x1 / scale + x_offset,
                y1 / scale + y_offset,
                x2 / scale + x_offset,
                y2 / scale + y_offset,
            )

        return RawMouthInput(
            detected=mouth.detected,
            center=restore_point(mouth.center),
            face_box=restore_box(mouth.face_box),
            mouth_landmarks=[
                restore_point(point)
                for point in mouth.mouth_landmarks
                if restore_point(point) is not None
            ],
            openness=mouth.openness,
        )

    @classmethod
    def _merge_detections(
        cls,
        first: Sequence[Detection],
        second: Sequence[Detection],
    ) -> List[Detection]:
        merged: List[Detection] = []

        for candidate in list(first) + list(second):
            replaced = False

            for index, current in enumerate(merged):
                if (
                    current.label.strip().lower()
                    != candidate.label.strip().lower()
                ):
                    continue

                if cls._box_iou(
                    current.box,
                    candidate.box,
                ) < 0.55:
                    continue

                if (
                    candidate.confidence
                    > current.confidence
                ):
                    merged[index] = candidate

                replaced = True
                break

            if not replaced:
                merged.append(candidate)

        return merged

    @staticmethod
    def _box_iou(
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

        first_area = max(
            0.0,
            ax2 - ax1,
        ) * max(
            0.0,
            ay2 - ay1,
        )
        second_area = max(
            0.0,
            bx2 - bx1,
        ) * max(
            0.0,
            by2 - by1,
        )
        union = (
            first_area
            + second_area
            - intersection
        )
        if union <= 0.0:
            return 0.0
        return intersection / union

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------

    def _update_event_banner(
        self,
        snapshot: MealSessionSnapshot,
        timestamp: float,
    ) -> None:
        """카운트 증가 순간 큰 배너를 약 1.2초간 표시한다."""
        if (
            snapshot.bite_count
            > self._previous_overlay_bite_count
        ):
            self._event_banner_text = (
                f"BITE +1   TOTAL {snapshot.bite_count}"
            )
            self._event_banner_until = timestamp + 1.2

        if (
            snapshot.drink_count
            > self._previous_overlay_drink_count
        ):
            self._event_banner_text = (
                f"DRINK +1   TOTAL {snapshot.drink_count}"
            )
            self._event_banner_until = timestamp + 1.2

        self._previous_overlay_bite_count = (
            snapshot.bite_count
        )
        self._previous_overlay_drink_count = (
            snapshot.drink_count
        )

    def _draw_test_overlay(
        self,
        frame: Any,
        observation: MealObservation,
        snapshot: MealSessionSnapshot,
        timestamp: float,
    ) -> None:
        """원거리 테스트용 핵심 상태 오버레이."""
        height, width = frame.shape[:2]
        meal_debug = self.meal_fsm.meal_evidence_debug
        active = observation.active_hand_observation()
        bite = observation.bite_session
        drink = observation.drink_session

        def short_state(value: Any, maximum: int = 12) -> str:
            text = str(getattr(value, "value", value)).upper()
            aliases = {
                "APPROACHING": "APPROACH",
                "NEAR_MOUTH": "NEAR",
                "HAND_NEAR_VESSEL": "HAND_NEAR",
                "VESSEL_AVAILABLE": "VESSEL",
                "PICKUP_SESSION": "PICKUP",
                "RETURN_CANDIDATE": "RETURN",
            }
            text = aliases.get(text, text)
            return text[:maximum]

        def fit_scale(text: str, maximum_width: int, base: float) -> float:
            scale = base
            while scale > 0.42:
                size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
                )[0]
                if size[0] <= maximum_width:
                    break
                scale -= 0.05
            return max(0.42, scale)

        cooldown = float(bite.debug.get("cooldown_remaining", 0.0))
        cancel = bite.cancel_reason or "-"
        dist = (
            f"{active.normalized_hand_mouth_distance:.2f}"
            if active is not None
            and active.normalized_hand_mouth_distance is not None
            else "-"
        )
        leave_hold = (
            float(active.debug.get("leave_candidate_duration", 0.0))
            if active is not None
            else 0.0
        )

        panel_height = min(height - 8, max(172, int(height * 0.38)))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0, frame)

        margin = 12
        gap = 12
        col_width = max(120, (width - margin * 2 - gap) // 2)
        left_x = margin
        right_x = margin + col_width + gap
        title_scale = max(0.58, min(0.82, width / 900.0))
        body_scale = max(0.46, min(0.68, width / 1050.0))
        line_gap = max(24, int(panel_height / 7.0))

        count_line = (
            f"CONF {snapshot.bite_count}  "
            f"MEAL {meal_debug['meal_bites']}  "
            f"REJ {meal_debug['rejected_bites']}  "
            f"PEND {meal_debug['pending_bites']}  "
            f"DRINK {snapshot.drink_count}"
        )
        cv2.putText(
            frame, count_line, (margin, 30), cv2.FONT_HERSHEY_SIMPLEX,
            fit_scale(count_line, width - margin * 2, title_scale),
            (255, 255, 255), 2, cv2.LINE_AA,
        )

        left_lines = [
            f"BITE  {short_state(bite.state)}",
            f"CD    {cooldown:.2f}s",
            f"CANCEL {cancel[:20]}",
            (
                f"A{int(bool(active and active.approaching_mouth))} "
                f"N{int(bool(active and active.near_mouth))} "
                f"L{int(bool(active and active.leaving_mouth))} "
                f"RAW{int(bool(active and active.debug.get('raw_leaving', False)))}"
            ),
            f"DIST {dist}  LH {leave_hold:.2f}s",
        ]
        right_lines = [
            f"DRINK {short_state(drink.state)}",
            (
                f"ACT {int(drink.session_active)} "
                f"MOUTH {int(drink.mouth_reached)} "
                f"LEFT {int(drink.left_mouth)}"
            ),
            f"MEAL  {short_state(snapshot.meal_state)}",
            f"OBJ   {str(meal_debug['last_object'])[:13]}",
            f"WHY   {str(meal_debug['last_reason'])[:18]}",
        ]

        y0 = 58
        for index, text in enumerate(left_lines):
            cv2.putText(
                frame, text, (left_x, y0 + index * line_gap),
                cv2.FONT_HERSHEY_SIMPLEX,
                fit_scale(text, col_width, body_scale),
                (255, 255, 255), 2, cv2.LINE_AA,
            )
        for index, text in enumerate(right_lines):
            cv2.putText(
                frame, text, (right_x, y0 + index * line_gap),
                cv2.FONT_HERSHEY_SIMPLEX,
                fit_scale(text, col_width, body_scale),
                (255, 255, 255), 2, cv2.LINE_AA,
            )

        if self._event_banner_text and timestamp <= self._event_banner_until:
            banner = self._event_banner_text
            scale = fit_scale(banner, width - 40, max(0.9, width / 700.0))
            size = cv2.getTextSize(
                banner, cv2.FONT_HERSHEY_SIMPLEX, scale, 3
            )[0]
            x = max(10, (width - size[0]) // 2)
            y = min(height - 45, max(panel_height + 55, height // 2))
            banner_overlay = frame.copy()
            cv2.rectangle(
                banner_overlay,
                (max(0, x - 18), max(panel_height, y - size[1] - 18)),
                (min(width, x + size[0] + 18), min(height, y + 15)),
                (0, 0, 0), -1,
            )
            cv2.addWeighted(banner_overlay, 0.72, frame, 0.28, 0.0, frame)
            cv2.putText(
                frame, banner, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 3, cv2.LINE_AA,
            )

        cv2.putText(
            frame, "TEST | T:DETAIL", (10, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    def _draw_debug_overlay(
        self,
        frame: Any,
        observation: MealObservation,
        snapshot: MealSessionSnapshot,
    ) -> None:
        height, width = frame.shape[:2]

        person_roi = observation.debug.get("person_roi")
        if person_roi is not None:
            self._draw_box(frame, person_roi, "PERSON ROI")

        face_roi = observation.debug.get("face_roi")
        if face_roi is not None:
            face_source = observation.debug.get(
                "face_detection_source", "face_roi"
            )
            self._draw_box(frame, face_roi, f"FACE [{face_source}]")

        if observation.virtual_plate.available and observation.virtual_plate.roi is not None:
            self._draw_box(frame, observation.virtual_plate.roi, "PLATE")
        if observation.mouth.roi is not None:
            self._draw_box(frame, observation.mouth.roi, "MOUTH")

        for detection in observation.detections:
            self._draw_box(
                frame, detection.box,
                f"{detection.label} {detection.confidence:.2f}",
            )

        for side, hand in observation.hands.items():
            if not hand.detected:
                continue
            for point in hand.landmarks:
                cv2.circle(
                    frame, (int(point[0]), int(point[1])),
                    2, (255, 255, 255), -1,
                )
            contact = hand.debug.get("bite_contact_point")
            if contact is not None:
                cv2.circle(
                    frame, (int(contact[0]), int(contact[1])),
                    7, (255, 255, 255), 2,
                )

        active = observation.active_hand_observation()
        bite = observation.bite_session
        drink = observation.drink_session
        vessel = observation.vessel
        meal_debug = self.meal_fsm.meal_evidence_debug

        def short(value: Any, maximum: int = 18) -> str:
            text = str(getattr(value, "value", value)).upper()
            aliases = {
                "APPROACHING": "APPROACH",
                "NEAR_MOUTH": "NEAR",
                "HAND_NEAR_VESSEL": "HAND_NEAR",
                "VESSEL_AVAILABLE": "VESSEL",
                "PICKUP_SESSION": "PICKUP",
                "RETURN_CANDIDATE": "RETURN",
            }
            return aliases.get(text, text)[:maximum]

        def fit(text: str, maximum_width: int, base: float = 0.50) -> float:
            scale = base
            while scale > 0.34:
                if cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
                )[0][0] <= maximum_width:
                    break
                scale -= 0.04
            return max(0.34, scale)

        dist = (
            f"{active.normalized_hand_mouth_distance:.3f}"
            if active is not None
            and active.normalized_hand_mouth_distance is not None
            else "-"
        )
        delta = (
            f"{active.distance_delta:.2f}"
            if active is not None else "-"
        )
        cooldown = float(bite.debug.get("cooldown_remaining", 0.0))
        cancel = bite.cancel_reason or "-"
        raw_leave = bool(active and active.debug.get("raw_leaving", False))
        leave_hold = float(
            active.debug.get("leave_candidate_duration", 0.0)
            if active else 0.0
        )

        left_lines = [
            f"MEAL {short(snapshot.meal_state)} | ACT {short(snapshot.current_action)} | FPS {self.current_fps:.1f}",
            f"CONF {snapshot.bite_count} | MEAL {meal_debug['meal_bites']} | REJ {meal_debug['rejected_bites']} | PEND {meal_debug['pending_bites']}",
            f"BITE {short(bite.state)} | HAND {short(bite.active_hand)} | DONE {int(bite.session_completed)}",
            f"CD {cooldown:.2f}s | CANCEL {cancel[:24]}",
            f"A {int(bool(active and active.approaching_mouth))} | N {int(bool(active and active.near_mouth))} | L {int(bool(active and active.leaving_mouth))} | RAW {int(raw_leave)}",
            f"DIST {dist} | DELTA {delta} | LEAVE HOLD {leave_hold:.2f}s",
            f"DWELL {bite.dwell_duration:.2f}s | LEAVE {bite.leave_duration:.2f}s | TOTAL {bite.total_duration:.2f}s",
            f"LOW {int(bool(active and active.debug.get('in_low_food_zone', False)))} | LOWCTX {int(bool(active and active.debug.get('low_zone_context_active', False)))}",
            f"POSE {short(active.hand_pose) if active else '-'} | HOLD {short(active.holding_kind) if active else '-'}",
        ]
        right_lines = [
            f"DRINK {snapshot.drink_count} | STATE {short(drink.state)} | ACTIVE {int(drink.session_active)}",
            f"MOUTH {int(drink.mouth_reached)} | LEFT {int(drink.left_mouth)} | RETURN {int(drink.return_confirmed)}",
            f"VESSEL {short(vessel.normalized_label or '-', 12)} | VIS {int(vessel.detected)} | MEM {int(vessel.remembered)}",
            f"HAND NEAR {int(vessel.hand_near)} | BLOCK {int(bite.blocked_by_drink)}",
            f"OBJ {short(meal_debug['last_object'], 15)} | LABEL {short(meal_debug['last_label'] or '-', 15)}",
            f"DEC {short(meal_debug['last_decision'], 18)}",
            f"WHY {short(meal_debug['last_reason'], 24)}",
            f"CTX {int(meal_debug['last_behavior_context'])} | MOUTH VIS {int(observation.mouth.detected)} | HANDS {observation.quality.hand_count}",
            f"FACE {short(observation.debug.get('face_detection_source', '-'), 22)} | Q {observation.quality.quality_score:.2f}",
        ]

        margin = 8
        gap = 10
        col_width = max(140, (width - margin * 2 - gap) // 2)
        line_count = max(len(left_lines), len(right_lines))
        panel_height = min(height, max(220, line_count * 24 + 18))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.74, frame, 0.26, 0.0, frame)

        line_gap = max(20, min(27, (panel_height - 12) // line_count))
        base_scale = max(0.39, min(0.54, width / 1450.0))
        right_x = margin + col_width + gap
        for index, text in enumerate(left_lines):
            y = 20 + index * line_gap
            cv2.putText(
                frame, text, (margin, y), cv2.FONT_HERSHEY_SIMPLEX,
                fit(text, col_width, base_scale), (255, 255, 255),
                1, cv2.LINE_AA,
            )
        for index, text in enumerate(right_lines):
            y = 20 + index * line_gap
            cv2.putText(
                frame, text, (right_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                fit(text, col_width, base_scale), (255, 255, 255),
                1, cv2.LINE_AA,
            )

        cv2.putText(
            frame, "Q quit | T test/detail | P pause | A/D 5s | J/L 1s",
            (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_box(
        frame: Any,
        box: Sequence[float],
        label: str,
    ) -> None:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in box
        ]
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def _create_logger(self) -> MealLogger:
        return MealLogger(
            session_id=self.meal_fsm.session_id,
            config_path=self.options.config_path,
        )

    def _reset_session(self) -> None:
        self.logger.close()
        self.vision_processor.reset()
        self.meal_fsm.reset()
        self.logger = self._create_logger()
        self.last_observation = None
        self.last_result = None
        self.previous_frame_time = None
        self._event_banner_text = ""
        self._event_banner_until = 0.0
        self._previous_overlay_bite_count = 0
        self._previous_overlay_drink_count = 0
        print(
            "[Meal] V3 비전, FSM, Logger 세션을 초기화했습니다."
        )

    def _handle_key(self, key: int) -> bool:
        if key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            return True

        if key in (
            ord("r"),
            ord("R"),
        ):
            self._reset_session()

        elif key in (
            ord("p"),
            ord("P"),
        ):
            self.paused = not self.paused
            print(f"[Meal] Pause: {self.paused}")

        elif key in (
            ord("s"),
            ord("S"),
        ):
            self._save_screenshot()

        elif key in (
            ord("t"),
            ord("T"),
        ):
            self.test_mode = not self.test_mode
            print(
                "[Meal] Overlay mode: "
                + (
                    "TEST"
                    if self.test_mode
                    else "DETAIL"
                )
            )

        elif key in (
            ord("a"),
            ord("A"),
        ):
            self._seek_relative(-5.0)

        elif key in (
            ord("d"),
            ord("D"),
        ):
            # D는 디버그 토글과 충돌하므로 영상 파일에서는 +5초,
            # 카메라에서는 기존 디버그 토글로 사용한다.
            if self.is_video_file:
                self._seek_relative(5.0)
            else:
                self.debug_overlay = (
                    not self.debug_overlay
                )
                print(
                    f"[Meal] Debug Overlay: "
                    f"{self.debug_overlay}"
                )

        elif key in (
            ord("j"),
            ord("J"),
        ):
            self._seek_relative(-1.0)

        elif key in (
            ord("l"),
            ord("L"),
        ):
            self._seek_relative(1.0)

        elif key in (
            ord("h"),
            ord("H"),
        ):
            self._seek_absolute_ms(0.0)

        return False

    def _show_paused_frame(self) -> int:
        if not self.options.display:
            time.sleep(0.05)
            return -1

        if self.last_output_frame is not None:
            cv2.imshow(
                self.window_name,
                self.last_output_frame,
            )

        return cv2.waitKey(30) & 0xFF

    def _save_screenshot(self) -> None:
        if self.last_output_frame is None:
            return

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        name = datetime.now().strftime(
            "meal_v3_%Y%m%d_%H%M%S_%f.jpg"
        )
        path = self.output_directory / name
        cv2.imwrite(
            str(path),
            self.last_output_frame,
        )
        print(f"[Meal] 화면 저장: {path}")

    def _open_capture(self) -> cv2.VideoCapture:
        source: Any = (
            self.options.source
            if self.options.source
            else self.options.camera_index
        )

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(
                "카메라 또는 영상 소스를 열 수 없습니다: "
                f"{source}"
            )

        camera_config = self.config.get(
            "camera",
            {},
        )
        width = int(
            camera_config.get(
                "frame_width",
                camera_config.get("width", 640),
            )
        )
        height = int(
            camera_config.get(
                "frame_height",
                camera_config.get("height", 480),
            )
        )
        fps = float(
            camera_config.get(
                "target_fps",
                camera_config.get("fps", 30),
            )
        )

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width,
        )
        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height,
        )
        capture.set(
            cv2.CAP_PROP_FPS,
            fps,
        )

        if self.is_video_file:
            frame_count = float(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            source_fps = float(
                capture.get(cv2.CAP_PROP_FPS)
            )
            if frame_count > 0.0 and source_fps > 0.0:
                self.video_duration_ms = (
                    frame_count / source_fps * 1000.0
                )

        return capture

    def _prepare_video_controls(self) -> None:
        """영상 파일 테스트용 탐색 트랙바를 만든다."""
        if (
            not self.options.display
            or not self.is_video_file
        ):
            return

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )
        maximum_seconds = max(
            1,
            int(round(self.video_duration_ms / 1000.0)),
        )
        cv2.createTrackbar(
            "Position (sec)",
            self.window_name,
            0,
            maximum_seconds,
            self._on_trackbar_seek,
        )

    def _on_trackbar_seek(
        self,
        seconds: int,
    ) -> None:
        if self._trackbar_updating:
            return
        self._seek_absolute_ms(float(seconds) * 1000.0)

    def _update_video_trackbar(self) -> None:
        if (
            not self.options.display
            or not self.is_video_file
            or self.capture is None
        ):
            return

        seconds = int(
            max(
                0.0,
                self.capture.get(
                    cv2.CAP_PROP_POS_MSEC
                ) / 1000.0,
            )
        )
        self._trackbar_updating = True
        try:
            cv2.setTrackbarPos(
                "Position (sec)",
                self.window_name,
                seconds,
            )
        finally:
            self._trackbar_updating = False

    def _resolve_frame_timestamp(
        self,
        runtime_timestamp: float,
    ) -> float:
        """
        카메라는 실제 경과시간을, 영상 파일은 원본 영상 시간을 사용한다.
        처리 속도가 느려도 Bite/Drink 시간 기준이 변하지 않는다.
        """
        if (
            not self.is_video_file
            or self.capture is None
        ):
            return runtime_timestamp

        timestamp = max(
            0.0,
            float(
                self.capture.get(
                    cv2.CAP_PROP_POS_MSEC
                )
            ) / 1000.0,
        )

        # 일부 코덱이 POS_MSEC를 0으로만 반환하는 경우 frame/fps로 보완
        if timestamp <= 0.0:
            fps = float(
                self.capture.get(cv2.CAP_PROP_FPS)
            )
            frame_position = float(
                self.capture.get(
                    cv2.CAP_PROP_POS_FRAMES
                )
            )
            if fps > 0.0:
                timestamp = frame_position / fps

        self._last_video_timestamp = timestamp
        return timestamp

    def _seek_relative(
        self,
        seconds: float,
    ) -> None:
        if (
            not self.is_video_file
            or self.capture is None
        ):
            return

        current_ms = float(
            self.capture.get(cv2.CAP_PROP_POS_MSEC)
        )
        self._seek_absolute_ms(
            current_ms + float(seconds) * 1000.0
        )

    def _seek_absolute_ms(
        self,
        target_ms: float,
    ) -> None:
        """
        영상 위치를 옮긴 뒤 이전 프레임의 세션 상태가 섞이지 않도록
        비전/FSM/Logger를 새 세션으로 초기화한다.
        """
        if (
            not self.is_video_file
            or self.capture is None
        ):
            return

        maximum = (
            self.video_duration_ms
            if self.video_duration_ms > 0.0
            else float("inf")
        )
        target_ms = min(
            maximum,
            max(0.0, float(target_ms)),
        )

        self.capture.set(
            cv2.CAP_PROP_POS_MSEC,
            target_ms,
        )
        self.frame_index = int(
            self.capture.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )
        self.previous_frame_time = None
        self.current_fps = 0.0
        self._last_video_timestamp = (
            target_ms / 1000.0
        )
        self._reset_session()

        print(
            "[Meal] 영상 위치 이동: "
            f"{target_ms / 1000.0:.1f}초"
        )

    def _prepare_writer(self) -> None:
        if (
            not self.options.save_output
            or self.capture is None
        ):
            return

        width = int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
        height = int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
        fps = float(
            self.capture.get(cv2.CAP_PROP_FPS)
        )
        if fps <= 1.0:
            fps = 20.0

        path = (
            self.options.output_path
            or Path("logs/meal")
            / (
                f"{self.meal_fsm.session_id}"
                "_output.mp4"
            )
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )
        self.writer = cv2.VideoWriter(
            str(path),
            fourcc,
            fps,
            (width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(
                f"출력 영상을 열 수 없습니다: {path}"
            )

    def _update_fps(
        self,
        timestamp: float,
    ) -> None:
        if self.previous_frame_time is None:
            self.previous_frame_time = timestamp
            return

        delta = (
            timestamp - self.previous_frame_time
        )
        self.previous_frame_time = timestamp

        if delta <= 0.0:
            return

        instant = 1.0 / delta
        if self.current_fps <= 0.0:
            self.current_fps = instant
        else:
            self.current_fps = (
                0.15 * instant
                + 0.85 * self.current_fps
            )

    def _limit_fps(self) -> None:
        target = self.options.target_fps
        if target is None or target <= 0.0:
            return

        frame_seconds = 1.0 / target
        time.sleep(max(0.0, frame_seconds * 0.05))

    # ------------------------------------------------------------------
    # Component creation / invocation
    # ------------------------------------------------------------------

    def _create_detector(
        self,
        model_path: Optional[str],
    ) -> Any:
        module = self._optional_import(
            "vision.detector"
        )
        if module is None:
            print(
                "[Meal] vision.detector가 없어 "
                "객체 검출 없이 실행합니다."
            )
            return None

        for name in (
            "YOLODetector",
            "ObjectDetector",
            "Detector",
        ):
            cls = getattr(module, name, None)
            if cls is None:
                continue

            kwargs_candidates = []
            if model_path:
                kwargs_candidates.extend(
                    [
                        {"model_path": model_path},
                        {"model": model_path},
                        {"weights": model_path},
                    ]
                )
            kwargs_candidates.append({})

            for kwargs in kwargs_candidates:
                try:
                    return cls(**kwargs)
                except TypeError:
                    continue
                except Exception as error:
                    raise ComponentLoadError(
                        f"{name} 초기화 실패: {error}"
                    ) from error

        factory = getattr(
            module,
            "create_detector",
            None,
        )
        if callable(factory):
            return (
                factory(model_path)
                if model_path
                else factory()
            )

        raise ComponentLoadError(
            "지원되는 Detector 클래스를 찾지 못했습니다."
        )

    def _create_hand_detector(self) -> Any:
        module = self._optional_import(
            "vision.hand"
        )
        if module is None:
            print(
                "[Meal] vision.hand가 없어 "
                "손 검출 없이 실행합니다."
            )
            return None

        for name in (
            "HandDetector",
            "MediaPipeHandDetector",
            "HandTracker",
        ):
            cls = getattr(module, name, None)
            if cls is None:
                continue

            for kwargs in (
                {"max_num_hands": 2},
                {"maximum_hands": 2},
                {},
            ):
                try:
                    return cls(**kwargs)
                except TypeError:
                    continue
                except Exception as error:
                    raise ComponentLoadError(
                        f"{name} 초기화 실패: {error}"
                    ) from error

        factory = getattr(
            module,
            "create_hand_detector",
            None,
        )
        if callable(factory):
            return factory()

        raise ComponentLoadError(
            "지원되는 HandDetector를 찾지 못했습니다."
        )

    def _create_face_detector(self) -> Any:
        modules = [
            self._optional_import(
                "vision.face_mesh"
            ),
            self._optional_import(
                "vision.face"
            ),
        ]

        for module in modules:
            if module is None:
                continue

            for name in (
                "FaceMeshDetector",
                "MouthDetector",
                "FaceDetector",
                "FaceMesh",
            ):
                cls = getattr(
                    module,
                    name,
                    None,
                )
                if cls is None:
                    continue

                try:
                    return cls()
                except TypeError:
                    continue
                except Exception as error:
                    raise ComponentLoadError(
                        f"{name} 초기화 실패: {error}"
                    ) from error

            for factory_name in (
                "create_face_mesh",
                "create_face_detector",
            ):
                factory = getattr(
                    module,
                    factory_name,
                    None,
                )
                if callable(factory):
                    return factory()

        print(
            "[Meal] 얼굴/입 검출기를 찾지 못했습니다."
        )
        return None

    def _run_detector(self, frame: Any) -> Any:
        if self.components.detector is None:
            return []
        return self._invoke_component(
            self.components.detector,
            frame,
            (
                "detect",
                "process",
                "predict",
                "__call__",
            ),
        )

    def _run_hand_detector(
        self,
        frame: Any,
    ) -> Any:
        if self.components.hand_detector is None:
            return []
        return self._invoke_component(
            self.components.hand_detector,
            frame,
            (
                "detect",
                "process",
                "find_hands",
                "findHands",
                "__call__",
            ),
        )

    def _run_face_detector(
        self,
        frame: Any,
    ) -> Any:
        if self.components.face_detector is None:
            return None
        return self._invoke_component(
            self.components.face_detector,
            frame,
            (
                "detect",
                "process",
                "find_face",
                "find_mouth",
                "__call__",
            ),
        )

    @staticmethod
    def _invoke_component(
        component: Any,
        frame: Any,
        method_names: Sequence[str],
    ) -> Any:
        last_error: Optional[Exception] = None

        for name in method_names:
            method = (
                component
                if (
                    name == "__call__"
                    and callable(component)
                )
                else getattr(component, name, None)
            )
            if not callable(method):
                continue

            try:
                return method(frame)
            except TypeError as error:
                last_error = error
                try:
                    return method(image=frame)
                except TypeError:
                    continue

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "호출 가능한 처리 메서드가 없습니다: "
            f"{type(component).__name__}"
        )

    # ------------------------------------------------------------------
    # Adapt raw outputs
    # ------------------------------------------------------------------

    def _adapt_detections(
        self,
        raw: Any,
    ) -> List[Detection]:
        values = self._unwrap_result_collection(
            raw,
            (
                "detections",
                "objects",
                "results",
                "boxes",
            ),
        )

        result: List[Detection] = []
        for value in values:
            detection = self._coerce_detection(value)
            if detection is not None:
                result.append(detection)
        return result

    def _adapt_hands(
        self,
        raw: Any,
        frame_size: Tuple[int, int],
    ) -> List[RawHandInput]:
        media_pipe_result = None
        hand_collection = raw

        if isinstance(raw, tuple) and len(raw) >= 2:
            media_pipe_result = raw[0]
            hand_collection = raw[1]

        values = self._unwrap_result_collection(
            hand_collection,
            (
                "hands",
                "hand_results",
                "landmarks",
                "multi_hand_landmarks",
            ),
        )

        handedness_values = []
        handedness_source = (
            media_pipe_result
            if media_pipe_result is not None
            else raw
        )

        if isinstance(
            handedness_source,
            Mapping,
        ):
            handedness_values = list(
                handedness_source.get(
                    "handedness",
                    handedness_source.get(
                        "multi_handedness",
                        [],
                    ),
                )
                or []
            )
        else:
            handedness_values = list(
                getattr(
                    handedness_source,
                    "multi_handedness",
                    getattr(
                        handedness_source,
                        "handedness",
                        [],
                    ),
                )
                or []
            )

        hands: List[RawHandInput] = []
        for index, value in enumerate(values):
            landmarks = self._extract_landmarks(
                value
            )
            if not landmarks:
                continue

            landmarks = self._to_pixel_points(
                landmarks,
                frame_size,
            )

            side = self._extract_hand_side(value)
            if (
                side == HandSide.UNKNOWN
                and index < len(handedness_values)
            ):
                side = self._extract_hand_side(
                    handedness_values[index]
                )
            if side == HandSide.UNKNOWN:
                side = (
                    HandSide.LEFT
                    if index == 0
                    else HandSide.RIGHT
                )

            tracking_id = self._read_value(
                value,
                (
                    "tracking_id",
                    "track_id",
                    "id",
                ),
                default=None,
            )
            confidence = self._read_value(
                value,
                (
                    "confidence",
                    "score",
                ),
                default=1.0,
            )

            hands.append(
                RawHandInput(
                    side=side,
                    landmarks=landmarks,
                    tracking_id=(
                        int(tracking_id)
                        if tracking_id is not None
                        else None
                    ),
                    confidence=float(confidence),
                )
            )

        return hands[:2]

    def _adapt_mouth(
        self,
        raw: Any,
        frame_size: Tuple[int, int],
    ) -> RawMouthInput:
        if raw is None:
            return RawMouthInput()

        source = raw

        if isinstance(raw, tuple):
            if len(raw) >= 2:
                faces = raw[1]
                if (
                    isinstance(
                        faces,
                        (list, tuple),
                    )
                    and faces
                ):
                    source = faces[0]
                elif faces:
                    source = faces
                else:
                    return RawMouthInput()
            elif raw:
                source = raw[0]

        elif isinstance(raw, list):
            if raw:
                source = raw[0]
            else:
                return RawMouthInput()

        detected = bool(
            self._read_value(
                source,
                (
                    "detected",
                    "face_detected",
                    "found",
                ),
                default=True,
            )
        )
        center = self._read_value(
            source,
            (
                "mouth_center",
                "center",
                "lip_center",
            ),
            default=None,
        )
        mouth_landmarks = self._read_value(
            source,
            (
                "mouth_landmarks",
                "lip_landmarks",
                "lips",
            ),
            default=[],
        )
        face_box = self._read_value(
            source,
            (
                "face_box",
                "bbox",
                "box",
            ),
            default=None,
        )
        openness = self._read_value(
            source,
            (
                "mouth_openness",
                "openness",
                "mouth_open_ratio",
            ),
            default=None,
        )

        all_landmarks = self._extract_landmarks(
            source
        )

        if (
            not mouth_landmarks
            and len(all_landmarks) >= 468
        ):
            mouth_indices = (
                61, 146, 91, 181, 84, 17,
                314, 405, 321, 375, 291,
                308, 324, 318, 402, 317,
                14, 87, 178, 88, 95, 78,
                191, 80, 81, 82, 13, 312,
                311, 310, 415,
            )
            mouth_landmarks = [
                all_landmarks[index]
                for index in mouth_indices
                if index < len(all_landmarks)
            ]

        mouth_points = self._to_pixel_points(
            self._normalize_points(
                mouth_landmarks
            ),
            frame_size,
        )

        if center is not None:
            center_points = self._to_pixel_points(
                self._normalize_points([center]),
                frame_size,
            )
            center_value = (
                center_points[0]
                if center_points
                else None
            )
        elif mouth_points:
            center_value = (
                sum(
                    point[0]
                    for point in mouth_points
                )
                / len(mouth_points),
                sum(
                    point[1]
                    for point in mouth_points
                )
                / len(mouth_points),
            )
        else:
            center_value = None
            detected = False

        return RawMouthInput(
            detected=(
                detected
                and center_value is not None
            ),
            center=center_value,
            face_box=self._normalize_box(
                face_box,
                frame_size,
            ),
            mouth_landmarks=mouth_points,
            openness=(
                float(openness)
                if openness is not None
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_import(name: str) -> Any:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            return None

    @staticmethod
    def _close_component(
        component: Any,
    ) -> None:
        if component is None:
            return

        for name in (
            "close",
            "release",
            "shutdown",
        ):
            method = getattr(
                component,
                name,
                None,
            )
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break

    @staticmethod
    def _unwrap_result_collection(
        raw: Any,
        keys: Sequence[str],
    ) -> List[Any]:
        if raw is None:
            return []

        if isinstance(raw, Mapping):
            for key in keys:
                if key in raw:
                    value = raw[key]
                    if value is None:
                        return []
                    if isinstance(
                        value,
                        (list, tuple),
                    ):
                        return list(value)
                    return [value]

        for key in keys:
            value = getattr(raw, key, None)
            if value is not None:
                if isinstance(
                    value,
                    (list, tuple),
                ):
                    return list(value)
                return [value]

        if isinstance(raw, tuple):
            for value in reversed(raw):
                if isinstance(
                    value,
                    (list, tuple),
                ):
                    return list(value)
                if isinstance(value, Mapping):
                    return (
                        MealApplication
                        ._unwrap_result_collection(
                            value,
                            keys,
                        )
                    )

        if isinstance(raw, list):
            return raw

        return [raw]

    @staticmethod
    def _coerce_detection(
        value: Any,
    ) -> Optional[Detection]:
        if isinstance(value, Detection):
            return value

        label = MealApplication._read_value(
            value,
            (
                "label",
                "class_name",
                "name",
            ),
            default=None,
        )
        confidence = MealApplication._read_value(
            value,
            (
                "confidence",
                "conf",
                "score",
            ),
            default=0.0,
        )
        box = MealApplication._read_value(
            value,
            (
                "box",
                "bbox",
                "xyxy",
            ),
            default=None,
        )
        track_id = MealApplication._read_value(
            value,
            (
                "track_id",
                "id",
            ),
            default=None,
        )

        if label is None or box is None:
            return None

        try:
            x1, y1, x2, y2 = list(box)[:4]
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

    @staticmethod
    def _extract_landmarks(
        value: Any,
    ) -> List[Point2D]:
        if isinstance(
            value,
            (list, tuple),
        ):
            direct = MealApplication._normalize_points(
                value
            )
            if direct:
                return direct

        landmarks = MealApplication._read_value(
            value,
            (
                "landmarks",
                "hand_landmarks",
                "multi_hand_landmarks",
                "points",
                "lmList",
            ),
            default=None,
        )

        if (
            landmarks is None
            and hasattr(value, "landmark")
        ):
            landmarks = getattr(
                value,
                "landmark",
            )

        if landmarks is None:
            return []

        if hasattr(landmarks, "landmark"):
            landmarks = landmarks.landmark

        return MealApplication._normalize_points(
            landmarks
        )

    @staticmethod
    def _normalize_points(
        values: Any,
    ) -> List[Point2D]:
        if values is None:
            return []

        try:
            iterable = list(values)
        except TypeError:
            iterable = [values]

        points: List[Point2D] = []

        for value in iterable:
            if isinstance(value, Mapping):
                x = value.get("x")
                y = value.get("y")
            elif (
                hasattr(value, "x")
                and hasattr(value, "y")
            ):
                x = value.x
                y = value.y
            else:
                try:
                    x, y = list(value)[:2]
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            try:
                points.append(
                    (float(x), float(y))
                )
            except (TypeError, ValueError):
                continue

        return points

    @staticmethod
    def _to_pixel_points(
        points: Sequence[Point2D],
        frame_size: Tuple[int, int],
    ) -> List[Point2D]:
        width, height = frame_size
        result = []

        for x, y in points:
            if (
                -0.05 <= x <= 1.05
                and -0.05 <= y <= 1.05
            ):
                result.append(
                    (x * width, y * height)
                )
            else:
                result.append((x, y))

        return result

    @staticmethod
    def _extract_hand_side(
        value: Any,
    ) -> HandSide:
        side = MealApplication._read_value(
            value,
            (
                "side",
                "handedness",
                "label",
                "classification",
            ),
            default=None,
        )

        if hasattr(side, "classification"):
            classifications = side.classification
            if classifications:
                side = getattr(
                    classifications[0],
                    "label",
                    None,
                )

        if isinstance(
            side,
            (list, tuple),
        ) and side:
            side = side[0]

        if hasattr(side, "label"):
            side = side.label

        normalized = str(side).strip().lower()
        if "left" in normalized:
            return HandSide.LEFT
        if "right" in normalized:
            return HandSide.RIGHT
        return HandSide.UNKNOWN

    @staticmethod
    def _normalize_box(
        value: Any,
        frame_size: Tuple[int, int],
    ) -> Optional[BoxXYXY]:
        if value is None:
            return None

        if isinstance(value, Mapping):
            if all(
                key in value
                for key in (
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                )
            ):
                raw = (
                    value["x1"],
                    value["y1"],
                    value["x2"],
                    value["y2"],
                )
            elif all(
                key in value
                for key in (
                    "x",
                    "y",
                    "w",
                    "h",
                )
            ):
                raw = (
                    value["x"],
                    value["y"],
                    value["x"] + value["w"],
                    value["y"] + value["h"],
                )
            else:
                return None
        else:
            try:
                raw = tuple(
                    list(value)[:4]
                )
            except (
                TypeError,
                ValueError,
            ):
                return None

        if len(raw) != 4:
            return None

        x1, y1, x2, y2 = [
            float(item)
            for item in raw
        ]
        width, height = frame_size

        if all(
            -0.05 <= item <= 1.05
            for item in (
                x1,
                y1,
                x2,
                y2,
            )
        ):
            return (
                x1 * width,
                y1 * height,
                x2 * width,
                y2 * height,
            )

        return (x1, y1, x2, y2)

    @staticmethod
    def _read_value(
        value: Any,
        names: Sequence[str],
        default: Any = None,
    ) -> Any:
        if isinstance(value, Mapping):
            for name in names:
                if name in value:
                    return value[name]
            return default

        for name in names:
            if hasattr(value, name):
                return getattr(value, name)

        return default

    @staticmethod
    def _load_config(
        path: Path,
    ) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"설정 파일을 찾을 수 없습니다: {path}"
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


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> RuntimeOptions:
    parser = argparse.ArgumentParser(
        description="V3 식사 행동 인식 실행",
    )
    parser.add_argument(
        "--source",
        default="",
        help=(
            "영상 파일 경로. "
            "비워두면 카메라를 사용합니다."
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="카메라 인덱스",
    )
    parser.add_argument(
        "--config",
        default="meal/meal_config.json",
        help="meal_config.json 경로",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO 모델 경로",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="OpenCV 화면을 표시하지 않습니다.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Debug Overlay를 끕니다.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="결과 영상을 저장합니다.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 영상 저장 경로",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="처리할 최대 프레임 수",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="처리 FPS 제한",
    )

    args = parser.parse_args(argv)

    return RuntimeOptions(
        source=str(args.source),
        camera_index=int(args.camera),
        config_path=Path(args.config),
        model_path=args.model,
        display=not bool(args.no_display),
        debug_overlay=not bool(args.no_debug),
        save_output=bool(args.save_output),
        output_path=(
            Path(args.output)
            if args.output
            else None
        ),
        maximum_frames=args.max_frames,
        target_fps=args.target_fps,
    )


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    options = parse_arguments(argv)
    application = MealApplication(options)
    return application.run()


if __name__ == "__main__":
    raise SystemExit(main())
