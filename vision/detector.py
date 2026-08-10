"""
YOLO 기반 객체 검출 모듈.

역할
----
- YOLO 모델을 한 번만 불러온다.
- 카메라 프레임에서 객체를 검출한다.
- 클래스 이름, 신뢰도, 바운딩 박스를 구조화해 반환한다.
- 식사 기능에서 필요한 객체 라벨 집합을 제공한다.

이 파일은 MealFSM이나 MealObservation을 직접 다루지 않는다.
""" 

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ultralytics import YOLO


BoundingBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """YOLO에서 검출된 객체 하나의 정보."""

    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int


@dataclass(frozen=True)
class DetectionResult:
    """한 프레임의 YOLO 검출 결과."""

    detections: List[Detection]
    labels: Set[str]
    person_detected: bool

    def has_label(self, label: str) -> bool:
        """특정 클래스가 검출되었는지 반환한다."""
        normalized_label = label.strip().lower()
        return normalized_label in self.labels


class YOLODetector:
    """
    Ultralytics YOLO 객체 검출기.

    Parameters
    ----------
    model_path:
        사용할 YOLO 모델 경로 또는 모델명.

    confidence:
        검출 결과로 인정할 최소 신뢰도.

    device:
        추론 장치.
        None이면 Ultralytics가 자동 선택한다.
        Raspberry Pi에서는 일반적으로 "cpu"를 사용한다.

    image_size:
        YOLO 입력 이미지 크기.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        confidence: float = 0.4,
        device: Optional[str] = None,
        image_size: int = 640,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence는 0.0 이상 1.0 이하이어야 합니다."
            )

        if image_size <= 0:
            raise ValueError("image_size는 양수여야 합니다.")

        self.model_path = str(model_path)
        self.confidence = confidence
        self.device = device
        self.image_size = image_size

        self._model = YOLO(self.model_path)
        self._class_names = self._normalize_class_names(
            self._model.names
        )

    @property
    def class_names(self) -> Dict[int, str]:
        """현재 모델의 클래스 ID와 이름을 반환한다."""
        return dict(self._class_names)

    def detect(self, frame: Any) -> DetectionResult:
        """
        OpenCV 프레임 한 장에서 객체를 검출한다.

        Parameters
        ----------
        frame:
            OpenCV가 반환한 BGR 이미지 배열.

        Returns
        -------
        DetectionResult
            객체별 검출 결과와 라벨 집합.
        """
        if frame is None:
            raise ValueError("검출할 frame이 None입니다.")

        predict_kwargs: Dict[str, Any] = {
            "source": frame,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }

        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self._model.predict(**predict_kwargs)

        if not results:
            return DetectionResult(
                detections=[],
                labels=set(),
                person_detected=False,
            )

        detections = self._extract_detections(results[0])
        labels = {detection.label for detection in detections}

        return DetectionResult(
            detections=detections,
            labels=labels,
            person_detected="person" in labels,
        )

    def detect_labels(self, frame: Any) -> Set[str]:
        """
        검출된 클래스 이름만 필요한 경우 사용한다.

        예:
            {"person", "cup", "cell phone"}
        """
        return self.detect(frame).labels

    def _extract_detections(self, result: Any) -> List[Detection]:
        """Ultralytics 결과 객체에서 Detection 목록을 생성한다."""
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return []

        detections: List[Detection] = []

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            coordinates = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (
                int(round(value))
                for value in coordinates
            )

            label = self._class_names.get(
                class_id,
                f"class_{class_id}",
            )

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                )
            )

        return detections

    @staticmethod
    def _normalize_class_names(
        names: Any,
    ) -> Dict[int, str]:
        """
        Ultralytics 클래스명 구조를 Dict[int, str]로 통일한다.
        """
        if isinstance(names, dict):
            return {
                int(class_id): str(label).strip().lower()
                for class_id, label in names.items()
            }

        if isinstance(names, (list, tuple)):
            return {
                class_id: str(label).strip().lower()
                for class_id, label in enumerate(names)
            }

        raise TypeError(
            "YOLO 모델의 클래스 이름 구조를 해석할 수 없습니다."
        )