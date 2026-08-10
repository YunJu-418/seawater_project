from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class PersonDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    track_id: Optional[int]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PersonTracker:
    def __init__(self, model_path: str, confidence: float, inference_size: int) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.inference_size = inference_size

    def track(self, frame: np.ndarray) -> list[PersonDetection]:
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.confidence,
            imgsz=self.inference_size,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        detections: list[PersonDetection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        ids = boxes.id
        ids_list = ids.int().cpu().tolist() if ids is not None else [None] * len(boxes)

        for box, track_id in zip(boxes, ids_list):
            xyxy = box.xyxy[0].int().cpu().tolist()
            conf = float(box.conf[0].cpu())
            detections.append(
                PersonDetection(
                    bbox=tuple(xyxy),
                    confidence=conf,
                    track_id=int(track_id) if track_id is not None else None,
                )
            )
        return detections


def inside_roi(point: tuple[float, float], roi: list[int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def choose_primary_person(
    detections: list[PersonDetection], roi: list[int]
) -> Optional[PersonDetection]:
    candidates = [d for d in detections if inside_roi(d.center, roi)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
    )
