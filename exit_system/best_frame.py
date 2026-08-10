from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .frame_buffer import BufferedFrame


@dataclass
class ScoredFrame:
    item: BufferedFrame
    total: float
    size_score: float
    not_cut_score: float
    sharpness_score: float
    brightness: float


def _clip_bbox(bbox: tuple[int, int, int, int], width: int, height: int):
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def score_frame(item: BufferedFrame, cfg: dict) -> ScoredFrame | None:
    if item.bbox is None:
        return None

    frame = item.frame
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(item.bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
    if area_ratio < float(cfg["minimum_area_ratio"]):
        return None
    if brightness < float(cfg["minimum_brightness"]):
        return None

    # 현관 설치 후 기대되는 최대 유효 크기를 화면의 약 35%로 두고 정규화.
    size_score = min(area_ratio / 0.35, 1.0)

    margin = int(cfg["border_margin_px"])
    touches_border = (
        x1 <= margin
        or y1 <= margin
        or x2 >= width - margin
        or y2 >= height - margin
    )
    not_cut_score = 0.25 if touches_border else 1.0

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(lap_var / 200.0, 1.0)

    w = cfg["weights"]
    total = (
        float(w["size"]) * size_score
        + float(w["not_cut"]) * not_cut_score
        + float(w["sharpness"]) * sharpness_score
    )

    return ScoredFrame(
        item=item,
        total=total,
        size_score=size_score,
        not_cut_score=not_cut_score,
        sharpness_score=sharpness_score,
        brightness=brightness,
    )


def select_best_frame(
    candidates: list[BufferedFrame], cfg: dict
) -> ScoredFrame | None:
    scored = [score_frame(item, cfg) for item in candidates]
    valid = [item for item in scored if item is not None]
    return max(valid, key=lambda item: item.total) if valid else None
