from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ClothingText:
    upper: str
    lower: str
    note: str

    @property
    def summary(self) -> str:
        return f"상의 {self.upper}, 하의 {self.lower}"


COLOR_NAMES = [
    ("검정", np.array([0, 0, 0]), np.array([180, 255, 55])),
    ("흰색", np.array([0, 0, 190]), np.array([180, 55, 255])),
    ("회색", np.array([0, 0, 56]), np.array([180, 55, 189])),
    ("빨강", np.array([0, 60, 50]), np.array([10, 255, 255])),
    ("주황", np.array([11, 70, 50]), np.array([24, 255, 255])),
    ("노랑", np.array([25, 60, 60]), np.array([35, 255, 255])),
    ("초록", np.array([36, 45, 35]), np.array([85, 255, 255])),
    ("파랑", np.array([86, 45, 35]), np.array([130, 255, 255])),
    ("보라", np.array([131, 40, 35]), np.array([160, 255, 255])),
    ("빨강", np.array([161, 60, 50]), np.array([180, 255, 255])),
]


def _dominant_color_name(crop: np.ndarray, cfg: dict) -> str:
    if crop.size == 0:
        return "확인 불가"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 너무 어둡거나 과노출된 픽셀을 제외한다. 채도가 낮은 픽셀은
    # 검정/흰색/회색 판정에 필요하므로 제거하지 않고 설정값을 실제 사용한다.
    valid = (
        (v >= int(cfg["value_ignore_low"]))
        & (v <= int(cfg["value_ignore_high"]))
    )
    saturation_threshold = int(cfg.get("saturation_ignore_threshold", 20))
    neutral = s < saturation_threshold
    chromatic = s >= saturation_threshold
    valid = valid & (neutral | chromatic)
    pixels = hsv[valid]
    if len(pixels) < 30:
        pixels = hsv.reshape(-1, 3)

    counts: dict[str, int] = {}
    for name, low, high in COLOR_NAMES:
        inside = np.all((pixels >= low) & (pixels <= high), axis=1)
        counts[name] = counts.get(name, 0) + int(inside.sum())

    if not counts or max(counts.values()) == 0:
        return "확인 불가"
    return max(counts, key=counts.get) + " 계열"


def analyze_clothing_colors(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    cfg: dict,
) -> ClothingText:
    """
    라즈베리파이 단독 처리용 경량 방식:
    사람 바운딩박스를 상의/하의 영역으로 나누고 대표 색상을 텍스트화한다.

    의복 종류(티셔츠, 셔츠, 청바지 등)는 별도 학습 모델 없이 신뢰성 있게
    구분하기 어려우므로 본 버전에서는 추측하지 않는다.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    person = frame[y1:y2, x1:x2]
    if person.shape[0] < int(cfg["minimum_crop_height"]):
        return ClothingText("확인 불가", "확인 불가", "사람 영상이 너무 작음")

    ph, pw = person.shape[:2]

    # 머리·배경 영향을 줄이기 위해 중앙 폭만 사용.
    left = int(pw * 0.18)
    right = int(pw * 0.82)

    # 일반적인 신체 비율에 맞춘 경량 휴리스틱.
    upper = person[int(ph * 0.18):int(ph * 0.52), left:right]
    lower = person[int(ph * 0.55):int(ph * 0.88), left:right]

    upper_name = _dominant_color_name(upper, cfg)
    lower_name = _dominant_color_name(lower, cfg)

    return ClothingText(
        upper=upper_name,
        lower=lower_name,
        note="영상 기반 대표 색상 추정",
    )
