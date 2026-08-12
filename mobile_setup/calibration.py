from __future__ import annotations

from math import hypot, isfinite
from typing import Any


INDOOR_OFFSET = 0.10
OUTDOOR_OFFSET = 0.10
MIN_LINE_LENGTH = 0.22
HARD_EDGE_MARGIN = 0.04
WARNING_EDGE_MARGIN = 0.08
ROI_MARGIN = 0.06


class CalibrationError(ValueError):
    pass


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{name} 좌표가 숫자가 아닙니다.")

    value = float(value)

    if not isfinite(value):
        raise CalibrationError(f"{name} 좌표가 유효하지 않습니다.")

    if value < 0.0 or value > 1.0:
        raise CalibrationError(f"{name} 좌표가 화면 범위를 벗어났습니다.")

    return value


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise CalibrationError(f"{name} 점 정보가 올바르지 않습니다.")

    return (
        _number(value.get("x"), f"{name}.x"),
        _number(value.get("y"), f"{name}.y"),
    )


def _shift_line(
    line: list[tuple[float, float]],
    nx: float,
    ny: float,
    amount: float,
) -> list[tuple[float, float]]:
    return [
        (x + nx * amount, y + ny * amount)
        for x, y in line
    ]


def _outside_frame(points: list[tuple[float, float]]) -> bool:
    return any(
        x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0
        for x, y in points
    )


def _inside_edge_margin(
    points: list[tuple[float, float]],
    margin: float,
) -> bool:
    return any(
        x < margin
        or x > 1.0 - margin
        or y < margin
        or y > 1.0 - margin
        for x, y in points
    )


def _round_point(point: tuple[float, float]) -> list[float]:
    return [
        round(point[0], 6),
        round(point[1], 6),
    ]


def build_calibration(payload: Any) -> dict[str, Any]:
    try:
        if not isinstance(payload, dict):
            raise CalibrationError("설정 데이터가 올바르지 않습니다.")

        boundary = payload.get("boundary_line")

        if not isinstance(boundary, dict):
            raise CalibrationError("현관 경계선 정보가 없습니다.")

        a = _point(boundary.get("a"), "경계선 시작점")
        b = _point(boundary.get("b"), "경계선 끝점")

        direction_raw = payload.get("direction", 1)

        if direction_raw not in (-1, 1):
            raise CalibrationError("외출 방향 값이 올바르지 않습니다.")

        direction = int(direction_raw)

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        line_length = hypot(dx, dy)

        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        if line_length < MIN_LINE_LENGTH:
            errors.append(
                {
                    "code": "BOUNDARY_LINE_TOO_SHORT",
                    "message": (
                        "현관 경계선이 너무 짧습니다. "
                        "사람이 통과하는 전체 폭을 포함하도록 선을 늘려 주세요."
                    ),
                }
            )

        if line_length == 0:
            nx = 0.0
            ny = 1.0
        else:
            tangent_x = dx / line_length
            tangent_y = dy / line_length
            nx = -tangent_y * direction
            ny = tangent_x * direction

        boundary_line = [a, b]

        indoor_line = _shift_line(
            boundary_line,
            nx,
            ny,
            -INDOOR_OFFSET,
        )

        outdoor_line = _shift_line(
            boundary_line,
            nx,
            ny,
            OUTDOOR_OFFSET,
        )

        if _outside_frame(outdoor_line):
            errors.append(
                {
                    "code": "OUTDOOR_LINE_OUT_OF_FRAME",
                    "message": (
                        "외출 영역이 화면 밖으로 벗어납니다. "
                        "카메라 각도를 바깥쪽이 더 보이도록 조정하거나 "
                        "경계선을 화면 안쪽으로 옮겨 주세요."
                    ),
                }
            )
        elif _inside_edge_margin(outdoor_line, HARD_EDGE_MARGIN):
            errors.append(
                {
                    "code": "OUTDOOR_SPACE_INSUFFICIENT",
                    "message": (
                        "현관 바깥쪽 감지 공간이 부족합니다. "
                        "카메라 각도를 바깥이 더 보이게 조정해 주세요."
                    ),
                }
            )
        elif _inside_edge_margin(outdoor_line, WARNING_EDGE_MARGIN):
            warnings.append(
                {
                    "code": "OUTDOOR_LINE_NEAR_EDGE",
                    "message": (
                        "외출 영역이 화면 가장자리에 가깝습니다. "
                        "정확한 감지를 위해 바깥쪽 공간을 조금 더 확보하는 것을 권장합니다."
                    ),
                }
            )

        if _outside_frame(indoor_line):
            errors.append(
                {
                    "code": "INDOOR_LINE_OUT_OF_FRAME",
                    "message": (
                        "실내 감지 영역이 화면 밖으로 벗어납니다. "
                        "카메라가 실내 이동 경로도 보이도록 각도를 조정해 주세요."
                    ),
                }
            )
        elif _inside_edge_margin(indoor_line, HARD_EDGE_MARGIN):
            errors.append(
                {
                    "code": "INDOOR_SPACE_INSUFFICIENT",
                    "message": (
                        "현관 안쪽 감지 공간이 부족합니다. "
                        "실내 이동 경로가 조금 더 보이도록 카메라를 조정해 주세요."
                    ),
                }
            )

        all_points = indoor_line + boundary_line + outdoor_line

        min_x = max(
            0.0,
            min(point[0] for point in all_points) - ROI_MARGIN,
        )
        min_y = max(
            0.0,
            min(point[1] for point in all_points) - ROI_MARGIN,
        )
        max_x = min(
            1.0,
            max(point[0] for point in all_points) + ROI_MARGIN,
        )
        max_y = min(
            1.0,
            max(point[1] for point in all_points) + ROI_MARGIN,
        )

        geometry = {
            "boundary_line": [
                _round_point(a),
                _round_point(b),
            ],
            "indoor_line": [
                _round_point(indoor_line[0]),
                _round_point(indoor_line[1]),
            ],
            "outdoor_line": [
                _round_point(outdoor_line[0]),
                _round_point(outdoor_line[1]),
            ],
            "exit_direction": [
                round(nx, 6),
                round(ny, 6),
            ],
            "roi": [
                [
                    round(min_x, 6),
                    round(min_y, 6),
                ],
                [
                    round(max_x, 6),
                    round(max_y, 6),
                ],
            ],
            "offsets": {
                "indoor": INDOOR_OFFSET,
                "outdoor": OUTDOOR_OFFSET,
            },
        }

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "geometry": geometry,
        }

    except CalibrationError as exc:
        return {
            "ok": False,
            "errors": [
                {
                    "code": "INVALID_INPUT",
                    "message": str(exc),
                }
            ],
            "warnings": [],
            "geometry": None,
        }
