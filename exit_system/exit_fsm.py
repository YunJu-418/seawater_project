from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import time
from typing import Optional


class ExitState(Enum):
    WAITING = auto()
    INDOOR_SIDE = auto()
    OUTWARD_MOVING = auto()
    DISAPPEARED = auto()
    EXIT_CONFIRMED = auto()
    COOLDOWN = auto()


@dataclass
class FSMEvent:
    name: str
    timestamp: float
    last_seen_time: float | None = None


class ExitFSM:
    """
    현관 카메라의 사람 중심점이
    실내선 → 실외선 순서로 이동한 뒤 사라지고,
    disappear_confirm_seconds 동안 재검출되지 않으면 외출 확정.
    """

    def __init__(
        self,
        indoor_line_x: int,
        outdoor_line_x: int,
        disappear_confirm_seconds: float,
        reset_seconds: float,
    ) -> None:
        self.indoor_line_x = indoor_line_x
        self.outdoor_line_x = outdoor_line_x
        self.disappear_confirm_seconds = disappear_confirm_seconds
        self.reset_seconds = reset_seconds

        self.state = ExitState.WAITING
        self.last_seen_time: Optional[float] = None
        self.last_center_x: Optional[float] = None
        self.outward_evidence = False
        self.cooldown_start: Optional[float] = None

    def update(self, center_x: float | None, now: float | None = None) -> list[FSMEvent]:
        now = now or time.time()
        events: list[FSMEvent] = []

        if center_x is not None:
            self.last_seen_time = now

            if self.state == ExitState.COOLDOWN:
                if self.cooldown_start and now - self.cooldown_start >= self.reset_seconds:
                    self._reset()
                return events

            # 실내선과 실외선의 대소관계가 어느 방향이든 대응.
            increasing_outward = self.outdoor_line_x > self.indoor_line_x

            on_indoor_side = (
                center_x <= self.indoor_line_x
                if increasing_outward
                else center_x >= self.indoor_line_x
            )
            crossed_outdoor = (
                center_x >= self.outdoor_line_x
                if increasing_outward
                else center_x <= self.outdoor_line_x
            )

            if on_indoor_side:
                self.state = ExitState.INDOOR_SIDE
                self.outward_evidence = False

            elif crossed_outdoor and self.state in {
                ExitState.INDOOR_SIDE,
                ExitState.OUTWARD_MOVING,
            }:
                self.state = ExitState.OUTWARD_MOVING
                self.outward_evidence = True
                events.append(FSMEvent("OUTWARD_CROSSING", now, self.last_seen_time))

            elif self.state == ExitState.INDOOR_SIDE:
                self.state = ExitState.OUTWARD_MOVING

            self.last_center_x = center_x
            return events

        # 미검출
        if (
            self.outward_evidence
            and self.last_seen_time is not None
            and self.state in {ExitState.OUTWARD_MOVING, ExitState.DISAPPEARED}
        ):
            self.state = ExitState.DISAPPEARED
            if now - self.last_seen_time >= self.disappear_confirm_seconds:
                self.state = ExitState.EXIT_CONFIRMED
                events.append(FSMEvent("EXIT_CONFIRMED", now, self.last_seen_time))
                self.state = ExitState.COOLDOWN
                self.cooldown_start = now

        return events

    def _reset(self) -> None:
        self.state = ExitState.WAITING
        self.last_seen_time = None
        self.last_center_x = None
        self.outward_evidence = False
        self.cooldown_start = None
