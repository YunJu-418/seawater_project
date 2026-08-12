from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import numpy as np


@dataclass
class BufferedFrame:
    timestamp: float
    frame: np.ndarray
    bbox: tuple[int, int, int, int] | None


class SampledFrameBuffer:
    def __init__(self, seconds: int, sample_interval: float) -> None:
        max_items = max(2, int(seconds / sample_interval) + 2)
        self.items: deque[BufferedFrame] = deque(maxlen=max_items)
        self.sample_interval = sample_interval
        self.last_sample_time = 0.0

    def maybe_add(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
        timestamp: float | None = None,
    ) -> None:
        now = timestamp or time.time()
        if now - self.last_sample_time < self.sample_interval:
            return
        self.items.append(BufferedFrame(now, frame.copy(), bbox))
        self.last_sample_time = now

    def before(self, end_time: float, lookback_seconds: float) -> list[BufferedFrame]:
        start_time = end_time - lookback_seconds
        return [
            item
            for item in self.items
            if start_time <= item.timestamp <= end_time and item.bbox is not None
        ]
