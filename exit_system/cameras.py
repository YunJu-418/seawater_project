from __future__ import annotations

import threading
import time
import sys
from typing import Optional

import cv2
import numpy as np


class LatestFrame:
    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()
            self._timestamp = time.time()

    def get(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._frame is None:
                return None, self._timestamp
            return self._frame.copy(), self._timestamp


class USBCameraThread:
    def __init__(self, device: int, width: int, height: int, fps: int) -> None:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
        self.capture = cv2.VideoCapture(device, backend)
        if not self.capture.isOpened():
            raise RuntimeError(f"USB 카메라를 열 수 없습니다: /dev/video{device}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

        self.latest = LatestFrame()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "USBCameraThread":
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                self.latest.set(frame)
            else:
                time.sleep(0.05)

    def read(self) -> tuple[Optional[np.ndarray], float]:
        return self.latest.get()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.capture.release()


class PiCameraThread:
    """Raspberry Pi Camera Module 3용 Picamera2 어댑터."""

    def __init__(self, width: int, height: int, fps: int) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2가 설치되지 않았습니다. Raspberry Pi OS에서 "
                "`sudo apt install python3-picamera2`를 실행하세요."
            ) from exc

        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.camera.configure(config)
        self.latest = LatestFrame()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "PiCameraThread":
        self.camera.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self) -> None:
        while self.running:
            rgb = self.camera.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.latest.set(bgr)

    def read(self) -> tuple[Optional[np.ndarray], float]:
        return self.latest.get()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.camera.stop()
