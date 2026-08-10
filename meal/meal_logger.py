"""
식사 행동 인식 로그 저장 모듈 V3.

저장 파일
---------
1. <session_id>_events.jsonl
   - Bite/Drink 확정
   - Bite/Drink 세션 상태 변경
   - MealFSM 상태 변경
   - 식사 시작/종료 이벤트

2. <session_id>_frames.jsonl
   - 설정에 따라 프레임별 MealObservation 저장
   - Bite Session, Drink Session, Vessel 상태 포함

3. <session_id>_snapshot.json
   - 가장 최근 MealSessionSnapshot 저장

4. <session_id>_summary.json
   - 세션 종료 시 최종 요약 저장

특징
----
- V3의 이벤트 기반 Bite/Drink 구조를 그대로 저장한다.
- Enum, dataclass, Path, set 등을 JSON 안전 형식으로 변환한다.
- 로그 쓰기 실패가 메인 식사 기능을 중단시키지 않는다.
- 지정된 이벤트 수마다 자동 flush한다.
- 컨텍스트 매니저와 close()를 지원한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence, TextIO
import json
import os
import time
import traceback

from meal.meal_state import (
    EventType,
    MealEvent,
    MealObservation,
    MealSessionSnapshot,
    dataclass_to_dict,
)


class MealLoggerError(RuntimeError):
    """MealLogger 초기화 또는 강제 저장 실패 예외."""


class MealLogger:
    """V3 식사 세션 로그 저장기."""

    def __init__(
        self,
        session_id: str,
        config_path: str | Path = "meal/meal_config.json",
        log_directory: str | Path | None = None,
    ) -> None:
        if not str(session_id).strip():
            raise ValueError("session_id must not be empty")

        self.session_id = self._sanitize_name(str(session_id))
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        logger_config = self.config.get("logger", {})
        configured_directory = logger_config.get(
            "directory",
            "logs/meal",
        )
        self.log_directory = Path(
            log_directory
            if log_directory is not None
            else configured_directory
        )

        self.enabled = bool(
            logger_config.get("enabled", True)
        )
        self.save_frame_events = bool(
            logger_config.get(
                "save_frame_events",
                False,
            )
        )
        self.save_state_changes = bool(
            logger_config.get(
                "save_state_changes",
                True,
            )
        )
        self.save_confirmed_events = bool(
            logger_config.get(
                "save_confirmed_events",
                True,
            )
        )
        self.save_debug_values = bool(
            logger_config.get(
                "save_debug_values",
                True,
            )
        )
        self.flush_every_events = max(
            1,
            int(
                logger_config.get(
                    "flush_every_events",
                    1,
                )
            ),
        )
        self.snapshot_interval_seconds = max(
            0.2,
            float(
                logger_config.get(
                    "snapshot_interval_seconds",
                    1.0,
                )
            ),
        )
        self.atomic_replace_retries = max(
            0,
            int(
                logger_config.get(
                    "atomic_replace_retries",
                    4,
                )
            ),
        )
        self.atomic_replace_retry_delay = max(
            0.01,
            float(
                logger_config.get(
                    "atomic_replace_retry_delay",
                    0.08,
                )
            ),
        )

        self.events_path = (
            self.log_directory
            / f"{self.session_id}_events.jsonl"
        )
        self.frames_path = (
            self.log_directory
            / f"{self.session_id}_frames.jsonl"
        )
        self.snapshot_path = (
            self.log_directory
            / f"{self.session_id}_snapshot.json"
        )
        self.summary_path = (
            self.log_directory
            / f"{self.session_id}_summary.json"
        )
        self.error_path = (
            self.log_directory
            / f"{self.session_id}_logger_errors.log"
        )

        self._events_file: Optional[TextIO] = None
        self._frames_file: Optional[TextIO] = None
        self._event_since_flush = 0
        self._frame_since_flush = 0
        self._closed = False
        self._finalized = False
        self._lock = RLock()

        self.event_count = 0
        self.frame_count = 0
        self.write_error_count = 0
        self.started_wall_time = self._utc_now_iso()
        self.last_event_wall_time: Optional[str] = None
        self.last_frame_wall_time: Optional[str] = None
        self._last_snapshot_monotonic: Optional[float] = None
        self._last_snapshot_signature: Optional[
            tuple[Any, ...]
        ] = None

        if self.enabled:
            self._open_files()

    def __enter__(self) -> "MealLogger":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        exc_traceback: Any,
    ) -> None:
        self.close()

    def log_events(
        self,
        events: Sequence[MealEvent],
        snapshot: Optional[MealSessionSnapshot] = None,
    ) -> int:
        """이벤트 목록을 JSONL로 저장하고 저장 개수를 반환한다."""

        if not self.enabled or self._closed:
            return 0

        written = 0
        for event in events:
            if not self._should_save_event(event):
                continue
            if self.log_event(event):
                written += 1

        if snapshot is not None:
            self.save_snapshot(snapshot)

        return written

    def log_event(
        self,
        event: MealEvent,
    ) -> bool:
        """MealEvent 하나를 events JSONL에 저장한다."""

        if not self.enabled or self._closed:
            return False

        event.clamp()
        payload = {
            "record_type": "meal_event",
            "schema_version": "3.0",
            "session_id": self.session_id,
            "logged_at": self._utc_now_iso(),
            "event": self._safe_json_value(event),
        }

        success = self._write_jsonl(
            file=self._events_file,
            path=self.events_path,
            payload=payload,
            is_event=True,
        )
        if success:
            self.event_count += 1
            self.last_event_wall_time = payload["logged_at"]

        return success

    def log_observation(
        self,
        observation: MealObservation,
        snapshot: Optional[MealSessionSnapshot] = None,
    ) -> bool:
        """설정이 켜진 경우 프레임별 V3 관찰 결과를 저장한다."""

        if (
            not self.enabled
            or self._closed
            or not self.save_frame_events
        ):
            return False

        payload = self._observation_payload(
            observation=observation,
            snapshot=snapshot,
        )

        success = self._write_jsonl(
            file=self._frames_file,
            path=self.frames_path,
            payload=payload,
            is_event=False,
        )
        if success:
            self.frame_count += 1
            self.last_frame_wall_time = payload["logged_at"]

        return success

    def log_update(
        self,
        observation: MealObservation,
        events: Sequence[MealEvent],
        snapshot: MealSessionSnapshot,
    ) -> Dict[str, int | bool]:
        """
        메인 루프에서 한 번에 호출하는 통합 저장 함수.

        반환값:
        {
            "events_written": int,
            "frame_written": bool,
            "snapshot_written": bool
        }
        """

        events_written = self.log_events(events)
        frame_written = self.log_observation(
            observation=observation,
            snapshot=snapshot,
        )
        snapshot_written = self._save_snapshot_if_needed(
            snapshot=snapshot,
            force=bool(events) or snapshot.finished,
        )

        if snapshot.finished and not self._finalized:
            self.finalize(
                snapshot=snapshot,
                extra_summary={
                    "final_observation": (
                        observation.as_summary()
                    ),
                },
            )

        return {
            "events_written": events_written,
            "frame_written": frame_written,
            "snapshot_written": snapshot_written,
        }

    def _save_snapshot_if_needed(
        self,
        snapshot: MealSessionSnapshot,
        force: bool = False,
    ) -> bool:
        """
        Snapshot을 매 프레임 저장하지 않는다.

        - 상태·카운트 이벤트가 있으면 즉시 저장
        - 그 외에는 기본 1초 주기로 저장
        """
        now = time.monotonic()
        signature = (
            snapshot.meal_state.value,
            snapshot.bite_state.value,
            snapshot.drink_state.value,
            int(snapshot.bite_count),
            int(snapshot.drink_count),
            bool(snapshot.finished),
        )
        changed = (
            signature != self._last_snapshot_signature
        )
        interval_passed = (
            self._last_snapshot_monotonic is None
            or now - self._last_snapshot_monotonic
            >= self.snapshot_interval_seconds
        )

        if not (
            force
            or changed
            or interval_passed
        ):
            return False

        success = self.save_snapshot(snapshot)
        if success:
            self._last_snapshot_monotonic = now
            self._last_snapshot_signature = signature
        return success

    def save_snapshot(
        self,
        snapshot: MealSessionSnapshot,
    ) -> bool:
        """최신 세션 상태를 원자적으로 JSON 파일에 저장한다."""

        if not self.enabled or self._closed:
            return False

        snapshot.clamp()
        payload = {
            "record_type": "meal_session_snapshot",
            "schema_version": "3.0",
            "session_id": self.session_id,
            "saved_at": self._utc_now_iso(),
            "snapshot": self._safe_json_value(snapshot),
            "logger": self.status(),
        }

        return self._atomic_write_json(
            self.snapshot_path,
            payload,
        )

    def finalize(
        self,
        snapshot: MealSessionSnapshot,
        extra_summary: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> bool:
        """최종 세션 요약을 저장하고 모든 버퍼를 flush한다."""

        if not self.enabled:
            return False

        if self._finalized:
            return True

        snapshot.clamp()
        duration = max(
            0.0,
            float(snapshot.updated_at)
            - float(snapshot.started_at),
        )

        bites_per_minute = (
            snapshot.bite_count / (duration / 60.0)
            if duration > 0.0
            else 0.0
        )
        drinks_per_minute = (
            snapshot.drink_count / (duration / 60.0)
            if duration > 0.0
            else 0.0
        )

        payload: Dict[str, Any] = {
            "record_type": "meal_session_summary",
            "schema_version": "3.0",
            "session_id": self.session_id,
            "saved_at": self._utc_now_iso(),
            "started_at": snapshot.started_at,
            "finished_at": snapshot.updated_at,
            "duration_seconds": duration,
            "meal_state": snapshot.meal_state.value,
            "bite_state": snapshot.bite_state.value,
            "drink_state": snapshot.drink_state.value,
            "bite_count": snapshot.bite_count,
            "drink_count": snapshot.drink_count,
            "bites_per_minute": round(
                bites_per_minute,
                4,
            ),
            "drinks_per_minute": round(
                drinks_per_minute,
                4,
            ),
            "last_bite_time": snapshot.last_bite_time,
            "last_drink_time": snapshot.last_drink_time,
            "last_activity_time": (
                snapshot.last_activity_time
            ),
            "finished": snapshot.finished,
            "logger": self.status(),
        }

        if extra_summary:
            payload["extra"] = self._safe_json_value(
                dict(extra_summary)
            )

        self.flush()
        success = self._atomic_write_json(
            self.summary_path,
            payload,
        )
        self.flush()

        if success:
            self._finalized = True

        return success

    def flush(self) -> None:
        """열려 있는 로그 파일 버퍼를 디스크에 반영한다."""

        with self._lock:
            for file in (
                self._events_file,
                self._frames_file,
            ):
                if file is None or file.closed:
                    continue

                try:
                    file.flush()
                    os.fsync(file.fileno())
                except OSError as error:
                    self._record_error(
                        "flush",
                        error,
                    )

            self._event_since_flush = 0
            self._frame_since_flush = 0

    def close(self) -> None:
        """flush 후 모든 파일을 안전하게 닫는다."""

        with self._lock:
            if self._closed:
                return

            self.flush()

            for file in (
                self._events_file,
                self._frames_file,
            ):
                if file is None or file.closed:
                    continue

                try:
                    file.close()
                except OSError as error:
                    self._record_error(
                        "close",
                        error,
                    )

            self._events_file = None
            self._frames_file = None
            self._closed = True

    def status(self) -> Dict[str, Any]:
        """현재 Logger 상태를 반환한다."""

        return {
            "enabled": self.enabled,
            "closed": self._closed,
            "finalized": self._finalized,
            "event_count": self.event_count,
            "frame_count": self.frame_count,
            "write_error_count": self.write_error_count,
            "events_path": str(self.events_path),
            "frames_path": str(self.frames_path),
            "snapshot_path": str(self.snapshot_path),
            "summary_path": str(self.summary_path),
            "started_wall_time": self.started_wall_time,
            "last_event_wall_time": (
                self.last_event_wall_time
            ),
            "last_frame_wall_time": (
                self.last_frame_wall_time
            ),
        }

    def _open_files(self) -> None:
        try:
            self.log_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._events_file = self.events_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )

            if self.save_frame_events:
                self._frames_file = self.frames_path.open(
                    "a",
                    encoding="utf-8",
                    buffering=1,
                )

        except OSError as error:
            self._closed = True
            raise MealLoggerError(
                f"Failed to open meal log files: {error}"
            ) from error

    def _write_jsonl(
        self,
        file: Optional[TextIO],
        path: Path,
        payload: Mapping[str, Any],
        is_event: bool,
    ) -> bool:
        with self._lock:
            if file is None or file.closed:
                self._record_error(
                    "write_jsonl",
                    RuntimeError(
                        f"Log file is not open: {path}"
                    ),
                )
                return False

            try:
                line = json.dumps(
                    self._safe_json_value(payload),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                file.write(line + "\n")

                if is_event:
                    self._event_since_flush += 1
                    if (
                        self._event_since_flush
                        >= self.flush_every_events
                    ):
                        file.flush()
                        os.fsync(file.fileno())
                        self._event_since_flush = 0
                else:
                    self._frame_since_flush += 1
                    if (
                        self._frame_since_flush
                        >= self.flush_every_events
                    ):
                        file.flush()
                        os.fsync(file.fileno())
                        self._frame_since_flush = 0

                return True

            except (
                OSError,
                TypeError,
                ValueError,
            ) as error:
                self._record_error(
                    "write_jsonl",
                    error,
                )
                return False

    def _atomic_write_json(
        self,
        path: Path,
        payload: Mapping[str, Any],
    ) -> bool:
        """
        Windows 파일 잠금에 대응하는 JSON 저장.

        1. 임시 파일 작성
        2. os.replace 재시도
        3. 계속 잠겨 있으면 대상 파일 직접 덮어쓰기
        """
        with self._lock:
            temporary = path.with_suffix(
                path.suffix + ".tmp"
            )
            safe_payload = self._safe_json_value(
                payload
            )

            try:
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with temporary.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        safe_payload,
                        file,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())

                last_error: Optional[
                    PermissionError
                ] = None

                for attempt in range(
                    self.atomic_replace_retries + 1
                ):
                    try:
                        os.replace(
                            temporary,
                            path,
                        )
                        return True
                    except PermissionError as error:
                        last_error = error
                        if (
                            attempt
                            >= self.atomic_replace_retries
                        ):
                            break
                        time.sleep(
                            self.atomic_replace_retry_delay
                            * (attempt + 1)
                        )

                # VS Code, 백신 등이 기존 JSON을 잠근 경우의 fallback
                try:
                    with path.open(
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(
                            safe_payload,
                            file,
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )
                        file.write("\n")
                        file.flush()
                        os.fsync(file.fileno())

                    try:
                        temporary.unlink(
                            missing_ok=True
                        )
                    except OSError:
                        pass

                    return True

                except (
                    OSError,
                    TypeError,
                    ValueError,
                ) as fallback_error:
                    self._record_error(
                        "atomic_write_json_fallback",
                        fallback_error,
                    )
                    if last_error is not None:
                        self._record_error(
                            "atomic_write_json_replace",
                            last_error,
                        )
                    return False

            except (
                OSError,
                TypeError,
                ValueError,
            ) as error:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

                self._record_error(
                    "atomic_write_json",
                    error,
                )
                return False

    def _observation_payload(
        self,
        observation: MealObservation,
        snapshot: Optional[MealSessionSnapshot],
    ) -> Dict[str, Any]:
        observation.clamp()

        payload: Dict[str, Any] = {
            "record_type": "meal_frame",
            "schema_version": "3.0",
            "session_id": self.session_id,
            "logged_at": self._utc_now_iso(),
            "timestamp": observation.timestamp,
            "frame_index": observation.frame_index,
            "summary": observation.as_summary(),
            "sessions": {
                "bite": self._safe_json_value(
                    observation.bite_session
                ),
                "drink": self._safe_json_value(
                    observation.drink_session
                ),
            },
            "vessel": self._safe_json_value(
                observation.vessel
            ),
        }

        if self.save_debug_values:
            payload["observation"] = (
                self._safe_json_value(observation)
            )

        if snapshot is not None:
            snapshot.clamp()
            payload["snapshot"] = (
                self._safe_json_value(snapshot)
            )

        return payload

    def _should_save_event(
        self,
        event: MealEvent,
    ) -> bool:
        confirmed_types = {
            EventType.BITE_CONFIRMED,
            EventType.DRINK_CONFIRMED,
            EventType.SESSION_STARTED,
            EventType.SESSION_FINISHED,
        }
        state_change_types = {
            EventType.BITE_STATE_CHANGED,
            EventType.DRINK_STATE_CHANGED,
            EventType.MEAL_STATE_CHANGED,
            EventType.REST_STARTED,
            EventType.END_CANDIDATE_STARTED,
        }

        if event.event_type in confirmed_types:
            return self.save_confirmed_events

        if event.event_type in state_change_types:
            return self.save_state_changes

        return True

    def _record_error(
        self,
        operation: str,
        error: BaseException,
    ) -> None:
        self.write_error_count += 1

        try:
            self.log_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.error_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        {
                            "logged_at": self._utc_now_iso(),
                            "operation": operation,
                            "error_type": (
                                type(error).__name__
                            ),
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        except OSError:
            # Logger 오류 기록 자체의 실패는 메인 기능에 전달하지 않는다.
            pass

    @classmethod
    def _safe_json_value(
        cls,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "__dataclass_fields__"):
            return cls._safe_json_value(
                dataclass_to_dict(value)
            )

        if isinstance(value, Mapping):
            return {
                str(cls._safe_json_value(key)):
                cls._safe_json_value(item)
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set, frozenset),
        ):
            return [
                cls._safe_json_value(item)
                for item in value
            ]

        if isinstance(value, float):
            if value != value:
                return None
            if value in {
                float("inf"),
                float("-inf"),
            }:
                return None
            return value

        if isinstance(value, (str, int, bool)):
            return value

        if hasattr(value, "item"):
            try:
                return cls._safe_json_value(
                    value.item()
                )
            except (TypeError, ValueError):
                pass

        return str(value)

    @staticmethod
    def _load_config(
        path: Path,
    ) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Meal config file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            value = json.load(file)

        if not isinstance(value, dict):
            raise ValueError(
                "meal_config.json root must be an object"
            )

        return value

    @staticmethod
    def _sanitize_name(
        value: str,
    ) -> str:
        allowed = []

        for character in value.strip():
            if (
                character.isalnum()
                or character in {"-", "_"}
            ):
                allowed.append(character)
            else:
                allowed.append("_")

        sanitized = "".join(allowed).strip("_")
        return sanitized or "meal_session"

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()


__all__ = [
    "MealLogger",
    "MealLoggerError",
]