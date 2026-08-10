from __future__ import annotations

from datetime import datetime
from pathlib import Path
import signal
import time

import cv2
import sys
import yaml

from exit_system.best_frame import select_best_frame
from exit_system.cameras import PiCameraThread, USBCameraThread
from exit_system.clothing import analyze_clothing_colors
from exit_system.event_logger import save_event
from exit_system.retention import cleanup_old_exit_data
from exit_system.exit_fsm import ExitFSM
from exit_system.frame_buffer import SampledFrameBuffer
from exit_system.kakao import KakaoNotifier
from exit_system.vision import PersonTracker, choose_primary_person

RUNNING = True


def stop_handler(*_args):
    global RUNNING
    RUNNING = False


def load_config(path: str = "exit_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _is_point(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            for item in value
        )
    )


def _is_line(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and _is_point(value[0])
        and _is_point(value[1])
    )


def build_motion_config(config: dict, entrance_cfg: dict) -> dict:
    mobile = config.get("mobile_setup_calibration")

    if isinstance(mobile, dict):
        indoor_line = mobile.get("indoor_line")
        outdoor_line = mobile.get("outdoor_line")
        boundary_line = mobile.get("boundary_line")
        direction = mobile.get("exit_direction")
        roi = mobile.get("tracking_roi", [[0.05, 0.05], [0.95, 0.95]])

        geometry_valid = (
            _is_line(indoor_line)
            and _is_line(outdoor_line)
            and _is_line(boundary_line)
            and _is_point(direction)
            and _is_line(roi)
        )

        if geometry_valid:
            nx = float(direction[0])
            ny = float(direction[1])
            length = (nx * nx + ny * ny) ** 0.5

            if length > 1e-9:
                nx /= length
                ny /= length

                def projection(point) -> float:
                    return (
                        float(point[0]) * nx
                        + float(point[1]) * ny
                    )

                indoor_threshold = sum(
                    projection(point)
                    for point in indoor_line
                ) / 2.0

                outdoor_threshold = sum(
                    projection(point)
                    for point in outdoor_line
                ) / 2.0

                if outdoor_threshold > indoor_threshold:
                    print("[CALIBRATION] Mobile arbitrary-angle geometry enabled")

                    return {
                        "mode": "mobile",
                        "indoor_threshold": indoor_threshold,
                        "outdoor_threshold": outdoor_threshold,
                        "direction": (nx, ny),
                        "indoor_line": indoor_line,
                        "outdoor_line": outdoor_line,
                        "boundary_line": boundary_line,
                        "roi": roi,
                    }

    print("[CALIBRATION] Legacy vertical-line geometry enabled")

    return {
        "mode": "legacy",
        "indoor_threshold": float(
            entrance_cfg["indoor_line_x"]
        ),
        "outdoor_threshold": float(
            entrance_cfg["outdoor_line_x"]
        ),
        "roi": list(
            map(int, entrance_cfg["door_roi"])
        ),
    }


def _point_to_pixel(point, frame) -> tuple[int, int]:
    height, width = frame.shape[:2]

    x = int(round(float(point[0]) * (width - 1)))
    y = int(round(float(point[1]) * (height - 1)))

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    return x, y


def resolve_door_roi(
    motion_cfg: dict,
    frame,
) -> list[int]:
    if motion_cfg["mode"] == "legacy":
        return list(motion_cfg["roi"])

    height, width = frame.shape[:2]
    first, second = motion_cfg["roi"]

    x1 = int(round(
        min(float(first[0]), float(second[0]))
        * (width - 1)
    ))
    y1 = int(round(
        min(float(first[1]), float(second[1]))
        * (height - 1)
    ))
    x2 = int(round(
        max(float(first[0]), float(second[0]))
        * (width - 1)
    ))
    y2 = int(round(
        max(float(first[1]), float(second[1]))
        * (height - 1)
    ))

    x1 = max(0, min(width - 2, x1))
    y1 = max(0, min(height - 2, y1))
    x2 = max(x1 + 1, min(width - 1, x2))
    y2 = max(y1 + 1, min(height - 1, y2))

    return [x1, y1, x2, y2]


def resolve_motion_value(
    person,
    motion_cfg: dict,
    frame,
) -> float | None:
    if person is None:
        return None

    x1, y1, x2, y2 = person.bbox
    center_x = float((x1 + x2) / 2.0)

    if motion_cfg["mode"] == "legacy":
        return center_x

    center_y = float(y2)
    height, width = frame.shape[:2]

    normalized_x = center_x / max(width - 1, 1)
    normalized_y = center_y / max(height - 1, 1)

    nx, ny = motion_cfg["direction"]

    return (
        normalized_x * nx
        + normalized_y * ny
    )


def format_korean_time(dt: datetime) -> str:
    ampm = "오전" if dt.hour < 12 else "오후"
    hour = dt.hour % 12 or 12
    return f"{dt.year}년 {dt.month}월 {dt.day}일 {ampm} {hour}시 {dt.minute:02d}분"


def draw_debug(
    frame,
    cfg,
    motion_cfg,
    person,
    state_name: str,
    motion_value: float | None,
):
    if motion_cfg["mode"] == "mobile":
        x1, y1, x2, y2 = resolve_door_roi(
            motion_cfg,
            frame,
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

        line_specs = (
            (
                "boundary_line",
                (255, 255, 255),
            ),
            (
                "indoor_line",
                (0, 255, 0),
            ),
            (
                "outdoor_line",
                (0, 0, 255),
            ),
        )

        for key, color in line_specs:
            point_a, point_b = motion_cfg[key]

            cv2.line(
                frame,
                _point_to_pixel(point_a, frame),
                _point_to_pixel(point_b, frame),
                color,
                2,
            )

    else:
        x1, y1, x2, y2 = map(
            int,
            cfg["door_roi"],
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

        indoor_x = int(cfg["indoor_line_x"])
        outdoor_x = int(cfg["outdoor_line_x"])

        cv2.line(
            frame,
            (indoor_x, 0),
            (indoor_x, frame.shape[0]),
            (0, 255, 0),
            2,
        )

        cv2.line(
            frame,
            (outdoor_x, 0),
            (outdoor_x, frame.shape[0]),
            (0, 0, 255),
            2,
        )

    if person:
        px1, py1, px2, py2 = person.bbox

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (255, 0, 255),
            2,
        )

        x1, y1, x2, y2 = person.bbox
        cx, cy = int((x1 + x2) / 2), int(y2)

        cv2.circle(
            frame,
            (cx, cy),
            6,
            (0, 255, 255),
            -1,
        )

    cv2.putText(
        frame,
        f"STATE: {state_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    motion_text = "NONE" if motion_value is None else f"{motion_value:.3f}"
    cv2.putText(frame, f"MOTION: {motion_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame


def main() -> None:
    cfg = load_config()
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    entrance_cfg = cfg["entrance_camera"]
    living_cfg = cfg["living_camera"]
    model_cfg = cfg["model"]
    motion_cfg = build_motion_config(cfg, entrance_cfg)

    entrance_device = 8 if sys.platform.startswith("linux") else int(entrance_cfg.get("device", 0))
    entrance_camera = USBCameraThread(
        device=entrance_device,
        width=int(entrance_cfg["width"]),
        height=int(entrance_cfg["height"]),
        fps=int(entrance_cfg["fps"]),
    ).start()

    living_camera = None
    if bool(living_cfg.get("enabled", False)):
        try:
            living_camera = PiCameraThread(
                width=int(living_cfg["width"]),
                height=int(living_cfg["height"]),
                fps=int(living_cfg["fps"]),
            ).start()
        except RuntimeError as exc:
            print(f"[LIVING CAMERA DISABLED] {exc}")

    entrance_tracker = PersonTracker(
        model_cfg["path"],
        float(model_cfg["confidence"]),
        int(model_cfg["inference_size"]),
    )
    living_tracker = (
        PersonTracker(model_cfg["path"], float(model_cfg["confidence"]), int(model_cfg["inference_size"]))
        if living_camera else None
    )

    fsm = ExitFSM(
        indoor_line_x=float(motion_cfg["indoor_threshold"]),
        outdoor_line_x=float(motion_cfg["outdoor_threshold"]),
        disappear_confirm_seconds=float(cfg["fsm"]["disappear_confirm_seconds"]),
        reset_seconds=float(cfg["fsm"]["reset_seconds"]),
    )
    frame_buffer = SampledFrameBuffer(
        seconds=int(cfg["frame_buffer"]["seconds"]),
        sample_interval=float(cfg["frame_buffer"]["sample_interval_seconds"]),
    )
    notifier = KakaoNotifier(cfg["kakao"])

    frame_count = 0
    last_entrance_person = None
    last_detection_time = 0.0
    living_approach_started = None
    living_approach = False
    process_every = max(1, int(model_cfg["process_every_n_frames"]))
    track_timeout = float(cfg["fsm"].get("track_timeout_seconds", 1.0))
    approach_hold = float(cfg["fsm"].get("approach_hold_seconds", 0.8))
    show_preview = bool(entrance_cfg.get("show_preview", False))

    Path("data/images").mkdir(parents=True, exist_ok=True)

    try:
        while RUNNING:
            entrance_frame, entrance_ts = entrance_camera.read()
            if entrance_frame is None:
                time.sleep(0.02)
                continue

            now = time.time()
            frame_count += 1
            process_now = frame_count % process_every == 0

            if process_now and living_camera and living_tracker:
                living_frame, _ = living_camera.read()
                if living_frame is not None:
                    living_person = choose_primary_person(
                        living_tracker.track(living_frame), living_cfg["entrance_roi"]
                    )
                    if living_person is not None:
                        living_approach_started = living_approach_started or now
                        living_approach = now - living_approach_started >= approach_hold
                    else:
                        living_approach_started = None
                        living_approach = False

            if process_now:
                detected = choose_primary_person(
                    entrance_tracker.track(entrance_frame), resolve_door_roi(motion_cfg, entrance_frame)
                )
                last_entrance_person = detected
                if detected is not None:
                    last_detection_time = now

            # 처리 프레임 사이에서는 직전 검출을 잠시 유지하되, timeout 이후에는 반드시 미검출 처리.
            if last_entrance_person is not None and now - last_detection_time > track_timeout:
                last_entrance_person = None

            bbox = last_entrance_person.bbox if last_entrance_person else None
            motion_value = resolve_motion_value(
                last_entrance_person,
                motion_cfg,
                entrance_frame,
            )
            frame_buffer.maybe_add(entrance_frame, bbox, entrance_ts)

            for event in fsm.update(motion_value, now):
                if event.name != "EXIT_CONFIRMED":
                    continue

                last_seen = event.last_seen_time or event.timestamp
                candidates = frame_buffer.before(
                    last_seen, float(cfg["frame_buffer"]["candidate_lookback_seconds"])
                )
                best = select_best_frame(candidates, cfg["best_frame"])
                exit_dt = datetime.fromtimestamp(event.timestamp)
                destination = "LLM 모듈 연동 전"

                image_path = None
                clothing_text = "영상 상태로 인해 확인 불가"
                if best is not None and best.item.bbox is not None:
                    clothing = analyze_clothing_colors(
                        best.item.frame, best.item.bbox, cfg["clothing"]
                    )
                    clothing_text = clothing.summary
                    image_path = str(Path("data/images") / exit_dt.strftime("exit_%Y%m%d_%H%M%S.jpg"))
                    cv2.imwrite(image_path, best.item.frame)

                message = (
                    "🚨 외출이 감지되었습니다.\n"
                    f"시간: {format_korean_time(exit_dt)}\n"
                    f"외출 장소: {destination}\n"
                    f"착장: {clothing_text}"
                )
                payload = {
                    "event": "EXIT_CONFIRMED",
                    "exit_time": exit_dt.isoformat(timespec="seconds"),
                    "destination": destination,
                    "clothing": clothing_text,
                    "best_image_path": image_path,
                    "living_approach_detected": living_approach,
                }
                event_path = save_event(payload)
                deleted_images, deleted_events = cleanup_old_exit_data(7)
                if deleted_images or deleted_events:
                    print(f"[RETENTION] deleted images={deleted_images}, events={deleted_events}")
                notifier.send_to_me(message, image_path)
                print(f"[EVENT SAVED] {event_path}")
                print(message)

            if show_preview:
                preview = draw_debug(
                    entrance_frame.copy(),
                    entrance_cfg,
                    motion_cfg,
                    last_entrance_person,
                    fsm.state.name,
                motion_value,
                )
                cv2.imshow("Exit Detection", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.005)
    finally:
        entrance_camera.stop()
        if living_camera:
            living_camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
