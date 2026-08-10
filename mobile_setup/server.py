from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml
from flask import Flask, Response, jsonify, redirect, render_template, request

from mobile_setup.calibration import build_calibration


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "exit_config.yaml"
RUN_EXIT_PATH = BASE_DIR / "run_exit.py"

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_exit_process: subprocess.Popen[Any] | None = None
_setup_camera = None


@app.get("/")
def index():
    return redirect("/setup-v2")


@app.get("/setup")
def setup():
    return render_template("setup.html")


@app.get("/setup-v2")
def setup_v2():
    return render_template("setup_v2.html")


@app.get("/camera/frame.jpg")
def camera_frame():
    global _setup_camera

    try:
        if _setup_camera is None or not _setup_camera.isOpened():
            device = 0 if sys.platform.startswith("win") else 8
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
            _setup_camera = cv2.VideoCapture(device, backend)
            _setup_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _setup_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ok, frame = _setup_camera.read()
        if not ok:
            raise RuntimeError("노트북 카메라 프레임을 읽지 못했습니다.")

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("JPEG 변환에 실패했습니다.")

        return Response(
            encoded.tobytes(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    except Exception as error:
        return jsonify({
            "message": "카메라 영상을 가져오지 못했습니다.",
            "error": str(error),
        }), 503

@app.get("/api/status")
def api_status():
    return jsonify(
        {
            "ok": True,
            "service": "mobile_setup",
            "version": 2,
            "steps": [
                "kakao",
                "camera",
                "entrance",
            ],
        }
    )


@app.post("/api/setup/entrance/validate")
def validate_entrance():
    payload = request.get_json(silent=True) or {}
    result = build_calibration(payload)

    return jsonify(result), 200 if result["ok"] else 400


@app.post("/api/setup/entrance")
def save_entrance():
    payload = request.get_json(silent=True) or {}
    result = build_calibration(payload)

    if not result["ok"]:
        return jsonify(result), 400

    current_config: dict[str, Any] = {}

    if CONFIG_PATH.exists():
        try:
            loaded = yaml.safe_load(
                CONFIG_PATH.read_text(encoding="utf-8-sig")
            )

            if isinstance(loaded, dict):
                current_config = loaded

        except (OSError, yaml.YAMLError) as exc:
            return (
                jsonify(
                    {
                        "ok": False,
                        "code": "CONFIG_READ_ERROR",
                        "message": f"기존 설정 파일을 읽을 수 없습니다: {exc}",
                    }
                ),
                500,
            )

        backup_name = (
            f"exit_config_before_mobile_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )

        try:
            shutil.copy2(
                CONFIG_PATH,
                BASE_DIR / backup_name,
            )
        except OSError as exc:
            return (
                jsonify(
                    {
                        "ok": False,
                        "code": "CONFIG_BACKUP_ERROR",
                        "message": f"설정 파일 백업에 실패했습니다: {exc}",
                    }
                ),
                500,
            )

    current_config["mobile_setup_calibration"] = {
        "version": 1,
        "source": "mobile_single_boundary",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **result["geometry"],
    }

    temporary_path = CONFIG_PATH.with_suffix(".yaml.tmp")

    try:
        temporary_path.write_text(
            yaml.safe_dump(
                current_config,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(CONFIG_PATH)

    except OSError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "CONFIG_WRITE_ERROR",
                    "message": f"설정 파일 저장에 실패했습니다: {exc}",
                }
            ),
            500,
        )

    return jsonify(
        {
            "ok": True,
            "message": "현관 출입 설정이 저장되었습니다.",
            "geometry": result["geometry"],
            "warnings": result["warnings"],
            "saved_section": "mobile_setup_calibration",
        }
    )


@app.post("/api/system/start")
def start_system():
    global _exit_process

    if os.environ.get("MOBILE_SETUP_ALLOW_START") != "1":
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "START_DISABLED",
                    "message": (
                        "설정은 저장되었지만 현재 노트북 테스트에서는 "
                        "감지 프로그램 자동 실행이 비활성화되어 있습니다."
                    ),
                }
            ),
            409,
        )

    if _exit_process is not None and _exit_process.poll() is None:
        return jsonify(
            {
                "ok": True,
                "already_running": True,
                "message": "외출 감지 프로그램이 이미 실행 중입니다.",
            }
        )

    if not RUN_EXIT_PATH.exists():
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "RUN_EXIT_NOT_FOUND",
                    "message": "run_exit.py 파일을 찾을 수 없습니다.",
                }
            ),
            404,
        )

    try:
        _exit_process = subprocess.Popen(
            [
                sys.executable,
                str(RUN_EXIT_PATH),
            ],
            cwd=str(BASE_DIR),
        )

    except OSError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "START_FAILED",
                    "message": f"외출 감지 실행에 실패했습니다: {exc}",
                }
            ),
            500,
        )

    return jsonify(
        {
            "ok": True,
            "pid": _exit_process.pid,
            "message": "외출 감지 프로그램을 시작했습니다.",
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
    )
