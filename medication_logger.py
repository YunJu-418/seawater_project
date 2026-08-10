import json
import os
from datetime import datetime, time


class MedicationLogger:
    def __init__(self, log_path="logs/medication_log.json"):
        self.log_path = log_path

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        if not os.path.exists(self.log_path):
            self._write_logs([])

        # 아침 / 점심 / 저녁 복약 시간대
        self.meal_times = {
            "morning": {
                "label": "아침",
                "start": time(6, 0),
                "end": time(10, 0)
            },
            "lunch": {
                "label": "점심",
                "start": time(11, 0),
                "end": time(14, 0)
            },
            "dinner": {
                "label": "저녁",
                "start": time(17, 0),
                "end": time(21, 0)
            }
        }

    def _read_logs(self):
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_logs(self, logs):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

    def reset_logs(self):
        """
        테스트용 로그 초기화.
        실제 main.py에서는 사용하지 않는 것을 권장.
        """
        self._write_logs([])

    def get_current_meal(self, now=None):
        """
        현재 시간이 아침/점심/저녁 복약 시간대인지 확인.
        해당 시간이 아니면 None 반환.
        """
        if now is None:
            now = datetime.now()

        current_time = now.time()

        for meal_key, meal_info in self.meal_times.items():
            if meal_info["start"] <= current_time <= meal_info["end"]:
                return meal_key

        return None

    def has_taken_today(self, meal_key, now=None):
        """
        오늘 해당 시간대 약을 이미 복용 기록했는지 확인.
        """
        if now is None:
            now = datetime.now()

        today = now.strftime("%Y-%m-%d")
        logs = self._read_logs()

        for log in logs:
            if (
                log.get("date") == today
                and log.get("meal") == meal_key
                and log.get("event") == "MEDICATION_TAKEN"
            ):
                return True

        return False

    def log_if_taken(self, vision_state, now=None):
        """
        vision_state가 MEDICATION_DONE일 때만 복용 기록 저장.

        반환 log_state:
        - LOGGED: 새 복용 기록 저장 완료
        - ALREADY_LOGGED: 오늘 해당 시간대 복용 기록이 이미 있음
        - NOT_DONE: MEDICATION_DONE 상태가 아님
        - NOT_MEDICATION_TIME: 아침/점심/저녁 복약 시간이 아님
        """
        if now is None:
            now = datetime.now()

        if vision_state != "MEDICATION_DONE":
            return {
                "log_state": "NOT_DONE",
                "meal": None,
                "meal_label": None,
                "taken_time": None,
                "vision_state": vision_state
            }

        meal_key = self.get_current_meal(now)

        if meal_key is None:
            return {
                "log_state": "NOT_MEDICATION_TIME",
                "meal": None,
                "meal_label": None,
                "taken_time": None,
                "vision_state": vision_state
            }

        meal_label = self.meal_times[meal_key]["label"]

        if self.has_taken_today(meal_key, now):
            return {
                "log_state": "ALREADY_LOGGED",
                "meal": meal_key,
                "meal_label": meal_label,
                "taken_time": None,
                "vision_state": vision_state
            }

        taken_time = now.strftime("%Y-%m-%d %H:%M:%S")

        logs = self._read_logs()
        logs.append({
            "date": now.strftime("%Y-%m-%d"),
            "time": taken_time,
            "meal": meal_key,
            "meal_label": meal_label,
            "event": "MEDICATION_TAKEN",
            "vision_state": vision_state
        })
        self._write_logs(logs)

        return {
            "log_state": "LOGGED",
            "meal": meal_key,
            "meal_label": meal_label,
            "taken_time": taken_time,
            "vision_state": vision_state
        }

    def get_today_logs(self, now=None):
        """
        오늘 복용 기록만 반환.
        LLM 담당자가 오늘 아침/점심/저녁 복용 여부를 확인할 때 사용 가능.
        """
        if now is None:
            now = datetime.now()

        today = now.strftime("%Y-%m-%d")
        logs = self._read_logs()

        return [
            log for log in logs
            if log.get("date") == today
        ]
