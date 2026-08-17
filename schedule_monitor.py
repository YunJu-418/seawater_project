"""
시간 기반 이벤트 발생원 — 스케줄 모니터.

비전이 아니라 '시계'가 발생시키는 이벤트를 담당한다:
  - 시간대(아침/점심/저녁) 종료가 다가오는데 약 미복용   → MEDICATION_MISSED
  - 시간대 종료가 다가오는데 식사 기록 없음              → MEAL_MISSED

또한 어댑터(llm/adapters.py)가 "지금이 어느 시간대인가",
"이 시간대에 이미 복용/식사했는가"를 물어볼 때의 조회처 역할도 한다.

시간대 정의는 medication-detector 브랜치 MedicationLogger.meal_times 와
동일하게 맞춰 두었다 (아침 06~10 / 점심 11~14 / 저녁 17~21).
가능하면 팀원의 MedicationLogger 객체를 medication_logger 인자로 넘겨서
한 곳(팀원 코드)의 시간표를 단일 기준으로 쓰는 것을 권장한다.

사용 예 (아무 메인 루프에서나):
    monitor = ScheduleMonitor()               # 또는 ScheduleMonitor(medication_logger=medication_logger)
    ...
    guidance.notify_all(monitor.tick())       # 매 프레임 호출해도 됨 (내부에서 하루 슬롯당 1회 제한)
"""

import json
import os
from datetime import datetime, time, timedelta

import config
from llm.events import GuidanceEvent


# (키, 라벨, 시작, 끝) — MedicationLogger.meal_times 와 동일
DEFAULT_SLOTS = [
    ("morning", "아침", time(6, 0), time(10, 0)),
    ("lunch", "점심", time(11, 0), time(14, 0)),
    ("dinner", "저녁", time(17, 0), time(21, 0)),
]

_DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "medication_log.json"
)
_DEFAULT_MEAL_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "meal_log.json"
)


def current_slot(now=None, slots=DEFAULT_SLOTS):
    """현재 시각이 속한 시간대. 없으면 (None, None, None)."""
    now = now or datetime.now()
    t = now.time()
    for key, label, start, end in slots:
        if start <= t <= end:
            return key, label, end
    return None, None, None


def rotate_log(path: str, keep_days: int | None = None, now=None) -> int:
    """
    로그 파일 로테이션 — keep_days보다 오래된 항목을 월별 보관 파일로 이동.

    활성 파일은 매 기록마다 전체를 읽고 다시 쓰는 구조라, 몇 년치가 쌓이면
    라즈베리파이에서 느려진다. 오래된 기록은 지우지 않고
    logs/archive/<파일명>_YYYY-MM.json 으로 옮겨 보존한다 (보호자 통계 활용 가능).

    Returns:
        보관 파일로 이동한 항목 수.
    """
    keep_days = config.LOG_RETENTION_DAYS if keep_days is None else keep_days
    if keep_days <= 0:
        return 0
    now = now or datetime.now()
    cutoff = (now - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    try:
        with open(path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        return 0

    keep, old = [], []
    for log in logs:
        (old if str(log.get("date", "")) < cutoff else keep).append(log)
    if not old:
        return 0

    archive_dir = os.path.join(os.path.dirname(path), "archive")
    os.makedirs(archive_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    by_month = {}
    for log in old:
        by_month.setdefault(str(log.get("date", "0000-00"))[:7], []).append(log)
    for month, items in by_month.items():
        apath = os.path.join(archive_dir, f"{base}_{month}.json")
        try:
            with open(apath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
        existing.extend(items)
        with open(apath, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    print(f"[ScheduleMonitor] 로그 로테이션: {os.path.basename(path)} "
          f"{len(old)}건 → archive/ 이동 (최근 {keep_days}일 유지)")
    return len(old)


def next_slot(now=None, slots=DEFAULT_SLOTS):
    """지금 이후 가장 가까운 시간대 (라벨, 시작 시각 문자열). 오늘 남은 게 없으면 내일 첫 시간대."""
    now = now or datetime.now()
    t = now.time()
    for _key, label, start, _end in slots:
        if t < start:
            return label, start.strftime("%H:%M")
    _key, label, start, _end = slots[0]
    return label, start.strftime("%H:%M")  # 내일 첫 시간대


class ScheduleMonitor:
    def __init__(
        self,
        medication_logger=None,
        medication_log_path: str = _DEFAULT_LOG_PATH,
        meal_log_path: str = _DEFAULT_MEAL_LOG_PATH,
        slots=DEFAULT_SLOTS,
        remind_before_end_min: float | None = None,
    ):
        """
        medication_logger: 팀원 MedicationLogger 객체 (duck typing).
                           주면 시간대/복용 여부를 팀원 코드 기준으로 조회한다.
        medication_log_path: logger 미제공 시 직접 읽을 복용 기록 JSON 경로.
        remind_before_end_min: 시간대 종료 몇 분 전부터 미복용/미식사를 알릴지.
                               None이면 config.GUIDANCE_REMIND_BEFORE_END_MIN.
        """
        self._logger = medication_logger
        self._log_path = medication_log_path
        self._meal_log_path = meal_log_path
        self._slots = slots
        self._before_min = (
            config.GUIDANCE_REMIND_BEFORE_END_MIN
            if remind_before_end_min is None
            else remind_before_end_min
        )
        # 시작 시 1회 + 이후 날짜가 바뀔 때마다(tick에서 감지) 오래된 기록을
        # 보관 파일로 이동. 24시간 켜두는 운용에서도 로테이션이 계속 동작한다.
        rotate_log(self._meal_log_path)
        rotate_log(self._log_path)
        self._last_rotate_date = datetime.now().strftime("%Y-%m-%d")

        # {(date, slot_key): "HH:MM"} — 식사 완료 기록.
        # 메모리에만 두면 재시작 시 사라지고 대화(챗)에서 조회도 못 하므로
        # meal_log.json 파일에도 함께 남기고, 시작 시 파일에서 복원한다.
        self._meal_done: dict = self._load_meal_log()
        self._handled: set = set()     # {(date, slot_key, kind)} — 알림/확인 완료 (하루 슬롯당 1회)

    # ---------------- 조회 (어댑터가 사용) ----------------

    def slot_status(self, now=None):
        """(slot_key, 라벨, 이 시간대 복용 여부). 시간대 밖이면 (None, None, False)."""
        now = now or datetime.now()
        key, label, _end = current_slot(now, self._slots)
        if key is None:
            return None, None, False
        return key, label, self._medication_taken(key, now)

    def next_slot(self, now=None):
        """지금 이후 가장 가까운 시간대 (라벨, 시작 시각). 어댑터의 시간대 밖 안내용."""
        return next_slot(now, self._slots)

    def meal_done_in_current_slot(self, now=None) -> bool:
        """지금 시간대에 이미 식사를 마쳤는가. (시간대 밖이면 항상 False — 식사는 시간대 제한 없음)"""
        now = now or datetime.now()
        key, _label, _end = current_slot(now, self._slots)
        if key is None:
            return False
        return (now.strftime("%Y-%m-%d"), key) in self._meal_done

    # ---------------- 기록 (식사 어댑터가 호출) ----------------

    def record_meal_done(self, now=None):
        """식사 종료 확정(SESSION_FINISHED) 시 호출 — 현재 시간대에 식사 완료 표시.

        메모리와 meal_log.json 파일 양쪽에 기록한다 (재시작 유지 + 대화 조회용).
        """
        now = now or datetime.now()
        key, label, _end = current_slot(now, self._slots)
        if key is None:
            return
        date = now.strftime("%Y-%m-%d")
        if (date, key) in self._meal_done:
            return  # 같은 시간대 중복 기록 방지
        hhmm = now.strftime("%H:%M")
        self._meal_done[(date, key)] = hhmm
        self._append_meal_log({
            "date": date, "time": hhmm,
            "meal": key, "meal_label": label,
            "event": "MEAL_DONE",
        })

    def slots_report(self, kind: str = "medication", now=None) -> list:
        """
        오늘의 시간대별 현황 (대화 시스템의 '사실 근거'로 사용).

        Returns:
            [{"key", "label", "taken", "time", "phase"}, ...]
            phase ∈ {"지남", "진행 중", "예정"} — 지금 기준 시간대 상태
        """
        now = now or datetime.now()
        date = now.strftime("%Y-%m-%d")
        t = now.time()

        if kind == "medication":
            entries = self._medication_entries_today(now)
        else:
            entries = {k: v for (d, k), v in self._meal_done.items() if d == date}

        report = []
        for key, label, start, end in self._slots:
            phase = "지남" if t > end else ("예정" if t < start else "진행 중")
            report.append({
                "key": key, "label": label,
                "taken": key in entries, "time": entries.get(key),
                "phase": phase,
            })
        return report

    # ---------------- 시간 기반 이벤트 발생 ----------------

    def tick(self, now=None):
        """
        메인 루프에서 주기적으로 호출한다 (매 프레임도 무방).

        시간대 종료 remind_before_end_min 분 전부터, 아직 안 한 항목을 안내:
            약 미복용   → (MEDICATION_MISSED, {시간대, 종료 시각})
            식사 기록 X → (MEAL_MISSED, {시간대, 종료 시각})
        같은 날 같은 시간대에는 각 1회만 발생한다.

        Returns:
            [(GuidanceEvent, context dict), ...]  (없으면 빈 리스트)
        """
        now = now or datetime.now()

        # 자정을 넘겨 날짜가 바뀌면 하루 1회 로그 로테이션 (재시작 없는 24시간 운용 대비)
        today = now.strftime("%Y-%m-%d")
        if today != self._last_rotate_date:
            self._last_rotate_date = today
            rotate_log(self._meal_log_path, now=now)
            rotate_log(self._log_path, now=now)

        key, label, end = current_slot(now, self._slots)
        if key is None:
            return []

        end_dt = datetime.combine(now.date(), end)
        remaining_min = (end_dt - now).total_seconds() / 60.0
        if remaining_min > self._before_min:
            return []

        date = now.strftime("%Y-%m-%d")
        context = {
            "meal_label": label,
            "slot_ends_at": end.strftime("%H:%M"),
            "time": now.strftime("%H:%M"),
        }
        out = []

        med_id = (date, key, "medication")
        if med_id not in self._handled:
            self._handled.add(med_id)  # 복용했든 안내했든 이 시간대는 1회로 종료
            if not self._medication_taken(key, now):
                out.append((GuidanceEvent.MEDICATION_MISSED, dict(context)))

        meal_id = (date, key, "meal")
        if meal_id not in self._handled:
            self._handled.add(meal_id)
            if (date, key) not in self._meal_done:
                out.append((GuidanceEvent.MEAL_MISSED, dict(context)))

        return out

    # ---------------- internal ----------------

    def _load_meal_log(self) -> dict:
        try:
            with open(self._meal_log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            return {
                (log["date"], log["meal"]): log.get("time")
                for log in logs if log.get("event") == "MEAL_DONE"
            }
        except Exception:
            return {}

    def _append_meal_log(self, entry: dict):
        try:
            os.makedirs(os.path.dirname(self._meal_log_path), exist_ok=True)
            try:
                with open(self._meal_log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
            logs.append(entry)
            with open(self._meal_log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ScheduleMonitor] 식사 기록 저장 실패: {e}")

    def _medication_entries_today(self, now) -> dict:
        """오늘 복용 기록 {slot_key: "HH:MM"} — 팀원 로그 파일 기준."""
        today = now.strftime("%Y-%m-%d")
        entries = {}
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            return entries
        for log in logs:
            if log.get("date") == today and log.get("event") == "MEDICATION_TAKEN":
                raw = str(log.get("time", ""))
                hhmm = raw.split()[-1][:5] if raw else None
                entries[log.get("meal")] = hhmm
        return entries

    def _medication_taken(self, slot_key, now) -> bool:
        if self._logger is not None:
            try:
                return bool(self._logger.has_taken_today(slot_key, now))
            except Exception:
                pass  # 팀원 로거 조회 실패 시 파일 직접 읽기로 폴백

        today = now.strftime("%Y-%m-%d")
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            return False
        for log in logs:
            if (
                log.get("date") == today
                and log.get("meal") == slot_key
                and log.get("event") == "MEDICATION_TAKEN"
            ):
                return True
        return False
