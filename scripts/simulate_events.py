"""
카메라·비전 없이, 팀원 브랜치가 실제로 만들어 내는 '이벤트 모양 그대로'를
흘려보내서 [이벤트 → 어댑터 → 문장 생성 → (음성)] 파이프라인을 검증하는 시뮬레이터.

이 스크립트가 곧 우리 파트의 정의다: 이벤트 in → 생성형 음성 out.
비전 코드는 한 줄도 import 하지 않는다. 시각도 가짜 시계로 주입해서
시간대(아침/점심/저녁) 기능까지 전부 검증한다.

실행:
    python scripts/simulate_events.py            # 문장 생성까지 (음성 X)
    python scripts/simulate_events.py --voice    # gTTS 음성 재생까지
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guidance_service import GuidanceService
from llm.adapters import MedicationAdapter, from_exit_events, from_meal_events
from schedule_monitor import ScheduleMonitor


def banner(title):
    print()
    print("─" * 62)
    print(f"  {title}")
    print("─" * 62)


def med_log(log_state, meal_label=None, taken_time=None):
    """MedicationLogger.log_if_taken() 반환 dict 와 동일한 모양."""
    return {
        "log_state": log_state,
        "meal": None,
        "meal_label": meal_label,
        "taken_time": taken_time,
        "vision_state": "MEDICATION_DONE" if log_state != "NOT_DONE" else "WAITING",
    }


def meal_event(event_type, bite_count=0, drink_count=0):
    """MealEvent(meal/meal_state.py) 와 동일한 모양 (duck typing)."""
    return SimpleNamespace(
        event_type=event_type, bite_count=bite_count, drink_count=drink_count
    )


def exit_event(name):
    """FSMEvent(exit_system/exit_fsm.py) 와 동일한 모양."""
    return SimpleNamespace(name=name, timestamp=0.0)


def write_med_logs(path, entries):
    """복용 기록 JSON 파일 생성 (medication_logger 저장 포맷과 동일)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def taken_entry(date, meal_key, meal_label, hhmmss):
    return {
        "date": date,
        "time": f"{date} {hhmmss}",
        "meal": meal_key,
        "meal_label": meal_label,
        "event": "MEDICATION_TAKEN",
        "vision_state": "MEDICATION_DONE",
    }


def main():
    use_voice = "--voice" in sys.argv
    tmp = tempfile.mkdtemp(prefix="sim_")
    DATE = "2026-08-11"

    # 시뮬레이션에선 쿨다운을 짧게 (기본 30초는 실제 운용값)
    guidance = GuidanceService(enable_voice=use_voice, event_cooldown_sec=1.0)

    print("=" * 62)
    print(" 이벤트 기반 생성형 음성 파이프라인 시뮬레이션")
    print("  (medication-detector / feature-meal-fsm / exit_system / 스케줄)")
    print("=" * 62)

    # ══════════ A. 정상 아침 복용 (기록 없음 → GRABBED → DONE) ══════════
    banner("1) 아침 8시, 첫 복용 — GRABBED는 조용, 완료 때 칭찬")
    log_path = os.path.join(tmp, "log_a.json")
    write_med_logs(log_path, [])
    monitor = ScheduleMonitor(medication_log_path=log_path)
    adapter = MedicationAdapter(monitor=monitor)
    now = datetime(2026, 8, 11, 8, 12)

    frames = [
        ("WAITING", med_log("NOT_DONE")),
        ("GRABBED", med_log("NOT_DONE")),          # 시간대 안 + 미복용 → 무음
        ("GRABBED", med_log("NOT_DONE")),
        ("MEDICATION_DONE", med_log("LOGGED", "아침", f"{DATE} 08:12:03")),  # ← 칭찬
        ("MEDICATION_DONE", med_log("ALREADY_LOGGED", "아침")),              # 연속 프레임 무음
    ]
    spoken = 0
    for vs, lr in frames:
        spoken += guidance.notify_all(adapter.translate(vs, lr, now=now), wait=True)
    print(f"  → 프레임 {len(frames)}장, 안내 {spoken}회 (기대 1: 복용 완료 칭찬)")

    # ══════════ B. 기능 2 — 이미 복용한 시간대에 다시 약통을 잡음 ══════════
    banner("2) [기능2] 아침 약 복용 기록 있는데 다시 GRABBED → 삼키기 전에 안내")
    log_path = os.path.join(tmp, "log_b.json")
    write_med_logs(log_path, [taken_entry(DATE, "morning", "아침", "08:12:03")])
    monitor = ScheduleMonitor(medication_log_path=log_path)
    adapter = MedicationAdapter(monitor=monitor)
    now = datetime(2026, 8, 11, 9, 30)

    frames = [("WAITING", None), ("GRABBED", None), ("GRABBED", None)]
    spoken = 0
    for vs, lr in frames:
        spoken += guidance.notify_all(adapter.translate(vs, lr, now=now), wait=True)
    print(f"  → 안내 {spoken}회 (기대 1: 중복 복용 — 잡는 순간 조기 안내)")

    # ══════════ C. 기능 3 — 시간대 밖 복용 시도 ══════════
    banner("3) [기능3] 오후 3시(시간대 밖)에 약통 GRABBED → 삼키기 전에 안내")
    adapter = MedicationAdapter(monitor=ScheduleMonitor(
        medication_log_path=os.path.join(tmp, "log_b.json")))
    now = datetime(2026, 8, 11, 15, 0)

    frames = [
        ("WAITING", None),
        ("GRABBED", None),                              # ← 시간대 밖 → 조기 안내
        ("MEDICATION_DONE", med_log("NOT_MEDICATION_TIME")),  # 완료 시점 중복 경고는 생략
    ]
    spoken = 0
    for vs, lr in frames:
        spoken += guidance.notify_all(adapter.translate(vs, lr, now=now), wait=True)
    print(f"  → 안내 {spoken}회 (기대 1: 정해진 시간 안내, 이중 안내 없음)")

    # ══════════ D. 기능 4 — 복용 확인 필요 ══════════
    banner("4) [기능4] 약을 잡았는데 MEDICATION_CHECK_NEEDED → 살펴보기 유도")
    log_path = os.path.join(tmp, "log_d.json")
    write_med_logs(log_path, [])
    adapter = MedicationAdapter(monitor=ScheduleMonitor(medication_log_path=log_path))
    now = datetime(2026, 8, 11, 12, 10)

    frames = [
        ("WAITING", None),
        ("GRABBED", None),
        ("MEDICATION_CHECK_NEEDED", None),   # ← 손/바닥 살펴보기 안내
        ("MEDICATION_CHECK_NEEDED", None),   # 연속 프레임 무음
        ("GRABBED", None),                   # 다시 잡음 확인 → 무음
    ]
    spoken = 0
    for vs, lr in frames:
        spoken += guidance.notify_all(adapter.translate(vs, lr, now=now), wait=True)
    print(f"  → 안내 {spoken}회 (기대 1: 손·바닥 살펴보기)")

    # ══════════ E. 식사 시작/종료 + 기능2(중복 식사) ══════════
    banner("5) 점심 식사 세션 + [기능2] 같은 시간대 두 번째 식사 시도")
    monitor = ScheduleMonitor(medication_log_path=os.path.join(tmp, "log_d.json"))
    now = datetime(2026, 8, 11, 12, 20)

    updates = [
        [meal_event("session_started")],                                 # 식사 시작
        [meal_event("bite_confirmed", bite_count=1)],                    # 무음
        [meal_event("session_finished", bite_count=18, drink_count=2)],  # 종료 + 기록
        [meal_event("session_started")],                                 # ← 중복 식사!
    ]
    spoken = 0
    for events in updates:
        spoken += guidance.notify_all(
            from_meal_events(events, monitor=monitor, now=now), wait=True
        )
    print(f"  → 안내 {spoken}회 (기대 3: 시작 + 종료 + 중복 식사 안심 안내)")

    # ══════════ F. 기능 1 — 시간대 종료 임박, 미복용·미식사 ══════════
    banner("6) [기능1] 점심 13:40 (종료 20분 전), 약도 식사도 안 함 → tick")
    log_path = os.path.join(tmp, "log_f.json")
    write_med_logs(log_path, [])
    monitor = ScheduleMonitor(medication_log_path=log_path)

    spoken = guidance.notify_all(monitor.tick(now=datetime(2026, 8, 11, 13, 40)), wait=True)
    spoken += guidance.notify_all(monitor.tick(now=datetime(2026, 8, 11, 13, 41)), wait=True)  # 재호출 → 무음
    print(f"  → 안내 {spoken}회 (기대 2: 약 재안내 + 식사 재안내, 하루 1회 제한)")

    import time as _t; _t.sleep(1.2)  # 직전 시나리오와 같은 이벤트 → 서비스 쿨다운(시뮬레이션 1초) 통과 대기
    banner("6-2) [기능1] 저녁 20:40 — 약은 먹었고 식사만 안 함")
    write_med_logs(log_path, [taken_entry(DATE, "dinner", "저녁", "18:05:00")])
    spoken = guidance.notify_all(monitor.tick(now=datetime(2026, 8, 11, 20, 40)), wait=True)
    print(f"  → 안내 {spoken}회 (기대 1: 식사 재안내만)")

    # ══════════ G. 외출 ══════════
    banner("7) 현관 이동 → 외출 확정")
    updates = [
        [exit_event("OUTWARD_CROSSING")],   # 아직 현관에 있음 → 겉옷 안내
        [exit_event("EXIT_CONFIRMED")],     # 이미 없음 → 무음 (카카오 알림 담당)
    ]
    spoken = 0
    for events in updates:
        spoken += guidance.notify_all(from_exit_events(events), wait=True)
    print(f"  → 안내 {spoken}회 (기대 1: 외출 확정은 카카오 몫, 음성 없음)")

    print()
    print("=" * 62)
    print(" 시뮬레이션 완료 — 실제 통합은 INTEGRATION.md 참고")
    print("=" * 62)


if __name__ == "__main__":
    main()
