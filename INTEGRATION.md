# 음성 안내 파트 통합 가이드 (v2)

이 브랜치(me)의 역할은 하나다: **이벤트 in → 생성형 음성 out.**

이벤트 발생원은 두 종류다.
- **비전 이벤트** → `llm/adapters.py` 가 번역 (팀원 코드 import 없이 duck typing)
- **시간 이벤트** → `schedule_monitor.py` 가 발생 (시간대 종료 임박 미복용·미식사)

문장 생성은 노인 의사소통 교재(Chapter 4, pp.183~186) 원칙을
프롬프트(`llm/prompts.py`)와 검증 코드(`guidance_generator._validate`) 양쪽에 반영했다:
한 문장 한 행동 / 테스트식 질문 금지 / 지시대명사 금지 / 유도형 표현 / 이름+존칭 / 지남력 지원.

## 공통 준비 (한 번)

me 브랜치의 다음 파일이 함께 있어야 한다 (PR 머지 또는 복사):

```
llm/  voice/  guidance_service.py  schedule_monitor.py  config.py  .env.example
```

Ollama 없이도 동작한다 — `USE_LLM=false` 또는 Ollama 미기동 시 폴백 문장으로 안내.

---

## 1) medication-detector 브랜치 → 약 복용 안내

**핵심: 경고는 삼키기 전에.** 복용 '시도'(GRABBED) 순간에 시간대를 검사해서
중복 복용·시간대 밖 복용을 미리 안내한다. 완료 후에 알리면 이미 늦기 때문.

`main.py` 수정 — import 3줄 + 초기화 2줄 + 루프 안 교체:

```python
# 상단 import 에 추가
from guidance_service import GuidanceService
from llm.adapters import MedicationAdapter
from schedule_monitor import ScheduleMonitor
```

```python
# main() 에서 medication_logger 만드는 곳 아래에 추가
guidance = GuidanceService(enable_voice=True)
monitor = ScheduleMonitor(medication_logger=medication_logger)   # 시간표 단일 기준 = 팀원 로거
adapter = MedicationAdapter(medication_logger=medication_logger)
```

```python
# 기존:
medication_log_result = medication_logger.log_if_taken(
    medication_result["state"]
)
# 바로 아래에 추가:
guidance.notify_all(adapter.translate(
    vision_state=medication_result["state"],
    log_result=medication_log_result,
))
guidance.notify_all(monitor.tick())   # 시간대 종료 임박 미복용·미식사 재안내
```

`r` 키로 `medication_detector.reset()` 하는 곳에 `adapter.reset()` 도 한 줄.

동작 정리:

| 상황 | 안내 시점 | 문장 예 |
|---|---|---|
| 첫 복용 GRABBED → DONE | 완료 때 | "약을 잘 드셨어요…" |
| 이미 복용한 시간대에 GRABBED | **잡는 순간** | "약은 조금 전에 드셨어요. 안심하고 내려놓으셔도 돼요." |
| 시간대 밖 GRABBED | **잡는 순간** | "약은 정해진 시간에 드시는 게 좋아요…" |
| MEDICATION_CHECK_NEEDED | 진입 순간 | "약이 손에 있는지 한번 봐 주세요. 바닥도 한번 살펴봐 주세요." |
| 시간대 종료 30분 전 미복용 | tick | "약 드실 시간이 지나가고 있어요…" |
| 같은 동작의 연속 프레임 | — | 무음 (edge trigger) |

## 2) feature/meal-fsm 브랜치 → 식사 안내

`main_meal.py` 수정 — import 3줄 + 초기화 2줄 + 루프 안 2줄:

```python
# 상단 import 에 추가
from guidance_service import GuidanceService
from llm.adapters import from_meal_events
from schedule_monitor import ScheduleMonitor
```

```python
# __init__ 에서 self.logger = self._create_logger() 아래에 추가
self.guidance = GuidanceService(enable_voice=True)
self.monitor = ScheduleMonitor()
```

```python
# run() 루프의 result = self.meal_fsm.update(observation) 아래에 추가:
self.guidance.notify_all(
    from_meal_events(result.events, monitor=self.monitor)
)
self.guidance.notify_all(self.monitor.tick())
```

- `SESSION_STARTED` → "식사를 시작하셨네요…" / `SESSION_FINISHED` → "식사를 잘 하셨어요…"
- **같은 시간대에 두 번째 식사 시작 → "조금 전에 식사를 하셨어요…"** (중복 식사 방지)
- 시간대 밖 식사는 제한 없음 — 그대로 시작 안내 (중복만 방지)
- `BITE_CONFIRMED` 등 순간 이벤트는 의도적으로 무음
- 시간대 종료 임박 미식사 → tick 이 재안내

## 3) main 브랜치 exit_system → 외출 안내

`run_exit.py` 수정 — import 2줄 + 초기화 1줄 + 루프 2줄:

```python
from guidance_service import GuidanceService
from llm.adapters import from_exit_events
```

```python
# notifier = KakaoNotifier(...) 근처에 추가
guidance = GuidanceService(enable_voice=True)
```

```python
# 기존 for event in fsm.update(...) 를 다음처럼:
events = fsm.update(motion_value, now)
guidance.notify_all(from_exit_events(events))
for event in events:
    ...  # 기존 EXIT_CONFIRMED → 카카오 알림 로직 그대로
```

- `OUTWARD_CROSSING` (아직 현관에 있음) → "겉옷을 챙겨 보세요"
- `EXIT_CONFIRMED` 는 **일부러 무음** — 사람이 사라진 뒤 확정되므로 카카오 알림 담당.

---

## 시간대 정의와 설정

- 아침 06~10 / 점심 11~14 / 저녁 17~21 — `MedicationLogger.meal_times` 와 동일.
  약 파트에서는 `medication_logger` 를 넘겨 **팀원 시간표를 단일 기준**으로 쓴다.
- `.env` 로 조정 가능:
  - `GUIDANCE_REMIND_BEFORE_END_MIN=30` — 종료 몇 분 전부터 재안내할지
  - `GUIDANCE_EVENT_COOLDOWN_SEC=30` — 같은 이벤트 재발화 최소 간격

## 반복 발행 걱정 안 해도 되는 이유 (삼중 안전장치)

1. **어댑터 edge trigger** — 같은 상태의 연속 프레임은 전이 순간만 통과.
2. **에피소드 플래그** — GRABBED 때 경고했으면 완료 시점 같은 경고 생략 (이중 안내 방지).
3. **서비스 쿨다운 + 하루 1회 제한** — 같은 이벤트 30초 내 무음,
   미복용·미식사 재안내는 시간대당 1회.

## 카메라 없이 검증하기

```bash
python scripts/simulate_events.py           # 기능 1~4 포함 7개 시나리오 재생
python scripts/simulate_events.py --voice   # 음성까지
python scripts/test_guidance.py             # 이벤트별 문장 생성 일람
python scripts/generate_fallback_audio.py   # 새 문장 폴백 mp3 재생성 (필수 1회)
```
