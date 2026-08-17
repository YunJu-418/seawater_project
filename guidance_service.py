"""
안내 서비스 (이벤트 ↔ LLM ↔ TTS 연결 계층).

우리 파트의 공식 입구는 notify(event, context) 하나다.

- 비전 루프(30fps)를 막지 않도록 LLM 호출 + TTS 재생은 별도 스레드에서 처리 (non-blocking)
- _busy 플래그로 안내 음성 겹침 방지
- 이벤트별 쿨다운으로, FSM이 같은 이벤트를 짧은 간격에 반복 발행해도 한 번만 말한다

사용 예 (팀원 브랜치의 메인 루프):
    from guidance_service import GuidanceService
    from llm.adapters import from_meal_events

    guidance = GuidanceService(enable_voice=True)
    ...
    result = meal_fsm.update(observation)
    guidance.notify_all(from_meal_events(result.events))
"""

import threading
import time
from datetime import datetime

import config
from llm.events import GuidanceEvent
from llm.guidance_generator import GuidanceGenerator
from voice.tts import VoiceGuide


class GuidanceService:
    def __init__(
        self,
        enable_voice: bool = True,
        event_cooldown_sec: float | None = None,
    ):
        """
        event_cooldown_sec: 같은 이벤트를 다시 말하기까지의 최소 간격(초).
                            None이면 config.GUIDANCE_EVENT_COOLDOWN_SEC 사용.
        """
        self.generator = GuidanceGenerator()
        self.voice = VoiceGuide() if enable_voice else None
        self._busy = threading.Event()
        self._cooldown = (
            config.GUIDANCE_EVENT_COOLDOWN_SEC
            if event_cooldown_sec is None
            else event_cooldown_sec
        )
        self._last_spoken: dict[GuidanceEvent, float] = {}

    def notify(
        self,
        event: GuidanceEvent,
        context: dict | None = None,
        wait: bool = False,
    ) -> bool:
        """
        이벤트 발생을 알린다. 기본 non-blocking.
        wait=True 는 테스트/시뮬레이션에서 출력 순서를 보장하고 싶을 때만 사용.

        Returns:
            실제로 안내를 시작했으면 True, (쿨다운/진행 중이라) 건너뛰었으면 False.
        """
        now = time.monotonic()
        last = self._last_spoken.get(event)
        if last is not None and now - last < self._cooldown:
            return False

        if self._busy.is_set():
            print(f"[GuidanceService] 안내 진행 중 → {event.value} 건너뜀")
            return False

        self._last_spoken[event] = now

        if context is None:
            context = {}
        context.setdefault("time", datetime.now().strftime("%H:%M"))
        if config.PATIENT_NAME:
            context.setdefault("patient_name", config.PATIENT_NAME)

        thread = threading.Thread(
            target=self._run, args=(event, context), daemon=True
        )
        thread.start()
        if wait:
            thread.join()
        return True

    def notify_all(self, pairs, wait: bool = False) -> int:
        """
        어댑터가 번역한 (이벤트, 컨텍스트) 목록을 순서대로 알린다.

        Returns:
            실제로 안내가 시작된 건수.
        """
        spoken = 0
        for event, context in pairs:
            if self.notify(event, context, wait=wait):
                spoken += 1
        return spoken

    def _run(self, event, context):
        self._busy.set()
        try:
            sentence, source = self.generator.generate(event, context)
            print(f'[GuidanceService] ({source}) {event.value} → "{sentence}"')

            if self.voice is not None:
                self.voice.speak(sentence, event=event)
        finally:
            self._busy.clear()
