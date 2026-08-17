"""
카메라 없이 LLM 안내 문장 생성 + TTS를 단독 테스트하는 스크립트.

실행:
    python scripts/test_guidance.py            # 문장 생성만 (음성 X)
    python scripts/test_guidance.py --voice    # 음성 출력까지

API 키가 없으면 자동으로 폴백 템플릿이 출력된다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.events import GuidanceEvent, ACTIVE_EVENTS
from llm.guidance_generator import GuidanceGenerator


def main():
    use_voice = "--voice" in sys.argv

    generator = GuidanceGenerator(cache_ttl_sec=0)  # 테스트에선 캐시 끔
    voice = None
    if use_voice:
        from voice.tts import VoiceGuide
        voice = VoiceGuide()

    print("=" * 60)
    print("이벤트별 안내 문장 생성 테스트")
    print("=" * 60)

    for event in GuidanceEvent:
        active = "활성" if event in ACTIVE_EVENTS else "정의만"
        sentence, source = generator.generate(event)
        print(f"[{active}] {event.value:24s} ({source:8s}) → {sentence}")

        if voice is not None and event in ACTIVE_EVENTS:
            voice.speak(sentence, event=event)

    print("=" * 60)
    print("테스트 완료")


if __name__ == "__main__":
    main()
