"""
폴백 템플릿 문장의 음성 파일을 미리 생성해 audio_cache/ 에 저장한다.

네트워크가 되는 환경에서 1회 실행해 두면,
시연 중 네트워크가 끊겨도 스피커 안내가 끊기지 않는다.

실행:
    python scripts/generate_fallback_audio.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm.fallback import FALLBACK_SENTENCES
from voice.tts import VoiceGuide, fallback_audio_path


def main():
    os.makedirs(config.AUDIO_CACHE_DIR, exist_ok=True)
    voice = VoiceGuide()
    print(f"엔진: {config.TTS_ENGINE} / 목소리: {config.TTS_VOICE}\n")

    done = 0
    for event, sentence in FALLBACK_SENTENCES.items():
        path = fallback_audio_path(event)
        print(f"생성 중: {event.value} → \"{sentence}\"")
        if voice.synthesize_to(sentence, path):
            done += 1
            print(f"  저장 완료: {path}")
        else:
            print("  !! 생성 실패 — 네트워크를 확인하세요")

    print(f"\n총 {done}/{len(FALLBACK_SENTENCES)}개 폴백 음성 캐싱 완료")


if __name__ == "__main__":
    main()
