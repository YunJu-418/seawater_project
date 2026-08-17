"""
edge-tts 한국어 목소리 시청회 — 같은 문장을 목소리별로 재생해서 골라보는 도구.

실행:
    python scripts/try_voices.py

마음에 드는 목소리를 .env 의 TTS_VOICE 에 적으면 된다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice.tts import VoiceGuide

VOICES = [
    ("ko-KR-SunHiNeural", "선희 — 따뜻한 여성 톤 (기본값)"),
    ("ko-KR-InJoonNeural", "인준 — 차분한 남성 톤"),
    ("ko-KR-HyunsuMultilingualNeural", "현수 — 남성 톤"),
]

SAMPLE = "약을 잘 드셨어요. 오늘도 건강하게 보내세요."


def main():
    voice = VoiceGuide()
    for name, desc in VOICES:
        print(f"\n▶ {desc}  ({name})")
        config.TTS_VOICE = name          # 이 목소리로 합성
        voice.speak(SAMPLE)
    print("\n마음에 드는 목소리를 .env 의 TTS_VOICE= 에 적어 주세요.")


if __name__ == "__main__":
    main()
