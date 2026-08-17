"""
TTS(Text-to-Speech) 및 오디오 재생 모듈.

- 1순위 edge-tts: 신경망 음성 (선희 등) — 자연스러운 억양, 배속 내장 (네트워크 필요)
- 2순위 gTTS: edge 실패 시 예비 엔진 (네트워크 필요)
- pygame.mixer로 재생 → LG그램(Windows)과 라즈베리파이(USB 스피커) 모두 동일 코드로 동작
- gTTS 실패 시(네트워크 장애) 미리 캐싱된 폴백 mp3를 재생
  (audio_cache/ 폴더, scripts/generate_fallback_audio.py 로 사전 생성)
"""

import os
import hashlib
import threading

import config
from llm.events import GuidanceEvent

try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False

try:
    import asyncio
    import edge_tts
    _EDGE_AVAILABLE = True
except ImportError:
    _EDGE_AVAILABLE = False

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.effects import speedup
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False


def fallback_audio_path(event: GuidanceEvent) -> str:
    """이벤트별 사전 캐싱된 폴백 음성 파일 경로"""
    return os.path.join(config.AUDIO_CACHE_DIR, f"fallback_{event.value}.mp3")


class VoiceGuide:
    def __init__(self):
        os.makedirs(config.AUDIO_CACHE_DIR, exist_ok=True)
        os.makedirs(config.TTS_TEMP_DIR, exist_ok=True)

        self._lock = threading.Lock()  # 안내 음성 겹침 방지
        self._mixer_ready = False
        self._speed_warned = False     # 배속 불가 안내는 1회만

        if config.TTS_ENGINE == "edge" and not _EDGE_AVAILABLE:
            print("[VoiceGuide] edge-tts 미설치 → gTTS로 대체합니다 (pip install edge-tts)")
        if not _EDGE_AVAILABLE and not _GTTS_AVAILABLE:
            print("[VoiceGuide] TTS 엔진 없음 → 캐싱된 폴백 음성만 사용 가능합니다.")
        if not _PYGAME_AVAILABLE:
            print("[VoiceGuide] pygame 미설치 → 음성 출력이 비활성화됩니다.")

    # ---------------- public ----------------

    def speak(self, sentence: str, event: GuidanceEvent | None = None):
        """
        문장을 음성으로 출력한다. (blocking — 별도 스레드에서 호출 권장)

        1) gTTS로 문장 변환 후 재생
        2) 실패 시 event의 캐싱된 폴백 mp3 재생
        """
        with self._lock:
            path = self._synthesize(sentence)

            if path is None and event is not None:
                cached = fallback_audio_path(event)
                if os.path.exists(cached):
                    print("[VoiceGuide] TTS 실패 → 캐싱된 폴백 음성 재생")
                    path = cached

            if path is None:
                print(f"[VoiceGuide] 음성 출력 불가. 텍스트 안내: {sentence}")
                return False

            return self._play(self._apply_speed(path))

    # ---------------- internal ----------------

    def _synthesize(self, sentence: str):
        """문장 → mp3. 같은 문장(+같은 목소리·배속)은 파일 캐시 재사용."""
        # 캐시 키에 엔진·목소리·배속 포함 → 목소리를 바꾸면 새로 생성됨
        key = f"{config.TTS_ENGINE}|{config.TTS_VOICE}|{config.TTS_SPEED}|{sentence}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        prefix = "tts_edge_" if config.TTS_ENGINE == "edge" and _EDGE_AVAILABLE else "tts_"
        path = os.path.join(config.TTS_TEMP_DIR, f"{prefix}{digest}.mp3")
        if os.path.exists(path):
            return path
        if self.synthesize_to(sentence, path):
            return path
        return None

    def synthesize_to(self, sentence: str, path: str) -> bool:
        """문장을 지정 경로에 mp3로 저장. edge → gTTS 순으로 시도.

        폴백 음성 사전 생성(generate_fallback_audio.py)에서도 사용한다.
        """
        if config.TTS_ENGINE == "edge" and _EDGE_AVAILABLE:
            if self._edge_save(sentence, path):
                return True
            print("[VoiceGuide] edge-tts 실패 → gTTS로 재시도")
        if _GTTS_AVAILABLE:
            try:
                gTTS(text=sentence, lang=config.TTS_LANG).save(path)
                return True
            except Exception as e:
                print(f"[VoiceGuide] gTTS 변환 실패: {e}")
        return False

    def _edge_save(self, sentence: str, path: str) -> bool:
        """edge-tts 합성. 배속은 엔진 내장 rate로 처리 (pydub/ffmpeg 불필요)."""
        speed = max(0.5, min(config.TTS_SPEED, 2.0))
        rate = f"{int(round((speed - 1.0) * 100)):+d}%"   # 1.25 → "+25%"
        try:
            communicate = edge_tts.Communicate(
                sentence, voice=config.TTS_VOICE, rate=rate
            )
            asyncio.run(communicate.save(path))
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception as e:
            print(f"[VoiceGuide] edge-tts 변환 실패: {e}")
            try:
                if os.path.exists(path):
                    os.remove(path)   # 실패로 생긴 빈 파일이 캐시로 오인되지 않게
            except OSError:
                pass
            return False

    def _apply_speed(self, path: str) -> str:
        """TTS_SPEED 배속본을 만들어 경로 반환. 배속본도 파일 캐싱 (문장당 1회 변환).

        pydub/ffmpeg가 없으면 원속으로 폴백하고 설치 방법을 1회 안내한다.
        음 높이는 유지한 채 재생 길이만 줄이는 방식(chunk 기반)이라 목소리가
        다람쥐처럼 높아지지 않는다.
        """
        speed = max(0.5, min(config.TTS_SPEED, 2.0))
        if abs(speed - 1.0) < 0.01:
            return path
        if "tts_edge_" in os.path.basename(path):
            return path   # edge 산출물은 합성 단계에서 이미 배속 반영됨

        if not _PYDUB_AVAILABLE:
            if not self._speed_warned:
                self._speed_warned = True
                print(f"[VoiceGuide] 배속({speed}x)을 쓰려면 pydub과 ffmpeg가 필요합니다.")
                print("    pip install pydub  +  윈도우: winget install ffmpeg"
                      "  /  파이: sudo apt install ffmpeg")
                print("    → 지금은 원속(1.0x)으로 재생합니다.")
            return path

        base, ext = os.path.splitext(path)
        sped_path = f"{base}_x{int(speed * 100)}{ext}"
        if os.path.exists(sped_path):
            return sped_path

        try:
            audio = AudioSegment.from_file(path)
            faster = speedup(audio, playback_speed=speed, chunk_size=150, crossfade=25)
            faster.export(sped_path, format=ext.lstrip("."))
            return sped_path
        except Exception as e:
            if not self._speed_warned:
                self._speed_warned = True
                print(f"[VoiceGuide] 배속 처리 실패({e}) → 원속으로 재생합니다. "
                      "ffmpeg 설치를 확인하세요.")
            return path

    def _play(self, path: str) -> bool:
        if not _PYGAME_AVAILABLE:
            return False
        try:
            if not self._mixer_ready:
                pygame.mixer.init()
                self._mixer_ready = True

            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            return True
        except Exception as e:
            print(f"[VoiceGuide] 재생 실패: {e}")
            return False
