"""
음성 입력(STT) — 마이크로 들은 말을 텍스트로 변환.

SpeechRecognition + 구글 무료 STT(ko-KR). 키 발급 불필요, 변환 순간만 인터넷 필요.
패키지 미설치·마이크 없음·네트워크 장애 어느 경우에도 예외로 죽지 않고 None을
반환하며, 호출 쪽(chat_demo 등)이 키보드 입력으로 폴백한다 — TTS의 폴백 mp3와
같은 철학(음성 계층은 어떤 상황에도 시스템을 멈추지 않는다).

라즈베리파이 배포 메모:
  - USB 마이크 필요 (파이에는 내장 마이크 없음)
  - sudo apt install portaudio19-dev python3-dev && pip install SpeechRecognition PyAudio
  - 네트워크 불안정 환경 대비 오프라인 STT(Vosk 등)는 향후 개선 항목
"""

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    sr = None
    _SR_AVAILABLE = False


def list_microphones():
    """장치 목록 출력 — 인식이 계속 실패하면 엉뚱한 마이크를 잡은 경우가 흔하다."""
    if not _SR_AVAILABLE:
        print("[STT] speech_recognition 미설치")
        return
    try:
        names = sr.Microphone.list_microphone_names()
    except Exception as e:
        print(f"[STT] 장치 목록 조회 실패: {e}")
        return
    print("[STT] 사용 가능한 마이크 장치:")
    for i, name in enumerate(names):
        print(f"    [{i}] {name}")
    print("    → 원하는 번호를 --mic-index N 으로 지정하세요")


def _pick_transcript(result, min_confidence: float = 0.45):
    """구글 STT show_all 결과에서 최선의 받아쓰기를 고른다.

    확신도(confidence)가 낮으면 None — 소음을 억지로 단어로 만든 파편
    ("내용", "시간 좀 지겹" 같은)이 대화 시스템에 흘러드는 것을 막는다.
    확신도 정보가 아예 없으면(구글이 생략하는 경우) 그대로 수용한다.
    """
    if not result or not isinstance(result, dict):
        return None
    alternatives = result.get("alternative") or []
    if not alternatives:
        return None
    best = alternatives[0]
    text = (best.get("transcript") or "").strip()
    if not text:
        return None
    confidence = best.get("confidence")
    if confidence is not None and confidence < min_confidence:
        return None
    return text


class SpeechListener:
    def __init__(self, language: str = "ko-KR", device_index: int | None = None):
        """
        device_index: 사용할 마이크 번호 (None=시스템 기본).
                      list_microphones() 로 번호 확인 가능.
        """
        self.language = language
        self.recognizer = None
        self.mic = None
        self.fail_streak = 0   # 연속 인식 실패 횟수 (진단 힌트용)

        if not _SR_AVAILABLE:
            print("[STT] speech_recognition 미설치 → 음성 입력 비활성 "
                  "(pip install SpeechRecognition PyAudio)")
            return
        try:
            self.recognizer = sr.Recognizer()
            self.mic = sr.Microphone(device_index=device_index)
            try:
                names = sr.Microphone.list_microphone_names()
                used = (names[device_index] if device_index is not None
                        else "시스템 기본 장치")
                print(f"[STT] 사용 마이크: {used}")
            except Exception:
                pass
            with self.mic as source:
                print("[STT] 주변 소음 측정 중... 1초만 조용히 해주세요")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print(f"[STT] 소음 기준값(energy_threshold): "
                  f"{self.recognizer.energy_threshold:.0f}")
            # 1초 침묵이면 발화가 끝난 것으로 판단 (노인 발화 속도 고려해 여유 있게)
            self.recognizer.pause_threshold = 1.0
            print("[STT] 마이크 준비 완료")
        except Exception as e:
            print(f"[STT] 마이크 초기화 실패: {e} → 음성 입력 비활성")
            self.recognizer = None
            self.mic = None

    @property
    def available(self) -> bool:
        return self.recognizer is not None and self.mic is not None

    def listen_once(self, timeout=None, phrase_time_limit: float = 10.0):
        """
        마이크에서 한 발화를 듣고 텍스트로 반환. 실패 시 None (사유는 출력).

        timeout: 이 시간(초) 안에 말이 시작되지 않으면 None. None이면 무한 대기
                 (상시 대기 모드용 — 스피커 재생 중에는 이 함수가 호출되지 않으므로
                  로봇이 자기 목소리를 듣는 문제가 구조적으로 없다)
        phrase_time_limit: 한 발화 최대 길이(초)
        """
        if not self.available:
            return None
        try:
            with self.mic as source:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
        except sr.WaitTimeoutError:
            return None  # 시간 안에 말이 없었음 — 정상 상황
        except Exception as e:
            print(f"[STT] 녹음 실패: {e}")
            return None

        try:
            result = self.recognizer.recognize_google(
                audio, language=self.language, show_all=True
            )
            text = _pick_transcript(result, min_confidence=0.45)
            if text is None:
                print("[STT] (확신이 낮은 인식 결과라 무시 — 다시 말씀해 주세요)")
                return None
            self.fail_streak = 0
            return text
        except sr.UnknownValueError:
            self.fail_streak += 1
            print("[STT] (말을 알아듣지 못함)")
            if self.fail_streak == 3:
                print("[STT] 인식이 계속 실패하면 엉뚱한 마이크를 잡았을 가능성이 큽니다.")
                print("      1) 윈도우 설정 > 소리 > 입력 에서 마이크 레벨이 움직이는지 확인")
                print("      2) 아래 장치 목록에서 번호를 골라 --mic-index N 으로 재실행")
                list_microphones()
            return None
        except sr.RequestError as e:
            print(f"[STT] 구글 STT 연결 실패 — 인터넷 확인: {e}")
            return None
