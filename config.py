"""
프로젝트 전역 설정.

- LG그램(Windows) / 라즈베리파이(Raspberry Pi OS) 양쪽에서 코드 수정 없이 동작하도록
  플랫폼을 자동 감지한다.
- OpenAI API 키는 환경변수 또는 .env 파일에서 읽는다. (코드에 직접 쓰지 말 것)
"""

import os
import platform


def _detect_raspberry_pi() -> bool:
    """라즈베리파이 여부 감지 (/proc/device-tree/model 우선, 실패 시 아키텍처로 판단)"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "raspberry pi" in f.read().lower()
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return platform.machine() in ("aarch64", "armv7l")


# .env 파일 지원 (없으면 조용히 넘어감)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---- 플랫폼 ----
IS_RASPBERRY_PI = _detect_raspberry_pi()
PLATFORM_NAME = "RaspberryPi" if IS_RASPBERRY_PI else platform.system()  # Windows / Darwin / Linux

# ---- LLM ----
# USE_LLM=false 이면 LLM을 아예 호출하지 않고 폴백 템플릿만 사용한다.
# USE_LLM=true 일 때 LLM_PROVIDER 로 백엔드를 선택한다.
#
#   LLM_PROVIDER=ollama  → 로컬 sLLM (Ollama). 비용 0원. 수행계획서의 sLLM 구현에 해당.
#                          개발 PC(GPU)에서는 즉답 수준, 라즈베리파이에서도 소형 모델 구동 가능.
#   LLM_PROVIDER=openai  → OpenAI API. 크레딧 충전 필요.
#
# 두 백엔드 모두 OpenAI 호환 프로토콜을 사용하므로 코드 경로는 동일하다.
USE_LLM = os.getenv("USE_LLM", "false").lower() in ("1", "true", "yes")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()   # "ollama" | "openai"

# openai 백엔드 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")   # 짧은 안내 문장 생성 용도 → 저가 모델로 충분

# ollama(로컬 sLLM) 백엔드 설정
#   기본 모델: EXAONE 3.5 2.4B — 한국어 성능 우수한 국산 sLLM, 라즈베리파이 5(8GB)에서도 양자화 구동 가능
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b")

LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "20"))  # 초과 시 폴백 템플릿 사용 (로컬 sLLM은 첫 로딩이 느릴 수 있음)
LLM_MAX_RETRY = 1

# ---- 안내 문장 검증 기준 (치매 환자 의사소통 원칙) ----
GUIDANCE_MAX_CHARS = 60      # 전체 길이 제한 (짧고 명확한 표현)
GUIDANCE_MAX_SENTENCES = 2   # 문장 수 제한 (한 문장에 한 가지 행동만)

# 같은 이벤트를 다시 음성 안내하기까지의 최소 간격(초).
# FSM/로거가 같은 상태를 프레임마다 반복 보고해도 안내가 한 번만 나가게 하는 안전장치.
GUIDANCE_EVENT_COOLDOWN_SEC = float(os.getenv("GUIDANCE_EVENT_COOLDOWN_SEC", "30"))

# 시간대(아침/점심/저녁) 종료 몇 분 전부터 미복용·미식사를 안내할지 (schedule_monitor)
GUIDANCE_REMIND_BEFORE_END_MIN = float(os.getenv("GUIDANCE_REMIND_BEFORE_END_MIN", "30"))

# TTS 엔진: "edge"(권장 — 자연스러운 신경망 음성) 또는 "gtts"(예비)
# edge 실패 시 자동으로 gtts → 캐싱 폴백 mp3 순으로 넘어간다 (3중 안전망)
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge").lower()
# edge 목소리 — scripts/try_voices.py 로 들어보고 고르세요
#   ko-KR-SunHiNeural  : 선희, 따뜻한 여성 톤 (돌봄 로봇 기본값)
#   ko-KR-InJoonNeural : 인준, 차분한 남성 톤
#   ko-KR-HyunsuMultilingualNeural : 현수, 남성 톤
TTS_VOICE = os.getenv("TTS_VOICE", "ko-KR-SunHiNeural")

# 음성 재생 배속 (1.0=원속). 개발·시연에선 1.2~1.3이 듣기 편하다.
# 단, 교재 원칙(천천히 분명하게)상 실제 환자 배포에서는 1.0~1.1 권장.
# edge 엔진은 배속이 내장되어 pydub/ffmpeg 없이 동작.
# gtts·폴백 mp3 경로만 pydub + ffmpeg 필요 (없으면 원속으로 자동 폴백)
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

# 안내 문장 생성 시 판단 근거(이벤트, 상황, 참고 정보, 생성 경로, 검증 결과)를
# 터미널에 출력할지 여부. 시연·디버깅용 설명 가능성(explainability) 기능.
GUIDANCE_EXPLAIN = os.getenv("GUIDANCE_EXPLAIN", "true").lower() == "true"

# 활성 로그 파일에 남길 최근 일수. 이보다 오래된 기록은 logs/archive/ 의
# 월별 파일로 자동 이동 (기록 보존 + 파일 비대화로 인한 속도 저하 방지). 0 = 끔.
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "90"))

# ── 대화(챗) 설정 ──────────────────────────────────────
# 잡담 시 sLLM에 넣어줄 최근 대화 턴 수 (슬라이딩 윈도우)
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "4"))
# 대화 답변은 기록 요약(아침/점심/저녁 3항목)이 들어갈 수 있어 안내보다 약간 여유
CHAT_MAX_CHARS = int(os.getenv("CHAT_MAX_CHARS", "100"))
CHAT_MAX_SENTENCES = int(os.getenv("CHAT_MAX_SENTENCES", "3"))
# 키워드에 안 걸리는 문장을 sLLM에게 분류시킬지 (돌려 말하기 대응: "속이 허하네"→식사).
# 끄면(false) 키워드 미매칭은 전부 잡담으로 처리 (분류용 LLM 호출 없음 → 더 빠름)
CHAT_HYBRID_INTENT = os.getenv("CHAT_HYBRID_INTENT", "true").lower() == "true"

# 환자 호칭 (교재 p.184 — 이름과 존칭으로 불러 주체성 강화)
# .env 에 PATIENT_NAME=김복순 처럼 설정하면 모든 안내 문장이 "김복순님," 으로 시작한다.
# 비워두면 호칭 없이 안내한다.
PATIENT_NAME = os.getenv("PATIENT_NAME", "").strip()

# ---- TTS / 오디오 ----
TTS_LANG = "ko"
AUDIO_CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")
TTS_TEMP_DIR = os.path.join(AUDIO_CACHE_DIR, "temp")

# ---- 복약 스케줄 (추후 FSM 담당 팀원이 사용할 자리) ----
# 예: [{"label": "아침 약", "hour": 8, "minute": 0}, ...]
MEDICATION_SCHEDULE = []
