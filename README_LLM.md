# LLM 안내 문장 생성 + 음성 안내 모듈

개발 보고서의 **⑤ LLM 기반 의사소통 맞춤형 안내 생성** + **⑦ TTS** 파트 구현.

## 구조

```
FSM 판단 결과 (main.py)
   → GuidanceService.notify(이벤트)     ← 별도 스레드 (비전 루프 안 멈춤)
      → GuidanceGenerator (OpenAI API + 응답 검증 + 캐싱)
         └ 실패 시 → 폴백 템플릿 (llm/fallback.py)
      → VoiceGuide (gTTS → pygame 재생)
         └ 실패 시 → audio_cache/ 의 사전 캐싱 mp3 재생
```

| 파일 | 역할 |
|---|---|
| `config.py` | 플랫폼 자동 감지(노트북/라즈베리파이), API 키·모델·검증 기준 설정 |
| `llm/events.py` | FSM → LLM 이벤트 정의. 현재 활성: `MEDICATION_DONE`. 나머지(미복용, 중복 복용, 식사, 외출)는 인터페이스만 정의 |
| `llm/prompts.py` | 치매 환자 의사소통 원칙 시스템 프롬프트 + 이벤트별 상황 프롬프트 |
| `llm/guidance_generator.py` | API 호출, 응답 검증(60자/2문장/부정 표현 금지), 5분 캐싱, 폴백 전환 |
| `llm/fallback.py` | 네트워크·API 장애 시 즉시 사용하는 고정 안내 문장 |
| `voice/tts.py` | gTTS 변환 + pygame 재생 (Windows·라즈베리파이 공통), 폴백 음성 캐시 |
| `guidance_service.py` | main.py와 연결되는 진입점. 스레드 처리 + 음성 겹침 방지 |

## 설치 및 실행 (LG그램 기준)

```bash
pip install -r requirements.txt

# API 키 설정
copy .env.example .env      # 그 후 .env 열어서 팀 OpenAI 키 입력

# (권장) 폴백 음성 사전 캐싱 — 네트워크 되는 곳에서 1회 실행
python scripts/generate_fallback_audio.py

# 카메라 없이 문장 생성 테스트
python scripts/test_guidance.py
python scripts/test_guidance.py --voice   # 스피커 출력까지

# 전체 시스템 실행 (노트북 웹캠 사용)
python main.py
```

API 키가 없어도 폴백 템플릿으로 항상 동작하므로, 키 발급 전에도 개발·테스트 가능.

### 현재 운영 모드: 로컬 sLLM (Ollama) — 비용 0원

수행계획서의 **sLLM 기반 대화 시스템** 항목을 그대로 구현한 모드.
OpenAI 크레딧 없이 실제 LLM 생성 문장을 사용한다.

준비 (개발 PC에서 1회):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull exaone3.5:2.4b        # LG EXAONE 3.5 2.4B — 한국어 성능 우수, 약 1.6GB
```

`.env` 설정:

```
USE_LLM=true
LLM_PROVIDER=ollama
```

이후 실행하면 `로컬 sLLM 모드 (Ollama, model=exaone3.5:2.4b)` 로그와 함께
LLM 생성 → 검증 → (실패 시) 폴백 순으로 동작한다.
Ollama가 꺼져 있어도 폴백으로 안전하게 전환되므로 시연이 끊기지 않는다.

다른 모델을 쓰려면 `ollama pull` 후 `.env`의 `OLLAMA_MODEL`만 변경
(예: `qwen2.5:3b-instruct`, `kanana-nano` 등).

### 대안 모드

| 모드 | .env 설정 | 비고 |
|---|---|---|
| 폴백 전용 | `USE_LLM=false` | LLM 호출 없음. 최후의 안전망 |
| OpenAI API | `USE_LLM=true`, `LLM_PROVIDER=openai`, 키 입력 | 크레딧 충전 시. 코드 수정 불필요 |

두 백엔드 모두 OpenAI 호환 프로토콜이라 `guidance_generator.py`의 호출 경로는 동일하다.

## 라즈베리파이 이전 시

- 코드 수정 불필요. `config.py`가 플랫폼을 자동 감지함.
- 로컬 sLLM: 파이 5(8GB)에서도 `exaone3.5:2.4b` 양자화 모델 구동 가능하나 생성에 수 초~십수 초 소요.
  캐싱(5분) 덕에 반복 이벤트는 즉답. 속도가 부족하면 파이에서는 `USE_LLM=false`(폴백 전용)로 두고,
  개발 PC를 같은 네트워크의 Ollama 서버로 쓰는 방법도 있음 (`OLLAMA_BASE_URL=http://<PC IP>:11434/v1`).
- USB 스피커(ada-3369) 연결 후 `raspi-config` 또는 데스크톱에서 기본 오디오 출력 장치를 USB로 지정하면 pygame이 그대로 사용.
- 폴백 음성 캐싱(`audio_cache/`)은 파이에서도 한 번 실행하거나, 노트북에서 생성한 폴더를 그대로 복사해도 됨.

## FSM 담당 팀원 연동 방법

새 판단 로직(미복용, 중복 복용, 식사, 외출)이 완성되면 해당 지점에서 한 줄만 호출:

```python
guidance.notify(GuidanceEvent.MEDICATION_MISSED)
guidance.notify(GuidanceEvent.DUPLICATE_MEDICATION, context={"last_medication_time": "08:10"})
```

프롬프트(`llm/prompts.py`)와 폴백 문장(`llm/fallback.py`)은 8개 이벤트 전부 이미 작성되어 있음.

---

## 이벤트 어댑터 구조 (2026-08 갱신)

우리 파트의 역할을 한 문장으로: **이벤트 in → 생성형 음성 out.**

팀원 브랜치 3곳(medication-detector / feature-meal-fsm / main의 exit_system)이
각자 만들어 내는 이벤트를 `llm/adapters.py` 가 duck typing 으로 번역해서
`GuidanceService.notify()` 로 흘려보낸다. 비전 코드는 import 하지 않는다.

- 이벤트 목록: `llm/events.py` (활성 6종 + 스케줄 기반 4종)
- 브랜치별 삽입 위치와 코드: **`INTEGRATION.md`**
- 카메라 없는 전체 파이프라인 검증: `python scripts/simulate_events.py`
- 반복 발행 안전장치: 어댑터 edge trigger + 이벤트별 쿨다운
  (`GUIDANCE_EVENT_COOLDOWN_SEC`, 기본 30초)

새 이벤트 2종(`medication_off_schedule`, `meal_start`)이 추가됐으므로
오프라인 폴백 음성을 다시 생성해 둘 것:
`python scripts/generate_fallback_audio.py`

## 지능형 대화 시스템 (수행계획서 주요기능 1 — 2026-08 추가)

`llm/chat.py` — 사용자 발화에 답하는 양방향 대화. 데모: `python scripts/chat_demo.py [--voice]`

- **정보성 질문** (약/식사/시간/날씨): 의도는 키워드 규칙으로 분류하고,
  사실은 코드가 기록(`logs/medication_log.json`, `logs/meal_log.json`)에서 확정,
  sLLM은 말투만 입힌다 (로그 검색+생성 = 수행계획서의 RAG 항목 충족).
  sLLM이 없거나 검증 탈락 시 기록으로 조립한 템플릿 답변이 나가므로 항상 정답.
- **일상 잡담**: 최근 4턴 슬라이딩 윈도우 + 교재 회상법 프롬프트로 sLLM 자유 대화.
- 검증은 chat 모드(3문장/80자, 시간대 언급 허용)로 완화하되 테스트식 질문·
  지어내기·의료 조언 차단은 유지. 판단 근거 블록도 동일하게 출력.
- 식사 완료가 이제 `logs/meal_log.json`에 영구 저장된다 (재시작 유지).

## 음성 입력 (STT — 2026-08 추가)

`voice/stt.py` + `chat_demo` 입력 모드 3종:
`--mic`(엔터 후 말하기, 개발·시연) / `--listen`(상시 대기, **라즈베리파이 최종형**) / 기본 키보드.
구글 무료 STT(ko-KR), 순차 루프라 로봇 발화 중엔 마이크가 닫혀 자기 목소리에
반응하지 않음. STT 불가 시 키보드로 자동 폴백. 파이에서는 USB 마이크 +
`sudo apt install portaudio19-dev` 후 pip 설치. 웨이크워드·오프라인 STT(Vosk)는 향후 개선.
