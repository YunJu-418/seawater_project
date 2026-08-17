"""
LLM 기반 안내 문장 생성 모듈.

흐름:
    FSM 이벤트 → OpenAI API 호출 → 응답 검증(의사소통 원칙) → 안내 문장 반환
                        └ 실패/검증 탈락 시 → 폴백 템플릿 반환

특징:
- API 키가 없거나 openai 패키지가 없어도 폴백으로 항상 동작 (시연 안정성)
- 같은 이벤트가 짧은 시간에 반복되면 최근 생성 문장을 재사용 (비용/지연 절감)
"""

import re
import time

import config
from llm.events import GuidanceEvent
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from llm.fallback import get_fallback

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# 이모지·기호 문자 범위 (음성 안내 문장에서 제거 대상)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # 이모지 본체 (기호, 표정, 사물 등)
    "\U00002600-\U000027BF"   # 기타 기호 (☀ ✅ ✨ 등)
    "\U0001F1E6-\U0001F1FF"   # 국기
    "\u2b50\u2764\ufe0f\u200d"  # 별, 하트, 변형 선택자
    "]+"
)


class GuidanceGenerator:
    def __init__(self, cache_ttl_sec: int = 300):
        """
        cache_ttl_sec: 같은 이벤트에 대해 생성 문장을 재사용하는 시간(초)
        """
        self.cache_ttl_sec = cache_ttl_sec
        self._cache = {}  # {event: (sentence, timestamp)}

        self.client = None
        self.model = None
        if not config.USE_LLM:
            print("[GuidanceGenerator] USE_LLM=false → 폴백 템플릿 모드로 동작합니다.")
        elif not _OPENAI_AVAILABLE:
            print("[GuidanceGenerator] openai 패키지 미설치 → 폴백 템플릿 모드로 동작합니다.")
        elif config.LLM_PROVIDER == "ollama":
            # 로컬 sLLM (Ollama). OpenAI 호환 엔드포인트를 사용하므로 SDK를 그대로 재사용.
            # api_key 는 형식상 필요할 뿐 실제로 검사되지 않는다.
            self.client = OpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key="ollama",
                timeout=config.LLM_TIMEOUT_SEC,
                max_retries=config.LLM_MAX_RETRY,
            )
            self.model = config.OLLAMA_MODEL
            print(f"[GuidanceGenerator] 로컬 sLLM 모드 (Ollama, model={self.model})")
        elif config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                print("[GuidanceGenerator] OPENAI_API_KEY 미설정 → 폴백 템플릿 모드로 동작합니다.")
            else:
                self.client = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                    timeout=config.LLM_TIMEOUT_SEC,
                    max_retries=config.LLM_MAX_RETRY,
                )
                self.model = config.LLM_MODEL
                print(f"[GuidanceGenerator] OpenAI API 모드 (model={self.model})")
        else:
            print(f"[GuidanceGenerator] 알 수 없는 LLM_PROVIDER='{config.LLM_PROVIDER}' → 폴백 템플릿 모드로 동작합니다.")

    # ---------------- public ----------------

    def generate(self, event: GuidanceEvent, context: dict | None = None):
        """
        안내 문장 생성.

        Returns:
            (sentence: str, source: str)  source ∈ {"llm", "cache", "fallback"}
        """
        # 1) 캐시 확인 (시간대가 다르면 다른 캐시 — 점심 문장이 저녁에 재사용되지 않도록)
        cached = self._get_cache(event, context)
        if cached is not None:
            self._explain(event, context, "cache", cached,
                          [f"최근 {self.cache_ttl_sec}초 내 같은 상황에서 검증을 통과한 문장을 재사용"])
            return cached, "cache"

        # 2) LLM 호출
        fallback_reasons = ["LLM 비활성 상태 → 사람이 작성한 폴백 문장 사용"]
        if self.client is not None:
            sentence = self._call_llm(event, context)
            if sentence is not None:
                failures = self._validation_failures(sentence, context)
                if not failures:
                    self._set_cache(event, sentence, context)
                    self._explain(event, context, "llm", sentence, [self._pass_summary(sentence)])
                    return sentence, "llm"
                # 통째로 버리기 전에 구제 시도: 뒷문장부터 잘라가며 통과분만 사용
                salvaged = self.salvage(sentence, context, mode="guidance")
                if salvaged is not None:
                    print(f"[GuidanceGenerator] 뒷문장 절삭으로 구제: '{salvaged}'")
                    self._set_cache(event, salvaged, context)
                    self._explain(event, context, "llm", salvaged, [
                        self._pass_summary(salvaged),
                        "원문 검증 탈락(" + " / ".join(failures) + ") → 뒷문장을 잘라 구제",
                    ])
                    return salvaged, "llm"
                print(f"[GuidanceGenerator] 검증 탈락 → 폴백 사용: '{sentence}'")
                fallback_reasons = [
                    "sLLM 생성 문장이 검증에서 탈락 → 폴백 문장으로 교체",
                    "탈락 사유: " + " / ".join(failures),
                ]
            else:
                fallback_reasons = ["sLLM 호출 실패(응답 없음/오류) → 폴백 문장 사용"]

        # 3) 폴백
        sentence = get_fallback(event)
        self._explain(event, context, "fallback", sentence, fallback_reasons)
        return sentence, "fallback"

    # ---------------- internal ----------------

    def _call_llm(self, event, context):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(event, context)},
                ],
                max_tokens=80,
                temperature=0.5,
            )
            sentence = response.choices[0].message.content.strip()
            return self.clean_sentence(sentence)
        except Exception as e:
            print(f"[GuidanceGenerator] LLM 호출 실패: {e}")
            return None

    def clean_sentence(self, sentence: str, max_sentences: int | None = None) -> str:
        """sLLM 원문 응답 정리 (안내/대화 공용).

        여러 줄 → 첫 라인, 따옴표 제거, 이모지 제거, 느낌표 완화,
        문장 수 초과 시 앞에서부터 허용 개수만 사용.
        """
        if max_sentences is None:
            max_sentences = config.GUIDANCE_MAX_SENTENCES
        # 로컬 sLLM이 여러 줄로 답하는 경우 → 첫 번째 유효 라인만 사용
        if "\n" in sentence:
            lines = [ln.strip() for ln in sentence.splitlines() if ln.strip()]
            sentence = lines[0] if lines else sentence
        # 따옴표 등 불필요한 감싸기 제거
        sentence = sentence.strip('"').strip("'").strip()
        # 이모지/기호 제거 (음성 문장이므로 — 제거로 구제, 남으면 검증에서 탈락)
        sentence = _EMOJI_RE.sub("", sentence)
        # 느낌표 반복(!!, !!!) → 마침표로 완화
        sentence = re.sub(r"!{1,}", ".", sentence)
        # 기호 제거 후 남는 이중 공백/이중 마침표 정리
        sentence = re.sub(r"\s{2,}", " ", sentence)
        sentence = re.sub(r"\.{2,}", ".", sentence).strip()
        # 문장 수 초과 시 앞에서부터 허용 개수만 사용 (소형 모델이 길게 쓰는 습관 구제)
        parts = re.findall(r"[^.!?。]+[.!?。]?", sentence)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > max_sentences:
            sentence = " ".join(parts[:max_sentences]).strip()
        return sentence

    def _validate(self, sentence: str, context: dict | None = None, mode: str = "guidance") -> bool:
        """치매 환자 의사소통 원칙 기반 응답 검증 (통과 여부만)"""
        return not self._validation_failures(sentence, context, mode)

    def _validation_failures(self, sentence: str, context: dict | None = None, mode: str = "guidance") -> list:
        """검증 규칙을 전부 적용하고, 위반한 규칙의 '사유 목록'을 반환한다 (빈 목록 = 통과).

        판단 근거 출력(GUIDANCE_EXPLAIN)에서 어떤 규칙에 걸렸는지 설명하기 위해
        불리언 대신 사유를 수집하는 구조로 되어 있다.
        """
        failures = []
        if not sentence:
            return ["빈 문장"]
        # 모드별 허용치: 대화(chat)는 기록 요약이 들어가 안내보다 약간 여유
        max_chars = config.GUIDANCE_MAX_CHARS if mode == "guidance" else config.CHAT_MAX_CHARS
        max_sents = config.GUIDANCE_MAX_SENTENCES if mode == "guidance" else config.CHAT_MAX_SENTENCES
        # 길이 제한 (짧고 명확한 표현)
        if len(sentence) > max_chars:
            failures.append(f"길이 초과 ({len(sentence)}자 > {max_chars}자)")
        # 문장 수 제한 (한 문장에 한 가지 행동/정보)
        sentences = [s for s in re.split(r"[.!?。]", sentence) if s.strip()]
        if len(sentences) > max_sents:
            failures.append(f"문장 수 초과 ({len(sentences)}개 > {max_sents}개)")
        # 부정 표현 금지 (유도형 표현 원칙)
        for word in ("하지 마", "하지마", "안 됩니다", "안됩니다", "금지"):
            if word in sentence:
                failures.append(f"부정·금지 표현 포함 ('{word}')")
                break
        # 기억 확인(테스트식) 질문 금지 — 교재 p.185 "불필요한 테스트는 삼가한다"
        m = re.search(r"\S*[았었셨]나요", sentence)
        if m:
            failures.append(f"기억 확인 질문 포함 ('{m.group()}' — 교재 p.185 테스트 금지)")
        # 지시대명사 금지 — 교재 p.186 (사물의 이름을 구체적으로 말하기)
        m = re.search(r"(이것|저것|그것|여기|저기|거기)", sentence)
        if m:
            failures.append(f"지시대명사 포함 ('{m.group()}' — 교재 p.186)")
        # 이모지 잔존 시 탈락 (TTS 부적합)
        if _EMOJI_RE.search(sentence):
            failures.append("이모지·기호 포함 (TTS 부적합)")
        # 영문자 포함 시 탈락 (외래어 금지 원칙 + TTS가 부자연스럽게 읽음)
        if re.search(r"[A-Za-z]", sentence):
            failures.append("영문자 포함 (한글 전용 원칙)")
        # 한자 포함 시 탈락 (sLLM이 간혹 한자를 섞음 — TTS 부적합)
        if re.search(r"[\u4E00-\u9FFF]", sentence):
            failures.append("한자 포함 (한글 전용 원칙)")
        # 프롬프트 예시 등에서 새어 나온 가짜 호칭 차단 (설정된 실제 호칭만 허용)
        for placeholder in ("홍길동", "김철수", "ㅇㅇㅇ"):
            if placeholder in sentence and placeholder != config.PATIENT_NAME:
                failures.append(f"가짜 호칭 포함 ('{placeholder}')")
                break
        # 시간대 사실 검증 — 컨텍스트가 허용한 시간대 이름만 말할 수 있다.
        # (sLLM이 시각만 보고 "지금은 점심 시간" 식으로 추측하는 것을 기계적으로 차단.
        #  컨텍스트에 시간대 정보가 없으면 시간대 이름 자체를 금지 — 원칙 7)
        if mode == "guidance":
            # (대화 모드에서는 기록 요약에 세 시간대가 모두 사실로 제공되므로 이 검사를 생략)
            allowed_slots = set()
            if context:
                for key in ("meal_label", "next_slot_label"):
                    if context.get(key):
                        allowed_slots.add(context[key])
            for slot in ("아침", "점심", "저녁"):
                if slot in sentence and slot not in allowed_slots:
                    allowed = ", ".join(allowed_slots) if allowed_slots else "없음"
                    failures.append(f"사실과 다른 시간대 언급 ('{slot}' — 허용: {allowed})")
        # 날씨 창작 차단 — 시스템은 날씨 정보를 모르므로 언급 자체가 지어낸 것
        for word in ("날씨", "춥", "쌀쌀", "추워", "더워", "무더위", "비가 오", "눈이 오"):
            if word in sentence:
                failures.append(f"확인되지 않은 날씨 정보 창작 ('{word}')")
                break
        # 줄바꿈/목록 형태 금지 (음성 안내 부적합)
        if "\n" in sentence or sentence.startswith(("-", "*", "1.")):
            failures.append("줄바꿈/목록 형식 (음성 부적합)")
        return failures

    def salvage(self, sentence: str, context: dict | None = None,
                mode: str = "guidance"):
        """검증 탈락 문장의 구제 — 뒷문장부터 하나씩 잘라가며 재검증한다.

        예: "외로우셨군요. 언제든 말씀해 주세요. TV를 보세요." (영문 포함, 길이 초과)
            → 셋째 문장 절삭 → 앞 두 문장이 검증 통과 → 그것만 사용.
        규칙을 완화하는 게 아니라, 문장의 '통과하는 앞부분'만 살리는 것이므로
        안전 기준은 그대로 유지된다. 첫 문장부터 위반이면 None (→ 폴백).
        """
        parts = re.findall(r"[^.!?。]+[.!?。]?", sentence)
        parts = [p.strip() for p in parts if p.strip()]
        parts = parts[:-1]  # 원문 전체는 이미 탈락했으므로 한 문장 자르고 시작
        while parts:
            candidate = " ".join(parts).strip()
            if not self._validation_failures(candidate, context, mode):
                return candidate
            parts = parts[:-1]
        return None

    def _pass_summary(self, sentence: str) -> str:
        n = len([s for s in re.split(r"[.!?。]", sentence) if s.strip()])
        return (f"검증 통과 — 길이 {len(sentence)}/{config.GUIDANCE_MAX_CHARS}자, "
                f"문장 {n}/{config.GUIDANCE_MAX_SENTENCES}개, "
                "금지어·테스트질문·지시대명사·한글전용·시간대·날씨 검사 이상 없음")

    # 판단 근거 출력용: 컨텍스트 키 → 한글 라벨
    _CONTEXT_LABELS = {
        "time": "현재 시각",
        "meal_label": "시간대",
        "next_slot_label": "다음 복약 시간대",
        "next_slot_starts_at": "다음 시간대 시작",
        "slot_ends_at": "시간대 종료",
        "taken_time": "복용 기록 시각",
        "bite_count": "섭취 횟수",
        "drink_count": "음수 횟수",
        "patient_name": "환자 호칭",
    }

    def _explain(self, event, context, source, sentence, notes):
        """이 안내 문장이 왜/어떻게 만들어졌는지 터미널에 출력 (GUIDANCE_EXPLAIN)."""
        if not config.GUIDANCE_EXPLAIN:
            return
        # 판단 상황: 이벤트 지시문에서 상황 설명 문장만 추출 (생성 지시 문구는 제외)
        prompt_head = build_user_prompt(event, context).split("\n")[0]
        situation = []
        for part in re.findall(r"[^.]+\.", prompt_head):
            if any(w in part for w in ("생성하세요", "형태로", "말하지 마세요", "넣지 마세요")):
                continue
            situation.append(part.strip())
            if len(situation) == 2:
                break
        ctx_items = []
        for key, label in self._CONTEXT_LABELS.items():
            if context and context.get(key):
                ctx_items.append(f"{label}={context[key]}")
        source_name = {
            "llm": f"로컬 sLLM 생성 (model={self.model})" if self.model else "LLM 생성",
            "cache": "캐시 재사용",
            "fallback": "폴백 템플릿 (사람이 작성)",
        }[source]

        print("┌─ 판단 근거 " + "─" * 46)
        print(f"│ 이벤트    : {event.value}")
        print(f"│ 판단 상황 : {' '.join(situation) if situation else '-'}")
        print(f"│ 참고 정보 : {' · '.join(ctx_items) if ctx_items else '-'}")
        print(f"│ 생성 경로 : {source_name}")
        for i, note in enumerate(notes):
            print(f"│ {'결과 판정' if i == 0 else '          '} : {note}")
        print("└" + "─" * 58)

    def _cache_key(self, event, context):
        label = context.get("meal_label") if context else None
        return (event, label)

    def _get_cache(self, event, context=None):
        key = self._cache_key(event, context)
        item = self._cache.get(key)
        if item is None:
            return None
        sentence, ts = item
        if time.time() - ts > self.cache_ttl_sec:
            del self._cache[key]
            return None
        return sentence

    def _set_cache(self, event, sentence, context=None):
        self._cache[self._cache_key(event, context)] = (sentence, time.time())
