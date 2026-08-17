"""
지능형 대화 시스템 (수행계획서 주요기능 1: 자연어 처리 기술을 활용한 일상 대화).

구조 (판단은 코드, 말투는 sLLM):

    사용자 발화
      → [의도 분류: 키워드 규칙]  ← sLLM에게 분류를 맡기지 않는다 (2.4B 신뢰 불가)
      → 정보성(약/식사/시간/날씨) → 코드가 기록·시계에서 사실 확정
                                     → sLLM은 그 사실을 말투만 입힘 (미니 RAG)
                                     → sLLM 실패 시 사실로 조립한 템플릿 답변
      → 잡담                       → sLLM 자유 대화 (최근 N턴 슬라이딩 윈도우 + 회상법)
                                     → sLLM 실패 시 정중한 폴백
      → [검증(chat 모드)] → 통과 문장만 반환 → (호출자가 TTS 재생)

원칙: 사실 판단(복용/식사 여부)은 절대 sLLM이 하지 않는다.
      기록 조회는 ScheduleMonitor.slots_report()가 담당하고,
      sLLM이 사실과 다른 말을 하면 검증이 잡아 템플릿으로 교체한다.
"""

import re
from collections import deque
from datetime import datetime

import config
from llm.guidance_generator import GuidanceGenerator
from llm.prompts import (
    CHAT_INTENT_SYSTEM_PROMPT,
    CHAT_QA_SYSTEM_PROMPT,
    CHAT_TALK_SYSTEM_PROMPT,
)


# 의도 종류
INTENT_MEDICATION = "medication_query"   # 약 복용 여부/시간
INTENT_MEAL = "meal_query"               # 식사 여부/시간
INTENT_TIME = "time_query"               # 지금 몇 시/오늘 며칠
INTENT_WEATHER = "weather_query"         # 날씨 (시스템이 모름 → 정직 템플릿)
INTENT_HEALTH = "health_concern"         # 아프다는 호소 (최우선 — 공감·휴식·보호자 안내)
INTENT_CLARIFY = "clarify"               # 지시어가 애매 → 되묻기 (교재: 단어 찾기 도움)
INTENT_SMALLTALK = "smalltalk"           # 그 외 일상 대화

_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]

# sLLM 분류 응답 단어 → 의도 (파싱 순서 중요: "식사 시간"이라 답해도 식사가 먼저 걸리게)
_INTENT_WORDS = {
    "건강": INTENT_HEALTH,
    "약": INTENT_MEDICATION,
    "식사": INTENT_MEAL,
    "날씨": INTENT_WEATHER,
    "시간": INTENT_TIME,
    "잡담": INTENT_SMALLTALK,
}

# 아픔 호소 표현 — 식사/약보다 먼저 검사한다 ("배가 아프네"에 식사 기록으로 답하지 않게)
_HEALTH_RE = re.compile(
    r"(아프|아파|쑤시|저리|어지럽|메스껍|체했|체한|토할|구역질|열이 나|열나"
    r"|두통|골이|숨이 차|가슴이 답답|쓰리|결리)"
)
# 지시대명사 — 무엇을 가리키는지 모호하면 추측하지 않고 되묻는다
_DEMONSTRATIVE_RE = re.compile(r"(그거|그것|이거|이것|저거|저것)")


class ChatAssistant:
    def __init__(self, monitor=None, generator: GuidanceGenerator | None = None):
        """
        monitor: schedule_monitor.ScheduleMonitor (기록 조회처).
                 None이면 기본 경로(logs/)로 생성.
        generator: 기존 GuidanceGenerator 재사용 (sLLM 클라이언트·검증·문장정리 공유).
        """
        if monitor is None:
            from schedule_monitor import ScheduleMonitor
            monitor = ScheduleMonitor()
        self.monitor = monitor
        self.generator = generator or GuidanceGenerator()
        # 잡담용 대화 이력 — 슬라이딩 윈도우. maxlen을 넘으면 오래된 턴부터 탈락.
        self._history = deque(maxlen=config.CHAT_HISTORY_TURNS)
        self._last_intent = None

    # ---------------- public ----------------

    def ask(self, text: str, now=None):
        """
        사용자 발화 1건 처리.

        Returns:
            (sentence: str, source: str, intent: str)
            source ∈ {"llm", "template", "fallback"}
        """
        now = now or datetime.now()
        text = (text or "").strip()
        if not text:
            return "네, 말씀해 보세요.", "template", INTENT_SMALLTALK

        intent, classifier = self._classify(text)
        self._last_intent = intent

        if intent == INTENT_HEALTH:
            sentence, source, facts = self._answer_health(text)
        elif intent == INTENT_CLARIFY:
            sentence, source = "약 말씀이세요, 식사 말씀이세요?", "template"
            facts = "지시어가 무엇을 가리키는지 불명확 → 되묻기 (교재 p.184: 단어 찾기를 도와줌)"
        elif intent == INTENT_MEDICATION:
            sentence, source, facts = self._answer_record("medication", text, now)
        elif intent == INTENT_MEAL:
            sentence, source, facts = self._answer_record("meal", text, now)
        elif intent == INTENT_TIME:
            sentence, source, facts = self._answer_time(now), "template", "시계"
        elif intent == INTENT_WEATHER:
            sentence, source = ("바깥 날씨는 제가 알기 어려워요. 창밖을 한번 보시겠어요?",
                                "template")
            facts = "날씨 정보 없음 (지어내지 않음)"
        else:
            sentence, source, facts = self._smalltalk(text)

        self._history.append((text, sentence))
        self._explain(text, intent, classifier, facts, source, sentence)
        return sentence, source, intent

    def reset_history(self):
        self._history.clear()

    # ---------------- 의도 분류 (키워드 규칙) ----------------

    def _classify(self, text: str):
        """하이브리드 의도 분류.

        1) 키워드 규칙 — 빠르고 확실 (약/밥/몇 시/날씨 등 명시적 어휘)
        2) 규칙에 안 걸리면 sLLM에게 한 단어 분류를 요청 — 돌려 말하기 대응
           ("속이 허하네" → 식사, "그거 삼켰던가" → 약)
        3) sLLM이 없거나 분류 실패 → 잡담

        Returns:
            (intent, 분류 주체 설명 문자열)
        """
        # 후속 질문 처리: "그럼 저녁은?" 처럼 시간대만 말하면 직전 의도를 잇는다
        if re.match(r"^(그럼|그러면|그리고)?\s*(아침|점심|저녁)\s*(은|는|약|밥)?\s*\??$", text):
            if self._last_intent in (INTENT_MEDICATION, INTENT_MEAL):
                return self._last_intent, "후속 질문 → 직전 의도 계승"

        intent = self._classify_rules(text)
        if intent is not None:
            return intent, "키워드 규칙 (LLM 아님)"

        if config.CHAT_HYBRID_INTENT and self.generator.client is not None:
            intent = self._classify_llm(text)
            corrected = self._sanity_check_llm_intent(intent, text)
            if corrected is not None:
                note = ("sLLM 분류 (키워드 없음 → 하이브리드 2단계)"
                        if corrected == intent
                        else "sLLM 분류를 보정 (계획·의향 발화 → 잡담)")
                return corrected, note

        return INTENT_SMALLTALK, "키워드 규칙 (해당 없음 → 잡담)"

    def _classify_rules(self, text: str):
        """1단계: 키워드 규칙. 확신할 수 있을 때만 의도 반환, 아니면 None."""
        # 아픔 호소가 최우선 — "불닭을 잘못 먹었나 배가 아프네"는 식사 질문이 아니다
        if _HEALTH_RE.search(text):
            return INTENT_HEALTH
        if "날씨" in text:
            return INTENT_WEATHER
        if "약" in text:                              # "약 먹었어?", "약 시간 언제야"
            return INTENT_MEDICATION
        if re.search(r"(밥|식사|끼니|배가 고프|배고프|출출|허기)", text):
            return INTENT_MEAL
        # 지시어 + 먹는 동사인데 약/밥이 명시 안 됨 ("그거 먹었나?") → 추측 대신 되묻기
        if _DEMONSTRATIVE_RE.search(text):
            if re.search(r"(먹|드셨|드시|삼)", text):
                return INTENT_CLARIFY
            if self._last_intent in (INTENT_MEDICATION, INTENT_MEAL, INTENT_CLARIFY):
                return INTENT_CLARIFY        # "그거 말고 그거" — 직전 주제 정정 시도
        # 먹는 동사로 끝나는 짧은 질문만 식사로 해석 ("먹었나?", "드셨던가")
        # 문장 중간의 '먹었'("불닭을 먹었나 배가...")은 여기 걸리지 않는다
        if re.search(r"(먹었|먹은|드셨)\S*\s*\??$", text):
            return INTENT_MEAL
        if re.search(r"(몇\s*시|며칠|무슨\s*요일|날짜|오늘이)", text):
            return INTENT_TIME
        return None

    def _answer_health(self, text: str):
        """아픔 호소 응대 — 의료 조언 금지, 공감 + 휴식 + 보호자 안내."""
        facts = "건강 호소 감지 (의료 조언 금지 원칙 → 공감·휴식·보호자 안내)"
        if self.generator.client is not None:
            raw = self._call_chat_llm(
                system=CHAT_TALK_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (f"환자가 몸이 불편하다고 말했습니다: \"{text}\"\n"
                                "공감 한 마디, 편히 쉬시라는 권유, 계속 불편하면 보호자에게 "
                                "알리라는 안내를 짧은 세 문장 이내로 말하세요. "
                                "진단이나 약 권유는 하지 마세요."),
                }],
                temperature=0.4,
            )
            accepted = self._accept(raw)
            if accepted is not None:
                return accepted, "llm", facts
        return ("많이 불편하시겠어요. 잠시 편안하게 쉬어 보세요. "
                "계속 아프시면 보호자에게 알려 주세요."), "template", facts

    def _classify_llm(self, text: str):
        """2단계: sLLM 한 단어 분류. 응답 파싱 실패 시 None."""
        # 직전 대화를 같이 보여줘야 "그거" 같은 지시어의 의도를 짐작할 수 있다
        recent = list(self._history)[-2:]
        convo = "\n".join(f"환자: {u}\n로봇: {b}" for u, b in recent)
        content = f"[직전 대화]\n{convo}\n\n[이번 발화]\n{text}" if convo else text
        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[
                    {"role": "system", "content": CHAT_INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=10,
                temperature=0.0,   # 분류는 창의성 불필요 — 항상 같은 답
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Chat] 의도 분류 LLM 호출 실패: {e}")
            return None
        return self._parse_intent_word(raw)

    @staticmethod
    def _sanity_check_llm_intent(intent, text: str):
        """sLLM 분류 결과 상식 검증.

        2.4B 분류기는 시간 냄새가 나는 단어("좀 있다가", "언제쯤")에 낚여서
        외출 계획 발화를 '시간 질문'으로 오분류한다. 시각을 묻는 게 아니라
        무언가를 하겠다는 계획·의향이면 잡담으로 교정한다 — 오분류가 남더라도
        잡담(sLLM 자유 대화)이 시계 읊기보다 항상 자연스러운 대응이다.
        """
        if intent == INTENT_TIME:
            # 시각을 묻는 게 아니라 계획·의향 발화 → 잡담
            if re.search(r"(야겠|갈까|가야|나가|다녀오|해볼까|하고 싶|싶다|싶네)", text):
                return INTENT_SMALLTALK
            # 문장에 시간 관련 단어가 하나도 없으면 시간 질문일 수 없다.
            # (분류기가 직전 대화 문맥에 끌려가 소음 파편까지 '시간'으로 찍는 것 차단)
            if not re.search(r"(시간|몇\s*시|시계|며칠|요일|날짜|언제|지금|오늘)", text):
                return INTENT_SMALLTALK
        return intent

    @staticmethod
    def _parse_intent_word(raw: str):
        for word, intent in _INTENT_WORDS.items():
            if word in raw:
                return intent
        return None

    def _accept(self, raw):
        """sLLM 응답 수용 판정: 통과 → 그대로, 탈락 → 뒷문장 절삭 구제, 실패 → None."""
        if raw is None:
            return None
        failures = self.generator._validation_failures(raw, mode="chat")
        if not failures:
            return raw
        salvaged = self.generator.salvage(raw, mode="chat")
        if salvaged is not None:
            print(f"[Chat] 뒷문장 절삭으로 구제: '{salvaged}' "
                  f"(원문 탈락: {' / '.join(failures)})")
            return salvaged
        print(f"[Chat] 검증 탈락 → 대체 문장 사용: '{raw}' ({' / '.join(failures)})")
        return None

    # ---------------- 정보성: 기록 기반 답변 (미니 RAG) ----------------

    def _answer_record(self, kind: str, question: str, now):
        report = self.monitor.slots_report(kind, now)
        facts = self._format_facts(kind, report)

        # 1) sLLM: 사실을 근거로 말투만 입힘
        if self.generator.client is not None:
            raw = self._call_chat_llm(
                system=CHAT_QA_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (f"[오늘의 기록]\n{facts}\n\n[환자의 질문]\n{question}\n\n"
                                "기록만 근거로 짧게 답하세요."),
                }],
                temperature=0.3,
            )
            accepted = self._accept(raw)
            if accepted is not None:
                return accepted, "llm", facts

        # 2) 템플릿: 사실을 코드가 직접 문장으로 조립 (sLLM 없이도 정답 보장)
        # 질문이 특정 시간대를 집으면 ("점심 약은?", "그럼 저녁은?") 그 시간대만 답한다
        focus = [s for s in ("아침", "점심", "저녁") if s in question]
        if focus:
            report = [r for r in report if r["label"] in focus]
        return self._template_answer(kind, report), "template", facts

    def _format_facts(self, kind: str, report: list) -> str:
        word = "약" if kind == "medication" else "식사"
        lines = []
        for r in report:
            if r["taken"]:
                t = f" ({r['time']})" if r.get("time") else ""
                lines.append(f"- {r['label']} {word}: 완료{t}")
            elif r["phase"] == "예정":
                lines.append(f"- {r['label']} {word}: 아직 시간 전")
            else:
                lines.append(f"- {r['label']} {word}: 기록 없음")
        return "\n".join(lines)

    def _template_answer(self, kind: str, report: list) -> str:
        """sLLM 없이 기록만으로 조립하는 정답 문장 (한 문장 = 한 시간대)."""
        word, josa = ("약", "은") if kind == "medication" else ("식사", "는")
        done = [r for r in report if r["taken"]]
        missed = [r for r in report if not r["taken"] and r["phase"] != "예정"]
        upcoming = [r for r in report if not r["taken"] and r["phase"] == "예정"]

        parts = []
        if done:
            labels = "과 ".join(r["label"] for r in done)
            parts.append(f"{labels} {word}{josa} 드셨어요.")
        if missed:
            labels = "과 ".join(r["label"] for r in missed)
            parts.append(f"{labels} {word}{josa} 아직이에요.")
        if not done and not missed:
            return f"오늘은 아직 {word} 시간 전이에요. 시간이 되면 알려드릴게요."
        if upcoming and len(parts) < 2:
            parts.append(f"{upcoming[0]['label']} {word}{josa} 아직 시간 전이에요.")
        return " ".join(parts[:config.CHAT_MAX_SENTENCES])

    # ---------------- 정보성: 시간/날짜 (시계 = 정답, LLM 불필요) ----------------

    def _answer_time(self, now) -> str:
        ampm = "오전" if now.hour < 12 else "오후"
        h = now.hour if 1 <= now.hour <= 12 else abs(now.hour - 12) or 12
        weekday = _WEEKDAYS[now.weekday()]
        return (f"지금은 {ampm} {h}시 {now.minute}분이에요. "
                f"오늘은 {now.month}월 {now.day}일 {weekday}요일이에요.")

    # ---------------- 잡담 (슬라이딩 윈도우 + 회상법) ----------------

    def _smalltalk(self, text: str):
        facts = f"대화 이력 {len(self._history)}턴 (최대 {config.CHAT_HISTORY_TURNS}턴 유지)"
        if self.generator.client is not None:
            messages = []
            for user_turn, bot_turn in self._history:
                messages.append({"role": "user", "content": user_turn})
                messages.append({"role": "assistant", "content": bot_turn})
            messages.append({"role": "user", "content": text})

            raw = self._call_chat_llm(
                system=CHAT_TALK_SYSTEM_PROMPT, messages=messages, temperature=0.7,
            )
            accepted = self._accept(raw)
            if accepted is not None:
                return accepted, "llm", facts

        return ("지금은 이야기를 나누기가 어려워요. 잠시 후에 다시 말 걸어 주세요.",
                "fallback", facts)

    # ---------------- internal ----------------

    def _call_chat_llm(self, system: str, messages: list, temperature: float):
        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=120,
                temperature=temperature,
            )
            raw = response.choices[0].message.content.strip()
            return self.generator.clean_sentence(raw, max_sentences=config.CHAT_MAX_SENTENCES)
        except Exception as e:
            print(f"[Chat] LLM 호출 실패: {e}")
            return None

    def _explain(self, question, intent, classifier, facts, source, sentence):
        """대화 판단 근거 출력 (안내 파이프라인의 판단 근거와 같은 철학)."""
        if not config.GUIDANCE_EXPLAIN:
            return
        source_name = {
            "llm": f"로컬 sLLM 생성 (model={self.generator.model})",
            "template": "코드가 사실로 조립한 템플릿 (LLM 미사용)",
            "fallback": "폴백 문장",
        }[source]
        print("┌─ 대화 판단 근거 " + "─" * 41)
        print(f"│ 질문      : {question}")
        print(f"│ 의도 분류 : {intent} ({classifier})")
        for i, line in enumerate(str(facts).split("\n")):
            print(f"│ {'조회 근거' if i == 0 else '          '} : {line}")
        print(f"│ 생성 경로 : {source_name}")
        print("└" + "─" * 58)
