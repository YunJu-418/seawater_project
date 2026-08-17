"""
지능형 대화 데모 — 세 가지 입력 모드.

    python scripts/chat_demo.py                    # 키보드 입력 (기본)
    python scripts/chat_demo.py --mic --voice      # 엔터 누르고 말하기 (개발·시연용)
    python scripts/chat_demo.py --listen --voice   # 상시 대기 (라즈베리파이 최종형)

- --mic    : 엔터를 누르면 듣기 시작 → 말하면 인식 → 답변. 소음 환경 시연에 안전.
- --listen : 항상 귀를 열어두고, 말소리가 감지되면 알아서 듣고 답한다.
             듣기→답변→말하기가 순차 루프라 로봇이 말하는 동안엔 마이크가 닫혀 있어
             자기 목소리에 반응하지 않는다. "종료"라고 말하면 끝.
- STT 실패(패키지/마이크/인터넷) 시 키보드 모드로 자동 폴백 — 데모가 죽지 않는다.

동작 확인용: "오늘 약 먹었어?" / "밥은?" / "심심하네" / "배가 아프네" / "그거 먹었나?"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.chat import ChatAssistant

QUIT_WORDS = ("종료", "그만할게", "그만 할게", "잘 가", "잘가")
# 한/영 전환을 잊고 q를 눌러도 종료되게 (한글 자판에서 q = ㅂ)
QUIT_KEYS = ("q", "quit", "exit", "종료", "그만", "ㅂ", "ㅃ")


def _drain_stdin():
    """듣기 중에 눌린 키들이 버퍼에 남아 다음 프롬프트를 통과시키는 것을 방지."""
    try:
        import msvcrt  # Windows
        while msvcrt.kbhit():
            msvcrt.getwch()
    except ImportError:
        try:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
        except Exception:
            pass


def _parse_mic_index():
    if "--mic-index" in sys.argv:
        try:
            return int(sys.argv[sys.argv.index("--mic-index") + 1])
        except (IndexError, ValueError):
            print("--mic-index 뒤에 장치 번호를 붙여주세요 (예: --mic-index 1)")
    return None


def main():
    use_voice = "--voice" in sys.argv
    mode = "listen" if "--listen" in sys.argv else ("mic" if "--mic" in sys.argv else "keyboard")

    voice = None
    if use_voice:
        from voice.tts import VoiceGuide
        voice = VoiceGuide()

    listener = None
    if mode in ("mic", "listen"):
        from voice.stt import SpeechListener, list_microphones
        if "--list-mics" in sys.argv:
            list_microphones()
            return
        listener = SpeechListener(device_index=_parse_mic_index())
        if not listener.available:
            print("→ 음성 입력을 쓸 수 없어 키보드 모드로 전환합니다.")
            mode = "keyboard"

    print("=" * 60)
    print(" 지능형 대화 데모 — 약/식사/시간은 기록으로 답하고,")
    print(" 그 외에는 일상 대화(회상법)로 이어갑니다.")
    if mode == "keyboard":
        print(" [키보드 모드] 입력 후 엔터. 종료: q")
    elif mode == "mic":
        print(" [마이크 모드] 엔터 → 말하기. 종료: q 입력")
    else:
        print(" [상시 대기 모드] 그냥 말을 거세요. 종료: \"종료\"라고 말하기")
    print("=" * 60)

    assistant = ChatAssistant()

    while True:
        # ── 입력 (모드별) ─────────────────────────────
        if mode == "keyboard":
            try:
                text = input("\n나> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("q", "quit", "exit", "종료"):
                break

        elif mode == "mic":
            _drain_stdin()   # 듣기 중 눌린 키 제거 — 프롬프트 자동 통과 방지
            try:
                cmd = input("\n(엔터=말하기 / q=종료, 한/영 무관) ")
            except (EOFError, KeyboardInterrupt):
                break
            if cmd.strip().lower() in QUIT_KEYS:
                break
            print("듣고 있어요... 말씀하세요")
            text = listener.listen_once(timeout=6)
            if text is None:
                continue
            print(f"나(음성)> {text}")

        else:  # listen — 상시 대기
            try:
                text = listener.listen_once(timeout=None)
            except KeyboardInterrupt:
                break
            if text is None:
                continue
            print(f"\n나(음성)> {text}")
            if any(w in text for w in QUIT_WORDS):
                farewell = "네, 대화를 마칠게요. 편안한 하루 보내세요."
                print(f"로봇> {farewell}")
                if voice is not None:
                    voice.speak(farewell)
                return

        # ── 답변 + 음성 (모든 모드 공통) ───────────────
        sentence, source, intent = assistant.ask(text)
        print(f"로봇({source})> {sentence}")
        if voice is not None:
            voice.speak(sentence)   # 재생이 끝나야 다음 듣기로 넘어감 (자기 목소리 차단)

    print("\n대화를 마칩니다. 편안한 하루 보내세요.")


if __name__ == "__main__":
    main()
