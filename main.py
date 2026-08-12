import argparse
import cv2

from vision.camera import Camera
from vision.hand import HandDetector
from vision.face import FaceDetector
from vision.medication import MedicationDetector
from utils.logger import MedicationLogger


def run_medication():
    """최신 약 복용 기능을 실행한다."""
    camera = Camera(0)
    hand_detector = HandDetector(max_num_hands=2)
    face_detector = FaceDetector()
    medication_detector = MedicationDetector()

    medication_logger = MedicationLogger(
        log_path="logs/medication_log.json"
    )

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            frame = cv2.flip(frame, 1)

            hand_result, hand_points = hand_detector.detect(frame)
            face_result, face_boxes = face_detector.detect(frame)

            medication_result = medication_detector.process(
                frame=frame,
                hand_points=hand_points,
                face_boxes=face_boxes,
            )

            medication_log_result = medication_logger.log_if_taken(
                medication_result["state"]
            )

            frame = hand_detector.draw(frame, hand_result)
            frame = face_detector.draw(frame, face_result)
            frame = medication_detector.draw_result(
                frame,
                medication_result,
            )

            cv2.putText(
                frame,
                f"Hands: {len(hand_points)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Faces: {len(face_boxes)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            cv2.imshow("Medication Care System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                medication_detector.reset()

    finally:
        camera.release()
        hand_detector.close()
        face_detector.close()
        cv2.destroyAllWindows()
        print("프로그램 종료 완료")

    return 0

def run_meal(meal_args=None):
    """기존 main_meal.py의 식사 기능을 호출한다."""
    from main_meal import main as meal_main

    if meal_args is None:
        meal_args = []

    return meal_main(meal_args)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Seawater 통합 실행 파일",
    )

    parser.add_argument(
        "--mode",
        choices=("medication", "meal"),
        default=None,
        help="실행 기능. 지정하지 않으면 터미널에서 선택합니다.",
    )

    parser.add_argument(
        "--meal-source",
        default="",
        help="식사 테스트용 영상 파일 경로",
    )

    parser.add_argument(
        "--meal-camera",
        type=int,
        default=0,
        help="식사 기능 카메라 인덱스",
    )

    parser.add_argument(
        "--meal-no-display",
        action="store_true",
        help="식사 기능 OpenCV 화면 끄기",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    mode = args.mode

    if mode is None:
        print()
        print("실행할 기능을 선택하세요.")
        print("1. Medication")
        print("2. Meal")

        choice = input("선택 (1/2): ").strip()

        if choice == "1":
            mode = "medication"
        elif choice == "2":
            mode = "meal"
        else:
            print("잘못된 선택입니다.")
            return 1

    if mode == "medication":
        return run_medication()

    if mode == "meal":
        meal_args = ["--camera", str(args.meal_camera)]

        if args.meal_source:
            meal_args.extend(["--source", args.meal_source])

        if args.meal_no_display:
            meal_args.append("--no-display")

        return run_meal(meal_args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
