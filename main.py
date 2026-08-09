import cv2
import math

from vision.camera import Camera
from vision.hand import HandDetector
from vision.face import FaceDetector
from vision.medication import MedicationDetector
from utils.logger import MedicationLogger


def main():
    camera = Camera(0)
    hand_detector = HandDetector(max_num_hands=2)
    face_detector = FaceDetector()
    medication_detector = MedicationDetector()
    logger = MedicationLogger()

    saved = False

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            frame = cv2.flip(frame, 1)

            hand_result, hand_points = hand_detector.detect(frame)
            face_result, face_boxes = face_detector.detect(frame)

            medication_result = medication_detector.process(
                frame = frame,
                hand_points = hand_points,
                face_boxes = face_boxes
            )

            frame = hand_detector.draw(frame, hand_result)
            frame = face_detector.draw(frame, face_result)
            frame = medication_detector.draw_result(frame, medication_result)

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

            if medication_result["state"] in ["MEDICATION_DONE", "MEDICATION DONE"]:
                if not saved:
                    logger.save_event(
                        event="Medication Completed",
                        state=medication_result["state"],
                    )
                    saved = True

            cv2.imshow("Medication Care System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                medication_detector.reset()
                saved = False


    finally:
        camera.release()
        hand_detector.close()
        face_detector.close()
        cv2.destroyAllWindows()
        print("프로그램 종료 완료")


if __name__ == "__main__":
    main()
