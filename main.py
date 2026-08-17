import cv2
import math

from vision.camera import Camera
from vision.hand import HandDetector
from vision.face import FaceDetector
from vision.medication import MedicationDetector
from utils.logger import MedicationLogger
from guidance_service import GuidanceService
from llm.events import GuidanceEvent


def calc_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return int(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def draw_hand_info(frame, hand_points):
    h, w, _ = frame.shape

    if len(hand_points) == 0:
        return frame, None

    hand = hand_points[0]
    index = hand[8]
    index_point = (int(index["x"] * w), int(index["y"] * h))

    cv2.circle(frame, index_point, 8, (0, 255, 255), -1)
    cv2.putText(
        frame,
        f"Index: {index_point}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return frame, index_point


def draw_face_info(frame, face_boxes):
    if len(face_boxes) == 0:
        return frame, None

    face = face_boxes[0]
    face_center = (
        face["x"] + face["w"] // 2,
        face["y"] + face["h"] // 2,
    )

    cv2.circle(frame, face_center, 8, (255, 0, 0), -1)
    cv2.putText(
        frame,
        f"Face: {face_center}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    return frame, face_center


def main():
    camera = Camera(0)
    hand_detector = HandDetector(max_num_hands=2)
    face_detector = FaceDetector()
    medication_detector = MedicationDetector()
    logger = MedicationLogger()
    guidance = GuidanceService(enable_voice=True)

    state = "WAITING"
    grabbed_count = 0
    done_count = 0
    saved = False

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            frame = cv2.flip(frame, 1)

            hand_result, hand_points = hand_detector.detect(frame)
            face_result, face_boxes = face_detector.detect(frame)
            medication_box = medication_detector.detect(frame)

            frame = hand_detector.draw(frame, hand_result)
            frame = face_detector.draw(frame, face_result)
            frame = medication_detector.draw(frame, medication_box)

            frame, index_point = draw_hand_info(frame, hand_points)
            frame, face_center = draw_face_info(frame, face_boxes)

            medication_center = medication_box["center"]

            hand_med_dist = calc_distance(index_point, medication_center)
            hand_face_dist = calc_distance(index_point, face_center)

            if hand_med_dist is not None and hand_med_dist < 80:
                grabbed_count += 1
            else:
                grabbed_count = 0

            if grabbed_count > 10:
                state = "GRABBED"

            if state == "GRABBED":
                if hand_face_dist is not None and hand_face_dist < 120:
                    done_count += 1
                else:
                    done_count = 0

                if done_count > 10:
                    state = "MEDICATION DONE"

                    if not saved:
                        logger.save_event(
                            event="Medication Completed",
                            state=state,
                        )
                        saved = True

                        # LLM 기반 맞춤형 음성 안내 (별도 스레드에서 처리 → 비전 루프 유지)
                        guidance.notify(GuidanceEvent.MEDICATION_DONE)

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

            if hand_med_dist is not None:
                cv2.putText(
                    frame,
                    f"Hand-Med Dist: {hand_med_dist}",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if hand_face_dist is not None:
                cv2.putText(
                    frame,
                    f"Hand-Face Dist: {hand_face_dist}",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                f"STATE: {state}",
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

            cv2.imshow("Medication Care System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        hand_detector.close()
        face_detector.close()
        cv2.destroyAllWindows()
        print("프로그램 종료 완료")


if __name__ == "__main__":
    main()
