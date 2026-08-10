import cv2
import mediapipe as mp


class FaceMeshDetector:
    """식사 행동 분석용 얼굴·입 랜드마크 검출기."""

    UPPER_LIP_INDEX = 13
    LOWER_LIP_INDEX = 14

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        faces = []

        if result.multi_face_landmarks:
            frame_height, frame_width = frame.shape[:2]

            for face_landmarks in result.multi_face_landmarks:
                upper_lip = face_landmarks.landmark[self.UPPER_LIP_INDEX]
                lower_lip = face_landmarks.landmark[self.LOWER_LIP_INDEX]

                mouth_x = (upper_lip.x + lower_lip.x) / 2.0
                mouth_y = (upper_lip.y + lower_lip.y) / 2.0

                xs = [landmark.x for landmark in face_landmarks.landmark]
                ys = [landmark.y for landmark in face_landmarks.landmark]

                x_min = max(0.0, min(xs))
                x_max = min(1.0, max(xs))
                y_min = max(0.0, min(ys))
                y_max = min(1.0, max(ys))

                face_width = max(x_max - x_min, 1e-6)

                faces.append(
                    {
                        "mouth_center": {
                            "x": mouth_x,
                            "y": mouth_y,
                            "pixel_x": int(mouth_x * frame_width),
                            "pixel_y": int(mouth_y * frame_height),
                        },
                        "face_box": {
                            "x": int(x_min * frame_width),
                            "y": int(y_min * frame_height),
                            "w": int((x_max - x_min) * frame_width),
                            "h": int((y_max - y_min) * frame_height),
                        },
                        "face_width": face_width,
                        "landmarks": face_landmarks,
                    }
                )

        return result, faces

    def draw(self, frame, result):
        if not result.multi_face_landmarks:
            return frame

        frame_height, frame_width = frame.shape[:2]

        for face_landmarks in result.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    thickness=1,
                    circle_radius=1,
                ),
            )

            upper_lip = face_landmarks.landmark[self.UPPER_LIP_INDEX]
            lower_lip = face_landmarks.landmark[self.LOWER_LIP_INDEX]

            mouth_x = int(
                ((upper_lip.x + lower_lip.x) / 2.0) * frame_width
            )
            mouth_y = int(
                ((upper_lip.y + lower_lip.y) / 2.0) * frame_height
            )

            cv2.circle(
                frame,
                (mouth_x, mouth_y),
                6,
                (0, 0, 255),
                -1,
            )

        return frame

    def close(self):
        self.face_mesh.close()