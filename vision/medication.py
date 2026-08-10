import cv2
import math


class MedicationDetector:
    def __init__(self):

        self.model_path = "models/medicine_yolov8n_best_v2.pt"
        self.yolo_conf = 0.5
        self.model = None
        
        self.zoom_scale = 4.0

        self.evidence_time_added = False

        self.check_evidence_count = 0
        self.check_evidence_threshold = 45

        self.check_score = 0
        self.check_score_threshold = 4

        self.revisible_count = 0
        self.revisible_threshold = 5

        self.evidence_time2_added = False
        self.evidence_both_hands_missing_added = False
        self.evidence_face_hands_missing_added = False

        self.both_hands_missing_count = 0
        self.face_hands_missing_count = 0
        self.mouth_in_evidence_count = 0
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print("YOLO 모델 로드 완료")
        except Exception as e:
            print("YOLO 모델 로드 실패")
            print(e)
            self.model = None

        
        self.state = "NO_MEDICATION_DETECTED"
        self.grabbed_count = 0
        self.mouth_count = 0

        self.hand_med_threshold = 80
        self.hand_face_threshold = 70
        self.grab_frame_threshold = 10
        self.mouth_frame_threshold = 15

        self.tracked_hand_point = None
        self.grabbed_missing_count = 0
        self.pill_visible_after_grab_count = 0
        self.last_face_center = None
        self.face_missing_count = 0
        self.face_missing_limit = 30
     

    def detect(self, frame, hand_points=None, face_boxes=None):
        if self.model is None:
            return None

        medication_box = self.detect_yolo_box_with_person_zoom(
            frame,
            hand_points,
            face_boxes
        )

        if medication_box is not None:
            return medication_box

        return None

    def get_person_roi(self, frame, hand_points, face_boxes):
        h, w, _ = frame.shape

        points = []

    # 1. 감지된 모든 손 landmark를 points에 추가
        if hand_points:
            
            for hand in hand_points:
                for landmark in hand:
                    px = int(landmark["x"] * w)
                    py = int(landmark["y"] * h)
                    points.append((px, py))

    # 2. 얼굴이 감지되면 얼굴 박스와 입 중심을 points에 추가
        if face_boxes:
            face = face_boxes[0]

            fx1 = face["x"]
            fy1 = face["y"]
            fx2 = face["x"] + face["w"]
            fy2 = face["y"] + face["h"]

            points.append((fx1, fy1))
            points.append((fx2, fy2))

        # FaceMesh 사용 시 입 중심 추가
            if "mouth_center" in face:
                mouth = face["mouth_center"]
                points.append((mouth["pixel_x"], mouth["pixel_y"]))

    # 3. 손도 얼굴도 없으면 ROI 없음
        if not points:
            return None

    # 4. 인식된 손/얼굴이 모두 들어오는 최소 범위 계산
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

    # 5. 최소 범위에 약간의 여백만 추가
        margin_x = 70
        margin_y_top = 90
        margin_y_bottom = 160

        x1 = max(0, int(min_x - margin_x))
        y1 = max(0, int(min_y - margin_y_top))
        x2 = min(w, int(max_x + margin_x))
        y2 = min(h, int(max_y + margin_y_bottom))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def detect_yolo_box_with_person_zoom(self, frame, hand_points, face_boxes):
        person_roi = self.get_person_roi(frame, hand_points, face_boxes)
        
        if person_roi is None:
            return None

        x1, y1, x2, y2 = person_roi

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        zoomed_roi = cv2.resize(
            roi,
            None,
            fx=self.zoom_scale,
            fy=self.zoom_scale,
            interpolation=cv2.INTER_LINEAR
        )

        results = self.model.predict(
            source=zoomed_roi,
            imgsz=640,
            conf=self.yolo_conf,
            verbose=False
        )

        if not results:
            return None

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        best_box = None
        best_conf = 0

        for box in result.boxes:
            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is None:
            return None

        bx1, by1, bx2, by2 = best_box.xyxy[0].tolist()

    # 확대된 ROI 좌표 → 원래 ROI 좌표
        bx1 = bx1 / self.zoom_scale
        by1 = by1 / self.zoom_scale
        bx2 = bx2 / self.zoom_scale
        by2 = by2 / self.zoom_scale

    # ROI 좌표 → 원본 frame 좌표
        bx1 = int(bx1 + x1)
        by1 = int(by1 + y1)
        bx2 = int(bx2 + x1)
        by2 = int(by2 + y1)

        box_w = bx2 - bx1
        box_h = by2 - by1

        if box_w <= 0 or box_h <= 0:
            return None

        center_x = bx1 + box_w // 2
        center_y = by1 + box_h // 2

        return {
            "x": bx1,
            "y": by1,
            "w": box_w,
            "h": box_h,
            "center": (center_x, center_y),
            "label": f"Drug Zoom {best_conf:.2f}",
            "conf": best_conf,
            "source": "person_zoom_yolo",
            "roi": (x1, y1, x2, y2)
        }


    def calculate_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return None

        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_index_finger_point(self, frame, hand_points, target_point=None):
        if not hand_points:
            return None

        h, w, _ = frame.shape

        selected_point = None
        selected_dist = None

        for hand in hand_points:
            index_finger = hand[8]

            index_point = (
                int(index_finger["x"] * w),
                int(index_finger["y"] * h)
            )

            if target_point is None:
                return index_point

            dist = self.calculate_distance(index_point, target_point)

            if selected_dist is None or dist < selected_dist:
                selected_dist = dist
                selected_point = index_point

        return selected_point

    def get_all_index_finger_points(self, frame, hand_points):
        if not hand_points:
            return []

        h, w, _ = frame.shape
        points = []

        for hand in hand_points:
            index_finger = hand[8]

            index_point = (
                int(index_finger["x"] * w),
                int(index_finger["y"] * h)
            )

            points.append(index_point)

        return points

    def get_closest_point(self, points, target_point):
        if not points or target_point is None:
            return None, None

        closest_point = None
        closest_dist = None

        for point in points:
            dist = self.calculate_distance(point, target_point)

            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_point = point

        return closest_point, closest_dist

    def get_face_center(self, face_boxes):
        if face_boxes:
            face = face_boxes[0]

            if "mouth_center" in face:
                mouth = face["mouth_center"]

                self.last_face_center = (
                    mouth["pixel_x"],
                    mouth["pixel_y"]
                )
            else:
                self.last_face_center = (
                    face["x"] + face["w"] // 2,
                    face["y"] + face["h"] // 2
                )

            self.face_missing_count = 0
            return self.last_face_center

        if self.last_face_center is not None and self.face_missing_count < self.face_missing_limit:
            self.face_missing_count += 1
            return self.last_face_center

        return None

    def update_state(
        self,
        hand_med_dist,
        hand_face_dist,
        index_point=None,
        medication_box=None,
        hands_visible_count=0,
        face_visible=False
    ):
        medication_visible = medication_box is not None

        if self.state == "MEDICATION_DONE":
            return self.state

        if self.state == "NO_MEDICATION_DETECTED":
            self.state = "WAITING"

        if self.state == "WAITING":
            if hand_med_dist is not None and hand_med_dist < self.hand_med_threshold:
                self.grabbed_count += 1
            else:
                self.grabbed_count = 0

            if self.grabbed_count >= self.grab_frame_threshold:
                self.state = "GRABBED"
                self.tracked_hand_point = index_point
                self.grabbed_missing_count = 0
                self.pill_visible_after_grab_count = 0
                self.check_evidence_count = 0
                self.check_score = 0

        elif self.state == "GRABBED":
            if medication_visible:
                self.pill_visible_after_grab_count += 1
                self.grabbed_missing_count = 0
                self.check_evidence_count = 0
                self.check_score = 0
            else:
                self.grabbed_missing_count += 1

            if index_point is not None:
                self.tracked_hand_point = index_point

            if hand_face_dist is not None and hand_face_dist < self.hand_face_threshold:
                self.mouth_count += 1
            else:
                self.mouth_count = 0

            if self.mouth_count >= self.mouth_frame_threshold:
                self.state = "MEDICATION_DONE"
                return self.state

            if (
                not medication_visible
                and self.grabbed_missing_count > 35
            ):
                
                self.state = "CHECK_EVIDENCE"
                self.check_evidence_count = 0
                self.check_score = 0
                self.revisible_count = 0

                self.evidence_time_added = False

                self.evidence_time2_added = False
                self.evidence_both_hands_missing_added = False
                self.evidence_face_hands_missing_added = False

                self.both_hands_missing_count = 0
                self.face_hands_missing_count = 0
                self.mouth_in_evidence_count = 0

                return self.state

        elif self.state == "CHECK_EVIDENCE":
            self.check_evidence_count += 1

            # CHECK_EVIDENCE 중 약이 다시 보이더라도 바로 복귀하지 않음
            # 약이 보이고, 추적 중인 손이 약 근처에 있을 때만 재인식 카운트 증가
            if (
                medication_visible
                and hand_med_dist is not None
                and hand_med_dist < self.hand_med_threshold
            ):
                self.revisible_count += 1
            else:
                self.revisible_count = 0

            if self.revisible_count >= self.revisible_threshold:
                self.state = "GRABBED"
                self.grabbed_missing_count = 0
                self.check_evidence_count = 0
                self.check_score = 0
                self.revisible_count = 0
                return self.state
            
            # CHECK_EVIDENCE에서 손-입은 제한적으로만 복용으로 인정
            # 약 안 보임 1단계 점수(+2)만 있는 상태에서만 DONE 허용
            if hand_face_dist is not None and hand_face_dist < self.hand_face_threshold:
                self.mouth_in_evidence_count += 1
            else:
                self.mouth_in_evidence_count = 0

            if (
                self.check_score == 2
                and self.mouth_in_evidence_count >= self.mouth_frame_threshold
            ):
                self.state = "MEDICATION_DONE"
                return self.state

            # 증거 1: 약이 일정 시간 이상 계속 안 보임 1단계
            if (
                not self.evidence_time_added
                and self.check_evidence_count > self.check_evidence_threshold
            ):
                self.check_score += 2
                self.evidence_time_added = True

            # 증거 2: 양손이 모두 화면 밖으로 나감
            if hands_visible_count == 0:
                self.both_hands_missing_count += 1
            else:
                self.both_hands_missing_count = 0

            if (
                not self.evidence_both_hands_missing_added
                and self.both_hands_missing_count >= 8
            ):
                self.check_score += 1
                self.evidence_both_hands_missing_added = True

            # 증거 3: 손과 얼굴이 같이 화면 밖으로 사라짐
            if hands_visible_count == 0 and not face_visible:
                self.face_hands_missing_count += 1
            else:
                self.face_hands_missing_count = 0

            if (
                not self.evidence_face_hands_missing_added
                and self.face_hands_missing_count >= 8
            ):
                self.check_score += 2
                self.evidence_face_hands_missing_added = True

            # 증거 4: 약이 더 오래 안 보임 2단계
            if (
                not self.evidence_time2_added
                and self.check_evidence_count > 90
            ):
                self.check_score += 2
                self.evidence_time2_added = True

            if index_point is not None:
                self.tracked_hand_point = index_point

            if self.check_score >= self.check_score_threshold:
                self.state = "MEDICATION_CHECK_NEEDED"
                return self.state

        elif self.state == "MEDICATION_CHECK_NEEDED":
            # 약이 다시 보이고 손이 약 근처에 있는 상태가
            # 여러 프레임 연속 유지될 때만 다시 GRABBED로 복귀
            if (
                medication_visible
                and hand_med_dist is not None
                and hand_med_dist < self.hand_med_threshold
            ):
                self.revisible_count += 1
            else:
                self.revisible_count = 0

            if self.revisible_count >= self.revisible_threshold:
                self.state = "GRABBED"

                self.check_evidence_count = 0
                self.check_score = 0
                self.revisible_count = 0

                self.evidence_time_added = False
                self.evidence_time2_added = False
                self.evidence_both_hands_missing_added = False
                self.evidence_face_hands_missing_added = False

                self.both_hands_missing_count = 0
                self.face_hands_missing_count = 0
                self.mouth_in_evidence_count = 0

                return self.state
            
        return self.state
    
    def process(self, frame, hand_points, face_boxes):

        if self.state == "MEDICATION_DONE":
            face_center = self.get_face_center(face_boxes)
            index_point = self.get_index_finger_point(frame, hand_points)

            return {
                "medication_box": None,
                "index_point": index_point,
                "face_center": face_center,
                "medication_center": None,
                "hand_med_dist": None,
                "hand_face_dist": self.calculate_distance(index_point, face_center),
                "state": self.state
            }
        
        medication_box = self.detect(frame, hand_points, face_boxes)
        face_center = self.get_face_center(face_boxes)

        all_index_points = self.get_all_index_finger_points(frame, hand_points)

        # 약이 안 보이는 경우
        if medication_box is None:
            # 약을 이미 집은 상태라면, 손에 가려진 것으로 보고 손 추적 계속
            if self.state in ["GRABBED", "CHECK_EVIDENCE", "MEDICATION_CHECK_NEEDED"]:
                index_point, _ = self.get_closest_point(
                    all_index_points,
                    self.tracked_hand_point
                )

                hand_face_dist = self.calculate_distance(index_point, face_center)

                state = self.update_state(
                    hand_med_dist=None,
                    hand_face_dist=hand_face_dist,
                    index_point=index_point,
                    medication_box=None,
                    hands_visible_count=len(all_index_points),
                    face_visible=bool(face_boxes)
                )

                return {
                    "medication_box": None,
                    "index_point": index_point,
                    "face_center": face_center,
                    "medication_center": None,
                    "hand_med_dist": None,
                    "hand_face_dist": hand_face_dist,
                    "state": state
                }

            # 아직 약을 집기 전이라면 약이 안 보이는 것은 복약 상황 아님
            self.state = "NO_MEDICATION_DETECTED"
            self.grabbed_count = 0
            self.mouth_count = 0
            self.tracked_hand_point = None
            self.grabbed_missing_count = 0
            self.pill_visible_after_grab_count = 0

            return {
                "medication_box": None,
                "index_point": None,
                "face_center": face_center,
                "medication_center": None,
                "hand_med_dist": None,
                "hand_face_dist": None,
                "state": self.state
            }

        # 약이 보이는 경우
        medication_center = medication_box["center"]

        if self.state == "NO_MEDICATION_DETECTED":
            self.state = "WAITING"

        if self.state == "WAITING":
            # 약을 집기 전에는 약과 가장 가까운 손을 선택
            index_point, _ = self.get_closest_point(
                all_index_points,
                medication_center
            )

        elif self.state in ["GRABBED", "CHECK_EVIDENCE"]:
            # 약을 집은 뒤에는 기존 추적 손과 가장 가까운 손을 계속 추적
            index_point, _ = self.get_closest_point(
                all_index_points,
                self.tracked_hand_point
            )

        elif self.state == "MEDICATION_CHECK_NEEDED":
            # 확인 필요 상태에서 약이 다시 보이면,
            # 다시 집는 손을 찾기 위해 약과 가장 가까운 손을 선택
            index_point, _ = self.get_closest_point(
                all_index_points,
                medication_center
            )

        else:
            index_point = self.get_index_finger_point(frame, hand_points)

        hand_med_dist = self.calculate_distance(index_point, medication_center)
        hand_face_dist = self.calculate_distance(index_point, face_center)

        state = self.update_state(
            hand_med_dist=hand_med_dist,
            hand_face_dist=hand_face_dist,
            index_point=index_point,
            medication_box=medication_box,
            hands_visible_count=len(all_index_points),
            face_visible=bool(face_boxes)
        )

        return {
            "medication_box": medication_box,
            "index_point": index_point,
            "face_center": face_center,
            "medication_center": medication_center,
            "hand_med_dist": hand_med_dist,
            "hand_face_dist": hand_face_dist,
            "state": state
        }

    def draw(self, frame, medication_box):
        if medication_box is None:
            return frame

        x = medication_box["x"]
        y = medication_box["y"]
        w = medication_box["w"]
        h = medication_box["h"]
        center = medication_box["center"]
        label = medication_box.get("label", "Medication")

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        cv2.circle(
            frame,
            center,
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        roi = medication_box.get("roi")

        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(
                frame,
                (rx1, ry1),
                (rx2, ry2),
                (255, 0, 255),
                2
            )

        return frame    


    def draw_debug_info(
        self,
        frame,
        index_point=None,
        face_center=None,
        medication_center=None,
        hand_med_dist=None,
        hand_face_dist=None
    ):
        if index_point is not None:
            cv2.circle(frame, index_point, 8, (255, 0, 0), -1)

        if face_center is not None:
            cv2.circle(frame, face_center, 8, (0, 255, 0), -1)

        if index_point is not None and medication_center is not None:
            cv2.line(frame, index_point, medication_center, (255, 255, 0), 2)

        if index_point is not None and face_center is not None:
            cv2.line(frame, index_point, face_center, (0, 255, 255), 2)

        cv2.putText(
            frame,
            f"STATE: {self.state}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        if hand_med_dist is not None:
            cv2.putText(
                frame,
                f"Hand-Med Dist: {int(hand_med_dist)}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        if hand_face_dist is not None:
            cv2.putText(
                frame,
                f"Hand-Face Dist: {int(hand_face_dist)}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        cv2.putText(
            frame,
            f"Pill Visible After Grab: {self.pill_visible_after_grab_count}",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 255),
            2
        )

        if self.state == "MEDICATION_CHECK_NEEDED":
            cv2.putText(
                frame,
                "MEDICATION CHECK NEEDED",
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                "Check if the pill has fallen.",
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        return frame

    def draw_result(self, frame, medication_result):
        medication_box = medication_result.get("medication_box")
        index_point = medication_result.get("index_point")
        face_center = medication_result.get("face_center")
        medication_center = medication_result.get("medication_center")
        hand_med_dist = medication_result.get("hand_med_dist")
        hand_face_dist = medication_result.get("hand_face_dist")

        frame = self.draw(frame, medication_box)

        frame = self.draw_debug_info(
            frame=frame,
            index_point=index_point,
            face_center=face_center,
            medication_center=medication_center,
            hand_med_dist=hand_med_dist,
            hand_face_dist=hand_face_dist
        )

        return frame

    def reset(self):
        self.state = "NO_MEDICATION_DETECTED"
        self.grabbed_count = 0
        self.mouth_count = 0
        self.tracked_hand_point = None
        self.grabbed_missing_count = 0
        self.pill_visible_after_grab_count = 0
        self.last_face_center = None
        self.face_missing_count = 0
        self.check_evidence_count = 0
        self.check_score = 0
        self.evidence_time_added = False
        self.revisible_count = 0
        self.evidence_time2_added = False
        self.evidence_both_hands_missing_added = False
        self.evidence_face_hands_missing_added = False

        self.both_hands_missing_count = 0
        self.face_hands_missing_count = 0
        self.mouth_in_evidence_count = 0
