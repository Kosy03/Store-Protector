import cv2
import time
from ultralytics import YOLO
#UI에서 설정한 conf값으로 맞춰야하는데, 성능 테스트를 위해서 그 기능은 배제하고 만들었어요.
#나중에 테스트 영상들에서 원하던 결과가 나오고 나서 추가하면 될 것 같아요.

class PersonTracker:
    def __init__(self, roi_ratio=[0.73, 0.50, 0.99, 0.66]):
        self.model = YOLO("yolov8n.pt")
        self.roi_ratio = roi_ratio

        #변수 3개
        self.is_person_inside = False
        self.visited_kiosk = False
        self.last_seen_time = 0

    def process_frame(self, frame):
        h, w, _ = frame.shape
        roi_x1 = int(w * self.roi_ratio[0])
        roi_y1 = int(h * self.roi_ratio[1])
        roi_x2 = int(w * self.roi_ratio[2])
        roi_y2 = int(h * self.roi_ratio[3])

        results = self.model.track(frame, persist=True, classes=[0], conf=0.45, verbose=False)

        zone_triggered = False
        theft_detected = False
        current_time = time.time()

        # 현재 프레임에 사람이 한 명이라도 잡혔는지 확인하는 변수
        person_detected_now = False

        annotated_frame = results[0].plot()

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            person_detected_now = True

            # 방금 매장에 처음 들어온 상태라면 변수 초기화
            if not self.is_person_inside:
                self.is_person_inside = True
                self.visited_kiosk = False

            # 사람이 보이니까 마지막 목격 시간 계속 업데이트!
            self.last_seen_time = current_time

            # ID가 튀든 말든 상관없이, 잡힌 사람 중 아무나 구역을 밟으면 OK
            for box in results[0].boxes:
                if box.xyxy is not None:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)

                    foot_x = int((x1 + x2) / 2)
                    foot_y = y2

                    if roi_x1 < foot_x < roi_x2 and roi_y1 < foot_y < roi_y2:
                        self.visited_kiosk = True
                        zone_triggered = True

        #화면에서 사람이 안 보이면 검사
        if not person_detected_now and self.is_person_inside:
            elapsed_time = current_time - self.last_seen_time

            # 5초 유예시간이 끝났는데도 안 돌아왔다면
            if elapsed_time > 5.0:
                # 키오스크 안 들르고 5초 이상 사라짐 = 도난 확정!
                if not self.visited_kiosk:
                    theft_detected = True

                # 사람이 완전히 나갔으므로 다음 사람을 위해 변수 리셋
                self.is_person_inside = False
                self.visited_kiosk = False

        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Kiosk Zone", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if theft_detected:
            cv2.putText(annotated_frame, "THEFT WARNING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, zone_triggered, theft_detected