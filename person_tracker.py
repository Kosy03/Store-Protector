import cv2
from ultralytics import YOLO


class PersonTracker:
    def __init__(self, roi_ratio=[0.73, 0.50, 0.99, 0.66]):
        # 사람 탐지용 모델 로드
        self.model = YOLO("yolov8n.pt")
        # 관심구역 비율 저장
        self.roi_ratio = roi_ratio

    def process_frame(self, frame):
        h, w, _ = frame.shape
        # 관심구역 픽셀 좌표 계산
        roi_x1 = int(w * self.roi_ratio[0])
        roi_y1 = int(h * self.roi_ratio[1])
        roi_x2 = int(w * self.roi_ratio[2])
        roi_y2 = int(h * self.roi_ratio[3])

        # YOLO 추적 실행
        results = self.model.track(frame, persist=True, classes=[0], conf=0.6, verbose=False)

        zone_triggered = False

        # 사람이 한 명이라도 탐지되었다면 발 위치 체크
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if box.xyxy is not None:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)

                    # 사각형 박스의 하단 중앙을 발 위치로 지정
                    foot_x = int((x1 + x2) / 2)
                    foot_y = y2

                    # 발이 관심구역 사각형 내부에 들어왔는지 검사
                    if roi_x1 < foot_x < roi_x2 and roi_y1 < foot_y < roi_y2:
                        zone_triggered = True

        # 시각화용 배경 프레임 가져오기
        annotated_frame = results[0].plot()

        # 박스 위에 관심구역 사각형과 텍스트 그리기
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Kiosk Zone", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 구역 진입 성공 시 화면에 표시
        if zone_triggered:
            cv2.putText(annotated_frame, "CUSTOMER DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 색상 변환 후 신호값과 함께 리턴
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, zone_triggered