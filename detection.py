import cv2
from ultralytics import YOLO
from collections import defaultdict

# ─── OpenVINO 모델 로드 ───────────────────────────────────
snack_model = YOLO("models/snack_best_openvino_model/")
drink_model = YOLO("models/drink_best_openvino_model/")

snack_names = snack_model.names
drink_names = drink_model.names

# ─── 결과 저장 딕셔너리 ───────────────────────────────────
detection_log = defaultdict(list)  # {frame_idx: ['꼬깔콘', '포카리스웨트', ...]}

# ─── 영상 열기 ────────────────────────────────────────────
cap = cv2.VideoCapture("videos/topview_snack.mp4")

FRAME_SKIP = 5
RESIZE_W   = 480
frame_idx  = 0
last_boxes = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    display = cv2.resize(frame, (1280, 720))

    if frame_idx % FRAME_SKIP == 0:
        small = cv2.resize(frame, (RESIZE_W, int(frame.shape[0] * RESIZE_W / frame.shape[1])))
        sx = display.shape[1] / small.shape[1]
        sy = display.shape[0] / small.shape[0]

        last_boxes = []
        detected_items = []  # 현재 프레임 탐지 목록

        for box in snack_model(small, verbose=False)[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.2:
                continue
            label = snack_names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            last_boxes.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), label, conf, (0, 200, 0)))
            detected_items.append(label)  # ✅ 딕셔너리에 저장

        for box in drink_model(small, verbose=False)[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.7:
                continue
            label = drink_names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            last_boxes.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), label, conf, (255, 100, 0)))
            detected_items.append(label)  # ✅ 딕셔너리에 저장

        # ─── 프레임별 결과 저장 ───────────────────────────
        detection_log[frame_idx] = detected_items

    # ─── 박스 그리기 ──────────────────────────────────────
    for (x1, y1, x2, y2, label, conf, color) in last_boxes:
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"{label} {conf:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─── 최종 결과 출력 ───────────────────────────────────────
print("\n📋 프레임별 탐지 결과:")
for fid, items in detection_log.items():
    if items:
        print(f"  Frame {fid:04d}: {items}")

# 전체 고유 품목 목록
all_items = list({item for items in detection_log.values() for item in items})
print(f"\n🛒 전체 감지된 품목: {all_items}")