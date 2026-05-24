import cv2
from ultralytics import YOLO
from collections import defaultdict

# ─── 라벨 한글 변환 맵 ───────────────────────────────────
LABEL_MAP = {
    'yangparing': '양파링',
    'ojingeoddankong': '오징어땅콩',
}

# ─── 모델 로드 ───────────────────────────────────────────
snack_model = YOLO("models/best.pt")
snack_names = snack_model.names

# ─── 결과 저장 딕셔너리 ───────────────────────────────────
detection_log = defaultdict(list)

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
        detected_items = []

        for box in snack_model(small, verbose=False)[0].boxes:
            conf = float(box.conf[0])
            if conf < 0.2:
                continue
            label = snack_names[int(box.cls[0])]          # ✅ 영어 원본 유지 (박스용)
            korean_label = LABEL_MAP.get(label, label)     # ✅ 한글 변환 (로그용)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            last_boxes.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), label, conf, (0, 200, 0)))
            detected_items.append(korean_label)

        detection_log[frame_idx] = detected_items

    # ─── 박스 그리기 (영어 label 사용) ───────────────────
    for (x1, y1, x2, y2, label, conf, color) in last_boxes:
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"{label} {conf:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─── 최종 결과 출력 (한글) ───────────────────────────────
print("\n📋 프레임별 탐지 결과:")
for fid, items in detection_log.items():
    if items:
        print(f"  Frame {fid:04d}: {items}")

all_items = list({item for items in detection_log.values() for item in items})
print(f"\n🛒 전체 감지된 품목: {all_items}")