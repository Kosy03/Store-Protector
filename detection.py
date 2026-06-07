import cv2
from ultralytics import YOLO
from collections import defaultdict, deque, Counter

# ─── 라벨 한글 변환 맵 ───────────────────────────────────
LABEL_MAP = {
    'yangparing': '양파링',
    'ojingeoddankong': '오징어땅콩',
    'cocacola':'코카콜라',
    'garamandeunBae':'갈아만든배',
    'kkokkalcorn':'꼬칼콘',
    'pocarisweat':'포카리스웨트'
}

# ─── 설정값 ───────────────────────────────────────────────
CONF_THRESHOLD  = 0.65   # ↑ 높일수록 오탐 감소 (0.5~0.65 권장)
IOU_THRESHOLD   = 0.45   # 겹치는 박스 제거 강도
FRAME_SKIP      = 3      # 몇 프레임마다 추론할지
RESIZE_W        = 640    # 추론용 축소 너비 (클수록 정확, 느림)
VOTE_WINDOW     = 4      # 투표에 사용할 최근 프레임 수
VOTE_MIN        = 3      # 몇 프레임 이상 감지돼야 확정할지

# ─── 모델 로드 ───────────────────────────────────────────
snack_model = YOLO("models/best.pt")
snack_names = snack_model.names

# ─── 결과 저장 ────────────────────────────────────────────
detection_log  = defaultdict(list)
vote_buffer    = deque(maxlen=VOTE_WINDOW)   # 최근 N프레임 감지 결과
stable_labels  = set()                       # 투표 통과한 확정 라벨

# ─── 영상 열기 ────────────────────────────────────────────
cap = cv2.VideoCapture("videos/test1.mp4")

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

        last_boxes     = []
        detected_items = []

        results = snack_model(
            small,
            verbose=False,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            augment=True        # TTA: 다각도로 추론해 앙상블
        )

        for box in results[0].boxes:
            conf  = float(box.conf[0])
            label = snack_names[int(box.cls[0])]
            korean_label = LABEL_MAP.get(label, label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            last_boxes.append((
                int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy),
                label, conf, korean_label
            ))
            detected_items.append(korean_label)

        # ─── 다중 프레임 투표 ─────────────────────────────
        vote_buffer.append(detected_items)

        vote_counts = Counter(
            item for frame_items in vote_buffer for item in frame_items
        )
        # VOTE_MIN번 이상 등장한 라벨만 확정
        stable_labels = {
            item for item, cnt in vote_counts.items()
            if cnt >= VOTE_MIN
        }

        detection_log[frame_idx] = list(stable_labels)

    # ─── 박스 그리기 (투표 통과한 것만 초록, 미통과는 회색) ──
    for (x1, y1, x2, y2, label, conf, korean_label) in last_boxes:
        passed = korean_label in stable_labels
        color  = (0, 200, 0) if passed else (160, 160, 160)
        thickness = 2 if passed else 1

        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            display,
            f"{label} {conf:.2f}{'✓' if passed else '?'}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    # ─── 확정 품목 화면 좌상단 표시 ──────────────────────
    for i, item in enumerate(sorted(stable_labels)):
        cv2.putText(display, f"[확정] {item}", (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.imshow("Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─── 최종 결과 출력 ───────────────────────────────────────
print("\n📋 프레임별 확정 탐지 결과:")
for fid, items in detection_log.items():
    if items:
        print(f"  Frame {fid:04d}: {items}")

all_items = list({item for items in detection_log.values() for item in items})
print(f"\n🛒 전체 감지된 품목: {all_items}")