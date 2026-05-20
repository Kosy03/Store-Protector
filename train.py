
from ultralytics import YOLO
import cv2

# ─── 모델 로드 ────────────────────────────────────────────
model = YOLO("models/old_best.pt")  # Colab에서 받은 old_best.pt 경로

# ─── 사진 불러오기 ────────────────────────────────────────
image_path = "test_images/snack4.jpg"  # 테스트할 사진 경로
frame = cv2.imread(image_path)

# ─── 추론 ─────────────────────────────────────────────────
results = model(frame, verbose=False)[0]

# ─── 바운딩 박스 그리기 ───────────────────────────────────
for box in results.boxes:
    conf = float(box.conf[0])
    if conf < 0.2:
        continue
    label = model.names[int(box.cls[0])]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    print(f"✅ 감지: {label} (신뢰도: {conf:.2f})")

# ─── 결과 화면 출력 ───────────────────────────────────────
# 이미지 크기는 그대로, 창만 줄이기
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 640, 640)  # 원하는 크기
cv2.imshow("Result", frame)
cv2.waitKey(0)  # 아무 키나 누르면 닫힘
cv2.destroyAllWindows()

# ─── 결과 이미지 저장 ─────────────────────────────────────
cv2.imwrite("test_images/result.jpg", frame)
print("✅ 결과 이미지 저장 완료: test_images/result.jpg")