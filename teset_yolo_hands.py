import cv2
import torch
import time
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hand_model = YOLO("models/hand_yolov8n.pt")
item_model = YOLO("models/best.pt")
ITEM_CLASSES = item_model.names

VIDEO_PATH = "videos/test_video.mp4"
GHOST_TTL = 0
NEAR_MARGIN = 10
EXCLUDE_ZONE = (389, 316, 465, 391)
PICK_SECONDS = 0.5  # 총 누적 시간 기준


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (areaA + areaB - interArea)


def is_near_or_overlap(handBox, itemBox, margin=NEAR_MARGIN):
    hx1, hy1, hx2, hy2 = handBox
    ix1, iy1, ix2, iy2 = itemBox
    return (hx1 - margin < ix2 and hx2 + margin > ix1 and
            hy1 - margin < iy2 and hy2 + margin > iy1)


def is_in_exclude_zone(coords):
    x1, y1, x2, y2 = coords
    ex1, ey1, ex2, ey2 = EXCLUDE_ZONE
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return ex1 < cx < ex2 and ey1 < cy < ey2


cap = cv2.VideoCapture(VIDEO_PATH)
paused = False
frame_num = 0
last_seen_items = {}
confirmed_picks = set()
accumulated_time = {}   # {class_name: 누적 초}
near_start_time = {}    # {hand_index: (class_name, start_time)}

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        h, w = frame.shape[:2]
        scale = 480 / h
        frame = cv2.resize(frame, (int(w * scale), 480))

        # 손 감지
        hand_results = hand_model(frame, verbose=False, imgsz=640, conf=0.3, device=DEVICE)[0]
        hand_boxes = [box.xyxy[0].tolist() for box in hand_results.boxes]

        # 물건 감지
        item_results = item_model(frame, verbose=False, imgsz=640, conf=0.7, device=DEVICE)[0]
        item_boxes = []
        for box in item_results.boxes:
            cls_id = int(box.cls[0])
            class_name = ITEM_CLASSES.get(cls_id, f"unknown_{cls_id}")
            coords = box.xyxy[0].tolist()
            if is_in_exclude_zone(coords):
                continue
            item_boxes.append((class_name, coords))

        # Ghost bbox 로직
        detected_names = set()
        for class_name, coords in item_boxes:
            last_seen_items[class_name] = [coords, GHOST_TTL]
            detected_names.add(class_name)

        for class_name in list(last_seen_items.keys()):
            if class_name not in detected_names:
                last_seen_items[class_name][1] -= 1
                if last_seen_items[class_name][1] <= 0:
                    del last_seen_items[class_name]
                else:
                    item_boxes.append((class_name, last_seen_items[class_name][0]))

        # 손 박스 (하늘색)
        for hbox in hand_boxes:
            x1, y1, x2, y2 = map(int, hbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, "hand", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # 집기 감지
        picking_items = set()
        picked_coords = set()
        now = time.time()

        for i, hbox in enumerate(hand_boxes):
            near_item = None
            for class_name, coords in item_boxes:
                if is_near_or_overlap(hbox, coords):
                    near_item = (class_name, coords)
                    break

            if near_item:
                class_name, coords = near_item

                # 처음 가까워진 순간만 시작 시점 기록
                if i not in near_start_time or near_start_time[i][0] != class_name:
                    near_start_time[i] = (class_name, now)

                current_session = now - near_start_time[i][1]
                total = accumulated_time.get(class_name, 0) + current_session

                x1, y1, x2, y2 = map(int, coords)

                if total >= PICK_SECONDS:
                    if class_name not in confirmed_picks:
                        confirmed_picks.add(class_name)
                        print(f"[PICK CONFIRMED] {class_name} (누적 {total:.2f}s)")
                    else:
                        # 두 번째 이후 근접 시간 업데이트 출력
                        prev_total = accumulated_time.get(class_name, 0)
                        if int(total * 10) > int(prev_total * 10):  # 0.1초마다 출력
                            print(f"[PICK UPDATE] {class_name} (누적 {total:.2f}s)")

                    accumulated_time[class_name] = total
                    picking_items.add(class_name)
                    picked_coords.add(tuple(coords))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"PICK CONFIRMED: {class_name} {total:.1f}s",
                                (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                else:
                    progress = min(total / PICK_SECONDS, 1.0)
                    bar_w = int((x2 - x1) * progress)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 10),
                                  (0, 255, 255), -1)
                    cv2.putText(frame, f"holding: {class_name} {total:.1f}s",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)

        # 나머지 물건 박스
        for class_name, coords in item_boxes:
            if tuple(coords) in picked_coords:
                continue
            x1, y1, x2, y2 = map(int, coords)
            is_ghost = class_name in last_seen_items and last_seen_items[class_name][1] < GHOST_TTL
            color = (0, 165, 255) if is_ghost else (0, 100, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 상태 표시
        status = f"PICKING: {', '.join(picking_items)}!" if picking_items else "Monitoring..."
        color = (0, 0, 255) if picking_items else (200, 200, 200)
        cv2.putText(frame, status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"frame:{frame_num}  SPACE:pause  S:save", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"hands:{len(hand_boxes)}  items:{len(item_boxes)}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 누적 시간 표시
        all_times = {**accumulated_time}
        for i, (cname, start) in near_start_time.items():
            session = now - start
            all_times[cname] = all_times.get(cname, 0) + session
        for j, (cname, total) in enumerate(all_times.items()):
            cv2.putText(frame, f"{cname}: {total:.1f}s", (10, 95 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow("Debug Boxes", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        paused = not paused
        print(f"[{'일시정지' if paused else '재생'}] frame:{frame_num}")
    elif key == ord("s"):
        fname = f"debug_frame_{frame_num}.jpg"
        cv2.imwrite(fname, frame)
        print(f"[저장] {fname}")

cap.release()
cv2.destroyAllWindows()
# 영상 끝난 후 최종 결과 출력
print("\n========== 최종 결과 ==========")
for class_name, total in accumulated_time.items():
    if total >= 20.0:
        print(f"[집은 물건] {class_name} (누적 {total:.2f}s)")

if not any(t >= 20.0 for t in accumulated_time.values()):
    print("집은 물건 없음")
print("================================")