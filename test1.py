import cv2
import torch
import ultralytics

VIDEO_PATH = "videos/도난영상.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
scale = 640 / w
frame = cv2.resize(frame, (640, int(h * scale)))

drawing = False
start = (0, 0)
end = (0, 0)
clone = frame.copy()

def draw_rect(event, x, y, flags, param):
    global drawing, start, end, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        frame = clone.copy()
        cv2.rectangle(frame, start, (x, y), (0, 0, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end = (x, y)
        cv2.rectangle(frame, start, end, (0, 0, 255), 2)
        print(f"EXCLUDE_ZONE = ({start[0]}, {start[1]}, {end[0]}, {end[1]})")

print('Python:', __import__('sys').version)
print('PyTorch:', torch.__version__)
print('OpenCV:', cv2.__version__)
print('Ultralytics:', ultralytics.__version__)
print('CUDA 사용가능:', torch.cuda.is_available())

cv2.namedWindow("Select Exclude Zone")
cv2.setMouseCallback("Select Exclude Zone", draw_rect)

while True:
    cv2.imshow("Select Exclude Zone", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()