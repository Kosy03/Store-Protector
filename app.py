import streamlit as st
import pandas as pd
import cv2
from ultralytics import YOLO
from collections import Counter
import time

# ─── 라벨 한글 변환 맵 ───────────────────────────────────
LABEL_MAP = {
    'yangparing': '양파링',
    'ojingeoddankong': '오징어땅콩',
    'kokkalkon': '꼬깔콘',
    'kkulttwigwabaegi': '꿀꽈배기'
}

# ─── 민감도별 confidence 임계값 ──────────────────────────
SENSITIVITY_MAP = {
    "낮음": 0.6,
    "보통": 0.4,
    "높음": 0.2,
}


# ─── IoU 계산 및 중복 제거 함수 ─────────────────────────
def iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def deduplicate_by_iou(boxes_raw, iou_threshold=0.5):
    kept = []
    boxes_raw = sorted(boxes_raw, key=lambda x: -x[5])
    for candidate in boxes_raw:
        duplicate = False
        for kept_box in kept:
            if candidate[4] == kept_box[4]:
                if iou(candidate[:4], kept_box[:4]) > iou_threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(candidate)
    return kept


# ─── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(page_title="Store-Protector UI", layout="wide")
st.title("Store-Protector: 무인 매장 실시간 보안 관제 시스템")
st.markdown("---")

# ─── 💡 글로벌 세션 상태 초기화 ──────────────────────────
if "current_sensitivity" not in st.session_state:
    st.session_state.current_sensitivity = "보통"
if "current_conf" not in st.session_state:
    st.session_state.current_conf = 0.4

# ─── 상단 레이아웃 ───────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("A화면: 매장 전경 및 행동 분석")
    st.image("test_images/store_view.png", use_container_width=True)


    @st.fragment
    def render_sensitivity_setting():
        selected = st.selectbox(
            "민감도 수준", options=["낮음", "보통", "높음"],
            index=["낮음", "보통", "높음"].index(st.session_state.current_sensitivity),
            key="sensitivity_selector"
        )
        st.session_state.current_sensitivity = selected
        st.session_state.current_conf = SENSITIVITY_MAP[selected]
        st.caption(f"현재 confidence 임계값: {st.session_state.current_conf}")


    render_sensitivity_setting()

with col2:
    st.subheader("B화면: 계산대 상품 인식")
    b_frame = st.empty()
    st.markdown("#### [ 계산대 인식 결과 ]")
    b_table = st.empty()
    b_status_msg = st.empty()  # 💡 상태 메시지 레이아웃 공간

st.markdown("---")

# ─── 하단 C화면 ──────────────────────────────────────────
st.subheader("C화면: 시스템 판독 결과 및 로그")
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    c_result = st.empty()

# ─── 사이드바 ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    st.write("카메라 A 상태: 🟢 정상")
    st.write("카메라 B 상태: 🟢 정상")
    st.divider()


    @st.fragment
    def render_sidebar_status():
        st.markdown(f"**사람 감지 민감도:** `{st.session_state.current_sensitivity}`")
        st.markdown(f"**Confidence 임계값:** `{st.session_state.current_conf}`")


    render_sidebar_status()

# ─── 🚀 영상 처리 및 AI 추론 루프 ───────────────────────
model = YOLO("models/best.pt")
snack_names = model.names

video_path = "videos/topview_snack.mp4"
cap = cv2.VideoCapture(video_path)

FRAME_SKIP = 5
RESIZE_W = 480
frame_idx = 0
last_boxes = []
last_detected = []

# 영상이 재생되는 동안의 초기 UI 메시지 설정
b_status_msg.info("⏳ 계산대 물품 확인 중")

pick_str = ['양파링', '오징어땅콩', '포카칩']
pick_display_str = ", ".join(pick_str)

with c_result.container():
    st.info(f"""
    ### 🔄 시스템 판독 대기 중...
    - 매대에서 집은 물품: {pick_display_str}
    """)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 영상이 끝나면 루프를 빠져나갑니다.

    frame_idx += 1
    display = cv2.resize(frame, (1280, 720))

    if frame_idx % FRAME_SKIP == 0:
        small = cv2.resize(frame, (RESIZE_W, int(frame.shape[0] * RESIZE_W / frame.shape[1])))
        sx = display.shape[1] / small.shape[1]
        sy = display.shape[0] / small.shape[0]

        raw_boxes = []
        for box in model(small, verbose=False)[0].boxes:
            conf = float(box.conf[0])
            if conf < st.session_state.current_conf:
                continue

            label = snack_names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_boxes.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), label, conf, (0, 200, 0)))

        last_boxes = deduplicate_by_iou(raw_boxes, iou_threshold=0.5)
        last_detected = sorted([LABEL_MAP.get(b[4], b[4]) for b in last_boxes])

    # ─── 바운딩 박스 그리기 ───────────────────────
    for (x1, y1, x2, y2, label, conf, color) in last_boxes:
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"{LABEL_MAP.get(label, label)} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ B화면 영상 표시
    rgb_frame = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    b_frame.image(rgb_frame, use_container_width=True)

    # ✅ 테이블 업데이트 (영상 재생 중에는 "인식 중...")
    if last_detected:
        counts = Counter(last_detected)
        check_data = {
            "물품명": list(counts.keys()),
            "수량": list(counts.values()),
            "상태": ["인식 중..."] * len(counts)
        }
        df = pd.DataFrame(check_data)
        df.index = df.index + 1
        b_table.table(df)

cap.release()

# ─── ⏱️ 영상 종료 후 5초 카운트다운 타이머 구간 ───────────────────
# 영상이 끝나면 테이블의 상태를 "인식 완료"로 먼저 바꿉니다.
if last_detected:
    counts = Counter(last_detected)
    check_data = {
        "물품명": list(counts.keys()),
        "수량": list(counts.values()),
        "상태": ["인식 완료 (계산 중)"] * len(counts)
    }
    b_table.table(pd.DataFrame(check_data))

# 5초 동안 대기하면서 초단위로 UI 변경
for seconds_left in range(5, 0, -1):
    b_status_msg.warning(f"⏳ **계산중...**")
    time.sleep(1)

# ─── 🛑 5초 경과 후 최종 "계산 완료" 및 C화면 전환 ──────────────────
b_status_msg.success("🎉 **계산 완료**")

# B화면 테이블 상태도 최종 "계산 완료"로 갱신
if last_detected:
    counts = Counter(last_detected)
    check_data = {
        "물품명": list(counts.keys()),
        "수량": list(counts.values()),
        "상태": ["계산 완료"] * len(counts)
    }
    b_table.table(pd.DataFrame(check_data))

# C화면 계산 일치/불일치 판독 및 결과 업데이트
pick_counter = Counter(pick_str)
detected_counter = Counter(last_detected)
missing_counter = pick_counter - detected_counter

if missing_counter:
    missing_items = [f"{item} ({count}개)" for item, count in missing_counter.items()]
    missing_str = ", ".join(missing_items)
    error_message = f"**⚠️ 누락 물품 발생: {missing_str}**"
    status_icon = "⚠️ 주의: 계산 불일치 감지!"
    box_type = st.error
else:
    error_message = "✅ 모든 물품이 정상 결제되었습니다."
    status_icon = "🟢 확인: 계산 일치"
    box_type = st.success

detected_str = ", ".join(last_detected) if last_detected else "탐지된 물품 없음"

with c_result.container():
    box_type(f"### {status_icon}")
    st.warning(f"""
    **[ 최종 판독 결과 ]**
    - 매대에서 집은 물품: {pick_display_str}
    - 계산대에서 결제한 물품: {detected_str}
    ---
    {error_message}
    """)

st.success("CCTV 영상 분석 및 최종 결제 검증 완료")