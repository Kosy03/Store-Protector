# ═══════════════════════════════════════════════════════
#  [전체 구조 안내 - 새로 추가된 설명 주석]
#  이 파일은 무인매장을 지키는 "두 명의 경비원 + 한 명의 판사"라고 생각하면 이해가 쉽습니다.
#    - A화면(경비원1) : 매장 전경을 보며 누가 무엇을 집었는지 계속 감시
#    - B화면(경비원2) : 계산대 위에서 실제로 무엇이 결제됐는지 확인
#    - C화면(판사)    : 두 경비원의 보고를 비교해서 "정상 결제 / 누락 / 도난"을 최종 판결
#  Streamlit은 사용자가 뭔가 조작할 때마다 이 파일을 처음부터 끝까지 다시 실행하는 구조라서,
#  버튼 상태나 로그처럼 "기억해야 할 값"은 st.session_state 라는 메모장에 따로 적어둡니다.
# ═══════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import cv2, torch, time, os
from collections import defaultdict, deque, Counter

from sympy import false
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# ═══════════════════════════════════════════════════════
#  경로 설정
# ═══════════════════════════════════════════════════════
#1. 계산 완료 상황
VIDEO_A_PATH = "videos/구매영상.mp4"
VIDEO_B_PATH = "videos/계산대.mp4"

#2. 계산 누락 상황
#VIDEO_A_PATH = "videos/구매누락.mp4"
#VIDEO_B_PATH = "videos/계산대_누락.mp4"

#3. 계산대에 갔지만 계산대 카메라(B카메라)에 찍힌 것이 없는 상황
#VIDEO_A_PATH = "videos/구매영상.mp4"
#VIDEO_B_PATH = "videos/빈계산대.mp4"

#4. 도난 상황
#VIDEO_A_PATH = "videos/도난영상.mp4"


# 계산대 위치 설정
CHECKOUT_ZONE     = (102, 42, 229, 150)#구매,누락
#CHECKOUT_ZONE     = (36, 97, 154, 195)#도난

MODEL_HAND   = "models/hand_yolov8n_new.pt"
# [설명] MODEL_HAND : 손(hand)만 전용으로 탐지하는 모델 (공개 데이터셋으로 학습됨)
MODEL_ITEM   = "models/best.pt"
# [설명] MODEL_ITEM : 직접 촬영/라벨링한 과자·음료 6종 상품 탐지 커스텀 모델

# ═══════════════════════════════════════════════════════
#  파라미터
# ═══════════════════════════════════════════════════════
# [설명] 아래 값들은 코드의 "튜닝 다이얼"입니다. 숫자를 조절하면 탐지 민감도/판정 기준이 바뀝니다.
GHOST_TTL         = 0
# [설명] 상품이 한 프레임 동안 안 잡혀도 직전 위치를 몇 프레임 더 "보이는 척" 유지할지(잔상 유예시간).
#        0이면 잔상 없이 즉시 사라짐 = 마치 불을 끄자마자 바로 어두워지는 것과 같음
NEAR_MARGIN       = 10
# [설명] 손 박스와 상품 박스가 "가깝다"고 인정해줄 여유 픽셀(margin). 두 박스 테두리를 살짝 부풀려서 겹침 판정
EXCLUDE_ZONE      = (567, 98, 668, 359)
# [설명] 오탐지가 잦은 영역(예: 매대 밖 배경)의 좌표. 이 구역 안의 상품 탐지 결과는 통째로 무시(금지구역)
PICK_SECONDS      = 0.5
# [설명] 손이 같은 상품 근처에 누적 몇 초 이상 머물러야 "집었다(PICK)"로 확정할지 기준 시간
PERSON_CONF       = 0.3
# [설명] 사람(person) 탐지를 인정하는 최소 신뢰도(confidence)
ALIAS_DIST        = 150
# [설명] 추적 ID가 끊겼다가 새로 발급될 때, 직전 위치와 이 거리(px) 이내면 "같은 사람"으로 보고 ID를 이어붙임
ALIAS_MEMORY_SECONDS = 2.0
# [Alias 수정/추가] Alias 매칭 후보로 남겨둘 "기억 시간". 기존에는 prev_person_boxes가
#        "직전 1프레임"의 위치만 들고 있어서, 단 1프레임만 가려져도 매칭 후보가 통째로
#        사라지는 문제가 있었음(가려짐이 발생하는 상황 자체를 못 잡는 모순). 이제는
#        person_last_box에 "최근 ALIAS_MEMORY_SECONDS초 이내에 본 사람"을 계속 들고 있다가
#        그 안에서 매칭을 시도함. LOST_TIMEOUT_SECONDS보다 짧게 잡아야, 도난 판정이 나기 전에
#        같은 사람으로 먼저 합쳐질 기회를 줄 수 있음

CHECKOUT_MIN_STAY = 4.0
# [설명] 계산대 구역 안에 최소 몇 초 머물러야 "계산하러 왔다"고 보고 B화면 검증을 시작할지
FINAL_MIN_SECONDS = 1.5
# [설명] 너무 짧게 스쳐 지나간 픽업 기록(노이즈)을 걸러내기 위한 최종 유효 시간 기준
B_PAYMENT_DELAY   = 3.0
# [설명] B화면(계산대)에서 상품이 처음 보인 뒤, 몇 초 더 지켜보고 나서 "결제 완료"로 확정할지
LOST_TIMEOUT_SECONDS = 1.0
# [설명] 사람이 화면에서 사라진 뒤 이만큼(초) 안 돌아오면 "퇴장"으로 간주(도난 판정의 트리거)
# [병합] person_tracker.py의 5초 부재 판정과 동일한 기준을 그대로 가져옴.
#        기존 MAX_LOST_FRAMES(프레임 수 기준)는 FRAME_SKIP=2, 영상 FPS에 따라
#        실제 "몇 초"인지가 달라지는 문제가 있어서(예: 30fps면 약 2초, 15fps면 약 4초),
#        영상이 바뀌어도 항상 동일하게 "1초"를 의미하는 시간 기준으로 통일함
FRAME_SKIP        = 2
# [설명] 매 N프레임마다 한 번만 추론을 돌려 연산량을 줄임 (2 = 한 프레임씩 건너뛰며 처리, 숨 고르기)
RESIZE_W          = 640
# [설명] 모든 좌표 계산의 기준이 되는 리사이즈 너비. 구역 선택 도구와 반드시 동일해야 좌표가 안 어긋남
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# [설명] GPU(CUDA)가 있으면 GPU로, 없으면 CPU로 자동 전환
USE_ALIAS = True   # True = 고도화(Alias 적용), False = Baseline(Alias 미적용)
unique_canonical_ids = set()
person_last_box = {}
# [Alias 수정] canonical_id -> (pbox, 마지막으로 본 시각). 기존 prev_person_boxes를 대체함.
#        prev_person_boxes는 매 프레임 통째로 교체되는 "직전 1프레임 스냅샷"이라
#        Alias 매칭의 재료로 쓰기에 수명이 너무 짧았음(가려진 순간 바로 후보가 0개가 됨).
#        이 딕셔너리는 사람을 볼 때마다 즉시 갱신되고, ALIAS_MEMORY_SECONDS 동안 후보로 남음
LABEL_MAP = {
    'yangparing':      '양파링',
    'ojingeoddankong': '오징어땅콩',
    'cocacola':        '코카콜라',
    'garamandeunBae':  '갈아만든배',
    'kkokkalcorn':     '꼬칼콘',
    'pocarisweat':     '포카리스웨트',
}
def kr(name): return LABEL_MAP.get(name, name)
# [설명] 모델이 출력하는 영문 클래스명을 화면 표시용 한글 이름으로 바꿔주는 번역기.
#        LABEL_MAP에 없는 이름이면 원래 영문 이름을 그대로 돌려줌(안전한 기본값)

ID_COLORS = [
    (255,80,80),(80,255,80),(80,80,255),(255,255,80),
    (255,80,255),(80,255,255),(200,130,50),(130,50,200)
]
def get_color(pid): return ID_COLORS[int(pid) % len(ID_COLORS)]
# [설명] 사람 ID마다 항상 같은 색이 배정되도록 색상표를 순환(%)시켜 고르는 함수.
#        ID가 색상표 개수보다 커지면 처음 색부터 다시 재사용(돌려쓰기)

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
# [설명] Windows에 기본 내장된 "맑은 고딕" 폰트 경로. OS가 바뀌면(예: Colab) 이 경로도 바꿔야 함

def put_text_kr(frame, text, pos, font_size=20, color=(255,255,255)):
    # [설명] cv2.putText는 한글(TTF/한글 글리프)을 그리지 못해 깨진 글자(□□□)로 나옴.
    #        그래서 ① OpenCV 프레임을 PIL 이미지로 변환 → ② PIL로 한글 텍스트를 그린 뒤
    #        ③ 다시 OpenCV 형식(numpy 배열, BGR)으로 되돌리는 우회 경로를 사용함
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
        # [설명] 폰트 파일을 못 찾을 경우(예: 다른 PC) 영문 기본 폰트로 대체해 프로그램이 죽지 않게 함
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # [설명] OpenCV는 BGR, PIL은 RGB 순서를 쓰므로 변환 시 채널 순서를 맞춰줌
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    # [설명] fill 색상도 BGR로 들어온 color를 다시 RGB 순서로 뒤집어 PIL에 전달
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # [설명] 그린 결과를 다시 numpy 배열 + BGR로 변환해 OpenCV 파이프라인에 그대로 합류시킴

TRAIL_COLOR = (0, 255, 65)   # 형광 초록 (BGR)

def draw_trail_transparent(frame, trail, alpha=0.6):
    # [설명] 이동 경로를 반투명한 "혜성 꼬리"처럼 그리는 함수.
    if len(trail) < 2:
        return frame
    overlay = frame.copy()
    # [설명] 원본 frame이 아니라 복사본(overlay) 위에 선을 그려서, 나중에 투명도를 섞을 재료로 사용
    for k in range(1, len(trail)):
        thickness = max(2, int(4 * k / len(trail)))
        # [설명] 경로의 뒷부분(최근 위치)일수록 선이 두꺼워져 "어느 쪽으로 움직였는지" 방향감을 줌
        cv2.line(overlay, trail[k-1], trail[k], TRAIL_COLOR, thickness)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    # [설명] overlay(선이 그려진 복사본)와 원본 frame을 alpha 비율로 섞어 반투명 효과를 냄
    #        (alpha=0.6 → 선이 60% 진하게, 원본 배경은 40% 비쳐 보임)

@st.cache_resource
def load_models():
    return YOLO(MODEL_HAND), YOLO(MODEL_ITEM), YOLO("yolov8n.pt")
    # [설명] @st.cache_resource: Streamlit은 상호작용마다 스크립트를 처음부터 재실행하는데,
    #        이 데코레이터가 없으면 클릭할 때마다 무거운 모델 가중치를 매번 새로 불러오게 됨.
    #        한 번 로드한 모델 객체를 캐시(보관)해두고 이후에는 재사용해 속도를 지켜줌
    #        세 번째 YOLO("yolov8n.pt")는 COCO로 사전학습된 범용 모델 - 여기서는 사람(person) 탐지용으로 사용

def is_near_or_overlap(hb, ib, m=NEAR_MARGIN):
    return hb[0]-m<ib[2] and hb[2]+m>ib[0] and hb[1]-m<ib[3] and hb[3]+m>ib[1]
    # [설명] 두 사각형(손 박스 hb, 상품 박스 ib)이 겹치거나 margin(m) 이내로 가까운지 판정하는
    #        AABB(축 정렬 사각형) 충돌 검사. 네 방향 모두 조건을 만족해야 "근접"으로 인정

def is_in_exclude_zone(c):
    cx,cy=(c[0]+c[2])/2,(c[1]+c[3])/2
    ex1,ey1,ex2,ey2=EXCLUDE_ZONE
    return ex1<cx<ex2 and ey1<cy<ey2
    # [설명] 박스의 중심점(cx,cy)이 EXCLUDE_ZONE 사각형 내부에 있는지 보는 "점-사각형 포함 검사"

def box_center(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)
# [설명] 박스 [x1,y1,x2,y2]의 정중앙 좌표를 구하는 작은 도우미 함수

def hand_in_person(hb, pb, m=20):
    hcx,hcy=box_center(hb)
    return pb[0]-m<hcx<pb[2]+m and pb[1]-m<hcy<pb[3]+m
    # [설명] 손 박스의 중심점이 사람 박스(margin 포함) 안에 들어오는지 보는 느슨한 포함 검사.
    #        이를 통해 "이 손은 누구의 손인가"를 연결함

def person_in_zone(pbox, zone):
    cx,cy=box_center(pbox); x1,y1,x2,y2=zone
    return x1<cx<x2 and y1<cy<y2
    # [설명] 사람 박스의 중심점이 특정 구역(zone, 예: 계산대) 안에 있는지 보는 함수.
    #        CHECKOUT_ZONE 진입 여부 판정에 사용됨

def add_log(msg, level="info"):
    icon={"info":"ℹ️","warn":"⚠️","danger":"🚨","ok":"✅"}.get(level,"ℹ️")
    st.session_state.event_log.append(f"{time.strftime('%H:%M:%S')} {icon} {msg}")
    # [설명] 이벤트 로그 한 줄을 만드는 함수. 시각(HH:MM:SS) + 심각도별 이모지 아이콘 + 메시지를
    #        조합해 session_state.event_log 리스트 뒤에 쌓음(사이드바 로그창에 표시됨)

def iou(b1, b2):
    x1,y1=max(b1[0],b2[0]),max(b1[1],b2[1])
    x2,y2=min(b1[2],b2[2]),min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0
    # [설명] IoU(Intersection over Union) = 두 박스가 겹치는 면적 ÷ 합쳐진 전체 면적.
    #        값이 1에 가까울수록 거의 같은 위치를 가리키는 박스라는 뜻

def dedup(boxes, thresh=0.3):
    kept=[]; boxes=sorted(boxes,key=lambda x:-x[5])
    # [설명] 신뢰도(conf, index 5)가 높은 순으로 먼저 정렬해, 더 확실한 탐지를 우선 채택
    for c in boxes:
        if not any(c[4]==k[4] and iou(c[:4],k[:4])>thresh for k in kept):
            kept.append(c)
            # [설명] 같은 클래스(c[4]==k[4])이면서 IoU가 thresh를 넘는, 즉 "같은 물건을 중복으로
            #        잡은" 박스는 버리고 가장 신뢰도 높은 하나만 남김(중복 탐지 제거)
    return kept

# ═══════════════════════════════════════════════════════
#  B화면 처리
# ═══════════════════════════════════════════════════════
# [설명] run_screen_b() 전체 요약:
#   계산대(B) 영상을 돌리면서 상품을 계속 탐지하고, 처음 뭔가 잡힌 시점부터
#   B_PAYMENT_DELAY초가 지나면 그때까지 인식된 모든 상품을 "결제 확정 목록"으로 굳힌다.
#   이걸 A화면에서 넘겨받은 pickup_items(매대에서 집은 목록)와 비교할 수 있도록
#   (확정된 상품 집합, 결과 메시지) 형태로 돌려준다.
def run_screen_b(item_model, pickup_items, b_frame_ph, b_status_msg,
                 b_table, b_delay, conf_thresh):
    cap_b = cv2.VideoCapture(VIDEO_B_PATH)
    if not cap_b.isOpened():
        return set(), "B영상 열기 실패"

    ITEM_CLASSES  = item_model.names
    # [설명] {클래스 인덱스: 클래스명} 형태의 딕셔너리 (모델에 내장된 매핑)
    first_detect  = {}
    # [설명] 클래스별로 "처음 감지된 시각"을 기록 (결제 대기시간 계산의 기준점)
    last_boxes    = []
    # [설명] 매 프레임 추론하지 않고 5프레임마다 한 번만 추론하므로, 추론을 안 하는 프레임에도
    #        화면에 박스를 계속 그려주기 위해 "가장 최근 추론 결과"를 보관해두는 캐시
    all_detected  = {}
    # [설명] 클래스별로 지금까지 본 것 중 "가장 높은 신뢰도"를 기록(최종 인식 결과 테이블용)
    payment_items = set()
    result_msg    = ""
    frame_idx     = 0
    pickup_kr     = [kr(i) for i in pickup_items]
    pick_display  = ", ".join(pickup_kr) if pickup_kr else "없음"
    b_status_msg.info("⏳ 계산대 물품 확인 중...")

    while cap_b.isOpened():
        ret, frame = cap_b.read()
        if not ret: break
        frame_idx += 1
        h, w = frame.shape[:2]
        display = cv2.resize(frame, (RESIZE_W, int(h * RESIZE_W / w)))
        # [설명] 화면 표시용으로는 RESIZE_W 기준으로 리사이즈(다른 화면들과 동일한 좌표계 유지)
        now = time.time()

        if frame_idx % 5 == 0:
            # [설명] 5프레임에 한 번만 모델 추론을 실행해 연산 부담을 줄임(나머지 프레임은 last_boxes 재사용)
            small = cv2.resize(frame, (480, int(h * 480 / w)))
            # [설명] 추론 자체는 더 작은 480px 폭으로 돌려 속도를 더 확보함
            sx = display.shape[1] / small.shape[1]
            sy = display.shape[0] / small.shape[0]
            # [설명] "추론용 작은 프레임" 좌표를 "화면 표시용 display 프레임" 좌표로 환산하기 위한 배율
            raw = []
            for box in item_model(small, verbose=False,
                                  conf=conf_thresh, device=DEVICE)[0].boxes:
                cn   = ITEM_CLASSES.get(int(box.cls[0]), f"cls{int(box.cls[0])}")
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                raw.append((int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy),
                            cn, conf, (0,200,0)))
                # [설명] small 좌표계에서 나온 박스를 sx,sy 배율로 키워 display 좌표계로 변환해 저장
                if cn not in first_detect:
                    first_detect[cn] = now
                    add_log(f"B화면: {kr(cn)} 감지됨", "info")
                if cn not in all_detected or conf > all_detected[cn]:
                    all_detected[cn] = conf
            last_boxes = dedup(raw)
            # [설명] 이번에 추론된 raw 박스들 중 중복(같은 상품 겹침)을 정리해 캐시에 저장

        for (x1,y1,x2,y2,cn,conf,color) in last_boxes:
            cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
            display = put_text_kr(display, f"{kr(cn)} {conf:.2f}",
                                  (x1,max(y1-28,0)), 18, color)
            # [설명] 추론을 안 한 프레임이라도 캐시된 last_boxes를 매 프레임 다시 그려서
            #        화면이 끊기지 않고 박스가 계속 보이는 것처럼 자연스럽게 유지함

        b_frame_ph.image(cv2.cvtColor(display,cv2.COLOR_BGR2RGB),
                         width='stretch')
        # [설명] Streamlit은 RGB를 기대하므로 BGR→RGB 변환 후 placeholder에 덮어써서 "실시간 영상"처럼 보이게 함

        if all_detected:
            df = pd.DataFrame({
                "물품명": [kr(cn) for cn in all_detected],
                "신뢰도": [f"{c:.2f}" for c in all_detected.values()],
                "상태":   ["인식 완료 ✅"] * len(all_detected)
            })
            df.index = range(1, len(df)+1)
            b_table.table(df)
            # [설명] 결제가 최종 확정되기 전까지는 "인식 완료" 상태로만 중간 표시(아직 결제 확정 아님)

        if first_detect and not payment_items:
            elapsed = now - min(first_detect.values())
            # [설명] 가장 먼저 잡힌 상품 기준 시각부터 경과 시간을 잼(여러 상품이 순차로 올라와도
            #        제일 처음 시작 시점을 기준으로 통일된 대기시간을 적용)
            b_status_msg.warning(f"⏳ 계산 중... {elapsed:.1f}s / {b_delay}s")
            if elapsed >= b_delay:
                # [설명] 충분히 기다렸다고 판단되면 그 시점까지 인식된 모든 상품을 "결제 확정"으로 굳힘
                payment_items = set(all_detected.keys())
                payment_kr    = [kr(i) for i in payment_items]
                missing_kr    = set(pickup_kr) - set(payment_kr)
                # [설명] 매대에서 집은 목록(pickup_kr) - 계산대에서 결제된 목록(payment_kr) = 누락 목록
                df = pd.DataFrame({
                    "물품명": [kr(cn) for cn in all_detected],
                    "신뢰도": [f"{c:.2f}" for c in all_detected.values()],
                    "상태":   ["계산 완료 ✅"] * len(all_detected)
                })
                df.index = range(1, len(df)+1)
                b_table.table(df)
                result_msg = ("✅ 모든 물품이 정상 결제되었습니다."
                              if not missing_kr
                              else f"⚠️ 누락 물품: {', '.join(missing_kr)}")
                add_log(result_msg, "ok" if not missing_kr else "danger")
                b_status_msg.success("🎉 계산 완료")

    cap_b.release()
    if not payment_items:
        msg = "🚨 B화면 물건 미인식 — 도난 의심"
        add_log(msg, "danger")
        return set(), msg
        # [설명] B화면 영상이 끝날 때까지 단 하나의 상품도 인식되지 못했다면
        #        "계산대 자체에 아무것도 안 올라왔다"는 뜻이므로 도난 의심으로 처리
    return payment_items, result_msg

# ═══════════════════════════════════════════════════════
#  페이지 설정
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Store-Protector UI", layout="wide")
# [설명] 브라우저 탭 제목과 전체 화면 폭(wide)을 설정. 가장 먼저, 단 한 번 실행되어야 하는 설정 함수
st.title("Store-Protector: 무인 매장 실시간 보안 관제 시스템")
st.markdown("---")

if "event_log"  not in st.session_state: st.session_state.event_log = []
if "running"    not in st.session_state: st.session_state.running = False
# [설명] Streamlit은 클릭 한 번마다 스크립트 전체를 재실행하기 때문에, 일반 변수는 매번 초기화되어
#        버립니다. session_state는 재실행을 넘어 값이 유지되는 "메모장" 같은 저장소이며,
#        이미 값이 있다면 덮어쓰지 않도록 if문으로 최초 1회만 초기화함

# ═══════════════════════════════════════════════════════
#  사이드바 — 설정 + 실행 버튼
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    st.write("카메라 A 상태: 🟢 정상")
    st.write("카메라 B 상태: 🟢 정상")
    # [설명] 실제 카메라 연결 상태를 체크하는 로직이 아니라, 데모용으로 항상 "정상"을 보여주는 고정 텍스트
    st.divider()

    st.markdown("**▶ 실행 전 설정하세요**")
    item_conf     = st.slider("물건 감지 Confidence", 0.0, 1.0, 0.4, 0.05,
                              help="낮을수록 더 많이 잡히나 오감지 증가")
    # [설명] 사용자가 직접 상품 탐지 민감도를 조절하는 슬라이더(기본값 0.4, 0.05 단위로 이동)
    st.divider()

    # ── 실행 / 중지 버튼 ──
    col_s, col_e = st.columns(2)
    start_btn = col_s.button("▶ 시작", type="primary",
                             disabled=st.session_state.running)
    stop_btn  = col_e.button("⏹ 중지",
                             disabled=not st.session_state.running)
    # [설명] 이미 실행 중이면 "시작" 버튼을 비활성화하고, 실행 중이 아니면 "중지" 버튼을 비활성화해
    #        중복 클릭으로 인한 오작동을 막음

    if start_btn:
        st.session_state.running = True
        st.rerun()
        # [설명] st.rerun()은 스크립트를 즉시 처음부터 다시 실행시켜, 바뀐 running 값이
        #        곧바로 화면(버튼 활성/비활성, 대기화면 등)에 반영되도록 강제함
    if stop_btn:
        st.session_state.running = False
        st.rerun()

    st.divider()
    st.markdown(f"**현재 Confidence:** `{item_conf}`")
    st.header("📋 이벤트 로그")
    log_box = st.empty()
    # [설명] st.empty()는 "나중에 내용을 채워 넣을 빈 자리"를 미리 잡아두는 placeholder.
    #        실제 로그 텍스트는 메인 루프 안에서 반복적으로 이 자리에 덮어써짐

# ═══════════════════════════════════════════════════════
#  메인 레이아웃
# ═══════════════════════════════════════════════════════
col1, col2 = st.columns(2)
with col1:
    st.subheader("A화면: 매장 전경 및 행동 분석")
    a_frame  = st.empty()
    a_status = st.empty()

with col2:
    st.subheader("B화면: 계산대 상품 인식")
    b_frame      = st.empty()
    st.markdown("#### [ 계산대 인식 결과 ]")
    b_table      = st.empty()
    b_status_msg = st.empty()
# [설명] 위 placeholder들(a_frame, b_frame 등)은 매 프레임마다 .image()/.table() 등으로
#        내용을 덮어써서, 마치 실시간 동영상처럼 보이게 만드는 핵심 장치

st.markdown("---")
st.subheader("C화면: 시스템 판독 결과 및 로그")
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    c_result = st.empty()
    # [설명] 화면을 1:2:1 비율로 나눠 가운데(c2) 컬럼만 사용 → 판독 결과 패널을 중앙에 넓게 배치

# 대기 상태 표시
if not st.session_state.running:
    with c_result.container():
        st.info("### ⏸ 대기 중\n- 사이드바에서 설정 후 **▶ 시작** 버튼을 눌러주세요.")
    b_status_msg.info("⏳ 시작 버튼을 눌러주세요...")
    st.stop()
    # [설명] st.stop()은 이 지점에서 스크립트 실행을 그대로 멈춤. 즉 "시작" 버튼을 눌러
    #        running=True가 되고 rerun되기 전까지는 아래의 무거운 메인 루프가 절대 실행되지 않음

# ═══════════════════════════════════════════════════════
#  실행 중 상태
# ═══════════════════════════════════════════════════════
with c_result.container():
    st.info("### 🔄 시스템 판독 대기 중...\n- 고객 행동을 분석하고 있습니다.")
b_status_msg.info("⏳ 고객의 키오스크 접근을 대기 중입니다...")

hand_model, item_model, person_model = load_models()
# [설명] 캐시된 모델을 가져옴(이미 로드되어 있다면 재사용, 처음이라면 이때 로드)
ITEM_CLASSES = item_model.names

# [설명] 아래는 메인 루프 전체에서 사람별로 "기억"해야 하는 상태들의 모음입니다.
#        하나씩 어떤 역할인지 짚어보면:
permanent_trails  = defaultdict(list)
# [설명] 사람 ID별 이동 경로 전체(점들의 리스트). 한 번 쌓이면 지워지지 않아 '영구' 경로(trail)
person_snapshots  = {}
# [설명] 사람 ID별 "신원 사진" 한 장. 등장 후 3초가 지난 안정된 순간에 캡처해 보관(도난/누락 보고서용)
person_first_seen = {}
# [설명] 사람 ID별 처음 감지된 시각. person_snapshots를 "3초 뒤에" 찍기 위한 기준 시계
last_seen_items   = {}
# [설명] 상품 클래스별 "마지막으로 본 위치 + 남은 잔상 프레임 수"(GHOST_TTL 로직용 버퍼)
confirmed_picks   = set()
# [설명] 이미 PICK으로 확정되어 로그에 한 번 찍힌 (사람,상품) 조합 - 매 프레임 중복 로그 방지용
accumulated_time  = {}
# [설명] (사람ID, 상품클래스) 키로 "지금까지 누적된 보유/근접 시간"을 저장
near_start_time   = {}
# [설명] 손 인덱스별로 "현재 어떤 사람이, 어떤 상품 곁에서, 언제부터" 머물기 시작했는지 기록(타이머 시작점)
person_picks      = defaultdict(set)
# [설명] 사람 ID별로 "한 번이라도 PICK이 확정된 상품 클래스" 집합(최종 보고용)
person_pick_time  = defaultdict(dict)
# [설명] 사람 ID별 상품별 최종 누적 시간 - FINAL_MIN_SECONDS 기준으로 노이즈를 걸러낼 때 사용
id_alias          = {}
# [설명] ByteTrack이 가려짐(occlusion) 등으로 새 ID를 발급했을 때, "새 ID -> 원래(정규) ID"로
#        연결해주는 별칭 사전. 이게 없으면 사람이 잠깐 가려졌다가 나타날 때마다 다른 사람으로 취급됨
checkout_enter    = {}
# [설명] 사람 ID별로 계산대 구역(CHECKOUT_ZONE)에 "진입한 시각"을 기록 → 체류시간 계산용
checkout_done     = set()
# [설명] 이미 계산대 체크를 한 번 마친 사람 ID 모음(같은 사람이 같은 구역에서 재트리거되는 것 방지)
# [요청 반영] 단, B화면에서 아무것도 인식이 안 된 경우는 여기서 다시 빼서(discard), 아래쪽
#        "화면에서 사라짐" 기반 도난 판정이 이 사람을 마저 검사할 수 있도록 길을 열어둠
b_check_attempted = set()
# [요청 반영] checkout_done과 별개로, "이 사람에 대해 B화면 검증을 이미 한 번 시도했는지"만
#        기록함. checkout_done은 위 이유로 다시 빠질 수 있기 때문에, B화면 검증이 같은 사람에게
#        반복 실행되는 것을 막으려면 이 전용 플래그가 따로 필요함
lost_counter      = defaultdict(int)
# [설명] 사람 ID별로 "연속으로 안 보인 프레임 수"를 세는 카운터. (이제는 아래 person_last_seen
#        기반 시간 판정을 사용하므로 이 카운터 자체는 더 이상 도난 판정에 쓰이지 않음)
person_last_seen  = {}
# [병합] person_tracker.py 스타일: 프레임 수 대신 "마지막으로 보인 실제 시각(time.time())"을
#        저장함. FRAME_SKIP이나 영상 FPS가 달라져도 "사라진 지 몇 초"라는 기준이 흔들리지 않음
prev_tracked_pids = set()
# [설명] 직전 프레임에 추적되고 있던 사람 ID 집합. 이번 프레임과 비교(차집합)해
#        "이번에 새로 사라진 사람"을 찾아내는 데 사용
b_triggered       = False
# [설명] B화면(계산대) 검증이 이미 진행 중인지 표시하는 플래그. 여러 사람이 동시에 계산대
#        조건을 만족해 run_screen_b()가 중복 실행되는 것을 막는 잠금장치 역할
theft_confirmed   = set()
# [요청 반영] 도난이 "최초로 확정된" 사람 ID 모음. 로그 기록/스냅샷 저장/snapshots 폴더 파일
#        저장/C화면 업데이트처럼 "한 번만 해도 되는 무거운 처리"는 여기 들어있는지로 구분해서
#        딱 한 번만 실행함 (반복 실행 시 같은 파일을 계속 덮어쓰거나 로그가 도배되는 것 방지)

def judge_departure(pid, frame):
    # [요청 반영] "이 사람이 매장을 떠난 것으로 확정됐다"는 전제 하에, 픽업 이력과 계산 여부를
    #        대조해서 도난인지 아닌지 판정하는 로직을 함수로 분리함. 이렇게 분리해두면
    #        ① 매 프레임 LOST_TIMEOUT_SECONDS 경과 체크 ② 영상이 완전히 끝나는 순간의
    #        마무리 점검, 이 두 군데에서 똑같은 로직을 중복 작성하지 않고 재사용할 수 있음
    picks = person_picks.get(pid, set())
    valid = {i for i in picks
             if person_pick_time[pid].get(i,0) >= FINAL_MIN_SECONDS}
    # [설명] 너무 짧게 스친 픽업(노이즈)은 제외하고, 충분히 오래 들고 있던 것만 "유효 픽업"으로 인정
    if valid and pid not in checkout_done:
        # [설명] 유효한 픽업이 있는데 계산대를 거치지 않고(checkout_done에 없음) 사라졌다면 도난으로 판단
        if pid not in theft_confirmed:
            # [요청 반영] 로그 기록/스냅샷/파일저장/C화면 갱신처럼 "한 번만 해도 되는
            #        무거운 처리"는 이 사람이 처음 확정되는 이 순간에만 실행함
            msg = f"🚨 도난! Person#{pid}: {', '.join([kr(i) for i in valid])}"
            add_log(msg, "danger")
            st.toast("도난 발생!", icon="🚨")

            # ── 도난 순간 A화면에 진한 경로 그리기 ──────
            theft_frame = frame.copy()
            # 빨간 테두리
            cv2.rectangle(theft_frame,(0,0),
                          (theft_frame.shape[1],theft_frame.shape[0]),
                          (0,0,255),20)
            # 해당 사람의 경로를 진하게 (alpha=1.0, 두께 3)
            trail = permanent_trails.get(pid, [])
            for k in range(1, len(trail)):
                progress = k / len(trail)
                thickness = max(2, int(6 * progress))
                cv2.line(theft_frame, trail[k-1], trail[k],
                         TRAIL_COLOR, thickness)
            # 도난 텍스트 오버레이
            theft_frame = put_text_kr(
                theft_frame,
                f"🚨 도난 Person#{pid}",
                (10, 60), 24, (0, 0, 255)
            )
            theft_rgb = cv2.cvtColor(theft_frame, cv2.COLOR_BGR2RGB)

            # ── 파일 저장 ────────────────────────────────
            os.makedirs("snapshots", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            if pid in person_snapshots:
                snap_bgr = cv2.cvtColor(
                    person_snapshots[pid], cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"snapshots/person_{pid}_{ts}.jpg", snap_bgr)
            cv2.imwrite(
                f"snapshots/theft_scene_{pid}_{ts}.jpg",
                theft_frame)

            # ── A화면 업데이트 ───────────────────────────
            a_frame.image(theft_rgb, width='stretch')
            # [설명] 이번 프레임 루프의 맨 끝에서도 a_frame이 다시 갱신되지만,
            #        도난 장면만큼은 즉시 한 번 먼저 화면에 띄워 사용자가 놓치지 않게 함

            # ── C화면: 인물 사진 위, 도난 순간 아래 ─────
            with c_result.container():
                st.error("### 🚨 도난 발생!")
                st.error(
                    f"**[ 도난 감지 결과 ]**\n"
                    f"- 감지된 인원: Person#{pid}\n"
                    f"- 미결제 물품: {', '.join([kr(i) for i in valid])}\n"
                    f"---\n"
                    f"**⚠️ 계산 없이 퇴장한 것으로 판단됩니다.**"
                )
                # 인물 스냅샷 (위)
                if pid in person_snapshots:
                    st.image(
                        person_snapshots[pid],
                        caption=f"👤 Person#{pid} 최초 감지 스냅샷",
                        width=200
                    )
                else:
                    st.warning("스냅샷 없음")
                # 도난 순간 A화면 (아래, 크게)
                st.image(
                    theft_rgb,
                    caption="🚨 도난 순간 A화면 (이동 경로 포함)",
                    width='stretch'
                )

            theft_confirmed.add(pid)
            # [요청 반영] 이 사람을 "확정된 도난자"로 표시해둠 (이후엔 아래 else에서 반복 알림)

        else:
            # [요청 반영] 학교 컴퓨터 환경이라 소리는 빼고, 토스트는 시간 간격 제한 없이
            #        이 사람이 화면에 안 보이는 동안 매 프레임 계속 반복해서 띄움
            #        (로그 기록/스냅샷/파일저장/C화면 갱신 같은 무거운 처리는 여전히 반복 안 함)
            st.toast(f"🚨 도난 발생: Person#{pid}", icon="⚠️")

        # [요청 반영] 도난이 확정된 사람은 prev_tracked_pids에서 빼지 않고 그대로 둠
        #        → 다음 프레임에도 계속 "사라진 사람"으로 잡혀서 반복 토스트가 가능해짐

    else:
        # [병합] person_tracker.py가 판정 후 is_person_inside=False로 리셋했던 것과 동일하게,
        #        도난이 아닌 것으로 결론 난 사람(픽업이 없거나 이미 계산함)은 추적 후보에서
        #        완전히 제거함. 이렇게 안 하면 다음 프레임부터도 prev_tracked_pids에 계속
        #        남아있어 똑같은 검사가 끝없이 반복되는 비효율이 있었음
        prev_tracked_pids.discard(pid)
        person_last_seen.pop(pid, None)

cap_a = cv2.VideoCapture(VIDEO_A_PATH)
current_pids = set()
# [요청 반영] 영상이 한 프레임도 처리되지 못하고 곧바로 끝나는 극단적인 경우에도, 아래
#        "영상 종료 후 마무리 점검" 코드가 current_pids를 안전하게 참조할 수 있도록 미리 빈 값으로 초기화
last_valid_frame = None
# [버그 수정] cap_a.read()는 영상이 끝나는 마지막 시도에서 (False, None)을 돌려주는데,
#        이때 frame이 None으로 덮어써진 채로 break되어버림. "영상 종료 후 마무리 점검"에서
#        judge_departure(pid, frame)을 호출하면 frame이 None이라 frame.copy()에서 에러가
#        났었음. 그래서 매 프레임 끝에서 "마지막으로 정상 처리된 프레임"을 별도로 저장해두고,
#        마무리 점검에서는 frame 대신 이걸 사용하도록 함
frame_num = 0

try:
    while cap_a.isOpened() and st.session_state.running:
        ret, frame = cap_a.read()
        if not ret: break
        frame_num += 1
        if frame_num % FRAME_SKIP != 0: continue
        # [설명] FRAME_SKIP 배수가 아닌 프레임은 통째로 건너뜀(연산량 절약, 영상 자체는 계속 흘러감)

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (RESIZE_W, int(h * RESIZE_W / w)))
        # [설명] 이후 모든 좌표 계산(구역, 박스 등)은 이 리사이즈된 프레임 기준으로 통일됨
        now = time.time()
        # [병합] person_tracker.py처럼, 이번 프레임의 기준 시각을 맨 앞에서 한 번만 구해둠.
        #        기존 코드는 now를 한참 아래(손-상품 처리 구간)에서야 처음 정의했는데,
        #        그보다 먼저 실행되는 사람 추적 구간(person_first_seen[...] = now)에서
        #        이미 now를 참조하고 있어서, 같은 프레임 안에서 시간 기준이 어긋날 수 있었음

        hand_res   = hand_model(frame, verbose=False, imgsz=1280,
                                conf=0.3, device=DEVICE)[0]
        item_res   = item_model(frame, verbose=False, imgsz=1280,
                                conf=item_conf, device=DEVICE)[0]
        person_res = person_model.track(frame, persist=True, verbose=False,
                                        imgsz=1280, conf=PERSON_CONF,
                                        classes=[0], tracker="bytetrack.yaml",
                                        device=DEVICE)[0]
        # [설명] 한 프레임에 세 모델을 모두 돌림: 손 탐지 / 상품 탐지 / 사람 탐지+추적.
        #        person_model.track(persist=True)는 프레임이 바뀌어도 같은 사람에게 같은 ID를
        #        이어서 부여하려고 시도하는 ByteTrack 추적기. classes=[0]은 COCO 클래스 중
        #        "사람(person)"만 걸러서 본다는 의미

        hand_boxes = [b.xyxy[0].tolist() for b in hand_res.boxes]
        item_boxes = []
        for b in item_res.boxes:
            cn = ITEM_CLASSES.get(int(b.cls[0]), f"cls{int(b.cls[0])}")
            c  = b.xyxy[0].tolist()
            if not is_in_exclude_zone(c): item_boxes.append((cn,c))
            # [설명] EXCLUDE_ZONE(오탐지 잦은 구역) 안에 중심점이 있는 탐지 결과는 통째로 버림

        det_names = {cn for cn,_ in item_boxes}
        for cn in list(last_seen_items):
            if cn not in det_names:
                last_seen_items[cn][1] -= 1
                if last_seen_items[cn][1] <= 0: del last_seen_items[cn]
                else: item_boxes.append((cn, last_seen_items[cn][0]))
                # [설명] 이번 프레임에 다시 안 잡힌 클래스는 남은 잔상 프레임(GHOST_TTL)을 1씩 깎으며
                #        그동안은 직전 위치를 그대로 item_boxes에 끼워 넣어 "잠깐 깜빡여도 안 사라지게" 보정
                #        (현재 GHOST_TTL=0이라 사실상 즉시 사라지지만, 값을 올리면 유예 효과가 생김)
            else:
                last_seen_items[cn] = [next(c for n,c in item_boxes if n==cn),
                                       GHOST_TTL]
                # [설명] 이번에 정상 탐지된 클래스는 위치와 잔상 카운터(GHOST_TTL)를 다시 최신화

        tracked_persons=[]; current_pids=set()
        # [Alias 수정] 기존 new_pb={} 선언을 제거함. prev_person_boxes를 매 프레임 통째로
        #        교체하던 방식 대신, person_last_box를 사람을 볼 때마다 즉시 갱신하는 방식으로
        #        바꿨기 때문에 한 프레임 동안 모아뒀다가 한 번에 교체하는 중간 그릇이 더 이상 필요 없음
        if person_res.boxes.id is not None:
            for box,tid in zip(person_res.boxes, person_res.boxes.id):
                tid=int(tid); pbox=box.xyxy[0].tolist()
                cx,cy=int(box_center(pbox)[0]),int(box_center(pbox)[1])
                canonical = tid
                if USE_ALIAS:
                    canonical = id_alias.get(tid, tid)
                    if tid not in id_alias:
                        candidates = {
                            pid: pb for pid, (pb, t) in person_last_box.items()
                            if now - t <= ALIAS_MEMORY_SECONDS
                        }
                        print(f"[DEBUG] 새 ID={tid} 위치=({cx},{cy}) | 후보 수={len(candidates)}")
                        matched = False
                        for pp, pb in candidates.items():
                            pcx, pcy = int(box_center(pb)[0]), int(box_center(pb)[1])
                            dx, dy = abs(cx - pcx), abs(cy - pcy)
                            print(f"[DEBUG]   후보={pp} 위치=({pcx},{pcy}) dx={dx} dy={dy} (기준 ALIAS_DIST={ALIAS_DIST})")
                            if dx < ALIAS_DIST and dy < ALIAS_DIST:
                                root = id_alias.get(pp, pp)
                                id_alias[tid] = root
                                canonical = root
                                matched = True
                                print(f"[DEBUG]   ✅ 매칭 성공: {tid} -> {root}")
                                for k in list(accumulated_time):
                                    if k[0] == pp and (root, k[1]) not in accumulated_time:
                                        accumulated_time[(root, k[1])] = accumulated_time[k]
                                break
                        if not matched:
                            print(f"[DEBUG]   ❌ 매칭 실패 — 새 ID {tid}로 확정")
                person_last_box[canonical] = (pbox, now)
                # [Alias 수정] 기존 new_pb[canonical]=pbox 자리를 대체함. 이 사람을 봤다는 사실과
                #        위치, 시각을 즉시 갱신해서 다음 매칭 시도의 재료로 항상 최신 상태를 유지함
                tracked_persons.append((canonical,pbox))
                unique_canonical_ids.add(canonical)
                permanent_trails[canonical].append((cx,cy))
                current_pids.add(canonical)
                person_last_seen[canonical]=now
                # [병합] "사라진 프레임 카운터를 0으로 리셋" 대신, "마지막으로 보인 시각"을
                #        현재 시각(now)으로 갱신함 (프레임 세는 방식 → 초 단위로 재는 방식)

                # 처음 감지 시 시각 기록 → 3초 후 스냅샷 저장
                if canonical not in person_snapshots:
                    if canonical not in person_first_seen:
                        person_first_seen[canonical] = now  # 첫 감지 시각 기록
                    elif now - person_first_seen[canonical] >= 3.0:
                        # 3초 경과 후 스냅샷 저장
                        x1s, y1s, x2s, y2s = map(int, pbox)
                        x1c = max(0, x1s - 20);
                        y1c = max(0, y1s - 20)
                        x2c = min(frame.shape[1], x2s + 20)
                        y2c = min(frame.shape[0], y2s + 20)
                        # [설명] 사람 박스보다 20px씩 여유를 둬서 살짝 더 넓게 자르되, 프레임 경계를
                        #        벗어나지 않도록 min/max로 클램핑(잘림 방지)
                        crop = frame[y1c:y2c, x1c:x2c]
                        if crop.size > 0:
                            person_snapshots[canonical] = cv2.cvtColor(
                                crop, cv2.COLOR_BGR2RGB)
                            # [설명] Streamlit의 st.image는 RGB를 기대하므로 저장 시점에 미리 변환해둠

        for pid in prev_tracked_pids - current_pids:
            # [설명] 직전 프레임엔 있었지만 이번 프레임엔 없는 ID들 = "이번에 사라진 사람들"
            elapsed_lost = now - person_last_seen.get(pid, now)
            # [병합] person_tracker.py와 동일하게, 프레임을 세는 대신 "마지막으로 본 시각"과
            #        "지금(now)"의 실제 시간 차이(초)를 직접 계산함
            if elapsed_lost >= LOST_TIMEOUT_SECONDS:
                # [설명] 단순히 잠깐 가려진 게 아니라 충분히 오래(LOST_TIMEOUT_SECONDS초) 안 보였다면
                #        진짜로 매장을 떠났다고 판단하고 판정 로직(judge_departure)을 실행
                judge_departure(pid, frame)

        prev_tracked_pids = current_pids | (prev_tracked_pids - current_pids)
        # [설명] 이번에 보인 사람들(current_pids) + 아직 LOST_TIMEOUT_SECONDS에 도달하지 않아
        #        "유예 기간 중"인 사라진 사람들을 합쳐서 다음 프레임 비교 기준으로 넘김
        #        (바로 제거하지 않고 경과 시간이 다 찰 때까지는 계속 추적 후보로 남겨둠)
        # [병합] MAX_LOST_FRAMES → LOST_TIMEOUT_SECONDS로 바뀐 데 맞춰 변수명만 갱신(동작은 동일)
        # [Alias 수정] 기존 "prev_person_boxes = new_pb" 줄을 제거함. person_last_box가 사람을
        #        볼 때마다 위에서 이미 즉시 갱신되므로, 루프 끝에서 따로 통째로 교체해줄 필요가 없음

        hand_to_pid = {}
        for hi,hb in enumerate(hand_boxes):
            for tid,pb in tracked_persons:
                if hand_in_person(hb,pb): hand_to_pid[hi]=tid; break
                # [설명] 손 박스 hi가 어느 사람(tid)의 몸통 범위 안에 있는지 찾아 연결(처음 맞는 사람으로 확정)

        picking_items=set(); picked_coords=set()
        # [병합] now는 이번 프레임 맨 앞에서 이미 한 번 구해뒀으므로 여기서 다시 구하지 않음
        #        (같은 프레임 안에서는 항상 같은 시각 기준을 써야 시간 계산이 어긋나지 않음)
        for hi,hbox in enumerate(hand_boxes):
            near_item = next(
                ((cn,c) for cn,c in item_boxes if is_near_or_overlap(hbox,c)),
                None)
            # [설명] 이 손과 가장 먼저 겹치거나 가까운 상품 하나를 찾음(없으면 None)
            if near_item:
                cn,coords=near_item; pid=hand_to_pid.get(hi); key=(pid,cn)
                prev=near_start_time.get(hi)
                if prev is None or prev[0]!=pid or prev[1]!=cn:
                    # [설명] 이 손이 "방금 막" 이 상품 곁에 머물기 시작했거나, 직전과 다른
                    #        사람/상품 조합으로 바뀐 경우 → 타이머를 새로 시작해야 함
                    if prev:
                        ok=(prev[0],prev[1])
                        accumulated_time[ok]=accumulated_time.get(ok,0)+(now-prev[2])
                        # [설명] 이전 조합의 누적시간을 먼저 정산(저장)해두고 새 타이머로 넘어감
                    near_start_time[hi]=(pid,cn,now)
                session=now-near_start_time[hi][2]
                # [설명] 이번 "근접 세션"이 시작된 후 지금까지 흐른 시간
                total=accumulated_time.get(key,0)+session
                # [설명] 과거에 쌓인 누적시간 + 이번 세션 시간 = 현재까지의 총 보유시간
                x1,y1,x2,y2=map(int,coords)
                if total >= PICK_SECONDS:
                    if key not in confirmed_picks:
                        confirmed_picks.add(key)
                        add_log(f"PICK: {'P#'+str(pid) if pid else '?'}"
                                f" -> {kr(cn)} ({total:.1f}s)", "info")
                        # [설명] 같은 (사람,상품) 조합에 대해 로그가 매 프레임 반복되지 않도록
                        #        confirmed_picks에 등록된 적 없을 때 딱 한 번만 기록
                    if pid is not None:
                        person_picks[pid].add(cn)
                        person_pick_time[pid][cn]=total
                    accumulated_time[key]=total
                    picking_items.add(cn); picked_coords.add(tuple(coords))
                    color=get_color(pid) if pid else (0,0,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                    lbl = (f"PICK {'P#'+str(pid)+' ' if pid else ''}"
                           f"{kr(cn)} {total:.1f}s")
                    frame = put_text_kr(frame,lbl,(x1,max(y1-40,0)),20,color)
                    # [설명] 기준 시간을 넘긴 "확정 픽업"은 두꺼운 박스 + PICK 라벨로 강조 표시
                else:
                    bw=int((x2-x1)*min(total/PICK_SECONDS,1.0))
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                    cv2.rectangle(frame,(x1,y2+4),(x1+bw,y2+10),(0,255,255),-1)
                    frame = put_text_kr(frame,
                                        f"holding: {kr(cn)} {total:.1f}s",
                                        (x1,max(y1-28,0)),18,(0,255,255))
                    # [설명] 아직 PICK_SECONDS에 못 미친 경우, 박스 아래에 진행률 바(bw)를 그려
                    #        "충전 중인 게이지"처럼 얼마나 더 들고 있어야 확정되는지 시각적으로 보여줌
            else:
                if hi in near_start_time:
                    p,cn,st2=near_start_time.pop(hi)
                    k=(p,cn)
                    accumulated_time[k]=accumulated_time.get(k,0)+(now-st2)
                    # [설명] 손이 더 이상 어떤 상품과도 가깝지 않게 되면, 진행 중이던 타이머를
                    #        멈추고 그동안 쌓인 시간을 누적시간(accumulated_time)에 정산해 저장

        cx1,cy1,cx2,cy2=CHECKOUT_ZONE
        cv2.rectangle(frame,(cx1,cy1),(cx2,cy2),(0,255,128),2)
        cv2.putText(frame,"COUNTER",(cx1,cy1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,128),1)
        # [설명] 계산대 구역을 화면에 초록색 사각형으로 표시해 사용자가 어디가 트리거 영역인지 알 수 있게 함

        for tid,pbox in tracked_persons:
            in_zone=person_in_zone(pbox,CHECKOUT_ZONE)
            if in_zone:
                if tid not in checkout_enter: checkout_enter[tid]=now
                stay=now-checkout_enter[tid]
                bw=int((cx2-cx1)*min(stay/CHECKOUT_MIN_STAY,1.0))
                cv2.rectangle(frame,(cx1,cy2+4),(cx1+bw,cy2+10),(0,255,128),-1)
                cv2.putText(frame,f"P#{tid} {stay:.1f}s",(cx1,cy2+22),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,128),1)
                # [설명] 계산대 안에 머문 시간(stay)을 PICK 게이지와 비슷하게 진행률 바로 표시
                if (stay>=CHECKOUT_MIN_STAY and tid not in b_check_attempted
                        and not b_triggered):
                    # [설명] ① 충분히 오래 머물렀고 ② 이 사람이 처음 계산을 시도하는 것이며
                    #        ③ 다른 사람의 B화면 검증이 동시에 돌고 있지 않을 때만 검증을 시작
                    # [요청 반영] checkout_done 대신 b_check_attempted로 트리거 조건을 바꿈
                    #        (checkout_done은 B화면 미인식 시 나중에 다시 빠질 수 있어서,
                    #        그걸로 막으면 같은 사람에게 B화면 검증이 반복 실행될 위험이 있음)
                    checkout_done.add(tid); b_check_attempted.add(tid); b_triggered=True
                    picks=person_picks.get(tid,set())
                    valid={i for i in picks
                           if person_pick_time[tid].get(i,0)>=FINAL_MIN_SECONDS}
                    pick_kr=[kr(i) for i in valid]
                    pick_display=", ".join(pick_kr) if pick_kr else "없음"
                    add_log(f"P#{tid} 계산대 진입 -> B화면 시작"
                            f" (픽업: {pick_display})", "info")
                    a_status.info(f"P#{tid} 계산대 진입 — B화면 실행 중...")
                    with c_result.container():
                        st.info(f"### 🔄 판독 대기 중...\n"
                                f"- 매대에서 집은 물품: {pick_display}")
                    b_items,b_msg=run_screen_b(
                        item_model, valid, b_frame, b_status_msg,
                        b_table, B_PAYMENT_DELAY, item_conf
                    )
                    # [설명] 여기서 run_screen_b()가 끝날 때까지 A화면의 메인 루프는 멈춰서 기다림
                    #        (동기 호출). B화면 검증이 끝나야 비로소 아래 판정 로직이 이어서 실행됨
                    b_kr=[kr(i) for i in b_items]
                    missing_kr=set(pick_kr)-set(b_kr)
                    detected_str=", ".join(b_kr) if b_kr else "탐지된 물품 없음"
                    if not b_items:
                        if pick_kr:
                            # [요청 반영] 집은 물품이 있는데 B화면에서 전혀 인식이 안 된 경우,
                            #        여기서 바로 "도난 의심"으로 단정 짓지 않음. checkout_done에서
                            #        이 사람을 다시 빼서(discard), 아래쪽의 "화면에서 사라짐" 기반
                            #        도난 판정(LOST_TIMEOUT_SECONDS)이 이 사람을 마저 검사할 수
                            #        있도록 넘겨줌. 그쪽 판정은 스냅샷·이동경로까지 증거로 같이
                            #        남기기 때문에 단순 "의심" 메시지보다 훨씬 탄탄한 결론이 됨
                            checkout_done.discard(tid)
                            add_log(f"P#{tid} B화면 미인식 — 매장 이탈 여부 계속 확인", "warn")
                            with c_result.container():
                                st.warning(
                                    "### ⏳ 확인 필요\n"
                                    "- B화면에서 물건이 인식되지 않았습니다.\n"
                                    "- 고객이 매장을 완전히 벗어나는지 계속 지켜보고,\n"
                                    "  실제로 화면에서 사라지면 그때 도난 여부를 최종 판정합니다."
                                )
                        else:
                            # [요청 반영] 애초에 집은 물품이 없으면 훔칠 것도 없으므로
                            #        도난 의심 자체를 띄우지 않음
                            with c_result.container():
                                st.info(
                                    "### ℹ️ 특이사항 없음\n"
                                    "- 집은 물품이 없어 결제 검증이 필요하지 않습니다."
                                )
                    elif missing_kr:
                        # 누락 시 경로 포함 A화면 생성
                        missing_frame = frame.copy()
                        cv2.rectangle(missing_frame,(0,0),
                                      (missing_frame.shape[1],missing_frame.shape[0]),
                                      (0,165,255),20)  # 주황 테두리
                        trail_m = permanent_trails.get(tid, [])
                        for k in range(1, len(trail_m)):
                            progress = k / len(trail_m)
                            thickness = max(2, int(6 * progress))
                            cv2.line(missing_frame, trail_m[k-1], trail_m[k],
                                     TRAIL_COLOR, thickness)
                        missing_frame = put_text_kr(
                            missing_frame,
                            f"⚠️ 누락 Person#{tid}",
                            (10, 60), 24, (0, 165, 255)
                        )
                        missing_rgb = cv2.cvtColor(missing_frame, cv2.COLOR_BGR2RGB)
                        # [설명] 도난 장면과 같은 패턴(테두리+경로+텍스트)을 색상만 주황으로 바꿔
                        #        "완전 도난"보다는 "결제 누락"이라는 경고 수위 차이를 시각적으로 구분

                        # 파일 저장
                        os.makedirs("snapshots", exist_ok=True)
                        ts_m = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"snapshots/missing_scene_{tid}_{ts_m}.jpg",
                                    missing_frame)
                        if tid in person_snapshots:
                            snap_bgr = cv2.cvtColor(
                                person_snapshots[tid], cv2.COLOR_RGB2BGR)
                            cv2.imwrite(
                                f"snapshots/person_{tid}_{ts_m}.jpg", snap_bgr)

                        with c_result.container():
                            st.error("### ⚠️ 주의: 계산 불일치 감지!")
                            st.warning(
                                f"**[ 최종 판독 결과 ]**\n"
                                f"- 매대에서 집은 물품: {pick_display}\n"
                                f"- 계산대에서 결제한 물품: {detected_str}\n"
                                f"---\n"
                                f"**⚠️ 누락 물품: {', '.join(missing_kr)}**"
                            )
                            # 인물 스냅샷 (위)
                            if tid in person_snapshots:
                                st.image(
                                    person_snapshots[tid],
                                    caption=f"👤 Person#{tid} 스냅샷",
                                    width=200
                                )
                            # 경로 포함 A화면 (아래, 크게)
                            st.image(
                                missing_rgb,
                                caption="⚠️ 누락 감지 순간 A화면 (이동 경로 포함)",
                                width='stretch'
                            )
                    else:
                        with c_result.container():
                            st.success("### 🟢 확인: 계산 일치")
                            st.success(
                                f"**[ 최종 판독 결과 ]**\n"
                                f"- 매대에서 집은 물품: {pick_display}\n"
                                f"- 계산대에서 결제한 물품: {detected_str}\n"
                                f"---\n"
                                f"✅ 모든 물품이 정상 결제되었습니다."
                            )
                        # [설명] pick_kr와 b_kr이 완전히 일치(누락 없음)하면 초록색 성공 메시지로 마무리
                    b_triggered=False
                    # [설명] 이번 사람에 대한 B화면 검증이 끝났으니 잠금을 풀어 다음 사람도 검증 가능하게 함
            else:
                checkout_enter.pop(tid,None)
                # [설명] 계산대 구역을 벗어나면 체류 타이머 기록을 지워, 다음에 다시 들어올 때
                #        0초부터 새로 측정되게 함(잠깐 스쳐 지나간 경우 카운트 안 되도록)

        # 사람 박스 + 투명 경로
        for tid,pbox in tracked_persons:
            color=get_color(tid); x1,y1,x2,y2=map(int,pbox)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"P#{tid}",(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            frame = draw_trail_transparent(
                frame, permanent_trails[tid], alpha=0.6)
            picks=person_picks.get(tid,set())
            if picks:
                frame=put_text_kr(
                    frame,f"[{', '.join([kr(p) for p in picks])}]",
                    (x1,max(y1-40,0)),16,color)
                # [설명] 그 사람이 지금까지 픽업한 상품 목록을 머리 위에 한글로 함께 표시

        for hi,hbox in enumerate(hand_boxes):
            x1,y1,x2,y2=map(int,hbox); pid=hand_to_pid.get(hi)
            color=get_color(pid) if pid else (255,200,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"hand(P#{pid})" if pid else "hand",
                        (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
            # [설명] 손마다 어느 사람 소속인지 색과 라벨로 표시(소속 불명이면 주황색+"hand"만)

        for cn,coords in item_boxes:
            if tuple(coords) in picked_coords: continue
            # [설명] 이미 위에서 "PICK" 박스로 강조해 그린 상품은 여기서 또 그리지 않음(중복 표시 방지)
            x1,y1,x2,y2=map(int,coords)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,100,255),2)
            frame=put_text_kr(frame,kr(cn),(x1,max(y1-25,0)),18,(0,100,255))

        picking_kr  = [kr(i) for i in picking_items]
        status_text = (f"PICKING: {', '.join(picking_kr)}!"
                       if picking_kr else "Monitoring...")
        status_color= (0,0,255) if picking_kr else (200,200,200)
        frame = put_text_kr(frame,status_text,(10,50),20,status_color)
        cv2.putText(frame,f"frame:{frame_num}  persons:{len(tracked_persons)}",
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
        # [설명] 화면 좌상단에 현재 픽업 상태(빨강 강조)와 프레임/인원 디버그 정보를 항상 표시

        last_valid_frame = frame.copy()
        # [버그 수정] 이번 프레임의 모든 그리기 작업이 끝난 시점의 frame을 별도로 보관해둠.
        #        다음 루프에서 cap_a.read()가 영상 끝에 도달해 frame이 None이 되더라도,
        #        이 변수는 영향받지 않고 "마지막으로 정상 처리된 프레임" 그대로 남아있음
        a_frame.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                      width='stretch')
        log_box.text("\n".join(st.session_state.event_log[-15:]))
        # [설명] 매 프레임 A화면을 최신 frame으로 덮어쓰고, 사이드바 로그도 최근 15줄만 잘라
        #        보여줘서 로그가 무한정 길어져도 화면이 느려지지 않게 함

    # [요청 반영] 영상이 끝나는 순간(또는 "⏹ 중지" 버튼으로 멈추는 순간), 아직
    #        LOST_TIMEOUT_SECONDS가 다 안 찼어도 더 이상 기다릴 영상이 없으므로, 그때까지
    #        화면에 안 보이던 사람은 여기서 한 번 더 마무리 점검함. 이렇게 안 하면 "확인 필요"
    #        같은 중간 상태에서 그대로 멈춰버리고 끝까지 최종 결론이 안 나는 경우가 생김
    #        (테스트 영상이 사람이 사라진 직후 곧바로 끝나버리는 경우 특히 자주 발생)
    for pid in prev_tracked_pids - current_pids:
        judge_departure(pid, last_valid_frame)

finally:
    cap_a.release()
    st.session_state.running = False
    # [설명] 루프가 정상 종료되든, break로 빠지든, 예외가 나든 항상 실행되는 정리(cleanup) 구간.
    #        영상 파일 핸들을 반드시 풀어주고, 다음에 "시작" 버튼을 다시 누를 수 있도록 상태를 되돌림
print(f"\n{'='*40}")
print(f"USE_ALIAS = {USE_ALIAS}")
print(f"영상 전체에서 발급된 고유 Person ID 개수: {len(unique_canonical_ids)}")
print(f"{'='*40}\n")
st.success(f"✅ CCTV 영상 분석 완료 (총 {frame_num} 프레임)")