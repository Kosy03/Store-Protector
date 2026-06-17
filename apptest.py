import streamlit as st
import pandas as pd
import cv2, torch, time, os
from collections import defaultdict, deque, Counter
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# ═══════════════════════════════════════════════════════
#  경로 설정
# ═══════════════════════════════════════════════════════
VIDEO_A_PATH = "videos/구매영상.mp4"
VIDEO_B_PATH = "videos/계산대.mp4"
#VIDEO_A_PATH = "videos/구매누락.mp4"
#VIDEO_B_PATH = "videos/계산대_누락.mp4"
#VIDEO_A_PATH = "videos/도난영상.mp4"
MODEL_HAND   = "models/hand_yolov8n.pt"
MODEL_ITEM   = "models/best.pt"

# ═══════════════════════════════════════════════════════
#  파라미터
# ═══════════════════════════════════════════════════════
GHOST_TTL         = 0
NEAR_MARGIN       = 10
EXCLUDE_ZONE      = (567, 98, 668, 359)
PICK_SECONDS      = 0.5
PERSON_CONF       = 0.3
ALIAS_DIST        = 80
CHECKOUT_ZONE     = (12, 55, 153, 182)
CHECKOUT_MIN_STAY = 6.0
FINAL_MIN_SECONDS = 6.0
B_PAYMENT_DELAY   = 8.0
MAX_LOST_FRAMES   = 30
FRAME_SKIP        = 2
RESIZE_W          = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    'yangparing':      '양파링',
    'ojingeoddankong': '오징어땅콩',
    'cocacola':        '코카콜라',
    'garamandeunBae':  '갈아만든배',
    'kkokkalcorn':     '꼬칼콘',
    'pocarisweat':     '포카리스웨트',
}
def kr(name): return LABEL_MAP.get(name, name)

ID_COLORS = [
    (255,80,80),(80,255,80),(80,80,255),(255,255,80),
    (255,80,255),(80,255,255),(200,130,50),(130,50,200)
]
def get_color(pid): return ID_COLORS[int(pid) % len(ID_COLORS)]

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

def put_text_kr(frame, text, pos, font_size=20, color=(255,255,255)):
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

TRAIL_COLOR = (0, 255, 65)   # 형광 초록 (BGR)

def draw_trail_transparent(frame, trail, alpha=0.6):
    if len(trail) < 2:
        return frame
    overlay = frame.copy()
    for k in range(1, len(trail)):
        thickness = max(2, int(4 * k / len(trail)))
        cv2.line(overlay, trail[k-1], trail[k], TRAIL_COLOR, thickness)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

@st.cache_resource
def load_models():
    return YOLO(MODEL_HAND), YOLO(MODEL_ITEM), YOLO("yolov8n.pt")

def is_near_or_overlap(hb, ib, m=NEAR_MARGIN):
    return hb[0]-m<ib[2] and hb[2]+m>ib[0] and hb[1]-m<ib[3] and hb[3]+m>ib[1]

def is_in_exclude_zone(c):
    cx,cy=(c[0]+c[2])/2,(c[1]+c[3])/2
    ex1,ey1,ex2,ey2=EXCLUDE_ZONE
    return ex1<cx<ex2 and ey1<cy<ey2

def box_center(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)

def hand_in_person(hb, pb, m=20):
    hcx,hcy=box_center(hb)
    return pb[0]-m<hcx<pb[2]+m and pb[1]-m<hcy<pb[3]+m

def person_in_zone(pbox, zone):
    cx,cy=box_center(pbox); x1,y1,x2,y2=zone
    return x1<cx<x2 and y1<cy<y2

def add_log(msg, level="info"):
    icon={"info":"ℹ️","warn":"⚠️","danger":"🚨","ok":"✅"}.get(level,"ℹ️")
    st.session_state.event_log.append(f"{time.strftime('%H:%M:%S')} {icon} {msg}")

def iou(b1, b2):
    x1,y1=max(b1[0],b2[0]),max(b1[1],b2[1])
    x2,y2=min(b1[2],b2[2]),min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def dedup(boxes, thresh=0.3):
    kept=[]; boxes=sorted(boxes,key=lambda x:-x[5])
    for c in boxes:
        if not any(c[4]==k[4] and iou(c[:4],k[:4])>thresh for k in kept):
            kept.append(c)
    return kept

# ═══════════════════════════════════════════════════════
#  B화면 처리
# ═══════════════════════════════════════════════════════
def run_screen_b(item_model, pickup_items, b_frame_ph, b_status_msg,
                 b_table, b_delay, conf_thresh):
    cap_b = cv2.VideoCapture(VIDEO_B_PATH)
    if not cap_b.isOpened():
        return set(), "B영상 열기 실패"

    ITEM_CLASSES  = item_model.names
    first_detect  = {}
    last_boxes    = []
    all_detected  = {}
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
        now = time.time()

        if frame_idx % 5 == 0:
            small = cv2.resize(frame, (480, int(h * 480 / w)))
            sx = display.shape[1] / small.shape[1]
            sy = display.shape[0] / small.shape[0]
            raw = []
            for box in item_model(small, verbose=False,
                                  conf=conf_thresh, device=DEVICE)[0].boxes:
                cn   = ITEM_CLASSES.get(int(box.cls[0]), f"cls{int(box.cls[0])}")
                conf = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                raw.append((int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy),
                            cn, conf, (0,200,0)))
                if cn not in first_detect:
                    first_detect[cn] = now
                    add_log(f"B화면: {kr(cn)} 감지됨", "info")
                if cn not in all_detected or conf > all_detected[cn]:
                    all_detected[cn] = conf
            last_boxes = dedup(raw)

        for (x1,y1,x2,y2,cn,conf,color) in last_boxes:
            cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
            display = put_text_kr(display, f"{kr(cn)} {conf:.2f}",
                                  (x1,max(y1-28,0)), 18, color)

        b_frame_ph.image(cv2.cvtColor(display,cv2.COLOR_BGR2RGB),
                         use_container_width=True)

        if all_detected:
            df = pd.DataFrame({
                "물품명": [kr(cn) for cn in all_detected],
                "신뢰도": [f"{c:.2f}" for c in all_detected.values()],
                "상태":   ["인식 완료 ✅"] * len(all_detected)
            })
            df.index = range(1, len(df)+1)
            b_table.table(df)

        if first_detect and not payment_items:
            elapsed = now - min(first_detect.values())
            b_status_msg.warning(f"⏳ 계산 중... {elapsed:.1f}s / {b_delay}s")
            if elapsed >= b_delay:
                payment_items = set(all_detected.keys())
                payment_kr    = [kr(i) for i in payment_items]
                missing_kr    = set(pickup_kr) - set(payment_kr)
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
                time.sleep(3)
                break

    cap_b.release()
    if not payment_items:
        msg = "🚨 B화면 물건 미인식 — 도난 의심"
        add_log(msg, "danger")
        return set(), msg
    return payment_items, result_msg

# ═══════════════════════════════════════════════════════
#  페이지 설정
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Store-Protector UI", layout="wide")
st.title("Store-Protector: 무인 매장 실시간 보안 관제 시스템")
st.markdown("---")

if "event_log"  not in st.session_state: st.session_state.event_log = []
if "running"    not in st.session_state: st.session_state.running = False

# ═══════════════════════════════════════════════════════
#  사이드바 — 설정 + 실행 버튼
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    st.write("카메라 A 상태: 🟢 정상")
    st.write("카메라 B 상태: 🟢 정상")
    st.divider()

    st.markdown("**▶ 실행 전 설정하세요**")
    item_conf     = st.slider("물건 감지 Confidence", 0.0, 1.0, 0.4, 0.05,
                              help="낮을수록 더 많이 잡히나 오감지 증가")
    checkout_stay = st.slider("계산대 머문 시간 기준(초)", 1.0, 10.0,
                              CHECKOUT_MIN_STAY, 0.5)
    final_sec     = st.slider("집기 인정 최소 시간(초)", 1.0, 15.0,
                              FINAL_MIN_SECONDS, 0.5)
    b_delay       = st.slider("B화면 계산완료 대기(초)", 1.0, 15.0,
                              B_PAYMENT_DELAY, 0.5)
    st.divider()

    # ── 실행 / 중지 버튼 ──
    col_s, col_e = st.columns(2)
    start_btn = col_s.button("▶ 시작", type="primary",
                             disabled=st.session_state.running)
    stop_btn  = col_e.button("⏹ 중지",
                             disabled=not st.session_state.running)

    if start_btn:
        st.session_state.running = True
        st.rerun()
    if stop_btn:
        st.session_state.running = False
        st.rerun()

    st.divider()
    st.markdown(f"**현재 Confidence:** `{item_conf}`")
    st.header("📋 이벤트 로그")
    log_box = st.empty()

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

st.markdown("---")
st.subheader("C화면: 시스템 판독 결과 및 로그")
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    c_result = st.empty()

# 대기 상태 표시
if not st.session_state.running:
    with c_result.container():
        st.info("### ⏸ 대기 중\n- 사이드바에서 설정 후 **▶ 시작** 버튼을 눌러주세요.")
    b_status_msg.info("⏳ 시작 버튼을 눌러주세요...")
    st.stop()

# ═══════════════════════════════════════════════════════
#  실행 중 상태
# ═══════════════════════════════════════════════════════
with c_result.container():
    st.info("### 🔄 시스템 판독 대기 중...\n- 고객 행동을 분석하고 있습니다.")
b_status_msg.info("⏳ 고객의 키오스크 접근을 대기 중입니다...")

hand_model, item_model, person_model = load_models()
ITEM_CLASSES = item_model.names

permanent_trails  = defaultdict(list)
person_snapshots  = {}
last_seen_items   = {}
confirmed_picks   = set()
accumulated_time  = {}
near_start_time   = {}
person_picks      = defaultdict(set)
person_pick_time  = defaultdict(dict)
prev_person_boxes = {}
id_alias          = {}
checkout_enter    = {}
checkout_done     = set()
lost_counter      = defaultdict(int)
prev_tracked_pids = set()
b_triggered       = False

cap_a = cv2.VideoCapture(VIDEO_A_PATH)
frame_num = 0

try:
    while cap_a.isOpened() and st.session_state.running:
        ret, frame = cap_a.read()
        if not ret: break
        frame_num += 1
        if frame_num % FRAME_SKIP != 0: continue

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (RESIZE_W, int(h * RESIZE_W / w)))

        hand_res   = hand_model(frame, verbose=False, imgsz=640,
                                conf=0.3, device=DEVICE)[0]
        item_res   = item_model(frame, verbose=False, imgsz=640,
                                conf=item_conf, device=DEVICE)[0]
        person_res = person_model.track(frame, persist=True, verbose=False,
                                        imgsz=640, conf=PERSON_CONF,
                                        classes=[0], tracker="bytetrack.yaml",
                                        device=DEVICE)[0]

        hand_boxes = [b.xyxy[0].tolist() for b in hand_res.boxes]
        item_boxes = []
        for b in item_res.boxes:
            cn = ITEM_CLASSES.get(int(b.cls[0]), f"cls{int(b.cls[0])}")
            c  = b.xyxy[0].tolist()
            if not is_in_exclude_zone(c): item_boxes.append((cn,c))

        det_names = {cn for cn,_ in item_boxes}
        for cn in list(last_seen_items):
            if cn not in det_names:
                last_seen_items[cn][1] -= 1
                if last_seen_items[cn][1] <= 0: del last_seen_items[cn]
                else: item_boxes.append((cn, last_seen_items[cn][0]))
            else:
                last_seen_items[cn] = [next(c for n,c in item_boxes if n==cn),
                                       GHOST_TTL]

        tracked_persons=[]; new_pb={}; current_pids=set()
        if person_res.boxes.id is not None:
            for box,tid in zip(person_res.boxes, person_res.boxes.id):
                tid=int(tid); pbox=box.xyxy[0].tolist()
                cx,cy=int(box_center(pbox)[0]),int(box_center(pbox)[1])
                canonical=id_alias.get(tid,tid)
                if tid not in id_alias and tid not in prev_person_boxes:
                    for pp,pb in prev_person_boxes.items():
                        pcx,pcy=int(box_center(pb)[0]),int(box_center(pb)[1])
                        if abs(cx-pcx)<ALIAS_DIST and abs(cy-pcy)<ALIAS_DIST:
                            root=id_alias.get(pp,pp)
                            id_alias[tid]=root; canonical=root
                            for k in list(accumulated_time):
                                if k[0]==pp and (root,k[1]) not in accumulated_time:
                                    accumulated_time[(root,k[1])]=accumulated_time[k]
                            break
                new_pb[canonical]=pbox
                tracked_persons.append((canonical,pbox))
                permanent_trails[canonical].append((cx,cy))
                current_pids.add(canonical)
                lost_counter[canonical]=0

                # 처음 감지 시 컬러 스냅샷 저장
                if canonical not in person_snapshots:
                    x1s,y1s,x2s,y2s = map(int, pbox)
                    x1c=max(0,x1s-20); y1c=max(0,y1s-20)
                    x2c=min(frame.shape[1],x2s+20)
                    y2c=min(frame.shape[0],y2s+20)
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size > 0:
                        person_snapshots[canonical] = cv2.cvtColor(
                            crop, cv2.COLOR_BGR2RGB)

        for pid in prev_tracked_pids - current_pids:
            lost_counter[pid] += 1
            if lost_counter[pid] >= MAX_LOST_FRAMES:
                picks = person_picks.get(pid, set())
                valid = {i for i in picks
                         if person_pick_time[pid].get(i,0) >= final_sec}
                if valid and pid not in checkout_done:
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
                    a_frame.image(theft_rgb, use_container_width=True)

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
                            use_container_width=True
                        )

        prev_tracked_pids = current_pids | (prev_tracked_pids - current_pids)
        prev_person_boxes = new_pb

        hand_to_pid = {}
        for hi,hb in enumerate(hand_boxes):
            for tid,pb in tracked_persons:
                if hand_in_person(hb,pb): hand_to_pid[hi]=tid; break

        now=time.time(); picking_items=set(); picked_coords=set()
        for hi,hbox in enumerate(hand_boxes):
            near_item = next(
                ((cn,c) for cn,c in item_boxes if is_near_or_overlap(hbox,c)),
                None)
            if near_item:
                cn,coords=near_item; pid=hand_to_pid.get(hi); key=(pid,cn)
                prev=near_start_time.get(hi)
                if prev is None or prev[0]!=pid or prev[1]!=cn:
                    if prev:
                        ok=(prev[0],prev[1])
                        accumulated_time[ok]=accumulated_time.get(ok,0)+(now-prev[2])
                    near_start_time[hi]=(pid,cn,now)
                session=now-near_start_time[hi][2]
                total=accumulated_time.get(key,0)+session
                x1,y1,x2,y2=map(int,coords)
                if total >= PICK_SECONDS:
                    if key not in confirmed_picks:
                        confirmed_picks.add(key)
                        add_log(f"PICK: {'P#'+str(pid) if pid else '?'}"
                                f" -> {kr(cn)} ({total:.1f}s)", "info")
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
                else:
                    bw=int((x2-x1)*min(total/PICK_SECONDS,1.0))
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                    cv2.rectangle(frame,(x1,y2+4),(x1+bw,y2+10),(0,255,255),-1)
                    frame = put_text_kr(frame,
                                        f"holding: {kr(cn)} {total:.1f}s",
                                        (x1,max(y1-28,0)),18,(0,255,255))
            else:
                if hi in near_start_time:
                    p,cn,st2=near_start_time.pop(hi)
                    k=(p,cn)
                    accumulated_time[k]=accumulated_time.get(k,0)+(now-st2)

        cx1,cy1,cx2,cy2=CHECKOUT_ZONE
        cv2.rectangle(frame,(cx1,cy1),(cx2,cy2),(0,255,128),2)
        cv2.putText(frame,"CHECKOUT",(cx1,cy1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,128),1)

        for tid,pbox in tracked_persons:
            in_zone=person_in_zone(pbox,CHECKOUT_ZONE)
            if in_zone:
                if tid not in checkout_enter: checkout_enter[tid]=now
                stay=now-checkout_enter[tid]
                bw=int((cx2-cx1)*min(stay/checkout_stay,1.0))
                cv2.rectangle(frame,(cx1,cy2+4),(cx1+bw,cy2+10),(0,255,128),-1)
                cv2.putText(frame,f"P#{tid} {stay:.1f}s",(cx1,cy2+22),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,128),1)
                if (stay>=checkout_stay and tid not in checkout_done
                        and not b_triggered):
                    checkout_done.add(tid); b_triggered=True
                    picks=person_picks.get(tid,set())
                    valid={i for i in picks
                           if person_pick_time[tid].get(i,0)>=final_sec}
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
                        b_table, b_delay, item_conf
                    )
                    b_kr=[kr(i) for i in b_items]
                    missing_kr=set(pick_kr)-set(b_kr)
                    detected_str=", ".join(b_kr) if b_kr else "탐지된 물품 없음"
                    if not b_items:
                        with c_result.container():
                            st.error("### 🛑 도난 의심\n"
                                     "- B화면에서 물건이 인식되지 않았습니다.")
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
                                use_container_width=True
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
                    b_triggered=False
            else:
                checkout_enter.pop(tid,None)

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

        for hi,hbox in enumerate(hand_boxes):
            x1,y1,x2,y2=map(int,hbox); pid=hand_to_pid.get(hi)
            color=get_color(pid) if pid else (255,200,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"hand(P#{pid})" if pid else "hand",
                        (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)

        for cn,coords in item_boxes:
            if tuple(coords) in picked_coords: continue
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

        a_frame.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),
                      use_container_width=True)
        log_box.text("\n".join(st.session_state.event_log[-15:]))

finally:
    cap_a.release()
    st.session_state.running = False

st.success(f"✅ CCTV 영상 분석 완료 (총 {frame_num} 프레임)")