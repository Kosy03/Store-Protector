import streamlit as st
import pandas as pd
import time

# 1. 페이지 설정 (넓은 화면 모드)
st.set_page_config(page_title="Store-Protector UI", layout="wide")

# 상단 제목
st.title("Store-Protector: 무인 매장 실시간 보안 관제 시스템")
st.markdown("---")

# 2. 상단 구역 (A화면 & B화면) 레이아웃 설정
col1, col2 = st.columns(2)

with col1:
    st.subheader("A화면: 매장 전경 및 행동 분석")
    # 실제 영상 대신 임시로 사진 배치해둠
    st.image("test_images/store_view.png", use_column_width=True)
    st.info("💡 상태: id: 1 고객 추적 중 | 행동: 상품 픽업 감지")

with col2:
    st.subheader("B화면: 계산대 상품 인식")
    # 실제 영상 대신 임시로 사진 배치해둠
    st.image("test_images/snack2.jpg", use_column_width=True)

    # B화면 아래에 인식된 물품 리스트 (표 형태)
    st.markdown("#### [ 계산대 인식 결과 ]")
    check_data = {
        "물품명": ["꼬깔콘", "꿀꽈배기"],
        "수량": [1, 1],
        "상태": ["일치", "일치"]
    }
    st.table(pd.DataFrame(check_data))

st.markdown("---")

# 3. 하단 구역 (C화면: 로그 및 경고창)
st.subheader("C화면: 시스템 판독 결과 및 로그")

# 기획안의 '주의!' 메시지를 재현하기 위해 3개 열로 나눔
c1, c2, c3 = st.columns([1, 2, 1])

with c2:
    # 조건에 따라 경고창 띄우기 (가짜 데이터 기반)
    mismatch_detected = True  # 나중에 로직과 연결

    if mismatch_detected:
        st.error("### ⚠️ 주의: 계산 불일치 감지!")
        st.warning("""
        **[ 분석 결과 ]**
        - 매대에서 집은 물품: 꼬깔콘, 꿀꽈배기, 포카칩
        - 계산대에서 결제한 물품: 꼬깔콘, 꿀꽈배기
        - **누락 물품: 포카칩 (1개)**
        """)
        st.button("해당 시점 영상 다시보기")
    else:
        st.success("정상 결제 완료: 모든 물품이 일치합니다.")

# 사이드바 (설정창)
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    st.write("카메라 A 상태: 🟢 정상")
    st.write("카메라 B 상태: 🟢 정상")
    st.divider()
    st.write("최근 감지된 고객 ID: 1")