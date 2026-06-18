# Store-Protector: 무인 매장 실시간 보안 관제 시스템

## 1. 프로젝트 소개

### 프로젝트 개요

Store-Protector는 AI 기반 객체 탐지 기술을 활용하여 무인 매장에서 발생할 수 있는 도난 및 구매 누락 상황을 실시간으로 감지하는 보안 관제 시스템입니다.

본 시스템은 두 개의 CCTV 영상을 동시에 분석하여 고객의 상품 집기 행동과 계산대 결제 내역을 비교하고, 정상 결제 여부를 판별합니다.

### 주요 기능

* YOLO 기반 사람, 손, 상품 객체 탐지
* ByteTrack 기반 고객 추적(Person Tracking)
* 상품 집기(Pick-up) 행동 인식
* 계산대 상품 인식
* 구매 상품과 결제 상품 비교
* 구매 누락 탐지
* 도난 의심 상황 탐지
* 실시간 Streamlit 대시보드 제공
* 이벤트 로그 기록
* 스냅샷 자동 저장

---

## 2. 실행 화면

### 메인 대시보드

> [실행 화면 캡처 삽입]
>
> * Streamlit 기반 실시간 관제 화면
> * A화면(매장 CCTV), B화면(계산대 CCTV), C화면(최종 판독 결과)

### 상품 집기 감지

> [실행 화면 캡처 삽입]
>
> * 고객이 상품을 집는 순간 감지
> * 상품명과 인식 시간을 표시

### 정상 결제 결과

> [실행 화면 캡처 삽입]
>
> * 매장에서 집은 상품과 계산대에서 결제한 상품이 일치하는 경우

### 구매 누락 탐지

> [실행 화면 캡처 삽입]
>
> * 계산되지 않은 상품이 존재하는 경우 경고 발생

### 도난 감지 결과

> [실행 화면 캡처 삽입]
>
> * 상품을 소지한 상태로 계산 없이 퇴장한 경우
> * 인물 스냅샷 및 이동 경로 표시

---

## 3. 시스템 구조

```text
A 카메라 (매장)
       │
       ▼
손/사람/상품 탐지
       │
       ▼
상품 집기 행동 분석
       │
       ▼
고객 추적 및 상품 기록
       │
       ▼
계산대 진입 여부 확인
       │
       ▼
B 카메라 (계산대)
       │
       ▼
상품 인식
       │
       ▼
구매 상품 ↔ 결제 상품 비교
       │
       ▼
정상 결제 / 구매 누락 / 도난 판정
```

---

## 4. 개발 환경 및 의존성

### 개발 환경

| 항목               | 내용              |
| ---------------- | --------------- |
| OS               | Windows 10 / 11 |
| Language         | Python 3.10+    |
| Framework        | Streamlit       |
| Deep Learning    | PyTorch         |
| Object Detection | YOLOv8          |
| Tracking         | ByteTrack       |
| IDE              | PyCharm         |

### 주요 라이브러리

```bash
streamlit
pandas
opencv-python
torch
ultralytics
numpy
Pillow
```

### requirements.txt

```txt
streamlit
pandas
opencv-python
torch
ultralytics
numpy
Pillow
```

---

## 5. 상세 설치 및 실행 방법

### 1) 프로젝트 클론

```bash
git clone https://github.com/Kosy03/Store-Protector.git
cd store-protector
```

### 2) 가상환경 생성

```bash
python -m venv venv
```

### 3) 가상환경 활성화

Windows

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
source venv/bin/activate
```

### 4) 패키지 설치

```bash
pip install -r requirements.txt
```

### 5) 모델 파일 준비

```text
models/
 ├─ hand_yolov8n.pt
 ├─ best.pt
```
'https://drive.google.com/drive/folders/1m2H2z4xVa-cv8yn14VTd8fYJNVhIQpvh?usp=sharing'

### 6) 영상 파일 준비

```text
videos/
 ├── 구매영상.mp4
 ├── 계산대.mp4
 ├── 구매누락.mp4
 ├── 계산대_누락.mp4
 └── 도난영상.mp4
```
'https://drive.google.com/drive/folders/1GCWH0JzN2JNwVEpAmFkeDhF2rk7qjRem?usp=sharing'

### 7) 실행

```bash
streamlit run app.py
```

### 8) 접속

브라우저에서 아래 주소 접속

```text
http://localhost:8501
```

---

## 6. 데이터 파이프라인

### Step 1. 영상 입력

* A 카메라(매장)
* B 카메라(계산대)

### Step 2. 객체 탐지

YOLO 모델을 활용하여

* 사람(Person)
* 손(Hand)
* 상품(Item)

을 탐지한다.

### Step 3. 고객 추적

ByteTrack을 활용하여

* 고객 ID 부여
* 고객 이동 경로 저장

을 수행한다.

### Step 4. 상품 집기 감지

손과 상품의 위치 관계를 분석하여

* 상품 접촉
* 일정 시간 이상 유지

시 집기로 판정한다.

### Step 5. 계산대 진입 감지

고객이 계산대 영역에 일정 시간 이상 머물면

* 결제 프로세스 시작

으로 판단한다.

### Step 6. 결제 상품 인식

B 카메라 영상을 분석하여

* 계산된 상품 목록 생성

한다.

### Step 7. 결과 비교

```text
집은 상품 == 결제 상품
```

→ 정상 결제

```text
집은 상품 > 결제 상품
```

→ 구매 누락

```text
집은 상품 존재
+
결제 상품 없음
```

→ 도난 의심

### Step 8. 결과 저장

* 이벤트 로그 생성
* 스냅샷 저장
* 최종 결과 출력

---

## 7. 프로젝트 폴더 구조

```text
Store-Protector
│
├── models/
│   ├── hand_yolov8n.pt
│   └── best.pt
│
├── videos/
│   ├── 구매영상.mp4
│   ├── 계산대.mp4
|   ├── 구매누락.mp4
│   ├── 계산대_누락.mp4
|   └── 도난영상.mp4
│
├── snapshots/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 8. 기대 효과

* 무인 매장 도난 방지
* 구매 누락 자동 탐지
* CCTV 모니터링 자동화
* 관리자 업무 부담 감소
* 실시간 보안 관제 지원

---

## 9. 팀원별 역할 분담

| 이름  | 담당 역할                                                                                                                                                                                                       |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 고서연 | 상품 데이터 수집, 상품 6종 촬영, 데이터 라벨링, YOLO 학습 데이터셋 구축, 객체 탐지 모델(best.pt) 학습 및 성능 검증                                                                                                                                 |
| 이하영 | Streamlit 기반 사용자 인터페이스(UI) 개발, 사람·손·상품 객체 탐지 로직 구현, ByteTrack 기반 사용자 추적 기능 구현, 상품 집기(Pick-up) 행동 분석 알고리즘 개발, 계산대 상품 인식 연동, 구매 상품-결제 상품 비교 로직 구현, 구매 누락 및 도난 탐지 기능 구현, 이벤트 로그 관리, 스냅샷 저장 기능 구현, 시스템 통합 및 테스트 |

### 역할 상세 설명

#### 고서연

* 상품 6종 데이터 촬영
* 객체 탐지용 데이터셋 구축
* 데이터 라벨링 수행
* YOLO 모델 학습
* 학습 모델(best.pt) 생성 및 성능 검증

#### 이하영

* 전체 시스템 설계 및 구현
* Streamlit 기반 실시간 관제 화면 개발
* YOLO를 활용한 사람, 손, 상품 탐지 기능 구현
* ByteTrack을 활용한 고객 추적 기능 구현
* 상품 집기 행동 분석 로직 구현
* 계산대 진입 감지 기능 구현
* 결제 상품 인식 및 비교 기능 구현
* 구매 누락 탐지 기능 구현
* 도난 감지 및 스냅샷 저장 기능 구현
* 이벤트 로그 및 최종 판독 결과 출력 기능 구현
* 시스템 통합 및 테스트


---

## 10. 향후 개선 사항

- 실시간 CCTV 연동
- 다중 고객 동시 분석 성능 향상
- 상품 종류 확대
- 모바일 알림 기능 추가
- 관리자 웹 대시보드 고도화
```
