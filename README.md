# Dance-with-Chun

2026학년도 1학기 인공지능융합 캡스톤디자인

# 바이오매스 탄화공정 모니터링 시스템

바이오매스 조성과 탄화 조건을 입력하면 바이오차의 물성·배출가스·품질등급·경제성을 예측하는 웹 기반 시스템입니다.

---

## 개요

저활용 바이오매스(축분·커피박·톱밥·왕겨 등)를 탄화하여 바이오차를 생산하는 공정에서, 생산 전에 품질과 가치를 미리 예측합니다.

- **입력**: 바이오매스 조성(5종), 탄화온도, 월
- **출력**: 물성 8종, 배출가스 8종, 품질등급(IBI·EBC), 바이오SRF 적합성, 톤당 가격, 탄소격리량

두 가지 데이터 환경으로 학습한 모델을 비교합니다.

- **원 데이터 모델 (Ori)**: 제공된 실측·시트 데이터 기반
- **생성 데이터 모델 (Gen)**: 물리모델 기반 생성 데이터로 학습 범위 확장

---

## 실행 방법

### 1. 패키지 설치

```
pip install fastapi uvicorn scikit-learn xgboost pandas openpyxl
```

### 2. 서버 실행

학습된 모델 파일(`model_ori.pkl`, `model_gen.pkl`)이 포함되어 있으므로 바로 실행할 수 있습니다.

```
python server.py
```

브라우저에서 접속:

- 원 데이터 모델: `http://localhost:8000/`
- 생성 데이터 모델: `http://localhost:8000/gen`

### 3. (선택) 모델 재학습

데이터 CSV가 있을 때, 모델을 다시 학습해 `.pkl`을 생성할 수 있습니다.

```
python train_ori.py    # model_ori.pkl 생성
python train_gen.py    # model_gen.pkl 생성
```

---

## 파일 구성

| 파일 | 설명 |
|---|---|
| `ori_model.py` | 원 데이터 모델 (OriModel) |
| `gen_model.py` | 생성 데이터 모델 (GenModel) |
| `server.py` | FastAPI 서버 |
| `train_ori.py` | 원 모델 학습 → `model_ori.pkl` |
| `train_gen.py` | 생성 모델 학습 → `model_gen.pkl` |
| `index_ori.html` | 원 데이터 대시보드 |
| `index_gen.html` | 생성 데이터 대시보드 |
| `model_ori.pkl` | 학습된 원 모델 |
| `model_gen.pkl` | 학습된 생성 모델 |

---

## 품질 등급

- **IBI**: International Biochar Initiative — 국제 바이오차 협회 기준
- **EBC**: European Biochar Certificate — 유럽 바이오차 인증
- **바이오SRF**: 고형연료 적합성 판정

---

## 기술 스택

- Python, scikit-learn, XGBoost (모델)
- FastAPI (서버)
- HTML / CSS / JavaScript (대시보드)
