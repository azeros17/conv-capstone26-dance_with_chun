# 한밭대학교 인공지능소프트웨어학과 (춘과함께춤을)

#### 바이오매스 탄화공정 모니터링 시스템 (Biomass Carbonization Process Monitoring System)

**팀 구성**

- 20221065 이정빈
- 20221078 천세춘


## Teamate Project Background

- ### 필요성

  * 바이오차는 토양 개량제(탄소 격리)와 고형연료로 쓰여 탄소중립과 수익 창출에 동시에 기여한다. 그러나 바이오차의 품질과 경제적 가치는 원료 조성과 탄화 조건에 따라 크게 달라지며, 생산 전에 이를 예측하기 어렵다. 

  * 생산자가 조성·온도만으로 품질 등급과 가치를 미리 알 수 있다면, 용도(비료/연료)와 공정 조건을 합리적으로 결정할 수 있다.

- ### 기존 해결책의 문제점

  * 바이오차 품질 측정은 실험실 분석에 의존해 시간과 비용이 크고, 생산 후에야 결과를 알 수 있다.

  * 품질 인증(IBI·EBC)과 탄소크레딧 산정 절차가 복잡해 다수 생산자가 수익화를 포기한다.


## System Design

- ### System Requirements

  * **입력**: 바이오매스 조성 5종(우분·돈분·커피박·톱밥·슬러지), 탄화온도(200 ~ 800°C), 월(1~12)

  * **출력**: 물성 8종(수율·고정탄소·H/C비·회분·발열량·BET·탄화점·함수율), 배출가스 8종, 품질등급(IBI·EBC), 바이오SRF 적합성, 톤당 가격, 탄소격리량

  * **예측 모델**

    + 성질 예측 모델: 조성·온도로 물성 예측. 
    + 가스 예측 모델: 조성·월로 배출가스 예측. 
    + 품질 등급: IBI(International Biochar Initiative), EBC(European Biochar Certificate), 바이오SRF 기준으로 등급·적합성 판정

  * **두 데이터 환경 비교**

    + 원 데이터 모델(Ori): 제공된 실측·시트 데이터 기반
    + 생성 데이터 모델(Gen): 물리모델 기반 생성 데이터로 학습 범위 확장

  * **시스템 구성**: FastAPI 서버 + HTML/CSS/JavaScript 웹 대시보드. 학습된 모델을 .pkl로 저장해 서버가 로드, 실시간 예측 결과를 대시보드에 표시

- ### System Dependencies

  * Python 3.12.7
  * fastapi
  * uvicorn
  * scikit-learn
  * xgboost
  * pandas
  * openpyxl


## Installing and Running

- 패키지 설치

```
pip install -r requirements.txt
```

- 서버 실행 (학습된 모델 포함, 바로 실행 가능)

```
python server.py
```

- 브라우저 접속

  * 원 데이터 모델: http://localhost:8000/
  * 생성 데이터 모델: http://localhost:8000/gen

- (선택) 모델 재학습

```
python train_ori.py    # model_ori.pkl 생성
python train_gen.py    # model_gen.pkl 생성
```


## Code Structure

```
003 Code/
├── ori_model.py        원 데이터 모델 (OriModel)
├── gen_model.py        생성 데이터 모델 (GenModel)
├── server.py           FastAPI 서버
├── train_ori.py        원 모델 학습 → model_ori.pkl
├── train_gen.py        생성 모델 학습 → model_gen.pkl
├── index_ori.html      원 데이터 대시보드
├── index_gen.html      생성 데이터 대시보드
├── model_ori.pkl       학습된 원 모델
├── model_gen.pkl       학습된 생성 모델
└── requirements.txt
```


## Conclusion

- 조성·온도·월 입력만으로 바이오차의 물성·배출가스·품질등급·경제성을 실시간 예측하는 웹 시스템을 구현했다.

- 원 데이터 모델과 생성 데이터 모델을 비교하여, 데이터 환경에 무관하게 "변수의 형태와 물리 법칙에 근거한 모델 설계" 방법론을 일관되게 적용했다.

- 향후 실측 데이터를 추가 확보하여 검증 신뢰도를 높이고, 실시간 IoT 센서와 연동해 공정 자동 제어로 확장할 수 있다.
