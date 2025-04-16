# DyeOptimAI - 의류 염색 공정 최적화 AI 시스템

## 프로젝트 개요
- 목적: 염색가공 공정에서 품질을 예측하고 공정을 최적화하는 AI 모델 개발
- 문제점: 숙련공의 노하우에 의존하는 염색 공정과 비효율적인 CCM 검사 과정
- 접근법: 데이터 기반 예측 모델과 최적화 알고리즘 적용

## 데이터셋
- 내용: 염색 가동 길이, 온도, 속도, 색차 등 11개 변수
- 출처: 염색설비 PLC, CCM 검사설비, 물량정보 PC key-in
- 형식: xlsx, csv
- 규모: 약 3,180만 개 데이터포인트 (157MB)

## 사용 알고리즘
1. 랜덤포레스트 회귀: 염색 품질 예측 모델
2. 유전 알고리즘: 최적 공정 조건 도출

## 설치 및 사용 방법
1. 저장소 클론:
   git clone https://github.com/owenminjong/apparel-process-optimization.git
   cd DyeOptimAI
2. 필요 패키지 설치: pip install -r requirements.txt
3. 데이터 다운로드: python -m src.preprocessing.download_data
4. 모델 학습 및 최적화: python main.py

## 폴더 구조
- data/: 데이터 파일 (자동 다운로드)
- notebooks/: 주피터 노트북 파일
- src/: 소스 코드
- preprocessing/: 데이터 전처리
- modeling/: 예측 모델
- optimization/: 최적화 알고리즘
- results/: 결과 및 시각화
   