# DyeOptimAI - 의류 염색 공정 최적화 AI 시스템

## 프로젝트 개요
- **시스템명**: DyeOptimAI
   - Dye: 염료, 염색하다
   - Optim: Optimization (최적화)
   - AI: Artificial Intelligence (인공지능)
- **목적**: 염색가공 공정에서 품질을 예측하고 공정을 최적화하는 AI 모델 개발
- **문제점**: 숙련공의 노하우에 의존하는 염색 공정과 비효율적인 CCM 검사 과정
- **접근법**: 데이터 기반 예측 모델과 최적화 알고리즘 적용
- **출처**: KAMP(Korea AI Manufacturing Platform) 제공 의류 공정최적화 AI 데이터셋

## 데이터셋
- **내용**: 염색 가동 길이, 온도, 속도, 색차 등 11개 변수
- **출처**: 염색설비 PLC, CCM 검사설비, 물량정보 PC key-in
- **형식**: xlsx, csv
- **규모**: 약 3,180만 개 데이터포인트 (157MB)

## 성능 및 안정성
- **모델 성능**:
   - 설명력(Adjusted R²): 96.93%
   - RMSE: 0.53
   - MAE: 0.32
- **모델 안정성**:
   - Adjusted R² 변동 계수: 0.27%
   - RMSE 변동 계수: 3.14%
- **최적화 성능**:
   - 목표 염색색차 DE 1.5에 대한 평균 오차: 0.0027
- **최적화 안정성**:
   - 최적화 결과 변동 계수: 0.00%
   - 주요 공정 변수의 일관된 최적값 도출

## 사용 알고리즘
1. **랜덤포레스트 회귀**: 염색 품질 예측 모델
2. **유전 알고리즘**: 최적 공정 조건 도출

## 요구사항
- Python 3.8+
- 필수 패키지: pandas, numpy, matplotlib, scikit-learn, seaborn, shap, geneticalgorithm, gdown, openpyxl

## 설치 및 사용 방법
1. **저장소 클론**:
   ```bash
   git clone https://github.com/owenminjong/apparel-process-optimization.git
   cd DyeOptimAI
   ```

2. **가상환경 생성 및 활성화**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **필요 패키지 설치**:
   ```bash
   pip install -r requirements.txt
   ```

4. **전체 파이프라인 실행**:
   ```bash
   python main.py
   ```

5. **개별 단계 실행**:
   - 데이터 다운로드: `python main.py --mode download`
   - 데이터 전처리: `python main.py --mode preprocess`
   - 모델 구축: `python main.py --mode model`
   - 공정 최적화: `python main.py --mode optimize --target 1.5`

6. **일관성 테스트 실행** (선택사항):
   ```bash
   python run_all_tests.py
   ```

7. **옵션**:
   - 목표 염색색차 DE 값 설정: `--target 1.5`
   - 강제 데이터 다운로드: `--force-download`
   - 특정 단계 건너뛰기: `--no-download`, `--no-preprocess`, `--no-model`, `--no-optimize`

## 프로젝트 구조
```
DyeOptimAI/
├── data/                # 데이터 파일
│   ├── LOT 물량.xlsx
│   ├── PRODUCTION_TREND.csv
│   ├── CCM 측정값.xlsx
│   └── preprocessed_data.csv
├── models/              # 저장된 모델 및 스케일러
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── variable_bounds.json
├── notebooks/           # 주피터 노트북 파일 (분석 탐색용)
├── results/             # 결과 및 시각화
│   ├── prediction_vs_actual.png
│   ├── shap_feature_importance.png
│   ├── optimization_convergence.png
│   └── summary_report.md
├── test_results/        # 일관성 테스트 결과
│   ├── model_consistency.png
│   ├── optimization_consistency.png
│   ├── parameter_consistency.png
│   └── consistency_report.md
├── src/                 # 소스 코드
│   ├── preprocessing/   # 데이터 전처리 코드
│   ├── modeling/        # 모델링 코드
│   └── optimization/    # 최적화 코드
├── main.py              # 메인 실행 스크립트
├── run_all_tests.py     # 일관성 테스트 스크립트
├── requirements.txt     # 필요 패키지 목록
└── README.md            # 프로젝트 설명서
```

## 주요 기능
1. **데이터 전처리**:
   - 데이터 로드 및 정제
   - 중복/이상치 제거
   - 파생변수 생성
   - 데이터셋 병합

2. **품질 예측 모델링**:
   - 랜덤포레스트 회귀 모델
   - 하이퍼파라미터 최적화
   - 모델 성능 평가 (Adjusted R², RMSE, MAE)
   - SHAP 기반 변수 중요도 분석

3. **공정 최적화**:
   - 유전 알고리즘 기반 최적화
   - 목표 품질값 달성을 위한 공정 변수 도출
   - 제약 조건 적용 가능
   - 수렴 과정 시각화

4. **결과 분석**:
   - 예측 vs 실제 시각화
   - 변수 중요도 시각화
   - 최적화 수렴 과정 시각화
   - 요약 보고서 생성

5. **일관성 테스트**:
   - 모델 안정성 평가
   - 최적화 결과 일관성 평가
   - 시스템 신뢰성 보고서 생성

## 컴포넌트 설계
프로젝트는 유지보수성 높은 컴포넌트 기반 아키텍처로 설계되었습니다:

1. **데이터 전처리 컴포넌트**:
   - `DataLoader`: 데이터 로드 담당
   - `DataCleaner`: 데이터 정제 및 이상치 제거
   - `FeatureEngineering`: 파생변수 생성
   - `DataMerger`: 데이터셋 병합

2. **모델링 컴포넌트**:
   - `DataSplitter`: 데이터 분할
   - `DataScaler`: 데이터 스케일링
   - `ModelBuilder`: 모델 구축 및 학습
   - `ModelEvaluator`: 모델 성능 평가
   - `FeatureAnalyzer`: 변수 중요도 분석

3. **최적화 컴포넌트**:
   - `VariableBoundsManager`: 변수 범위 관리
   - `InputConverterFactory`: 입력 변환 함수 생성
   - `ObjectiveFunctionFactory`: 목적 함수 생성
   - `GeneticOptimizer`: 유전 알고리즘 최적화

## 주요 발견사항
1. **변수 중요도**:
   - '투입중량/길이', '단위중량(kg)', '진행온도'가 염색 품질에 가장 큰 영향을 미침
   - 포속 관련 변수들은 상대적으로 낮은 영향력을 보임

2. **최적 공정 조건**:
   - 목표 염색색차 DE 1.5에 대한 최적 공정 변수 조합 도출
   - 단위중량(kg): 987.54
   - 투입중량(kg): 295.80
   - 염색길이(m): 534.21
   - 투입중량/길이: 16.18
   - 진행온도: 20.32

3. **시스템 신뢰성**:
   - 다양한 랜덤 시드에서도 96.9% 이상의 설명력 유지
   - 최적화 결과의 높은 일관성 (변동 계수 0.00%)
   - 예측 오차 0.0027로 매우 정확한 최적화 성능

## 향후 개선 사항
- 앙상블 접근법 구현으로 모델 안정성 추가 향상
- 실시간 모니터링 및 예측을 위한 웹 인터페이스 개발
- 배치 단위 처리 기능 강화
- 다양한 제품/컬러에 대한 개별 모델 구축
- Docker 컨테이너화 및 CI/CD 파이프라인 구축

## 참고 문헌
- KAMP(Korea AI Manufacturing Platform) 의류 공정최적화 AI 데이터셋 가이드북
- 유전 알고리즘 관련 문헌
- 랜덤포레스트 및 머신러닝 관련 문헌

## 라이센스
이 프로젝트에 사용된 데이터셋은 KAMP의 이용약관에 따라 사용됩니다.
출처: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 의류 공정최적화 AI 데이터셋, 스마트제조혁신추진단(㈜임픽스), 2023.08.18.., www.kamp-ai.kr