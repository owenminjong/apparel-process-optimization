# DyeOptimAI 웹 결과 자동화 시스템

이 문서는 DyeOptimAI 시스템의 웹 결과 자동화 기능에 대한 설명입니다. 이 기능을 통해 AI 모델 실행 결과가 웹 인터페이스에 자동으로 반영됩니다.

## 개요

DyeOptimAI 웹 결과 자동화 시스템은 모델 실행 및 최적화 결과를 JSON 형식으로 저장하고, 웹페이지에서 이를 동적으로 로드하여 표시하는 기능을 제공합니다. 이를 통해 최신 분석 결과를 항상 웹 인터페이스에서 확인할 수 있습니다.

## 파일 구조

- `main.py`: 결과 JSON 파일을 생성하는 메인 스크립트
- `results/web_results.json`: 자동 생성되는 결과 데이터 파일
- `index.html`: 결과를 표시하는 웹 페이지
- `js/script.js`: 탭 전환 등 기본 웹 기능을 처리하는 스크립트

## 자동 업데이트 메커니즘

1. `main.py` 실행 시 `generate_web_results()` 함수가 호출됩니다.
2. 이 함수는 분석 결과를 수집하여 `results/web_results.json` 파일을 생성합니다.
3. 웹페이지 로드 시 내장된 JavaScript 코드가 이 JSON 파일을 불러옵니다.
4. 파싱된 데이터를 사용하여 웹페이지의 해당 요소들이 업데이트됩니다.

## 자동 업데이트되는 요소

### 1. 개요 탭
- 모델 성능 지표: Adjusted R², RMSE, MAE
- 목표 염색색차 DE 값
- 마지막 업데이트 시간/날짜

### 2. 유전 알고리즘 수렴 탭
- 목표 염색색차 DE 값

### 3. 예측 vs 실제 탭
- 모델 성능 지표 테이블(Adjusted R², RMSE, MAE)

### 4. 최적화 결과 탭
- 목표 염색색차 DE 값
- 예측 염색색차 DE 값
- 오차 값
- 최적 공정 변수 테이블(모든 변수의 최적값)

### 5. 일관성 테스트 탭
- 테스트 횟수
- 모델 일관성 테이블(각 지표의 평균과 변동 계수)
- 최적화 일관성 테이블(예측값과 오차의 평균 및 변동 계수)

## 자동 업데이트 되지 않는 요소

다음 요소들은 자동 업데이트되지 않으며, 필요시 수동으로 업데이트해야 합니다:

1. **분석 내용 및 인사이트**: 상관관계 분석, SHAP 중요도 분석 등의 텍스트 내용
2. **순위 정보**: 변수 중요도 순위 등 순서가 있는 정보
3. **그래프 이미지**: 모든 시각화 이미지는 결과 폴더에서 직접 로드됨
4. **해석 및 설명 텍스트**: 분석 결과에 대한 해석이나 설명

## JSON 파일 구조

`web_results.json` 파일은 다음과 같은 구조를 가집니다:

```json
{
  "model_performance": {
    "adjusted_r2": 0.9693,
    "rmse": 0.5337,
    "mae": 0.3248
  },
  "optimization_results": {
    "target_value": 1.5,
    "predicted_value": 1.5026,
    "error": 0.0026,
    "optimal_parameters": {
      "단위중량(kg)": 987.5369,
      "투입중량(kg)": 295.7952,
      "염색길이(m)": 534.2140,
      "투입중량/길이": 16.1770,
      "투입중량/액량": 0.0392,
      "공정진행시간(%)": 100.0000,
      "진행온도": 20.3241,
      "포속1": 1835.3465,
      "포속3": 502.1508,
      "포속4": 1095.2471
    }
  },
  "consistency_results": {
    "model_r2_mean": 0.9693,
    "model_r2_std": 0.0026,
    "model_r2_variability": 0.27,
    "model_rmse_mean": 0.5337,
    "model_rmse_std": 0.0167,
    "model_rmse_variability": 3.14,
    "optimization_predicted_mean": 1.6200,
    "optimization_predicted_std": 0.0000,
    "optimization_variability": 0.00,
    "n_tests": 3
  },
  "last_updated": "2025-04-19 14:49:01"
}