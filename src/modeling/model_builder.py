# src/modeling/model_builder.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
import os
import shap
import joblib

# 결과 디렉토리 생성
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

class DataSplitter:
    """데이터를 학습 및 테스트 세트로 분리하는 컴포넌트"""

    def __init__(self, test_size=0.2, random_state=123):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, X, y):
        """데이터를 학습 및 테스트 세트로 분리합니다."""
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)


class DataScaler:
    """데이터 스케일링을 위한 컴포넌트"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train, X_test):
        """학습 데이터에 맞춰 스케일러를 학습하고, 학습 및 테스트 데이터를 변환합니다."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # DataFrame으로 변환하여 컬럼명 유지
        X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

        return X_train_scaled_df, X_test_scaled_df

    def transform(self, X):
        """새로운 데이터를 변환합니다."""
        return self.scaler.transform(X)

    def save(self, path='models/scaler.pkl'):
        """스케일러를 저장합니다."""
        joblib.dump(self.scaler, path)
        print(f"스케일러가 {path}에 저장되었습니다.")

    @classmethod
    def load(cls, path='models/scaler.pkl'):
        """저장된 스케일러를 로드합니다."""
        scaler_instance = cls()
        scaler_instance.scaler = joblib.load(path)
        return scaler_instance


class ModelBuilder:
    """랜덤포레스트 회귀 모델을 구축하는 컴포넌트"""

    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def build_model(self, X_train, y_train):
        """랜덤포레스트 회귀 모델을 구축하고 최적화합니다."""
        # 하이퍼파라미터 정의
        parameters = {
            'max_depth': [2, 4, 6, 8],
            'n_estimators': [20, 30, 50, 100, 150],
            'min_samples_leaf': [2, 4, 6, 8, 10]
        }

        # K-fold 교차검증 설정
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # 모델 생성
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state),
            parameters,
            cv=kfold,
            verbose=1,
            n_jobs=-1
        )

        # 모델 학습
        grid_search.fit(X_train, y_train.values.ravel())

        # 최적 파라미터 저장
        self.best_params = grid_search.best_params_
        print(f"랜덤포레스트 회귀모델 best 파라미터: {self.best_params}")

        # 최적 모델 추출
        self.model = grid_search.best_estimator_
        # 최종 학습
        self.model.fit(X_train, y_train.values.ravel())

        return self.model

    def save_model(self, path='models/rf_model.pkl'):
        """모델을 저장합니다."""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        joblib.dump(self.model, path)
        print(f"모델이 {path}에 저장되었습니다.")

    @classmethod
    def load_model(cls, path='models/rf_model.pkl'):
        """저장된 모델을 로드합니다."""
        model_instance = cls()
        model_instance.model = joblib.load(path)
        return model_instance


class ModelEvaluator:
    """모델 성능을 평가하는 컴포넌트"""

    def __init__(self):
        self.metrics = None

    def evaluate(self, model, X_test, y_test):
        """모델 성능을 평가합니다."""
        # 모델 예측
        y_pred = model.predict(X_test)

        # 성능 평가 지표 계산
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # 메트릭 저장
        self.metrics = {
            'adjusted_r2': adj_r2,
            'rmse': rmse,
            'mae': mae
        }

        # 결과 출력
        print(f"Adjusted R² score: {adj_r2:.3f}")
        print(f"RMSE score: {rmse:.3f}")
        print(f"MAE score: {mae:.3f}")

        return y_pred, self.metrics

    def plot_predictions(self, y_pred, y_test, path='results/prediction_vs_actual.png'):
        """실제값과 예측값을 비교하는 플롯을 생성합니다."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, y_test)
        plt.title('예측값 vs 실제값 (랜덤포레스트 회귀)')
        plt.xlabel('예측값')
        plt.ylabel('실제값')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"예측 vs 실제 플롯이 {path}에 저장되었습니다.")


class FeatureAnalyzer:
    """변수 중요도를 분석하는 컴포넌트"""

    def __init__(self):
        self.shap_values = None

    def analyze_importance(self, model, X_train, path='results/shap_feature_importance.png'):
        """SHAP 값을 사용하여 변수 중요도를 분석합니다."""
        # SHAP Explainer 생성
        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer.shap_values(X_train)

        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X_train, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"SHAP 변수 중요도 플롯이 {path}에 저장되었습니다.")

        return self.shap_values


class ModelPipeline:
    """전체 모델링 파이프라인을 관리하는 컴포넌트"""

    def __init__(self, test_size=0.2, random_state=123, cv=5):
        self.data_splitter = DataSplitter(test_size, random_state)
        self.data_scaler = DataScaler()
        self.model_builder = ModelBuilder(cv, random_state)
        self.model_evaluator = ModelEvaluator()
        self.feature_analyzer = FeatureAnalyzer()
        self.model = None
        self.X_cols = None

    def run(self, data_path='data/preprocessed_data.csv', target_col='염색색차 DE'):
        """전체 모델링 파이프라인을 실행합니다."""
        # 데이터 로드
        df = pd.read_csv(data_path)
        print(f"데이터 로드 완료: {df.shape[0]}행, {df.shape[1]}열")

        # 독립변수와 종속변수 정의
        # 가이드북 기준 독립변수 선택
        self.X_cols = ['단위중량(kg)', '투입중량(kg)', '염색길이(m)', '투입중량/길이', '투입중량/액량',
                       '공정진행시간(%)', '진행온도', '포속1', '포속3', '포속4']

        X = df[self.X_cols]
        y = df[[target_col]]

        # 1. 데이터 분리
        print("\n1. 데이터 분리 중...")
        X_train, X_test, y_train, y_test = self.data_splitter.split_data(X, y)

        # 2. 데이터 스케일링
        print("\n2. 데이터 스케일링 중...")
        X_train_scaled, X_test_scaled = self.data_scaler.fit_transform(X_train, X_test)

        # 3. 모델 구축
        print("\n3. 모델 구축 중...")
        self.model = self.model_builder.build_model(X_train_scaled, y_train)

        # 4. 모델 평가
        print("\n4. 모델 평가 중...")
        y_pred, metrics = self.model_evaluator.evaluate(self.model, X_test_scaled, y_test)
        self.model_evaluator.plot_predictions(y_pred, y_test)

        # 5. 변수 중요도 분석
        print("\n5. 변수 중요도 분석 중...")
        shap_values = self.feature_analyzer.analyze_importance(self.model, X_train_scaled)

        # 6. 모델 및 스케일러 저장
        print("\n6. 모델 및 스케일러 저장 중...")
        self.model_builder.save_model()
        self.data_scaler.save()

        print("\n모델링 완료!")

        # 주요 결과 반환
        return {
            'model': self.model,
            'scaler': self.data_scaler.scaler,
            'X_cols': self.X_cols,
            'metrics': metrics,
            'shap_values': shap_values
        }


def run_modeling_pipeline(data_path='data/preprocessed_data.csv'):
    """모델링 파이프라인 실행을 위한 편의 함수"""
    pipeline = ModelPipeline()
    return pipeline.run(data_path)


if __name__ == "__main__":
    results = run_modeling_pipeline()