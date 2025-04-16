import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

def build_model(X, y):
    """
    랜덤포레스트 회귀 모델을 구축합니다.
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 하이퍼파라미터 설정
    parameters = {
        'max_depth': [2, 4, 6, 8],
        'n_estimators': [20, 50, 100, 150],
        'min_samples_leaf': [2, 4, 6, 8, 10]
    }

    # 모델 생성 및 학습
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf_model, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    # 최적 모델 추출
    best_model = grid_search.best_estimator_

    # 모델 평가
    y_pred = best_model.predict(X_test_scaled)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"최적 파라미터: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    return best_model, scaler

if __name__ == "__main__":
    # TODO: 전처리된 데이터 로드 및 모델 구축 코드 구현
    pass