# src/preprocessing/data_preprocessor.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

from data_loader import load_all_data
from data_cleaner import (clean_workload_data, clean_operation_data, clean_ccm_data,
                          remove_outliers, remove_unnecessary_columns)
from feature_engineering import create_workload_features, create_operation_features
from data_merger import merge_datasets

def preprocess_data():
    """전체 데이터 전처리 프로세스를 실행합니다."""
    print("데이터 로드 중...")
    workload, operation, ccm = load_all_data()

    if workload is None or operation is None or ccm is None:
        print("데이터 로드 실패. 전처리를 중단합니다.")
        return None

    print("\n1. 데이터 정제 시작...")
    workload_cleaned = clean_workload_data(workload)
    operation_cleaned = clean_operation_data(operation)
    ccm_cleaned = clean_ccm_data(ccm)

    print("\n2. 파생변수 생성 중...")
    workload_with_features = create_workload_features(workload_cleaned)
    operation_with_features = create_operation_features(operation_cleaned)

    print("\n3. 데이터셋 병합 중...")
    merged_data = merge_datasets(workload_with_features, operation_with_features, ccm_cleaned)

    if merged_data is None:
        print("데이터셋 병합 실패. 전처리를 중단합니다.")
        return None

    print("\n4. 이상치 제거 중...")
    data_without_outliers = remove_outliers(merged_data)

    print("\n5. 불필요한 열 제거 중...")
    # 예: '투입액량(L)'과 '목표온도'가 불필요한 변수라고 판단되면 제거
    excluded_columns = ['투입액량(L)', '목표온도']
    final_data = remove_unnecessary_columns(data_without_outliers, excluded_columns)

    print(f"\n전처리 완료. 최종 데이터셋: {final_data.shape[0]}행, {final_data.shape[1]}열")

    # 전처리된 데이터 저장
    final_data.to_csv('data/preprocessed_data.csv', index=False, encoding='utf-8-sig')
    print("전처리된 데이터가 'data/preprocessed_data.csv'에 저장되었습니다.")

    return final_data

if __name__ == "__main__":
    preprocessed_data = preprocess_data()
    if preprocessed_data is not None:
        print("\n전처리된 데이터 미리보기:")
        print(preprocessed_data.head())